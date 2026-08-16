import os
import time

# Ensure offline loading and memory configuration
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", message=".*calling .generate\\(\\) with the `input_ids` being on a device type different.*")

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from accelerate import dispatch_model
from .base import BaseLLMClient


def module_bytes(module):
    """Total size in bytes of a module's parameters and buffers."""
    return (
        sum(p.numel() * p.element_size() for p in module.parameters())
        + sum(b.numel() * b.element_size() for b in module.buffers())
    )


def build_balanced_device_map(model, devices=("cuda:0", "cuda:1")):
    """
    Build a module-path -> device map that places vision tower on CPU
    (since text chat does not require image processing) and balances the 60
    language model layers across cuda:0 and cuda:1 with safe headroom on both GPUs.
    """
    lm = model.model.language_model
    device_map = {}

    # Keep vision tower modules on CPU to save 1.14GB of GPU VRAM
    vision_tower = getattr(model.model, "vision_tower", None)
    if vision_tower is not None:
        device_map["model.vision_tower"] = "cpu"

    embed_vision = getattr(model.model, "embed_vision", None)
    if embed_vision is not None:
        device_map["model.embed_vision"] = "cpu"

    embed_tokens = getattr(lm, "embed_tokens", None)
    if embed_tokens is not None:
        device_map["model.language_model.embed_tokens"] = devices[0]

    if getattr(lm, "rotary_emb", None) is not None:
        device_map["model.language_model.rotary_emb"] = devices[0]

    norm = getattr(lm, "norm", None)
    if norm is not None:
        device_map["model.language_model.norm"] = devices[1]

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        device_map["lm_head"] = devices[0]

    # Assign layers 0..28 (29 layers) to cuda:0 and layers 29..59 (31 layers) to cuda:1
    layers = lm.layers
    for i in range(len(layers)):
        if i <= 28:
            device_map[f"model.language_model.layers.{i}"] = devices[0]
        else:
            device_map[f"model.language_model.layers.{i}"] = devices[1]

    print(
        f"Balanced device map: cuda:0 (embed + 29 layers = 31.04GB, ~700MB headroom), "
        f"cuda:1 (31 layers + norm = 30.36GB, ~1.4GB headroom), vision=cpu (saved 1.14GB VRAM)"
    )
    return device_map


class AcceleratedGemma4LLMClient(BaseLLMClient):
    """
    Gemma4 31B-it client that uses Hugging Face Accelerate for loading and
    cross-GPU dispatch instead of manual layer sharding.

    Benefits:
      - Faster load: weights are dispatched once to their target devices.
      - Faster inference: Accelerate's AlignDevicesHook only moves the
        tensors that cross a device boundary, replacing the per-forward
        attribute scan of the manual sharder.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Initializing AcceleratedGemma4LLMClient from: {model_path}")

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        print(f"Processor loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        print("Loading model weights (streamed to CPU RAM)...")
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        print(f"Model weights loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        print("Building balanced device map and dispatching across GPUs...")
        device_map = build_balanced_device_map(self.model)
        dispatch_model(self.model, device_map=device_map)
        print(f"Device dispatch completed in {time.time() - t0:.1f}s")

    def generate(self, prompt, system_prompt=None, history=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for role, content in history:
                role_mapped = "assistant" if role in ["assistant", "bot", "model"] else "user"
                messages.append({"role": role_mapped, "content": content})

        messages.append({"role": "user", "content": prompt})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # The first layers (embed_tokens, vision tower) live on cuda:0
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        max_new_tokens = kwargs.get("max_new_tokens", 160)
        temperature = kwargs.get("temperature", 0.2)
        do_sample = kwargs.get("do_sample", True) if temperature > 0.0 else False

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id or 0,
            )

        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        del inputs, outputs
        torch.cuda.empty_cache()
        return response.strip()

    def generate_stream(self, prompt, system_prompt=None, history=None, **kwargs):
        from transformers import TextIteratorStreamer
        from threading import Thread

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for role, content in history:
                role_mapped = "assistant" if role in ["assistant", "bot", "model"] else "user"
                messages.append({"role": role_mapped, "content": content})

        messages.append({"role": "user", "content": prompt})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        max_new_tokens = kwargs.get("max_new_tokens", 160)
        temperature = kwargs.get("temperature", 0.2)
        do_sample = kwargs.get("do_sample", True) if temperature > 0.0 else False

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id or 0,
        )

        def _run():
            with torch.inference_mode():
                self.model.generate(**generation_kwargs)

        thread = Thread(target=_run)
        thread.start()

        for new_text in streamer:
            yield new_text

        torch.cuda.empty_cache()

