import os
import time

# Ensure offline loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    Build a module-path -> device map that balances actual weight bytes
    across GPUs, keeping shared modules (embedding, vision tower, head) on
    cuda:0 and the final norm on cuda:1.
    """
    lm = model.model.language_model
    load = {device: 0.0 for device in devices}
    device_map = {}

    def place(module, path):
        nonlocal device
        device = "cuda:0"
        device_map[path] = devices[0]
        load[devices[0]] += module_bytes(module)

    vision_tower = getattr(model.model, "vision_tower", None)
    if vision_tower is not None:
        device_map["model.vision_tower"] = devices[0]
        load[devices[0]] += module_bytes(vision_tower)

    embed_vision = getattr(model.model, "embed_vision", None)
    if embed_vision is not None:
        device_map["model.embed_vision"] = devices[0]
        load[devices[0]] += module_bytes(embed_vision)

    embed_tokens = getattr(lm, "embed_tokens", None)
    if embed_tokens is not None:
        device_map["model.language_model.embed_tokens"] = devices[0]
        load[devices[0]] += module_bytes(embed_tokens)

    if getattr(lm, "rotary_emb", None) is not None:
        device_map["model.language_model.rotary_emb"] = devices[0]

    norm = getattr(lm, "norm", None)
    if norm is not None:
        device_map["model.language_model.norm"] = devices[1]
        load[devices[1]] += module_bytes(norm)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        device_map["lm_head"] = devices[0]

    layers = lm.layers
    layer_sizes = [module_bytes(layer) for layer in layers]
    for i, (layer, size) in enumerate(zip(layers, layer_sizes)):
        device = min(load, key=load.get)
        device_map[f"model.language_model.layers.{i}"] = device
        load[device] += size

    print(
        f"Balanced device map: cuda:0={sum(v == devices[0] for v in device_map.values())} modules "
        f"({load[devices[0]] / 1e9:.1f}GB), cuda:1={sum(v == devices[1] for v in device_map.values())} modules "
        f"({load[devices[1]] / 1e9:.1f}GB)"
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

        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        do_sample = kwargs.get("do_sample", True) if temperature > 0.0 else False

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id or 0,
            )

        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return response.strip()
