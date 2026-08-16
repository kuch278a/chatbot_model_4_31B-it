import os
import sys

# Ensure offline loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Pillow Resampling monkeypatch
import PIL.Image
if not hasattr(PIL.Image, "Resampling"):
    class ResamplingMock:
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 1
        BOX = 4
        HAMMING = 5
    PIL.Image.Resampling = ResamplingMock

import warnings
warnings.filterwarnings("ignore", message=".*calling .generate\\(\\) with the `input_ids` being on a device type different.*")

import torch
import types
from transformers import AutoProcessor, AutoModelForMultimodalLM
from .base import BaseLLMClient

def register_device_dispatch(module):
    original_forward = module.forward
    def custom_forward(self, *args, **kwargs):
        # 1. Determine target device: first check parameters
        device = None
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # If no parameters, find device from inputs (args/kwargs)
            def find_tensor_device(x):
                if isinstance(x, torch.Tensor):
                    return x.device
                elif isinstance(x, list) or isinstance(x, tuple):
                    for item in x:
                        dev = find_tensor_device(item)
                        if dev is not None:
                            return dev
                elif isinstance(x, dict):
                    for v in x.values():
                        dev = find_tensor_device(v)
                        if dev is not None:
                            return dev
                return None
                
            for arg in args:
                device = find_tensor_device(arg)
                if device is not None:
                    break
            if device is None:
                for v in kwargs.values():
                    device = find_tensor_device(v)
                    if device is not None:
                        break
                        
        if device is None:
            # Fallback
            return original_forward(*args, **kwargs)
        
        # 2. Ensure all raw tensor attributes, parameters, and buffers are on the correct device
        for name in dir(self):
            try:
                attr = getattr(self, name)
                if isinstance(attr, torch.Tensor) and attr.device != device:
                    # Move to device
                    if name in self._buffers:
                        self._buffers[name] = attr.to(device)
                    elif name in self._parameters:
                        self._parameters[name] = torch.nn.Parameter(attr.to(device))
                    else:
                        setattr(self, name, attr.to(device))
            except Exception:
                pass
                
        if hasattr(self, "embed_scale") and isinstance(self.embed_scale, torch.Tensor):
            self.embed_scale = self.embed_scale.to(device)
                
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif isinstance(x, list):
                return [to_device(item) for item in x]
            elif isinstance(x, tuple):
                return tuple(to_device(item) for item in x)
            elif isinstance(x, dict):
                return {k: to_device(v) for k, v in x.items()}
            return x

        new_args = [to_device(arg) for arg in args]
        new_kwargs = {k: to_device(v) for k, v in kwargs.items()}
        return original_forward(*new_args, **new_kwargs)

    module.forward = types.MethodType(custom_forward, module)


class Gemma4LLMClient(BaseLLMClient):
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Initializing Gemma4LLMClient from: {model_path}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        
        # Load model on CPU
        print("Loading model weights onto CPU RAM...")
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        # Partition model across GPUs
        print("Partitioning model layers across GPU 0 and GPU 1...")
        self._shard_model()
        
    def _shard_model(self):
        device0, device1 = "cuda:0", "cuda:1"

        def param_bytes(module):
            total = 0
            for p in module.parameters():
                total += p.numel() * p.element_size()
            for b in module.buffers():
                total += b.numel() * b.element_size()
            return total

        # Keep vision tower modules on CPU to save 1.14GB of GPU VRAM
        if hasattr(self.model.model, "vision_tower") and self.model.model.vision_tower is not None:
            # Keep on CPU
            pass

        if hasattr(self.model.model, "embed_vision") and self.model.model.embed_vision is not None:
            # Keep on CPU
            pass

        embed_tokens = getattr(self.model.model.language_model, "embed_tokens", None)
        if embed_tokens is not None:
            embed_tokens.to(device0)
            register_device_dispatch(embed_tokens)

        if hasattr(self.model.model.language_model, "rotary_emb") and self.model.model.language_model.rotary_emb is not None:
            register_device_dispatch(self.model.model.language_model.rotary_emb)

        norm = getattr(self.model.model.language_model, "norm", None)
        if norm is not None:
            norm.to(device1)
            register_device_dispatch(norm)

        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:
            lm_head.to(device0)
            register_device_dispatch(lm_head)

        # Assign layers 0..28 (29 layers) to cuda:0 and layers 29..59 (31 layers) to cuda:1
        layers = self.model.model.language_model.layers
        device_counts = {device0: 0, device1: 0}
        for i, layer in enumerate(layers):
            target_device = device0 if i <= 28 else device1
            layer.to(target_device)
            register_device_dispatch(layer)
            device_counts[target_device] += 1

        print(
            f"Layer balancing complete: {device0}={device_counts[device0]} layers (31.04GB, ~700MB headroom), "
            f"{device1}={device_counts[device1]} layers (30.36GB, ~1.4GB headroom), vision=cpu (saved 1.14GB VRAM)"
        )
        print("Model sharding completed successfully.")

    def generate(self, prompt, system_prompt=None, history=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        if history:
            for role, content in history:
                # Map roles correctly to what template expects (user / assistant)
                role_mapped = "assistant" if role in ["assistant", "bot", "model"] else "user"
                messages.append({"role": role_mapped, "content": content})
                
        messages.append({"role": "user", "content": prompt})
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Send inputs to GPU 0 (since the first layers are on cuda:0)
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        
        # Run generation
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
                pad_token_id=self.processor.tokenizer.pad_token_id or 0
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
            enable_thinking=False
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
            pad_token_id=self.processor.tokenizer.pad_token_id or 0
        )

        def _run():
            with torch.inference_mode():
                self.model.generate(**generation_kwargs)

        thread = Thread(target=_run)
        thread.start()

        for new_text in streamer:
            yield new_text

        torch.cuda.empty_cache()
