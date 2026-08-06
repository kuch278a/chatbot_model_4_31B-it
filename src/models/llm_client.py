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
        # 1. Move vision tower and embed_vision to GPU 0 if present
        if hasattr(self.model.model, "vision_tower") and self.model.model.vision_tower is not None:
            self.model.model.vision_tower.to("cuda:0")
            register_device_dispatch(self.model.model.vision_tower)
            
        if hasattr(self.model.model, "embed_vision") and self.model.model.embed_vision is not None:
            self.model.model.embed_vision.to("cuda:0")
            register_device_dispatch(self.model.model.embed_vision)
            
        # 2. Move embed_tokens to GPU 0 and register rotary embedding dispatch
        if hasattr(self.model.model.language_model, "embed_tokens") and self.model.model.language_model.embed_tokens is not None:
            self.model.model.language_model.embed_tokens.to("cuda:0")
            register_device_dispatch(self.model.model.language_model.embed_tokens)
            
        if hasattr(self.model.model.language_model, "rotary_emb") and self.model.model.language_model.rotary_emb is not None:
            register_device_dispatch(self.model.model.language_model.rotary_emb)
            
        # 3. Distribute 60 layers evenly
        layers = self.model.model.language_model.layers
        num_layers = len(layers)
        for i, layer in enumerate(layers):
            device = "cuda:0" if i < num_layers // 2 else "cuda:1"
            layer.to(device)
            register_device_dispatch(layer)
            
        # 4. Move norm and head to GPU 1
        if hasattr(self.model.model.language_model, "norm") and self.model.model.language_model.norm is not None:
            self.model.model.language_model.norm.to("cuda:1")
            register_device_dispatch(self.model.model.language_model.norm)
            
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            self.model.lm_head.to("cuda:1")
            register_device_dispatch(self.model.lm_head)
            
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
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        do_sample = kwargs.get("do_sample", True) if temperature > 0.0 else False
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id or 0
            )
            
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return response.strip()
