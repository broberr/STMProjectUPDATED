# src/models/minicpm_captioner.py
from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .base_vlm import BaseVLMCaptioner, resolve_torch_dtype


class MiniCPMCaptioner(BaseVLMCaptioner):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        max_new_tokens: int = 120,
    ):
        super().__init__(model_id, device, dtype, load_in_4bit, max_new_tokens)

        torch_dtype = resolve_torch_dtype(dtype)

        model_kwargs = {
            "device_map": "auto" if device != "cpu" else None,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if load_in_4bit and device != "cpu":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    def caption(self, image_bgr, prompt: str) -> str:
        rgb = image_bgr[:, :, ::-1]
        img = Image.fromarray(rgb.astype(np.uint8))

        msgs = [{"role": "user", "content": prompt}]
        with torch.no_grad():
            # Many MiniCPM checkpoints expose .chat(image=..., msgs=..., tokenizer=...)
            out = self.model.chat(image=img, msgs=msgs, tokenizer=self.tokenizer)
        return str(out).strip()