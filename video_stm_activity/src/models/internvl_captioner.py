# src/models/internvl_captioner.py
from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

from .base_vlm import BaseVLMCaptioner, resolve_torch_dtype


class InternVLCaptioner(BaseVLMCaptioner):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,   # InternVL2: keep False
        max_new_tokens: int = 120,
    ):
        super().__init__(model_id, device, dtype, load_in_4bit, max_new_tokens)

        torch_dtype = resolve_torch_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # ✅ Use an IMAGE processor (AutoProcessor may resolve to a tokenizer here)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=None,          # avoid meta tensors
            low_cpu_mem_usage=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def caption(self, image_bgr, prompt: str) -> str:
        rgb = image_bgr[:, :, ::-1]
        img = Image.fromarray(rgb.astype(np.uint8))

        # Convert PIL -> pixel_values tensor
        inputs = self.image_processor(images=img, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise RuntimeError(f"InternVL image_processor returned keys {list(inputs.keys())}, expected 'pixel_values'")

        pixel_values = inputs["pixel_values"].to(self.device)

        # Match model dtype (important for bfloat16/fp16)
        if pixel_values.dtype != self.model.dtype:
            pixel_values = pixel_values.to(self.model.dtype)

        gen_cfg = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": False,
        }

        with torch.no_grad():
            out = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config=gen_cfg,
            )

        return str(out).strip()