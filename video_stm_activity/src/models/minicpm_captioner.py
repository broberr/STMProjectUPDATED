# src/models/minicpm_captioner.py
from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

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
            "trust_remote_code": True,
            "device_map": "auto" if device != "cpu" else None,
        }

        if load_in_4bit and device != "cpu":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    def caption(self, image_bgr, prompt: str) -> str:
        rgb = image_bgr[:, :, ::-1]
        img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")

        # ✅ force float pixel_values using the model’s processor
        proc = self.processor(images=img, return_tensors="pt")

        # MiniCPM processors usually return pixel_values
        if "pixel_values" in proc:
            pixel_values = proc["pixel_values"]
        elif "image" in proc:
            pixel_values = proc["image"]
        else:
            raise RuntimeError(f"MiniCPM processor returned keys {list(proc.keys())}; expected pixel_values")

        # Keep on CPU float32 to avoid Byte CPU ops (model will move what it needs)
        pixel_values = pixel_values.float().cpu()

        msgs = [{"role": "user", "content": prompt}]

        with torch.no_grad():
            # Many MiniCPM checkpoints accept either image=... or pixel_values=...
            try:
                out = self.model.chat(
                    pixel_values=pixel_values,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                )
            except TypeError:
                # fallback: some builds still use image=
                out = self.model.chat(
                    image=pixel_values,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                )

        return str(out).strip()