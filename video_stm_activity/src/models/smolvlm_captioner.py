# src/models/smolvlm_captioner.py
from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

from .base_vlm import BaseVLMCaptioner, make_bnb_4bit_config


class SmolVLMCaptioner(BaseVLMCaptioner):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        max_new_tokens: int = 120,
    ):
        super().__init__(model_id, device, dtype, load_in_4bit, max_new_tokens)

        model_kwargs = {
            "device_map": "auto" if self.device != "cpu" else None,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
        }

        if self.load_in_4bit and self.device != "cpu":
            model_kwargs["quantization_config"] = make_bnb_4bit_config(self.torch_dtype)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model.eval()

    def caption(self, image_bgr, prompt: str) -> str:
        rgb = image_bgr[:, :, ::-1]
        img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = "<image>\n" + prompt

        inputs = self.processor(text=text, images=img, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out_ids[0][prompt_len:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()