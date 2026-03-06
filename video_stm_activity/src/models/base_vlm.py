# src/models/base_vlm.py
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from transformers import BitsAndBytesConfig


def resolve_torch_dtype(dtype: str):
    if not dtype:
        return None
    d = dtype.lower().strip()
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16", "half"):
        return torch.float16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def make_bnb_4bit_config(torch_dtype: torch.dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )


class BaseVLMCaptioner(ABC):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        max_new_tokens: int = 120,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = resolve_torch_dtype(dtype)
        self.load_in_4bit = bool(load_in_4bit) and device != "cpu"
        self.max_new_tokens = int(max_new_tokens)

    @abstractmethod
    def caption(self, image_bgr, prompt: str) -> str:
        raise NotImplementedError

    def close(self):
        return