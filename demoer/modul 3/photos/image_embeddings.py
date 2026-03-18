from __future__ import annotations

import math
import os
from functools import lru_cache

from PIL import Image, ImageOps


def get_image_embed_model() -> str:
    return os.getenv("IMAGE_EMBED_MODEL", "openai/clip-vit-base-patch32")


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


@lru_cache(maxsize=1)
def load_model():
    try:
        import torch
        from transformers import AutoProcessor, CLIPModel
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Mangler dependencies til rigtige image embeddings. "
            "Installer mindst 'torch' og 'transformers'."
        ) from exc

    model_name = get_image_embed_model()
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return model, processor, torch


def image_to_vector(image: Image.Image) -> list[float]:
    model, processor, torch = load_model()

    image = ImageOps.exif_transpose(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    vector = features[0].cpu().tolist()
    return l2_normalize([float(value) for value in vector])


def text_to_vector(text: str) -> list[float]:
    model, processor, torch = load_model()

    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    vector = features[0].cpu().tolist()
    return l2_normalize([float(value) for value in vector])
