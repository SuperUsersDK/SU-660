from __future__ import annotations

import math
import os
import re
import threading
from functools import lru_cache

from PIL import Image, ImageOps

# CLIP-processoren skalerer den korteste side til 224 px og skærer et kvadrat
# ud fra midten. Både ViT-B/32 og ViT-L/14 bruger 224 px.
MODEL_INPUT_SIZE = 224


def get_image_embed_model() -> str:
    return os.getenv("IMAGE_EMBED_MODEL", "openai/clip-vit-base-patch32")


def get_compare_embed_model() -> str:
    """Den ekstra model, som /compare sammenligner med standardmodellen."""
    return os.getenv("COMPARE_IMAGE_EMBED_MODEL", "openai/clip-vit-large-patch14")


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


# Agenten kan kalde flere tools parallelt, og webappen kører forespørgsler i tråde.
# To samtidige indlæsninger af samme model fejler i PyTorch, så indlæsningen låses.
_LOAD_LOCK = threading.Lock()


def load_model(model_name: str | None = None):
    # Navnet slås op før cachen, så en ændret IMAGE_EMBED_MODEL ikke rammer
    # en gammel cachet model.
    with _LOAD_LOCK:
        return _load_model(model_name or get_image_embed_model())


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        import torch
        from transformers import AutoProcessor, CLIPModel
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Mangler dependencies til rigtige image embeddings. "
            "Installer mindst 'torch' og 'transformers'."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return model, processor, torch


def prepare_image(image: Image.Image) -> Image.Image:
    """Drej efter EXIF-orientering og sørg for præcis tre farvekanaler."""
    return ImageOps.exif_transpose(image).convert("RGB")


def image_to_vector(image: Image.Image, model_name: str | None = None) -> list[float]:
    model, processor, torch = load_model(model_name)

    inputs = processor(images=prepare_image(image), return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    vector = features[0].cpu().tolist()
    return l2_normalize([float(value) for value in vector])


def texts_to_vectors(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Embed tekster i samme CLIP-rum som billederne, så tekst kan søge i billeder."""
    model, processor, torch = load_model(model_name)

    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    return [
        l2_normalize([float(value) for value in row])
        for row in features.cpu().tolist()
    ]


def text_to_vector(text: str, model_name: str | None = None) -> list[float]:
    return texts_to_vectors([text], model_name)[0]


def patch_size(model_name: str | None = None) -> int:
    """Feltstørrelsen står i modelnavnet, fx 'patch32' eller 'patch14'."""
    match = re.search(r"patch(\d+)", model_name or get_image_embed_model())
    return int(match.group(1)) if match else 32


def model_input_preview(image: Image.Image) -> tuple[Image.Image, float]:
    """Returnér det 224×224 udsnit, modellen ser, og hvor stor en andel af billedet det er.

    Efterligner CLIP-processoren: korteste side skaleres til 224 px (bicubic),
    og der skæres et kvadrat ud fra midten. Alt uden for kvadratet ser modellen aldrig.
    """
    image = prepare_image(image)
    width, height = image.size
    crop = ImageOps.fit(
        image,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    shortest = min(width, height)
    kept_fraction = (shortest * shortest) / (width * height) if width and height else 1.0
    return crop, kept_fraction
