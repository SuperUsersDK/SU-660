"""De 10 eksempelkategorier, som fetch_sample_photos.py henter, og som taggingen bruger.

Kategorierne er valgt, så de er visuelt meget forskellige og danner tydelige klynger
på kortet i 2D og 3D. Prompten er den engelske sætning, CLIP matcher billederne mod,
både når scriptet frasorterer tvivlsomme billeder, og når billederne tagges.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleCategory:
    tag: str
    prompt: str
    commons_categories: tuple[str, ...]
    color: str


SAMPLE_CATEGORIES: tuple[SampleCategory, ...] = (
    SampleCategory("bil", "a photo of a car", ("Quality images of automobiles",), "#d32f2f"),
    SampleCategory("fly", "a photo of an airplane", ("Quality images of aircraft",), "#1976d2"),
    SampleCategory("skib", "a photo of a ship on the water", ("Quality images of ships",), "#00838f"),
    SampleCategory("fugl", "a photo of a bird", ("Quality images of birds",), "#f9a825"),
    SampleCategory("insekt", "a close-up photo of an insect", ("Quality images of insects",), "#455a64"),
    SampleCategory("blomst", "a close-up photo of a flower", ("Quality images of flowers",), "#c2185b"),
    SampleCategory("svamp", "a photo of a mushroom", ("Quality images of fungi",), "#8d6e63"),
    SampleCategory("kirke", "a photo of a church building", ("Quality images of churches",), "#5e35b1"),
    SampleCategory("mad", "a photo of food on a plate", ("Quality images of food",), "#ef6c00"),
    SampleCategory("bjerg", "a photo of a mountain landscape", ("Quality images of mountains",), "#2e7d32"),
)
