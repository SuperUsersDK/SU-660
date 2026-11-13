"""Beregninger over alle vektorer: 2D-kort (t-SNE) og næsten-dubletter."""

from __future__ import annotations

import numpy as np


def project_2d(vectors: list[list[float]], *, seed: int = 42) -> list[tuple[float, float]]:
    """Projicér vektorerne ned i 2D med t-SNE og skalér til intervallet 0..1.

    t-SNE bevarer naboskaber: billeder, der ligger tæt i 512 dimensioner, lander
    tæt på kortet. Afstande mellem klynger og akserne betyder derimod ingenting.
    """
    from sklearn.manifold import TSNE

    data = np.asarray(vectors, dtype=np.float32)
    if len(data) < 3:
        return [(0.5, 0.5) for _ in range(len(data))]

    perplexity = min(30.0, (len(data) - 1) / 3)
    coords = TSNE(
        n_components=2,
        metric="cosine",
        init="pca",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(data)

    low = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - low, 1e-9)
    scaled = (coords - low) / span
    return [(float(x), float(y)) for x, y in scaled]


def weighted_position(
    positions: list[tuple[float, float]],
    weights: list[float],
) -> tuple[float, float]:
    """Placér en søgning på kortet som et vægtet gennemsnit af dens bedste hits.

    t-SNE kan ikke placere nye punkter, så søgeteksten får en tilnærmet position.
    """
    total = sum(weights) or 1.0
    x = sum(pos[0] * weight for pos, weight in zip(positions, weights)) / total
    y = sum(pos[1] * weight for pos, weight in zip(positions, weights)) / total
    return x, y


def find_duplicates(
    filenames: list[str],
    vectors: list[list[float]],
    *,
    threshold: float = 0.95,
    limit: int = 60,
) -> list[tuple[str, str, float]]:
    """Find billedpar med cosine-lighed over tærsklen, højeste først.

    Vektorerne er normaliserede, så et matrixprodukt giver alle cosine-ligheder på én gang.
    """
    if len(vectors) < 2:
        return []
    data = np.asarray(vectors, dtype=np.float32)
    similarity = data @ data.T
    rows, cols = np.triu_indices(len(data), k=1)
    scores = similarity[rows, cols]
    keep = scores >= threshold
    order = np.argsort(-scores[keep])[:limit]
    pairs_rows, pairs_cols, pairs_scores = rows[keep][order], cols[keep][order], scores[keep][order]
    return [
        (filenames[i], filenames[j], float(score))
        for i, j, score in zip(pairs_rows, pairs_cols, pairs_scores)
    ]


def mix_vectors(
    base: list[float],
    direction: list[float],
    weight: float,
    *,
    reference: list[list[float]] | None = None,
) -> list[float]:
    """Regning med embeddings: billede + vægt × tekst, normaliseret igen.

    Negativ vægt trækker billedet væk fra teksten ("minus solnedgang").

    Billede-billede-ligheder spreder sig 4-7 gange mere end tekst-billede-ligheder
    (tekst og billeder ligger forskudt i CLIP-rummet, "the modality gap"). Med
    `reference` (alle billedvektorer) skaleres begge led efter deres spredning over
    samlingen, så vægt 1 betyder, at billedet og teksten trækker lige meget.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    direction_arr = np.asarray(direction, dtype=np.float64)
    if reference is not None and len(reference) > 1:
        matrix = np.asarray(reference, dtype=np.float64)
        base_spread = float((matrix @ base_arr).std()) or 1.0
        direction_spread = float((matrix @ direction_arr).std()) or 1.0
        base_arr = base_arr / base_spread
        direction_arr = direction_arr / direction_spread
    mixed = base_arr + weight * direction_arr
    norm = np.linalg.norm(mixed)
    if norm == 0:
        return list(map(float, base))
    return [float(value) for value in mixed / norm]
