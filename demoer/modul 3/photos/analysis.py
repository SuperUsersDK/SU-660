"""Beregninger over alle vektorer: kort i 2D/3D (t-SNE/PCA) og næsten-dubletter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Projection:
    """Koordinater for hvert billede, plus det der skal til for at placere nye punkter.

    `coords` er skaleret til 0..1 pr. akse i 2D og centreret i -1..1 (samme skala på
    alle akser) i 3D, så PCA's proportioner bevares.
    """

    coords: list[tuple[float, ...]]
    method: str
    explained_variance: list[float] | None = None
    # Kun PCA kan placere nye punkter præcist: (x - mean) @ components.T, derefter skalering.
    mean: np.ndarray | None = None
    components: np.ndarray | None = None
    offset: np.ndarray | None = None
    scale: np.ndarray | None = None

    def transform(self, vector: list[float]) -> tuple[float, ...] | None:
        if self.components is None:
            return None
        raw = (np.asarray(vector, dtype=np.float64) - self.mean) @ self.components.T
        return tuple(float(v) for v in (raw - self.offset) / self.scale)


def _scale(raw: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dims == 2:
        offset = raw.min(axis=0)
        scale = np.maximum(raw.max(axis=0) - offset, 1e-9)
    else:
        offset = (raw.max(axis=0) + raw.min(axis=0)) / 2
        scale = np.full(raw.shape[1], max(float(np.abs(raw - offset).max()), 1e-9))
    return (raw - offset) / scale, offset, scale


def tsne_perplexity(n: int) -> float:
    """Perplexity er groft sagt, hvor mange naboer hvert punkt tager hensyn til.

    Den skal være mindre end en typisk klynge, ellers flyder klyngerne sammen.
    n/20 giver fx 10 for 200 billeder (20 pr. kategori) og 30 for store samlinger.
    """
    return float(min(30.0, max(5.0, n / 20), (n - 1) / 3))


def project(
    vectors: list[list[float]],
    *,
    dims: int = 2,
    method: str = "tsne",
    seed: int = 42,
) -> Projection:
    """Projicér vektorerne ned i 2 eller 3 dimensioner.

    t-SNE bevarer naboskaber: billeder, der ligger tæt i 512 dimensioner, lander
    tæt på kortet. Afstande mellem klynger og akserne betyder derimod ingenting.

    PCA er en lineær projektion på de akser, hvor vektorerne varierer mest. Akser og
    afstande betyder noget, men klynger smelter ofte sammen, fordi kun en lille del af
    variationen kan vises i 2-3 akser.
    """
    if dims not in (2, 3):
        raise ValueError("dims skal være 2 eller 3")
    if method not in ("tsne", "pca"):
        raise ValueError("method skal være 'tsne' eller 'pca'")

    data = np.asarray(vectors, dtype=np.float64)
    if len(data) < dims + 1:
        center = 0.5 if dims == 2 else 0.0
        return Projection(coords=[(center,) * dims for _ in range(len(data))], method=method)

    if method == "pca":
        mean = data.mean(axis=0)
        _, singular, vt = np.linalg.svd(data - mean, full_matrices=False)
        components = vt[:dims]
        raw = (data - mean) @ components.T
        variance = singular**2
        explained = [float(v) for v in variance[:dims] / variance.sum()]
        scaled, offset, scale = _scale(raw, dims)
        return Projection(
            coords=[tuple(float(v) for v in row) for row in scaled],
            method=method,
            explained_variance=explained,
            mean=mean,
            components=components,
            offset=offset,
            scale=scale,
        )

    from sklearn.manifold import TSNE

    raw = TSNE(
        n_components=dims,
        metric="cosine",
        init="pca",
        perplexity=tsne_perplexity(len(data)),
        random_state=seed,
    ).fit_transform(data.astype(np.float32))
    scaled, _, _ = _scale(raw.astype(np.float64), dims)
    return Projection(coords=[tuple(float(v) for v in row) for row in scaled], method=method)


def project_2d(vectors: list[list[float]], *, seed: int = 42) -> list[tuple[float, float]]:
    """2D t-SNE skaleret til 0..1 (bruges af det oprindelige kort)."""
    return project(vectors, dims=2, method="tsne", seed=seed).coords  # type: ignore[return-value]


def weighted_position(
    positions: list[tuple[float, ...]],
    weights: list[float],
) -> tuple[float, ...]:
    """Placér en søgning på kortet som et vægtet gennemsnit af dens bedste hits.

    t-SNE kan ikke placere nye punkter, så søgeteksten får en tilnærmet position.
    """
    total = sum(weights) or 1.0
    dims = len(positions[0]) if positions else 2
    return tuple(
        sum(pos[axis] * weight for pos, weight in zip(positions, weights)) / total
        for axis in range(dims)
    )


def find_duplicates(
    filenames: list[str],
    vectors: list[list[float]],
    *,
    threshold: float = 0.9,
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
