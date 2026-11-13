from __future__ import annotations

import math
import os
from pathlib import Path
import random
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PHOTO_DIR = ROOT / "demoer/modul 3/photos"
if str(PHOTO_DIR) not in sys.path:
    sys.path.insert(0, str(PHOTO_DIR))

from analysis import find_duplicates, mix_vectors, project_2d, weighted_position  # noqa: E402
from image_embeddings import model_input_preview, patch_size  # noqa: E402
from photo_store import COLLECTION_NAME, collection_for_model  # noqa: E402
from tagging import classify, softmax  # noqa: E402


def unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


class FakeQdrant:
    """Minimal Qdrant-erstatning med scroll og query_points over faste punkter."""

    def __init__(self, points: dict[str, tuple[list[float], str]]):
        self.points = points
        self.queries: list[dict] = []

    def _point(self, name: str, with_vectors: bool) -> SimpleNamespace:
        vector, tag = self.points[name]
        return SimpleNamespace(
            id=name,
            payload={"filename": name, "tag": tag},
            vector=vector if with_vectors else None,
        )

    def scroll(self, *, collection_name, limit=10, offset=None, with_payload=True, with_vectors=False, scroll_filter=None):
        names = sorted(self.points)
        if scroll_filter is not None:
            wanted = scroll_filter.must[0].match.value
            names = [name for name in names if name == wanted]
        return [self._point(name, with_vectors) for name in names], None

    def query_points(self, *, collection_name, query, limit, with_payload, query_filter=None):
        self.queries.append({"query": query, "filter": query_filter})
        excluded = set()
        if query_filter is not None:
            excluded = {cond.match.value for cond in query_filter.must_not or []}
        scored = [
            SimpleNamespace(
                payload={"filename": name, "tag": tag},
                score=sum(a * b for a, b in zip(vector, query)),
            )
            for name, (vector, tag) in self.points.items()
            if name not in excluded
        ]
        scored.sort(key=lambda hit: -hit.score)
        return SimpleNamespace(points=scored[:limit])

    def collection_exists(self, name):
        return name == COLLECTION_NAME


class AnalysisTests(unittest.TestCase):
    def test_duplicates_are_found_above_threshold_highest_first(self):
        names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        vectors = [unit(1, 0, 0), unit(1, 0.01, 0), unit(1, 0.2, 0), unit(0, 0, 1)]
        pairs = find_duplicates(names, vectors, threshold=0.97)
        self.assertEqual([(a, b) for a, b, _ in pairs], [("a.jpg", "b.jpg"), ("b.jpg", "c.jpg"), ("a.jpg", "c.jpg")])
        self.assertTrue(all(pairs[i][2] >= pairs[i + 1][2] for i in range(len(pairs) - 1)))

    def test_mix_moves_towards_text_and_stays_normalized(self):
        base, text = unit(1, 0), unit(0, 1)
        mixed = mix_vectors(base, text, 0.5)
        self.assertAlmostEqual(math.hypot(*mixed), 1.0)
        self.assertGreater(mixed[1], 0)
        away = mix_vectors(base, text, -0.5)
        self.assertLess(away[1], 0)

    def test_mix_with_reference_balances_weak_text_signal(self):
        # Billederne spreder sig kraftigt langs x, men kun svagt langs y (teksten).
        reference = [unit(1, 0.05 * i) for i in range(-3, 4)] + [unit(-1, 0.05)]
        base, text = unit(1, 0), unit(0, 1)
        raw = mix_vectors(base, text, 1.0)
        balanced = mix_vectors(base, text, 1.0, reference=reference)
        self.assertGreater(balanced[1], raw[1])

    def test_weighted_position_prefers_heavy_hits(self):
        x, y = weighted_position([(0.0, 0.0), (1.0, 1.0)], [3.0, 1.0])
        self.assertAlmostEqual(x, 0.25)
        self.assertAlmostEqual(y, 0.25)

    def test_projection_returns_one_point_per_vector_in_unit_square(self):
        rng = random.Random(0)
        vectors = [unit(*[rng.gauss(0, 1) for _ in range(8)]) for _ in range(25)]
        coords = project_2d(vectors)
        self.assertEqual(len(coords), 25)
        self.assertTrue(all(0.0 <= v <= 1.0 for point in coords for v in point))


class EmbeddingHelperTests(unittest.TestCase):
    def test_model_preview_is_center_crop_of_shortest_side(self):
        crop, kept = model_input_preview(Image.new("RGB", (300, 200), "blue"))
        self.assertEqual(crop.size, (224, 224))
        self.assertAlmostEqual(kept, 2 / 3)

    def test_patch_size_is_read_from_model_name(self):
        self.assertEqual(patch_size("openai/clip-vit-base-patch32"), 32)
        self.assertEqual(patch_size("openai/clip-vit-large-patch14"), 14)

    def test_each_model_has_its_own_collection(self):
        self.assertEqual(collection_for_model("openai/clip-vit-base-patch32"), COLLECTION_NAME)
        self.assertEqual(
            collection_for_model("openai/clip-vit-large-patch14"),
            f"{COLLECTION_NAME}__openai_clip_vit_large_patch14",
        )


class TaggingTests(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(softmax([1.0, 2.0, 3.0])), 1.0)

    def test_each_image_gets_the_closest_label(self):
        labels = [unit(1, 0), unit(0, 1)]
        predictions = classify([unit(1, 0.1), unit(0.1, 1)], labels, ["vrag", "fisk"])
        self.assertEqual([tag for tag, _ in predictions], ["vrag", "fisk"])
        self.assertTrue(all(score > 0.5 for _, score in predictions))


class PhotoAppRouteTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeQdrant({
            "a.jpg": (unit(1, 0, 0), "vrag"),
            "b.jpg": (unit(1, 0.01, 0), "vrag"),
            "c.jpg": (unit(0, 1, 0), "fisk"),
        })
        with patch.dict(os.environ), patch("dotenv.load_dotenv", return_value=True):
            self.module = runpy.run_path(str(PHOTO_DIR / "app.py"))
        self.globals = self.module["index"].__globals__
        self.patches = patch.dict(self.globals, {
            "QdrantClient": lambda **kwargs: self.fake,
            "text_to_vector": lambda text, model_name=None: unit(0, 1, 0),
            "library_path": lambda name: Path(name),
        })
        self.patches.start()
        self.client = TestClient(self.module["app"])

    def tearDown(self):
        self.patches.stop()

    def test_text_search_can_filter_on_tag(self):
        response = self.client.post("/search-text", data={"query": "a fish", "tag": "fisk"})
        self.assertEqual(response.status_code, 200)
        condition = self.fake.queries[-1]["filter"].must[0]
        self.assertEqual((condition.key, condition.match.value), ("tag", "fisk"))

    def test_duplicates_page_lists_near_identical_pair(self):
        response = self.client.get("/duplicates?threshold=0.95")
        self.assertEqual(response.status_code, 200)
        self.assertIn("1 par over 0.95", response.text)

    def test_mix_excludes_the_base_image(self):
        response = self.client.get("/mix?base=a.jpg&text=fish&weight=1")
        self.assertEqual(response.status_code, 200)
        excluded = self.fake.queries[-1]["filter"].must_not[0].match.value
        self.assertEqual(excluded, "a.jpg")
        self.assertIn("c.jpg", response.text)

    def test_compare_explains_missing_collection(self):
        response = self.client.get("/compare")
        self.assertIn("findes ikke endnu", response.text)
        self.assertIn("--model", response.text)


class ThumbnailSafetyTests(unittest.TestCase):
    def test_paths_outside_data_dir_are_rejected(self):
        with patch.dict(os.environ), patch("dotenv.load_dotenv", return_value=True):
            module = runpy.run_path(str(PHOTO_DIR / "app.py"))
        client = TestClient(module["app"])
        self.assertEqual(client.get("/thumb/..%2F..%2F..%2F..%2F.env").status_code, 404)
        self.assertEqual(client.get("/model-view", params={"file": "../../../../.env"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
