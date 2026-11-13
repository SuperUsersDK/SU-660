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

from analysis import find_duplicates, project, project_2d, weighted_position  # noqa: E402
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

    def test_weighted_position_prefers_heavy_hits(self):
        x, y = weighted_position([(0.0, 0.0), (1.0, 1.0)], [3.0, 1.0])
        self.assertAlmostEqual(x, 0.25)
        self.assertAlmostEqual(y, 0.25)

    def test_3d_tsne_is_centered_in_unit_cube(self):
        rng = random.Random(1)
        vectors = [unit(*[rng.gauss(0, 1) for _ in range(8)]) for _ in range(25)]
        projection = project(vectors, dims=3, method="tsne")
        self.assertEqual({len(p) for p in projection.coords}, {3})
        self.assertTrue(all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for p in projection.coords for v in p))
        self.assertIsNone(projection.transform(vectors[0]))

    def test_pca_reports_variance_and_places_new_points_exactly(self):
        rng = random.Random(2)
        vectors = [unit(*[rng.gauss(0, 1) for _ in range(8)]) for _ in range(30)]
        projection = project(vectors, dims=3, method="pca")
        self.assertEqual(len(projection.explained_variance), 3)
        self.assertTrue(0 < sum(projection.explained_variance) <= 1)
        self.assertGreaterEqual(projection.explained_variance[0], projection.explained_variance[1])
        # Et kendt punkt placeres samme sted, som det fik i projektionen.
        for got, expected in zip(projection.transform(vectors[4]), projection.coords[4]):
            self.assertAlmostEqual(got, expected)

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


class SamplePhotoTests(unittest.TestCase):
    def test_clip_filter_keeps_only_confident_own_category(self):
        from fetch_sample_photos import accept

        self.assertTrue(accept([0.1, 0.8, 0.1], own_index=1, min_confidence=0.6))
        self.assertFalse(accept([0.1, 0.5, 0.4], own_index=1, min_confidence=0.6))
        self.assertFalse(accept([0.7, 0.2, 0.1], own_index=1, min_confidence=0.1))

    def test_cleanup_removes_only_files_made_by_the_script(self):
        import tempfile

        from fetch_sample_photos import remove_previous_samples

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("bil_001_Volvo.jpg", "fugl_050_Ugle.jpg", "IMG_4160.JPG", "bil_ferie.jpg"):
                (folder / name).write_bytes(b"x")
            self.assertEqual(remove_previous_samples(folder), 2)
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ["IMG_4160.JPG", "bil_ferie.jpg"])

    def test_resume_keeps_first_images_and_numbers_them_without_gaps(self):
        import tempfile

        from fetch_sample_photos import existing_samples, keep_first

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("bil_001_A.jpg", "bil_003_C.jpg", "bil_004_D.jpg", "bil_005_E.jpg", "fly_001_X.jpg"):
                (folder / name).write_bytes(b"x")
            kept = keep_first(existing_samples(folder, "bil"), 3)
            self.assertEqual([p.name for p in kept], ["bil_001_A.jpg", "bil_002_C.jpg", "bil_003_D.jpg"])
            self.assertEqual(
                sorted(p.name for p in folder.iterdir()),
                ["bil_001_A.jpg", "bil_002_C.jpg", "bil_003_D.jpg", "fly_001_X.jpg"],
            )

    def test_label_set_follows_environment(self):
        from sample_categories import SAMPLE_CATEGORIES
        from tagging import get_labels

        with patch.dict(os.environ, {"PHOTO_LABEL_SET": ""}):
            self.assertEqual(list(get_labels()), [c.tag for c in SAMPLE_CATEGORIES])
        with patch.dict(os.environ, {"PHOTO_LABEL_SET": "dykning"}):
            self.assertIn("vrag", get_labels())
        with patch.dict(os.environ, {"PHOTO_LABEL_SET": "ukendt"}):
            with self.assertRaises(ValueError):
                get_labels()

    def test_sample_categories_are_distinct(self):
        from sample_categories import SAMPLE_CATEGORIES

        self.assertEqual(len(SAMPLE_CATEGORIES), 10)
        self.assertEqual(len({c.tag for c in SAMPLE_CATEGORIES}), 10)
        self.assertEqual(len({c.color for c in SAMPLE_CATEGORIES}), 10)


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

    def _upload(self, tag: str = "") -> str:
        import io
        import re

        image_data = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(image_data, format="JPEG")
        response = self.client.post(
            "/search",
            files={"image": ("sample.jpg", image_data.getvalue(), "image/jpeg")},
            data={"tag": tag},
        )
        self.assertEqual(response.status_code, 200)
        match = re.search(r'name="upload_id" value="([0-9a-f]+)"', response.text)
        self.assertIsNotNone(match)
        return match.group(1)

    def test_changing_tag_reuses_uploaded_image_without_reembedding(self):
        embeds = []
        with patch.dict(self.globals, {"image_to_vector": lambda image: embeds.append(1) or unit(1, 0, 0)}):
            upload_id = self._upload()
            response = self.client.post("/search", data={"upload_id": upload_id, "tag": "fisk"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(embeds), 1)
        condition = self.fake.queries[-1]["filter"].must[0]
        self.assertEqual(condition.match.value, "fisk")
        self.assertIn(f"/upload/{upload_id}", response.text)
        self.assertEqual(self.client.get(f"/upload/{upload_id}").status_code, 200)

    def test_text_search_remembers_upload_and_image_search_remembers_text(self):
        with patch.dict(self.globals, {"image_to_vector": lambda image: unit(1, 0, 0)}):
            upload_id = self._upload(tag="vrag")
            text_page = self.client.post(
                "/search-text",
                data={"query": "a fish", "tag": "fisk", "upload_id": upload_id, "image_tag": "vrag"},
            ).text
            image_page = self.client.post(
                "/search",
                data={"upload_id": upload_id, "tag": "", "query": "a fish", "text_tag": "fisk"},
            ).text

        for page in (text_page, image_page):
            self.assertIn(f'name="upload_id" value="{upload_id}"', page)
            self.assertIn('name="query" value="a fish"', page)
        self.assertIn("<option value='fisk' selected>", text_page)
        self.assertIn("<option value='vrag' selected>", text_page)

    def test_forgotten_upload_asks_for_a_new_image(self):
        response = self.client.post("/search", data={"upload_id": "unknown", "tag": "fisk"})
        self.assertIn("Vælg et billede at uploade", response.text)
        self.assertEqual(self.fake.queries, [])

    def test_3d_pca_map_has_three_coordinates_and_exact_query_position(self):
        self.fake.points["d.jpg"] = (unit(0, 0, 1), "koralrev")
        self.fake.points["e.jpg"] = (unit(0.5, 0.5, 0.7), "dykker")
        self.globals["_MAP_CACHE"].clear()
        data = self.client.get("/api/map", params={"mode": "3d-pca"}).json()
        self.assertEqual(data["dims"], 3)
        self.assertEqual(len(data["explained"]), 3)
        self.assertEqual({len(p["pos"]) for p in data["points"]}, {3})

        found = self.client.get("/api/map/search", params={"q": "fish", "mode": "3d-pca", "k": 2}).json()
        self.assertTrue(found["exact"])
        self.assertEqual(len(found["position"]), 3)
        self.assertEqual(found["hits"][0]["filename"], "c.jpg")

    def test_unknown_map_mode_is_rejected(self):
        self.assertEqual(self.client.get("/api/map", params={"mode": "4d"}).status_code, 400)

    def test_map_page_uses_local_three_js(self):
        page = self.client.get("/map").text
        self.assertIn('"three": "/static/three/three.module.min.js"', page)
        self.assertNotIn("cdn.", page)
        self.assertEqual(self.client.get("/static/three/three.module.min.js").status_code, 200)
        self.assertEqual(self.client.get("/static/three/addons/controls/OrbitControls.js").status_code, 200)

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
