from __future__ import annotations

import io
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from chunking.common import chunk_fixed_words, keyword_score, retrieve


ROOT = Path(__file__).resolve().parents[1]


class ClassroomDemoTests(unittest.TestCase):
    def test_exercise_can_load_the_workshop_document(self):
        exercise = runpy.run_path(str(ROOT / "øvelser/modul 2/chunking_template.py"))
        self.assertTrue(exercise["load_document"]().strip())

    def test_termination_question_can_retrieve_its_policy_evidence(self):
        demo = runpy.run_path(str(ROOT / "chunking/07_parameter_grid_eval.py"))
        case = demo["QUERY_SET"][-1]
        chunks = chunk_fixed_words(demo["BENCHMARK_DOCS"], chunk_words=18, overlap_words=4)
        hits = retrieve(case["question"], chunks, top_k=3)
        self.assertIn("policy_md", [chunk.doc_id for chunk, _ in hits])
        self.assertGreater(keyword_score(case["question"], hits, case["expected_keywords"]), 0)

    def test_photo_configuration_precedes_model_use(self):
        events = []

        def configure(*args, **kwargs):
            events.append("env")
            os.environ["IMAGE_EMBED_MODEL"] = "classroom-test-model"
            return True

        def embed(image):
            self.assertEqual(os.getenv("IMAGE_EMBED_MODEL"), "classroom-test-model")
            self.assertEqual(image.size, (8, 8))
            events.append("embed")
            return [1.0, 0.0]

        def query(**kwargs):
            events.append("query")
            return SimpleNamespace(points=[])

        photo_dir = ROOT / "demoer/modul 3/photos"
        with patch.dict(os.environ), patch.object(sys, "path", [str(photo_dir), *sys.path]):
            with patch("dotenv.load_dotenv", side_effect=configure):
                module = runpy.run_path(str(photo_dir / "app.py"))
            globals_ = module["search"].__globals__
            with patch.dict(globals_, {
                "image_to_vector": embed,
                "QdrantClient": lambda **kwargs: SimpleNamespace(query_points=query),
            }):
                with TestClient(module["app"]) as client:
                    self.assertIn("classroom-test-model", client.get("/").text)
                    image_data = io.BytesIO()
                    Image.new("RGB", (8, 8), "blue").save(image_data, format="JPEG")
                    response = client.post("/search", files={
                        "image": ("sample.jpg", image_data.getvalue(), "image/jpeg"),
                    })
                    self.assertEqual(response.status_code, 200)
        self.assertEqual(events, ["env", "embed", "query"])


if __name__ == "__main__":
    unittest.main()
