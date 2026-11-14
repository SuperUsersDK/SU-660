from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "demoer/modul 2/chunking_lab"
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from chunkers import STRATEGIES, run_strategy  # noqa: E402

TEXTS = json.loads((LAB_DIR / "texts/texts.json").read_text(encoding="utf-8"))


def sample(text_id: str) -> str:
    item = next(t for t in TEXTS if t["id"] == text_id)
    return (LAB_DIR / "texts" / item["file"]).read_text(encoding="utf-8")


class ChunkerTests(unittest.TestCase):
    def test_every_chunk_points_to_its_exact_place_in_the_text(self):
        for item in TEXTS:
            text = sample(item["id"])
            for strategy in STRATEGIES:
                try:
                    chunks = run_strategy(strategy.id, text)
                except ValueError:
                    continue
                with self.subTest(text=item["id"], strategy=strategy.id):
                    self.assertTrue(chunks)
                    for chunk in chunks:
                        self.assertEqual(text[chunk.start:chunk.end], chunk.text)
                        self.assertEqual(chunk.text, chunk.text.strip())

    def test_every_sample_answer_exists_in_its_text(self):
        for item in TEXTS:
            text = sample(item["id"])
            for question in item["questions"]:
                self.assertIn(question["expected"], text, question["question"])

    def test_fixed_size_respects_size_and_overlap_repeats_text(self):
        text = "abcdefghij " * 40
        fixed = run_strategy("fixed", text, {"size": 100})
        self.assertTrue(all(len(c.text) <= 100 for c in fixed))
        overlap = run_strategy("overlap", text, {"size": 100, "overlap": 30})
        self.assertGreater(len(overlap), len(fixed))
        self.assertLess(overlap[1].start, overlap[0].end)

    def test_sentences_never_cut_inside_a_sentence(self):
        text = "Første sætning er her. Anden sætning kommer så! Tredje? Fjerde sætning slutter teksten."
        for chunk in run_strategy("sentences", text, {"max_chars": 100}):
            self.assertIn(chunk.text[-1], ".!?")

    def test_markdown_sections_keep_heading_path(self):
        chunks = run_strategy("markdown", sample("rejsepolitik"))
        car = next(c for c in chunks if c.text.startswith("### Kørsel i egen bil"))
        self.assertEqual(car.meta["heading"], "Rejse- og udgiftspolitik › Transport › Kørsel i egen bil")
        self.assertNotIn("## Hotel", car.text)

    def test_python_source_gives_one_chunk_per_function_or_class(self):
        chunks = run_strategy("code", sample("lager"))
        names = [c.meta["name"] for c in chunks]
        self.assertEqual(names, ["imports og konstanter", "Vare", "pris_med_moms", "beregn_rabat", "genbestillingspunkt", "Lager"])
        self.assertTrue(chunks[1].text.startswith("@dataclass"))

    def test_structure_strategies_refuse_unsuitable_text(self):
        with self.assertRaises(ValueError):
            run_strategy("markdown", "Bare almindelig tekst uden overskrifter.")
        with self.assertRaises(ValueError):
            run_strategy("code", "Dette er ikke Python (")

    def test_parameters_are_clamped_to_allowed_range(self):
        chunks = run_strategy("fixed", "x" * 500, {"size": 1})
        self.assertEqual(len(chunks[0].text), 50)


class ChunkLabApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util

        # Foto-demoen har også en app.py, så modulet indlæses under et entydigt navn.
        spec = importlib.util.spec_from_file_location("chunking_lab_app", LAB_DIR / "app.py")
        lab = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = lab
        with patch("dotenv.load_dotenv", return_value=True):
            spec.loader.exec_module(lab)
        cls.lab = lab
        cls.client = TestClient(lab.app)

    def test_setup_lists_texts_and_strategies(self):
        data = self.client.get("/api/setup").json()
        self.assertEqual(len(data["texts"]), 4)
        self.assertEqual([s["id"] for s in data["strategies"]], [s.id for s in STRATEGIES])

    def test_chunk_preview_needs_no_embedding(self):
        with patch.object(self.lab.embedder, "embed_passages", side_effect=AssertionError("må ikke embedde")):
            data = self.client.post("/api/chunk", json={"text_id": "faq", "strategy": "paragraphs"}).json()
        self.assertEqual(data["stats"]["chunks"], len(data["chunks"]))

    def test_custom_text_keeps_its_exact_characters(self):
        text = "\n\n  Første afsnit.\n\nAndet afsnit.\n"
        data = self.client.post("/api/chunk", json={"text": text, "strategy": "paragraphs"}).json()
        self.assertEqual([text[c["start"]:c["end"]] for c in data["chunks"]], ["Første afsnit.", "Andet afsnit."])

    def test_unsuitable_strategy_is_reported_not_crashed(self):
        response = self.client.post("/api/chunk", json={"text_id": "faq", "strategy": "code"})
        self.assertEqual(response.status_code, 422)
        self.assertTrue(response.json()["not_applicable"])

    def test_projection_places_known_vector_inside_map(self):
        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]]
        projection = self.lab.Projection.fit(vectors)
        for vector in vectors:
            self.assertTrue(all(-1e-9 <= v <= 1 + 1e-9 for v in projection.transform(vector)))


if __name__ == "__main__":
    unittest.main()
