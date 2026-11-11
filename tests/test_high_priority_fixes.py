from __future__ import annotations

import contextlib
import io
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from chunking.common import chunk_fixed_chars, chunk_fixed_words


ROOT = Path(__file__).resolve().parents[1]
ingest = runpy.run_path(str(ROOT / "demoer/modul 2/big_demo/ingest.py"))
token_demo = runpy.run_path(str(ROOT / "demoer/modul 2/chunking_demo.py"))
token_solution = runpy.run_path(str(ROOT / "øvelser/modul 2/løsning/chunking_løsning.py"))


class ChunkWindowTests(unittest.TestCase):
    def test_invalid_windows_fail_before_processing_even_empty_documents(self):
        for size, overlap in [(0, 0), (-1, 0), (2, -1), (2, 2), (2, 3)]:
            for name, call in [
                ("words", lambda: chunk_fixed_words([], size, overlap)),
                ("chars", lambda: chunk_fixed_chars([], size, overlap)),
                ("token demo", lambda: token_demo["chunk_by_tokens"](
                    {"id": "x", "text": "one two three"}, chunk_tokens=size,
                    overlap_tokens=overlap, strategy_name="test")),
                ("token solution", lambda: token_solution["chunk_fixed_size"](
                    "one two three", chunk_tokens=size, overlap_tokens=overlap)),
            ]:
                with self.subTest(name=name, size=size, overlap=overlap):
                    with self.assertRaises(ValueError):
                        call()

    def test_valid_windows_preserve_overlap_and_final_content(self):
        doc = {"id": "x", "text": "one two three four five"}
        self.assertEqual(
            [c.text for c in chunk_fixed_words([doc], 3, 1)],
            ["one two three", "three four five"],
        )
        self.assertEqual(
            [c.text for c in chunk_fixed_chars([{"id": "x", "text": "abcdefg"}], 4, 1)],
            ["abcd", "defg"],
        )


class LocalEmbeddings(Embeddings):
    """Exercise real Qdrant ingestion without API keys, downloads, or network calls."""

    def embed_documents(self, texts):
        if any("EMBED_FAIL" in text for text in texts):
            raise RuntimeError("simulated embedding failure")
        return [[1.0, float(len(text) % 17 + 1)] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class IngestSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".ingest-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "docs"
        self.directory.mkdir()
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        self.collection = "test_sync"
        settings = SimpleNamespace(
            openai_api_key="test-only", qdrant_url="unused", qdrant_api_key="",
            collection_name=self.collection, embedding_model="local-test",
        )
        self.overrides = {
            "_get_settings": lambda: settings,
            "QdrantClient": lambda **kwargs: self.client,
            "OpenAIEmbeddings": lambda **kwargs: LocalEmbeddings(),
        }

    def sync(self, directory=None, force=False):
        with patch.dict(ingest["main"].__globals__, self.overrides):
            with contextlib.redirect_stdout(io.StringIO()):
                ingest["main"](str(directory or self.directory), force=force)

    def points(self):
        return self.client.scroll(self.collection, limit=1000, with_payload=True)[0]

    def write(self, name, text):
        (self.directory / name).write_text(text, encoding="utf-8")

    def test_edits_and_deletions_replace_old_chunks_including_force(self):
        for force in (False, True):
            with self.subTest(force=force):
                self.write("edited.md", "Original fact. " * 250)
                self.write("removed.md", "Deleted fact.")
                self.sync()
                old_ids = {p.id for p in self.points()}
                self.assertGreater(len(old_ids), 2)
                self.write("edited.md", "Replacement fact.")
                (self.directory / "removed.md").unlink()
                self.sync(force=force)
                points = self.points()
                self.assertEqual([p.payload["page_content"] for p in points], ["Replacement fact."])
                self.assertTrue(old_ids.isdisjoint({p.id for p in points}))

    def test_unchanged_run_is_idempotent(self):
        self.write("notes.md", "A useful fact.")
        self.sync()
        ids = {p.id for p in self.points()}
        self.sync()
        self.assertEqual(ids, {p.id for p in self.points()})

    def test_empty_directory_removes_its_last_document(self):
        self.write("notes.md", "A useful fact.")
        self.sync()
        (self.directory / "notes.md").unlink()
        self.sync()
        self.assertEqual(self.points(), [])

    def test_empty_source_file_removes_its_old_content(self):
        self.write("notes.md", "Original fact.")
        self.sync()
        self.write("notes.md", "")
        self.sync(force=True)
        self.assertEqual(self.points(), [])

    def test_initially_empty_directory_does_not_create_a_collection(self):
        self.sync()
        self.assertFalse(self.client.collection_exists(self.collection))

    def test_missing_directory_does_not_prune(self):
        self.write("notes.md", "A useful fact.")
        self.sync()
        ids = {p.id for p in self.points()}
        with self.assertRaises(RuntimeError):
            self.sync(self.directory / "missing")
        self.assertEqual(ids, {p.id for p in self.points()})

    def test_embedding_failure_preserves_previous_chunks(self):
        self.write("notes.md", "Original fact.")
        self.sync()
        ids = {p.id for p in self.points()}
        self.write("notes.md", "EMBED_FAIL")
        with self.assertRaisesRegex(RuntimeError, "simulated embedding failure"):
            self.sync()
        self.assertEqual(ids, {p.id for p in self.points()})

    def test_load_failure_after_a_successful_write_does_not_prune(self):
        self.write("a.md", "Original A.")
        self.write("b.md", "Original B.")
        self.sync()
        ids = {p.id for p in self.points()}
        self.write("a.md", "Replacement A.")
        original_loader = ingest["load_document_file"]

        def failing_loader(path):
            if path.name == "b.md":
                raise RuntimeError("simulated load failure")
            return original_loader(path)

        self.overrides["load_document_file"] = failing_loader
        with self.assertRaisesRegex(RuntimeError, "simulated load failure"):
            self.sync()
        self.assertTrue(ids.issubset({p.id for p in self.points()}))

    def test_directories_with_identical_filenames_are_isolated(self):
        self.write("notes.md", "Identical content.")
        other = Path(self.temp.name) / "other"
        other.mkdir()
        (other / "notes.md").write_text("Identical content.", encoding="utf-8")
        self.sync()
        self.sync(other)
        self.assertEqual(len(self.points()), 2)
        (self.directory / "notes.md").unlink()
        self.sync()
        points = self.points()
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].payload["metadata"]["ingest_root"], str(other.resolve()))

    def test_legacy_points_are_pruned_with_pagination_and_unknown_points_preserved(self):
        self.client.create_collection(
            self.collection, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        legacy = [models.PointStruct(
            id=str(uuid.uuid4()), vector=[1.0, 1.0],
            payload={"metadata": {"source": str(self.directory / "deleted.md")}},
        ) for _ in range(270)]
        unknown_id = str(uuid.uuid4())
        legacy.append(models.PointStruct(
            id=unknown_id, vector=[1.0, 1.0], payload={"metadata": {"source_file": "unknown.md"}},
        ))
        self.client.upsert(self.collection, legacy)
        self.sync()
        self.assertEqual([p.id for p in self.points()], [unknown_id])


if __name__ == "__main__":
    unittest.main()
