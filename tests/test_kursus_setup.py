from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import kursus_setup as setup  # noqa: E402


class KursusSetupTests(unittest.TestCase):
    def test_tiktoken_cache_is_checked_by_url_hash(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": tmp}):
            self.assertEqual(setup.tiktoken_missing(), ["o200k_base", "cl100k_base"])
            url = setup.TIKTOKEN_FILES["o200k_base"]
            (Path(tmp) / hashlib.sha1(url.encode()).hexdigest()).write_text("x")
            self.assertEqual(setup.tiktoken_missing(), ["cl100k_base"])

    def test_scan_points_to_menu_item_when_qdrant_is_down(self):
        with patch.object(setup, "qdrant_get", return_value=None):
            checks = {c.name: c for c in setup.scan(check_openai=False)}
        self.assertEqual(checks["Qdrant"].status, setup.MISSING)
        self.assertEqual(checks["Qdrant"].fix, "2")
        # Uden Qdrant kan collections ikke tjekkes, så de springes over i stedet for at fejle.
        self.assertNotIn("Billeder i Qdrant", checks)

    def test_scan_reports_missing_photos_in_qdrant(self):
        def fake_get(path, timeout=3):
            if path == "/collections":
                return {"result": {"collections": []}}
            return {"status": 404}

        with patch.object(setup, "qdrant_get", side_effect=fake_get), \
                patch.object(setup, "photo_files", return_value=[Path(f"bil_{i:03d}_x.jpg") for i in range(10)]):
            checks = {c.name: c for c in setup.scan(check_openai=False)}
        self.assertEqual(checks["Billeder i Qdrant"].status, setup.MISSING)
        self.assertEqual(checks["Billeder i Qdrant"].fix, "5")
        self.assertEqual(checks["RAG-dokumenter (modul 2)"].fix, "6")

    def test_env_is_read_even_without_python_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("# kommentar\nKURSUS_TEST_NOEGLE='abc'\n", encoding="utf-8")
            real_import = __import__

            def no_dotenv(name, *args, **kwargs):
                if name == "dotenv":
                    raise ModuleNotFoundError(name)
                return real_import(name, *args, **kwargs)

            with patch.object(setup, "ENV_FILE", env), patch("builtins.__import__", side_effect=no_dotenv), \
                    patch.dict(os.environ, {}, clear=False):
                setup.load_env()
                self.assertEqual(os.environ.get("KURSUS_TEST_NOEGLE"), "abc")

    def test_selection_accepts_numbers_ranges_and_all(self):
        self.assertEqual(setup.parse_selection("1,3", 4), [0, 2])
        self.assertEqual(setup.parse_selection(" 3 1 ", 4), [0, 2])
        self.assertEqual(setup.parse_selection("alle", 3), [0, 1, 2])
        self.assertIsNone(setup.parse_selection("5", 4))
        self.assertIsNone(setup.parse_selection("x", 4))

    def _cleanup_with(self, answers):
        deleted = []
        items = [
            setup.CleanupItem("Model a", 10, "", lambda: deleted.append("a")),
            setup.CleanupItem("Qdrant: b", 0, "", lambda: deleted.append("b")),
        ]
        replies = iter(answers)
        with patch.object(setup, "cleanup_items", return_value=items), patch.object(setup, "photo_files", return_value=[]):
            setup.cleanup(ask=lambda prompt: next(replies))
        return deleted

    def test_cleanup_deletes_only_after_explicit_yes(self):
        self.assertEqual(self._cleanup_with(["2", "ja"]), ["b"])
        self.assertEqual(self._cleanup_with(["alle", "ja"]), ["a", "b"])

    def test_cleanup_deletes_nothing_without_confirmation(self):
        self.assertEqual(self._cleanup_with([""]), [])            # Enter = fortryd
        self.assertEqual(self._cleanup_with(["alle", "nej"]), [])
        self.assertEqual(self._cleanup_with(["alle", ""]), [])
        self.assertEqual(self._cleanup_with(["9"]), [])            # ugyldigt valg

    def test_cleanup_never_lists_foreign_qdrant_collections(self):
        def fake_get(path, timeout=3):
            if path == "/collections":
                names = ["modul3_photos_demo", "chunk_lab__x", "rag_data", "kundens_data", "modul3_demo"]
                return {"result": {"collections": [{"name": n} for n in names]}}
            return {"result": {"points_count": 1}}

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": tmp}), \
                patch.object(setup, "qdrant_get", side_effect=fake_get), \
                patch.object(setup, "hf_repo_dir", return_value=None), \
                patch.object(setup, "photo_files", return_value=[]):
            labels = [item.label for item in setup.cleanup_items()]
        self.assertNotIn("Qdrant: kundens_data", labels)
        self.assertIn("Qdrant: rag_data", labels)
        self.assertIn("Qdrant: modul3_demo", labels)

    def test_menu_rejects_unknown_choice(self):
        self.assertFalse(setup.run_choice("42"))


if __name__ == "__main__":
    unittest.main()
