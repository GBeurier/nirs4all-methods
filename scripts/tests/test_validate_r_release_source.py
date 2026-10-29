"""A packaging recovery must never silently select a different numerical core."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("r_release_guard", Path(__file__).parents[1] / "validate_r_release_source.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class RecoverySourceTest(unittest.TestCase):
    def test_repair_preserves_tag_and_refuses_numeric_or_wrong_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()

            git("init")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.invalid")
            desc = repo / "bindings/r/n4m/DESCRIPTION"
            desc.parent.mkdir(parents=True)
            desc.write_text("Package: n4m\nVersion: 1.3.10\n")
            git("add", ".")
            git("commit", "-m", "immutable source")
            git("tag", "v1.3.10")
            desc.write_text(desc.read_text() + "Suggests: jsonlite\n")
            git("add", ".")
            git("commit", "-m", "declare test dependency")
            expected = git("rev-parse", "HEAD")
            self.assertEqual(guard.validate(repo, "v1.3.10", expected)["packaging_source"], expected)
            with self.assertRaises(ValueError):
                guard.validate(repo, "v1.3.10", "0" * 40)
            with self.assertRaises(ValueError):
                guard.validate(repo, "main", expected)
            (repo / "kernel.cpp").write_text("different numerical source\n")
            git("add", ".")
            git("commit", "-m", "forbidden numerical change")
            with self.assertRaisesRegex(ValueError, "outside R packaging"):
                guard.validate(repo, "v1.3.10", git("rev-parse", "HEAD"))


if __name__ == "__main__":
    unittest.main()
