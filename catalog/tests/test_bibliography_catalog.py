"""Regression tests for the lossless, offline bibliography migration."""

from __future__ import annotations

import json
import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "catalog" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bibliography_catalog as catalog  # noqa: E402
import generate_bibliography as generated  # noqa: E402
import validate_bibliography as review  # noqa: E402


class BibliographyCatalogTest(unittest.TestCase):
    def test_actual_baseline_is_lossless_and_reviewable(self) -> None:
        data = catalog.build_catalog()
        legacy = catalog.extract_legacy_bibliography()
        self.assertEqual(len(data["publications"]), len(legacy) + 15)
        self.assertEqual(
            len([e for e in data["publications"] if e["provenance"]["fields_status"] == "unstructured_legacy"]),
            len(legacy),
        )
        self.assertEqual(
            len([e for e in data["publications"] if e["provenance"]["fields_status"] == "reviewed_structured"]),
            15,
        )
        self.assertEqual(review.validate(data), [])
        self.assertEqual(data["publications"][0]["method_key"], sorted(legacy)[0])
        self.assertIsNone(data["publications"][0]["doi"])

    def test_ast_extraction_accepts_existing_public_fixture(self) -> None:
        fixture = (
            ROOT / "catalog" / "tests" / "fixtures" / "legacy_bibliography_excerpt.py"
        )
        extracted = catalog.extract_legacy_bibliography(fixture)
        self.assertEqual(list(extracted), ["pls", "pcr"])
        self.assertIn("SIMPLS", extracted["pls"]["paper"])

    def test_rendering_and_offline_report_are_deterministic(self) -> None:
        data = catalog.build_catalog()
        first = generated.outputs(data)
        second = generated.outputs(json.loads(catalog.canonical_json(data)))
        self.assertEqual(first, second)
        report = json.loads(first[generated.LINK_REPORT])
        self.assertEqual(report["network_access"], "not_attempted")
        self.assertTrue(
            all(item["offline_syntax_valid"] for item in report["checked_links"])
        )
        self.assertEqual(
            report["missing_structured_links"], 2 * len(data["publications"])
        )
        self.assertEqual(report["legacy_citations_scanned"], len(data["publications"]))
        self.assertTrue(
            any(item["field"] == "legacy_doi" for item in report["checked_links"])
        )

    def test_migration_reads_the_requested_git_blob_not_the_worktree(self) -> None:
        revision = catalog.source_revision()
        data = catalog.build_catalog(revision=revision)
        legacy_blob = catalog.git_blob_text(revision, catalog.LEGACY_PATH)
        self.assertEqual(data["baseline"]["source_revision"], revision)
        self.assertEqual(
            data["baseline"]["legacy_python_sha256"],
            catalog.sha256_text(legacy_blob),
        )
        self.assertEqual(
            data["publications"],
            catalog.build_catalog(revision=revision)["publications"],
        )

    def test_shallow_clone_reports_required_history_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            origin = root / "origin"
            shallow = root / "shallow"
            origin.mkdir()
            subprocess.run(["git", "init"], cwd=origin, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=origin,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Bibliography Test"],
                cwd=origin,
                check=True,
            )
            tracked = origin / "source.txt"
            tracked.write_text("first\n", encoding="utf-8")
            subprocess.run(["git", "add", "source.txt"], cwd=origin, check=True)
            subprocess.run(["git", "commit", "-m", "first"], cwd=origin, check=True)
            first = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=origin,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            tracked.write_text("second\n", encoding="utf-8")
            subprocess.run(["git", "commit", "-am", "second"], cwd=origin, check=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", f"file://{origin}", str(shallow)],
                check=True,
                capture_output=True,
            )
            with self.assertRaisesRegex(catalog.ProvenanceError, "fetch-depth: 0"):
                catalog.git_blob_text(first, Path("source.txt"), repo=shallow)

    def test_schema_enforces_const_min_length_and_array_items(self) -> None:
        data = catalog.build_catalog()
        schema = json.loads(
            (ROOT / "catalog" / "schema" / "bibliography_v1.json").read_text(
                encoding="utf-8"
            )
        )
        entry = data["publications"][0]
        entry["provenance"]["fields_status"] = "reviewed"
        entry["method_key"] = ""
        entry["authors"] = [""]
        errors = review.validate_schema(data, schema)
        self.assertTrue(any("oneOf" in error for error in errors))
        self.assertTrue(any("minLength" in error for error in errors))

    def test_offline_report_scans_raw_legacy_urls_and_dois(self) -> None:
        data = catalog.build_catalog()
        data["publications"][0]["legacy_citation"] = (
            "Legacy source https://example.invalid/path. doi:10.1234/example."
        )
        report = generated.link_report(data)
        found = {
            (item["field"], item["value"])
            for item in report["checked_links"]
            if item["id"] == data["publications"][0]["id"]
        }
        self.assertIn(("legacy_url", "https://example.invalid/path"), found)
        self.assertIn(("legacy_doi", "10.1234/example"), found)
        data["publications"][0]["legacy_citation"] = "Broken https://"
        self.assertFalse(
            generated.link_report(data)["checked_links"][0]["offline_syntax_valid"]
        )

    def test_generated_bibtex_contains_importable_lossless_note_records(self) -> None:
        data = catalog.build_catalog()
        bibtex = generated.render_bibtex(data)
        self.assertEqual(bibtex.count("@misc{"), 73)
        self.assertIn("howpublished = {Historical method bibliography catalog}", bibtex)
        self.assertIn("note = {", bibtex)
        self.assertNotRegex(bibtex, r"(?m)^%.*@")

    def test_generated_bibtex_is_accepted_by_installed_bibtex(self) -> None:
        """Exercise BibTeX itself, not a regex approximation of its grammar."""
        bibtex_binary = shutil.which("bibtex")
        self.assertIsNotNone(bibtex_binary, "BibTeX must be installed for this gate")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "bibliography.bib").write_text(
                generated.render_bibtex(catalog.build_catalog()), encoding="utf-8"
            )
            (root / "bibliography.aux").write_text(
                "\\relax\n\\citation{*}\n\\bibstyle{plain}\n\\bibdata{bibliography}\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [bibtex_binary, "bibliography"], cwd=root, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "bibliography.bbl").exists())

    def test_generated_page_preformats_legacy_citations(self) -> None:
        page = generated.render_page(catalog.build_catalog())
        self.assertNotIn("orphan: true", page)
        self.assertTrue(page.startswith("# Global bibliography"))
        self.assertIn("````text", page)
        self.assertIn("avoid MyST auto-linkification", page)
        self.assertIn("Source provenance", page)
        self.assertIn("parity/REFERENCES.md", page)
        self.assertIn("Reviewed finetuning references", page)

    def test_rendered_reviewed_doi_and_url_are_clickable(self) -> None:
        data = copy.deepcopy(catalog.build_catalog())
        reviewed = next(
            entry
            for entry in data["publications"]
            if entry["provenance"]["fields_status"] == "reviewed_structured"
        )
        reviewed["doi"] = "10.1234/reviewed-example"
        reviewed["url"] = "https://example.invalid/reviewed"
        page = generated.render_page(data)
        self.assertIn("[DOI: 10.1234/reviewed-example](https://doi.org/10.1234/reviewed-example)", page)
        self.assertIn("[URL](https://example.invalid/reviewed)", page)

    def test_review_rejects_invented_doi(self) -> None:
        data = catalog.build_catalog()
        data["publications"][0]["doi"] = "10.1000/example"
        self.assertTrue(
            any(
                "unreviewed structured citation fact" in error
                for error in review.validate(data)
            )
        )

    def test_review_rejects_tampered_entry_provenance_source(self) -> None:
        data = catalog.build_catalog()
        data["publications"][0]["provenance"]["source"] = "invented"
        self.assertTrue(
            any("provenance source" in error for error in review.validate(data))
        )

    def test_review_rejects_reviewed_field_or_entry_digest_tampering(self) -> None:
        data = catalog.build_catalog()
        reviewed = next(
            entry
            for entry in data["publications"]
            if entry["provenance"]["fields_status"] == "reviewed_structured"
        )
        reviewed["title"] = "Invented title"
        reviewed["provenance"]["entry_sha256"] = "0" * 64
        reviewed["provenance"]["source_sha256"] = "0" * 64
        errors = review.validate(data)
        self.assertTrue(any("does not match its reviewed BibTeX entry" in error for error in errors))
        self.assertTrue(any("entry digest" in error for error in errors))
        self.assertTrue(any("source digest" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
