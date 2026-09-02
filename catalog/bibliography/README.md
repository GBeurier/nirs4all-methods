# Bibliography catalog

`methods_bibliography_v1.json` is a generated, versioned migration of the
historical `docs/_extras/methods_bibliography.py` dictionary. The old source
contains prose `paper` strings, not verified structured records. Schema v1
therefore retains each string verbatim and sets authors, publication title,
year, journal, DOI, URL, and BibTeX to `null` with an explicit
`unstructured_legacy` provenance marker.

This is intentional: supplying plausible metadata from memory would weaken the
scientific record. Add structured fields only with a reviewed source and keep
the original string/provenance. `CITATION.cff` is separately preserved verbatim
as the software citation. Donor/parity source digests pin the existing public
`parity/REFERENCES.md` and registry lockfile without duplicating their mapping.

Rebuild and review locally (all commands are offline):

```bash
python3 catalog/scripts/migrate_bibliography.py
python3 catalog/scripts/generate_bibliography.py
python3 catalog/scripts/validate_bibliography.py
python3 catalog/scripts/migrate_bibliography.py --check
python3 catalog/scripts/generate_bibliography.py --check
```

Rollback is a normal source rollback: delete the four generated docs artifacts
and restore this catalog/schema/scripts together. The legacy Python dictionary
remains untouched and is still read by the existing methods-page builder, so
this migration is backward-readable and has no runtime compatibility switch.
