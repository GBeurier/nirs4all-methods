# Developer documentation

The public developer documentation describes the supported library:

- [Architecture](../ARCHITECTURE.md) and [architecture overview](../architecture/overview.md).
- [Build](build.md), [testing](testing.md), and [release process](release_process.md).
- [ABI policy](../abi/stability_policy.md) and [serialization](../architecture/serialization.md).
- [Contribution workflow](workflow.md) and the repository's `CONTRIBUTING.md`.

Private specifications, implementation plans, investigation logs, review transcripts,
release checklists, and design prototypes belong in the ignored `.private-dev/`
directory at the repository root. Its `README.md` indexes current work and dated
archives. It is intentionally absent from the published documentation and source
distribution. Do not force-add private notes.

The 2026-09-05 archive preserves original relative paths and contents, including
the historical phase roadmaps, merge evidence, namespace and dashboard proposals,
and finetuning plans. These records explain past decisions; released source,
public contracts, and passing release gates define current behavior.

Some historical files remain public because active tools consume them as
contracts: `proposals/namespace/_rename_map.tsv` supports generation and ABI
doc-lint; the three `docs/architecture/aom_robust_hpo_*.csv` manifests and
`roadmap/phase-7f-aom-robust-hpo-portfolio.md` are inputs to the AOM integration
contract tests. Their contents remain unchanged from the published baseline.
They are executable acceptance evidence, not disposable planning notes; archival
copies are retained privately as well.
