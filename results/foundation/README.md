# Phase 1 foundation-audit results

This directory contains machine-generated, lightweight evidence only:

- `reference_audit.json` is produced by `scripts/audit_references.py` and records citation coverage plus official DataCite/arXiv metadata comparisons.
- `mamba_130m_kappa.json` is produced by `src/utils/measure_mamba_kappa.py` from the immutable model revision declared in `configs/reproducibility.yaml`.
- `anonymity_audit.json` is produced by `scripts/audit_anonymity.ps1` and checks tracked content, Git authors, remotes, and PDF author metadata.
- `legacy_reference_inventory.json` is produced by `scripts/audit_legacy_tex.py` from the official arXiv source and records every orphaned legacy reference.
- `clasp_artifact_audit.json` is produced by `scripts/audit_clasp_artifact.py` and verifies the public anonymous CLASP artifact endpoint and essential files.
- `anonymous_repository_audit.json` is produced by `scripts/audit_anonymous_repository.py` and checks the public review mirror for required files, planning-material leakage, and identity terms.
- `anonymous_quickstart_reproduction.json` records a clean download, locked environment creation, and independent execution of the published Phase 1 quickstart.
- `docker_validation.json` records an observed build identifier, the pinned base-image digest, and the containerized test result. The image identifier is informational because BuildKit manifest metadata may vary across rebuilds.
- `phase8_figure_manifest.json` binds the four publication vector PDFs to the exact tracked input files and SHA-256 hashes used to generate them.

No historical headline metric is imported into this directory.
