# Phase 1 foundation-audit results

This directory contains machine-generated, lightweight evidence only:

- `reference_audit.json` is produced by `scripts/audit_references.py` and records citation coverage plus official DataCite/arXiv metadata comparisons.
- `mamba_130m_kappa.json` is produced by `src/utils/measure_mamba_kappa.py` from the immutable model revision declared in `configs/reproducibility.yaml`.
- `anonymity_audit.json` is produced by `scripts/audit_anonymity.ps1` and checks tracked content, Git authors, remotes, and PDF author metadata.
- `legacy_reference_inventory.json` is produced by `scripts/audit_legacy_tex.py` from the official arXiv source and records every orphaned legacy reference.
- `clasp_artifact_audit.json` is produced by `scripts/audit_clasp_artifact.py` and verifies the public anonymous CLASP artifact endpoint and essential files.
- `docker_validation.json` records an observed build identifier, the pinned base-image digest, and the containerized test result. The image identifier is informational because BuildKit manifest metadata may vary across rebuilds.

No historical headline metric is imported into this directory.
