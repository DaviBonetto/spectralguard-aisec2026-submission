"""Verify the anonymous CLASP artifact exposes reproducibility essentials."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def listing(base: str, path: str = "") -> list[dict[str, object]]:
    url = f"{base}/files"
    if path:
        url += "?" + urllib.parse.urlencode({"path": path})
    request = urllib.request.Request(
        url, headers={"User-Agent": "SpectralGuard-Artifact-Audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="hispikes-91C0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = f"https://anonymous.4open.science/api/repo/{args.repo_id}"
    root = listing(base)
    code = listing(base, "hispikes")
    archive = listing(base, "archive")
    root_names = {str(item["name"]) for item in root}
    code_names = {str(item["name"]) for item in code}
    archive_names = {str(item["name"]) for item in archive}
    required = {
        "README.md": "README.md" in root_names,
        "uv.lock": "uv.lock" in root_names,
        "pyproject.toml": "pyproject.toml" in root_names,
        "feature_extraction.py": "feature_extraction.py" in code_names,
        "inject_hispa.py": "inject_hispa.py" in code_names,
        "SETUP.md": "SETUP.md" in archive_names,
        "xgboost_hyperparam.json": "xgboost_hyperparam.json" in archive_names,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": f"https://anonymous.4open.science/r/{args.repo_id}/",
        "api_base": base,
        "root_entry_count": len(root),
        "code_entry_count": len(code),
        "archive_entry_count": len(archive),
        "required_artifacts": required,
        "passed": all(required.values()),
        "phase_3_note": (
            "The Phase 1 audit established artifact availability only. The completed "
            "common-protocol comparison is released separately under results/phase3."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
