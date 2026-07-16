"""Audit the public anonymous mirror without GitHub credentials."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_PATHS = {
    "README.md",
    "uv.lock",
    "paper/main.tex",
    "paper/references.bib",
    "scripts/verify_foundation_result.py",
    "src/utils/measure_mamba_kappa.py",
    "results/foundation/mamba_130m_kappa.json",
}
IDENTITY_PATTERNS = (
    "Davi" + "Bonetto",
    "Davi" + " Bonetto",
    "Davi" + "b",
    "davi.bonetto100" + "@gmail.com",
    "github.com/" + "DaviBonetto",
    "C:\\Users\\" + "Davib",
)


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SpectralGuard-Anonymous-Audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def list_files(base: str) -> list[str]:
    queue = [""]
    files: list[str] = []
    while queue:
        path = queue.pop(0)
        url = f"{base}/files"
        if path:
            url += "?" + urllib.parse.urlencode({"path": path})
        for item in json.loads(request_bytes(url)):
            full = "/".join(
                part for part in (item.get("path", ""), item["name"]) if part
            )
            if "size" in item:
                files.append(full)
            else:
                queue.append(full)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = f"https://anonymous.4open.science/api/repo/{args.repo_id}"
    files = list_files(base)
    hits: list[str] = []
    for path in files:
        encoded = "/".join(
            urllib.parse.quote(part, safe="") for part in path.split("/")
        )
        content = request_bytes(f"{base}/file/{encoded}").decode(
            "utf-8", errors="ignore"
        )
        for pattern in IDENTITY_PATTERNS:
            if pattern.casefold() in content.casefold():
                hits.append(f"{path}::{pattern}")
    missing = sorted(REQUIRED_PATHS - set(files))
    passed = (
        not missing
        and not hits
        and "plano" not in {path.split("/", 1)[0] for path in files}
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": f"https://anonymous.4open.science/r/{args.repo_id}/",
        "public_api_base": base,
        "file_count": len(files),
        "required_paths_missing": missing,
        "identity_hits": hits,
        "planning_directory_exposed": any(
            path == "plano" or path.startswith("plano/") for path in files
        ),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
