"""Audit the public anonymous mirror without GitHub credentials."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.error
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


def identity_patterns(extra_tokens: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    patterns = [
        ("windows-user-path", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.I)),
        ("unix-user-path", re.compile(r"/(?:home|Users)/[^/\s]+/", re.I)),
        (
            "personal-email-provider",
            re.compile(r"@(?:gmail|outlook|hotmail|yahoo)[.]", re.I),
        ),
    ]
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    owner = re.search(r"github[.]com[/:]([^/]+)", remote, re.I)
    if owner:
        patterns.append(
            ("source-remote-owner", re.compile(re.escape(owner.group(1)), re.I))
        )
    patterns.extend(
        (f"forbidden-token-{index}", re.compile(re.escape(token), re.I))
        for index, token in enumerate(extra_tokens, start=1)
        if token
    )
    return patterns


def request_bytes(url: str, *, attempts: int = 5) -> bytes:
    """Fetch an API response, respecting transient rate limits."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "SpectralGuard-Anonymous-Audit/1.0"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == attempts - 1:
                raise
            retry_after = error.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else 2**attempt
            time.sleep(min(delay, 30))
    raise RuntimeError("anonymous repository request exhausted all attempts")


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
    parser.add_argument("--forbidden-token", action="append", default=[])
    args = parser.parse_args()
    base = f"https://anonymous.4open.science/api/repo/{args.repo_id}"
    files = list_files(base)
    hits: list[str] = []
    patterns = identity_patterns(args.forbidden_token)
    for path in files:
        encoded = "/".join(
            urllib.parse.quote(part, safe="") for part in path.split("/")
        )
        content = request_bytes(f"{base}/file/{encoded}").decode(
            "utf-8", errors="ignore"
        )
        for label, pattern in patterns:
            if pattern.search(content):
                hits.append(f"{path}::{label}")
    missing = sorted(REQUIRED_PATHS - set(files))
    passed = not missing and not hits
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": f"https://anonymous.4open.science/r/{args.repo_id}/",
        "public_api_base": base,
        "file_count": len(files),
        "required_paths_missing": missing,
        "identity_hits": hits,
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
