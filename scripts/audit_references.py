"""Audit LaTeX citation coverage and arXiv metadata through DataCite."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def normalize(value: str) -> str:
    value = value.replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKD", value)
    return " ".join(
        "".join(char for char in value if char.isalnum() or char.isspace())
        .lower()
        .split()
    )


def bib_entries(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for chunk in re.split(r"(?=@[A-Za-z]+\s*\{)", text):
        header = re.match(r"@[A-Za-z]+\s*\{\s*([^,]+),", chunk)
        if not header:
            continue
        fields: dict[str, str] = {}
        for match in re.finditer(
            r"(?ms)^\s*([A-Za-z]+)\s*=\s*\{(.*?)\}\s*,?\s*$", chunk
        ):
            fields[match.group(1).lower()] = " ".join(match.group(2).split())
        entries[header.group(1).strip()] = fields
    return entries


def citations(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite[A-Za-z*]*\{([^}]+)\}", text):
        keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def datacite(arxiv_id: str) -> dict[str, object]:
    doi = f"10.48550/arXiv.{arxiv_id}"
    request = urllib.request.Request(
        f"https://api.datacite.org/dois/{doi}",
        headers={"User-Agent": "SpectralGuard-Reference-Audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["data"]["attributes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tex = args.tex.read_text(encoding="utf-8")
    entries = bib_entries(args.bib.read_text(encoding="utf-8"))
    cited = citations(tex)
    missing = sorted(cited - entries.keys())
    orphaned = sorted(entries.keys() - cited)
    checks: list[dict[str, object]] = []
    divergences: list[str] = []
    for key, fields in sorted(entries.items()):
        arxiv_id = fields.get("eprint")
        if not arxiv_id:
            continue
        official = datacite(arxiv_id)
        official_title = official["titles"][0]["title"]
        official_authors = [creator["name"] for creator in official["creators"]]
        bib_authors = [name.strip() for name in fields.get("author", "").split(" and ")]
        title_match = normalize(fields.get("title", "")) == normalize(official_title)
        authors_match = [normalize(name) for name in bib_authors] == [
            normalize(name) for name in official_authors
        ]
        year_match = fields.get("year") == str(official["publicationYear"])
        if not (title_match and authors_match and year_match):
            divergences.append(key)
        checks.append(
            {
                "key": key,
                "arxiv_id": arxiv_id,
                "official_doi": official["doi"],
                "official_url": official["url"],
                "title_match": title_match,
                "authors_match": authors_match,
                "year_match": year_match,
            }
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bib_entry_count": len(entries),
        "cited_key_count": len(cited),
        "missing_bib_entries": missing,
        "orphaned_bib_entries": orphaned,
        "metadata_divergences": divergences,
        "official_metadata_checks": checks,
        "passed": not missing and not orphaned and not divergences,
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
