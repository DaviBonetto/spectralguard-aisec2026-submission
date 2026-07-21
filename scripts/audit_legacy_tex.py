"""Inventory citation coverage in a legacy self-contained TeX source."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--source-arxiv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.tex.read_text(encoding="utf-8")
    bibitems = re.findall(r"\\bibitem\{([^}]+)\}", text)
    cited: set[str] = set()
    for match in re.finditer(r"\\cite[A-Za-z*]*\{([^}]+)\}", text):
        cited.update(key.strip() for key in match.group(1).split(","))
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_arxiv": args.source_arxiv,
        "bibitem_count": len(bibitems),
        "unique_citation_count": len(cited),
        "orphaned_bibitems": sorted(set(bibitems) - cited),
        "missing_bibitems": sorted(cited - set(bibitems)),
        "disposition": (
            "Legacy references are not imported wholesale. Each reference enters the "
            "new bibliography only when cited and after official metadata validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    if report["missing_bibitems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
