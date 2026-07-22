"""Verify the Phase 8 manuscript, vector figures, and page-budget contract."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = PAPER / "main.tex"
PDF = PAPER / "main.pdf"
LOG = PAPER / "main.log"
MANIFEST = ROOT / "results" / "foundation" / "phase8_figure_manifest.json"
EXPECTED_FIGURES = {
    "evaluation_pipeline.pdf",
    "theory_counterexample.pdf",
    "detection_adaptation.pdf",
    "availability_cost.pdf",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_pages() -> int | None:
    executable = shutil.which("pdfinfo")
    if executable is None:
        return None
    try:
        output = subprocess.run(
            [executable, str(PDF)], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("pdfinfo did not report a page count")
    return int(match.group(1))


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    normalized = " ".join(tex.split())
    lowered = normalized.lower()

    assert PDF.is_file() and PDF.read_bytes().startswith(b"%PDF-")
    assert PDF.stat().st_size > 100_000
    assert b"/Author" not in PDF.read_bytes()
    assert tex.count(r"\includegraphics") == 4
    assert set(re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex)) == EXPECTED_FIGURES
    for label in ("fig:overview", "fig:theory", "fig:detection", "fig:availability"):
        assert rf"\label{{{label}}}" in tex
    for rq in ("RQ1", "RQ2", "RQ3", "RQ4"):
        assert rq in tex
    for section in (
        "Background and Related Work",
        "Experimental Methodology",
        "Results and Analysis",
        "Discussion",
        "Limitations",
        "Conclusion",
    ):
        assert rf"\section{{{section}}}" in tex
    for phrase in (
        "fixed-test prompt-cluster bootstrap interval",
        "five training-seed fits on the same split",
        "floor-saturated",
        "Only 13/300 prompts",
        "synthetic three-turn sessions cannot estimate a production false-block rate",
        "HumanEval",
    ):
        assert phrase in normalized, phrase
    assert "MMLU" not in tex
    assert "12-page" not in lowered and "twelve-page" not in lowered
    windows_root = r"[A-Za-z]:[\\/]+" + "Users" + r"[\\/]"
    unix_teamspace = "/" + "teamspace/"
    unix_home = "/" + "home/" + "zeus/"
    assert not re.search(windows_root, tex, flags=re.I)
    assert unix_teamspace not in tex and unix_home not in tex

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert set(manifest["figures"]) == EXPECTED_FIGURES
    for name, record in manifest["figures"].items():
        figure = PAPER / "figures" / name
        assert figure.is_file() and figure.read_bytes().startswith(b"%PDF-")
        assert figure.stat().st_size > 5_000
        assert record["sha256"] == sha256(figure)
        assert record["format"] == "vector_pdf"
        assert record["sources"]
        for source in record["sources"]:
            path = ROOT / source["path"]
            assert path.is_file()
            assert source["sha256"] == sha256(path)
    raster = [path for path in (PAPER / "figures").iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    assert raster == []

    bib_entries = re.findall(r"^@\w+\{", (PAPER / "references.bib").read_text(encoding="utf-8"), re.MULTILINE)
    assert len(bib_entries) >= 10
    if LOG.is_file():
        log = LOG.read_text(encoding="utf-8", errors="replace")
        assert "undefined references" not in log.lower()
        assert "undefined citations" not in log.lower()
        assert "overfull \\hbox" not in log.lower()
    pages = pdf_pages()
    if pages is not None:
        assert 8 <= pages <= 12, pages
    print(f"PHASE 8 MANUSCRIPT: PASS ({pages or 'page count unavailable'} pages, 4 vector figures)")


if __name__ == "__main__":
    main()
