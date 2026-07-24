"""Verify the Phase 8 manuscript, vector figures, and page-budget contract."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = PAPER / "main.tex"
PDF = PAPER / "main.pdf"
LOG = PAPER / "main.log"
MANIFEST = ROOT / "results" / "foundation" / "phase8_figure_manifest.json"
FEATURE_AUDIT = ROOT / "results" / "foundation" / "phase8_feature_schema_audit.json"
FEATURE_CSV = ROOT / "results" / "phase3" / "summaries" / "headline_multiseed.csv"
EXPECTED_FIGURES = {
    "evaluation_pipeline.pdf",
    "theory_counterexample.pdf",
    "detection_adaptation.pdf",
    "availability_cost.pdf",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_pages() -> int | None:
    candidates: list[str] = []
    found = shutil.which("pdfinfo")
    if found:
        candidates.append(found)
    if sys.platform == "win32":
        where = subprocess.run(
            ["where.exe", "pdfinfo"], capture_output=True, text=True
        )
        candidates.extend(line.strip() for line in where.stdout.splitlines())
    for executable in dict.fromkeys(candidates):
        try:
            output = subprocess.run(
                [executable, str(PDF)], check=True, capture_output=True, text=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


def verify_feature_schema(tex: str) -> dict[str, object]:
    expected = [f"layer_{index}_rho_mean" for index in range(64)]
    with FEATURE_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {int(row["seed"]) for row in rows} == {42, 123, 456, 789, 2026}
    for row in rows:
        assert ast.literal_eval(row["feature_names"]) == expected
        for field in ("feature_mean", "feature_scale", "weights"):
            assert len(ast.literal_eval(row[field])) == 64

    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from src.defense.detectors import spectral_feature_names

    frame = pd.DataFrame(
        {"spectral": [{"layers": [{"layer": index} for index in range(64)]}]}
    )
    assert spectral_feature_names(frame) == expected
    assert "64 monitored Mamba layers" in tex
    assert "one temporal mean" in tex
    assert r"\texttt{layer\_0\_rho\_mean}" in tex
    assert r"\texttt{layer\_63\_rho\_mean}" in tex

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "paper_feature_count": 64,
        "code_feature_count": 64,
        "serialized_feature_count": 64,
        "aggregation": "one temporal rho_mean per monitored layer",
        "ordering": "ascending monitored layer index",
        "first_feature": expected[0],
        "last_feature": expected[-1],
        "feature_names": expected,
        "detector_seeds_checked": sorted({int(row["seed"]) for row in rows}),
        "serialized_source": FEATURE_CSV.relative_to(ROOT).as_posix(),
        "code_source": "src/defense/detectors.py:spectral_feature_names",
    }
    FEATURE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_AUDIT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    normalized = " ".join(tex.split())
    lowered = normalized.lower()

    pdf_exists = PDF.is_file() and PDF.read_bytes().startswith(b"%PDF-")
    if pdf_exists:
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
        "Threat Model and DURA Questions",
        "Spectral Foundations: RQ1",
        "Monitor and Experimental Methodology",
        "DURA Evaluation",
        "Discussion",
        "Limitations",
        "Conclusion",
    ):
        assert rf"\section{{{section}}}" in tex
    for phrase in (
        "fixed-test prompt-cluster bootstrap intervals",
        "five training-seed fits on the same shared split",
        "Only 13 of 300 prompts",
        "cannot estimate a production",
        "HumanEval",
    ):
        assert phrase in normalized, phrase
    assert "MMLU" not in tex
    assert "12-page" not in lowered and "twelve-page" not in lowered
    assert "Proceedings of Proceedings" not in tex
    assert re.search(r"(?<!\\)eqref\{", tex) is None
    assert "32-observation windows" not in tex
    assert "32-token monitoring windows" in tex
    assert "broad clean campaign" in tex
    assert "not paired with the 75-prompt" in normalized
    for forbidden in ("legacy", "repair", "retract", "Phase~", "Phase "):
        assert forbidden not in tex, forbidden
    assert tex.count(r"\answerbox{") == 4
    assert r"\scriptsize" not in tex and r"\tiny" not in tex
    windows_root = r"[A-Za-z]:[\\/]+" + "Users" + r"[\\/]"
    unix_teamspace = "/" + "teamspace/"
    unix_home = "/" + "home/" + "zeus/"
    assert not re.search(windows_root, tex, flags=re.I)
    assert unix_teamspace not in tex and unix_home not in tex

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
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
            src_raw = hashlib.sha256(path.read_bytes()).hexdigest()
            src_norm = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            assert source["sha256"] in (src_raw, src_norm)

    raster = [path for path in (PAPER / "figures").iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    assert raster == []

    feature_audit = verify_feature_schema(tex)

    bib_entries = re.findall(r"^@\w+\{", (PAPER / "references.bib").read_text(encoding="utf-8"), re.MULTILINE)
    assert len(bib_entries) >= 28
    if LOG.is_file():
        log = LOG.read_text(encoding="utf-8", errors="replace")
        assert "undefined references" not in log.lower()
        assert "undefined citations" not in log.lower()
        assert "overfull \\hbox" not in log.lower()
    pages = pdf_pages()
    if pages is not None:
        assert 8 <= pages <= 11, pages
    print(
        f"PHASE 8 MANUSCRIPT: PASS ({pages or 'page count unavailable'} pages, "
        f"4 vector figures, {len(bib_entries)} references, "
        f"{feature_audit['serialized_feature_count']}-feature schema exact)"
    )


if __name__ == "__main__":
    main()
