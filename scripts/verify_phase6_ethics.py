"""Verify the Phase 6 ethics, disclosure, and ACM submission invariants."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
BIB = ROOT / "paper" / "references.bib"
README = ROOT / "README.md"


def _between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except IndexError as exc:
        raise AssertionError(f"missing required boundary: {start!r} -> {end!r}") from exc


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert tex.startswith(r"\documentclass[sigconf]{acmart}")
    assert "nonacm" not in tex
    for field in (
        r"\setcopyright{acmlicensed}",
        r"\copyrightyear{2018}",
        r"\acmYear{2018}",
        r"\acmDOI{XXXXXXX.XXXXXXX}",
        r"\acmISBN{978-1-4503-XXXX-X/2018/06}",
    ):
        assert field in tex, f"missing required ACM field: {field}"

    bibliography = tex.index(r"\bibliography{references}")
    open_science = tex.index(r"\section{Open Science}")
    ethics_start = tex.index(r"\section{Ethical Considerations}")
    genai = tex.index(r"\section{Generative AI Usage Statement}")
    assert bibliography < open_science < ethics_start < genai

    ethics = _between(
        tex,
        r"\section{Ethical Considerations}",
        r"\section{Generative AI Usage Statement}",
    )
    normalized_ethics = " ".join(ethics.split())
    required = (
        r"\cite{bailey2012menlo}",
        "no human subjects",
        "public benchmarks",
        "controlled sessions",
        "live target",
        "Menlo principles",
        "disclosure",
    )
    for phrase in required:
        assert phrase in normalized_ethics, f"missing ethics requirement: {phrase}"

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", ethics)
    assert len(words) >= 120, f"ethics analysis is too short: {len(words)} words"
    assert "@article{bailey2012menlo" in bib
    assert "doi     = {10.1109/MSP.2012.52}" in bib
    assert "Anonymous research artifact" in readme
    assert "anonymity invariants" in readme

    print(f"PHASE 6 ETHICS: PASS ({len(words)} words; ACM fields retained)")


if __name__ == "__main__":
    main()
