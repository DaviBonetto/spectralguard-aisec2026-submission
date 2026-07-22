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
    genai = tex.index(r"\section*{Generative AI Usage Statement}")
    assert bibliography < open_science < ethics_start < genai

    ethics = _between(
        tex,
        r"\section{Ethical Considerations}",
        r"\section*{Generative AI Usage Statement}",
    )
    normalized_ethics = " ".join(ethics.split())
    required = (
        r"\subsection{Responsible Disclosure and Release}",
        r"\cite{bailey2012menlo}",
        r"\cite{lemercier2026hispa}",
        "stakeholders",
        "dual-use risk",
        "IRB/ERB",
        "not human-subjects research",
        "already publicly disclosed",
        "did not probe a live product",
        "does not claim a coordinated or delayed disclosure",
        "MIT license",
        "residual incremental risk",
    )
    for phrase in required:
        assert phrase in normalized_ethics, f"missing ethics requirement: {phrase}"

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", ethics)
    assert len(words) >= 350, f"ethics analysis is too short: {len(words)} words"
    assert "@article{bailey2012menlo" in bib
    assert "doi     = {10.1109/MSP.2012.52}" in bib
    assert "Phase 6 ethics and responsible release" in readme
    assert "Phases 0--6 are complete" in readme

    print(f"PHASE 6 ETHICS: PASS ({len(words)} words; ACM fields retained)")


if __name__ == "__main__":
    main()
