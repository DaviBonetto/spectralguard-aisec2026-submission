"""Verify tracked or freshly generated Phase 4 semantic evidence."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.phase4_acceptance import validate_phase4_acceptance  # noqa: E402


if __name__ == "__main__":
    report = validate_phase4_acceptance()
    if report["status"] != "PASS":
        raise SystemExit(f"PHASE 4 RESULTS: FAIL {report['checks']}")
    print("PHASE 4 RESULTS: PASS")
