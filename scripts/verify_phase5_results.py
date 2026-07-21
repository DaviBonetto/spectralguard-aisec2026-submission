"""Verify tracked or freshly generated Phase 5 semantic evidence."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.phase5_acceptance import validate_phase5_acceptance  # noqa: E402


if __name__ == "__main__":
    report = validate_phase5_acceptance()
    if report["status"] != "PASS":
        raise SystemExit(f"PHASE 5 RESULTS: FAIL {report['checks']}")
    print("PHASE 5 RESULTS: PASS")
