# Phase 5 availability and adversarial-benign evidence

This directory contains the compact, publication-facing evidence from the
completed Phase 5 campaign. The 21.3 MB raw experiment archive remains outside
Git; `provenance.json` records both archive hashes, all verified internal hash
counts, the 1,520 raw session-condition rows, the 900 held-out prompt rows, and
their raw-file hashes.

The execution is complete and internally valid, but the scientific outcome is
negative. The original single-alert Algorithm 2 blocks all 320 clean test
sessions. The validation-selected graduated Algorithm 2b (`k=4,w=4`) preserves
the frozen malicious-session hard-block rate of 100%, yet only 8/320 clean test
sessions survive (2.5%, bootstrap 95% CI [0.94%, 4.38%]). Matched benign
controls and adversarial-benign sessions both have 0/320 survival. The latter
preserves benign task score, but is not attack-specific: the detector is already
near saturation on clean and matched-control windows.

The graduated policy reduces clean blocked-window rate from 98.74% to 73.24%,
but this does not restore usable session availability. Monitoring increases
measured latency by 328.53% (4.29x total runtime) over 60 validation sessions.
Accordingly, `outcome.json` marks the availability objective as not achieved,
and the paper must not claim that Algorithm 2b resolves denial of service.

The result is nevertheless valuable: it directly falsifies an unsafe
deployment claim under the frozen detector and shows why per-token or
per-window false-positive reporting is insufficient. `attacks/` records the
five-seed adversarial-benign search and frozen controls; `summaries/` contains
the validation tradeoff, held-out session survival, benignity, and overhead;
`manifests/` records the H100 environment, frozen detector, and execution gate;
`figures/` contains the validation frontier.

Verify all tracked hashes, matrices, frozen-detector identity, portable paths,
and explicit negative interpretation on CPU:

```powershell
.\.venv\Scripts\python.exe scripts\verify_phase5_results.py
```

Expected output:
`PHASE 5 RESULTS: PASS (complete negative availability result)`.
