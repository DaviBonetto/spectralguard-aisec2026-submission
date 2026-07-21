# Phase 3, Phase 4, and Phase 5 runtime workspace

This directory is the local, Git-ignored workspace used by the Lightning AI
notebooks. Phase 3 uses the top-level runtime folders; Phase 4 keeps its
checkpoints and evidence under `phase4/`; Phase 5 uses `phase5/`. Shared
downloads remain reusable.

The only files the researcher needs to download are:

- `exports/phase3_experiments.zip`
- `exports/phase3_final.zip`
- `exports/phase4_experiments.zip`
- `exports/phase4_final.zip`
- `exports/phase5_experiments.zip`
- `exports/phase5_final.zip`

If the 2.4-hour Phase 4 session reserve is reached, download
`exports/phase4_handoff.tar.gz`; restoring it continues without repeating
completed crafting or prompt-condition pairs.

If the 2.7-hour Phase 5 reserve is reached, download
`exports/phase5_handoff.tar.gz`; restoring it continues without repeating
completed attack seeds, session-condition traces, or held-out prompt scoring.

Model weights, Hugging Face credentials, caches, partial files, and notebook
state are excluded from both archives.

`phase3_experiments.zip` is created only after every required full-model cell
has completed. `phase3_final.zip` is created only after the split-aware analysis
has run. Neither filename is evidence of scientific success by itself; the
manifest and the executable 3.1--3.18 acceptance gates determine closure.
The same rule applies to Phases 4 and 5: only their semantic gates establish
closure. Phase 5 uses one fifth and final notebook for both the H100 experiment
and the leakage-safe CPU analysis/export.
