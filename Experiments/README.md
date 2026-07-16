# Phase 3 runtime workspace

This directory is the local, Git-ignored workspace used by the two Phase 3
Lightning AI notebooks. The notebooks create `checkpoints/`, `datasets/`,
`logs/`, `raw/`, `summaries/`, `figures/`, `manifests/`, and `exports/` here.

The only files the researcher needs to download are:

- `exports/phase3_experiments.zip`
- `exports/phase3_final.zip`

Model weights, Hugging Face credentials, caches, partial files, and notebook
state are excluded from both archives.

`phase3_experiments.zip` is created only after every required full-model cell
has completed. `phase3_final.zip` is created only after the split-aware analysis
has run. Neither filename is evidence of scientific success by itself; the
manifest, analysis gates, and Newton final audit determine closure.
