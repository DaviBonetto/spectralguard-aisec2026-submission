# Phase 3 confirmatory evidence

This directory contains the lightweight, publication-facing outputs derived
from the completed H100 campaign. The raw 550 MB experiment archive remains
outside Git; its integrity and provenance are recorded in `provenance.json`.

The strengthened semantic gate passes all 18 Phase 3 requirements. In
particular, the GPT-2 lexical control is below the preregistered 0.65 ceiling
(AUC 0.533496), all cross-model transfer cells contain 200 held-out examples,
adaptive v2/v3 uses the same 300 prompts with a 2,000-resample paired interval,
and the rho/accuracy regressions report McFadden pseudo-R2.

The confirmatory results do not support a superiority claim. Under the common
protocol, SpectralGuard reaches F1 0.806142 and mean AUC 0.887294, whereas the
CLASP-style BOE+XGBoost reimplementation reaches F1 0.968877 and mean AUC
0.996506. This comparison uses CLASP's published design and hyperparameters,
not official pretrained weights. The five prescribed SpectralGuard seed runs
are identical because the fitted logistic estimator is deterministic on the
fixed data; this zero between-seed variation must be disclosed rather than
presented as independent data variability.

`summaries/detector_uncertainty.csv` resolves the two nonexchangeable sources
of uncertainty. SpectralGuard F1 is 0.806142 with a 95% cluster-bootstrap CI
[0.795311, 0.817602] over 1,000 fixed test prompt IDs and 2,000 resamples. The
CLASP-style mean F1 is 0.968900 with a descriptive 95% t interval
[0.965550, 0.972249] over five stochastic training seeds on the same shared
split. The latter is seed-to-seed training variation, not independent sampling
uncertainty; the two intervals must not be drawn or interpreted as equivalent
error bars.

The adaptive v2/v3 accuracy difference is -0.00333 (95% paired bootstrap CI
[-0.02333, 0.01667], McNemar p=1.0). McFadden pseudo-R2 ranges from 0.00890 to
0.03375 across the four main conditions, so rho alone has weak explanatory
power for task correctness. The legacy sign reversal was not reproduced: all
three clean-minus-condition mean delta-rho estimates are positive.

`summaries/` contains the tables, `figures/` contains the five-restart Pareto
band, and `manifests/phase3_acceptance.json` maps requirements 3.1--3.18 to
their evidence. No model weights, credentials, notebook state, local absolute
paths, or author identity are included.
