# Results Interpretation

This document separates claims from evidence for the final report. The current
experiments use the frozen manifest in `configs/final_experiment_manifest.yaml`.

## Claim 1: Q0 outperformed the parameter-matched classical model on the transient PDE.

Evidence: mean transient relative L2 error was `0.1235` for `q0_separable` and
`0.2074` for `classical_matched`. Mean PDE residual RMS was `0.1434` for Q0 and
`0.1989` for the matched classical model.

Supporting tables/figures: `results/metrics/final_transient_summary.csv`,
`results/figures/transient_accuracy.png`.

Competing explanation: the result may reflect a useful trigonometric inductive
bias from angle encoding, not uniquely quantum computation.

Limitation: the training budget is small and the seed standard deviation for the
matched classical model is large.

Confidence: moderate for this frozen protocol; low for a broader claim.

## Claim 2: Q0 did not outperform the standard classical PINN on the transient PDE.

Evidence: mean transient relative L2 error was `0.0868` for
`classical_standard` versus `0.1235` for Q0. The standard classical model also ran
in about `0.18 s` per seed, versus about `23.15 s` for Q0.

Supporting tables/figures: `results/metrics/final_transient_summary.csv`,
`results/figures/accuracy_vs_runtime.png`.

Competing explanation: the standard classical model has many more trainable
parameters (`2241` versus `343`).

Limitation: no hyperparameter tuning was performed for any model after protocol
freeze.

Confidence: high for this protocol.

## Claim 3: Entanglement created a nonseparable quantum feature, but that did not improve transient accuracy.

Evidence: Q0 mixed interaction scores were approximately zero for both quantum
features. Q1 had one feature with mean interaction score about `3.80`, but Q1
mean transient relative L2 error was `0.2137`, worse than Q0 (`0.1235`) and the
standard classical model (`0.0868`).

Supporting tables/figures: `results/metrics/feature_diversity.csv`,
`results/figures/mixed_interaction_comparison.png`.

Competing explanation: Q1 may need a different optimizer or longer training to
convert richer features into solution accuracy.

Limitation: Q1 was deliberately not retuned because the comparison isolates the
effect of adding one CNOT under the same protocol.

Confidence: high that Q1 changed representation; moderate that it was not useful
under this training budget.

## Claim 4: Quantum parameter efficiency was not computational efficiency.

Evidence: Q0 and Q1 matched the parameter budget of `classical_matched` but took
about `23-25 s` per seed. Classical baselines took less than `0.3 s` per seed.

Supporting tables/figures: `results/metrics/final_transient_summary.csv`,
`results/metrics/steady_summary.csv`, `results/figures/accuracy_vs_runtime.png`.

Competing explanation: simulator overhead dominates this small problem; hardware
runtime is not tested.

Limitation: no memory profiling beyond package-level capability was finalized.

Confidence: high for simulator runtime; no claim for hardware.

## Claim 5: The preferred circuit changed between transient and steady tasks only weakly.

Evidence: In the transient task, Q0 had lower mean physical error than Q1. In the
steady task, Q1 had the lowest mean relative L2 error among the four models
(`0.0716`), but the standard classical model was close (`0.0869`) and far faster.

Supporting tables/figures: `results/metrics/final_transient_summary.csv`,
`results/metrics/steady_summary.csv`, `results/figures/steady_comparison.png`.

Competing explanation: steady Q1's advantage may reflect seed-level optimization
luck under a small epoch budget rather than an entanglement mechanism.

Limitation: steady Q1 feature effective rank is `1.0`, so the result does not
show richer steady representation.

Confidence: low-moderate.

## Justified Circuit-Design Rule

For this reduced thermo-optic PINN, use the shallow separable quantum feature
layer only as a compact parameter-matched control, not as a default replacement
for a conventional classical PINN. Add entanglement only if interaction-score and
spectral evidence show that nonseparable features reduce physical error under
the same training protocol. Parameter reduction should be reported separately
from simulator runtime.
