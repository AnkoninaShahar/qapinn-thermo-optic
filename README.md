# Explainable QAPINNs for a Reduced Thermo-Optic Phase Shifter

This repository contains a reproducible final submission package for a controlled
comparison of classical PINNs and two-qubit QAPINNs on a reduced thermo-optic
phase-shifter model.

The study tests a normalized one-dimensional thermal PDE. Optical phase and
thermal crosstalk are computed after the thermal prediction. The final comparison
uses four models:

- `classical_matched`: parameter-matched classical PINN;
- `classical_standard`: larger conventional classical PINN;
- `q0_separable`: separable two-qubit QAPINN;
- `q1_entangled`: Q0 plus one CNOT gate.

The submission does not claim quantum advantage, hardware speedup, or fabricated
device calibration.

## Team

- Amoggh Bellad <amogghb@gmail.com>
- Shahar Ankonina <shahar.ankonina05@gmail.com>

## Model

The thermal PDE is

```text
theta_tau = theta_xx + S(xi) u(tau)
theta(-1, tau) = theta(1, tau) = 0
theta(xi, 0) = 0
```

with normalized `xi in [-1, 1]`, `tau in [0, 1]`, a Gaussian heater source, and a
step heater waveform. The model output transform is

```text
theta_hat(xi, tau) = tau * (1 - xi^2) * N(xi, tau)
```

so the initial and boundary conditions are enforced exactly.

## Environment

Use Python 3.11. Newer Python versions may not have compatible wheels for the
pinned PyTorch and PennyLane stack.

```bash
conda create -y -p ./.venv311 python=3.11 pip
.venv311/bin/python -m pip install -r requirements-lock.txt
```

Verify imports:

```bash
.venv311/bin/python - <<'PY'
import numpy, torch, pennylane
print(numpy.__version__, torch.__version__, pennylane.__version__)
PY
```

Expected stack:

```text
numpy 1.26.4
torch 2.2.2
pennylane 0.35.1
```

## Validation Commands

Run the focused validation tests:

```bash
.venv311/bin/python -m pytest -q tests/test_models_losses.py tests/test_physics_reference.py tests/test_quantum_feasibility.py
```

Validate the independent Crank-Nicolson reference solution:

```bash
.venv311/bin/python scripts/validate_physics.py
```

Run the quantum second-derivative feasibility gate:

```bash
.venv311/bin/python scripts/test_quantum_derivatives.py
```

Run the final transient experiment matrix:

```bash
.venv311/bin/python scripts/run_final_transient.py
```

This trains:

- `classical_matched`
- `classical_standard`
- `q0_separable`
- `q1_entangled`

for seeds `11`, `23`, and `37`, then writes
`results/metrics/final_transient_metrics.csv` and
`results/metrics/final_transient_summary.csv`.

Run the steady Poisson experiment and regenerate analysis figures:

```bash
.venv311/bin/python scripts/run_steady.py
.venv311/bin/python scripts/analyze_results.py
```

Build the technical report PDF:

```bash
.venv311/bin/python scripts/build_report_pdf.py
```

## Outputs

Machine-readable records are written to:

- `results/metrics/reference_validation.json`
- `results/metrics/quantum_derivative_gate.json`
- `results/metrics/final_transient_metrics.csv`
- `results/metrics/final_transient_summary.csv`
- `results/metrics/steady_metrics.csv`
- `results/metrics/steady_summary.csv`
- `results/metrics/feature_diversity.csv`
- `results/metrics/fourier_spectral_summary.csv`
- `results/runs/*_seed*.json`
- `results/runs/*_arrays.npz`
- `results/runs/*_explainability.npz`

Feature-map and residual-map PNGs are written beside each run record in
`results/runs/`.

The final report and slide deck are included as generated artifacts:

- `report/technical_report.md`
- `report/technical_report.pdf`
- `slides/qapinn_submission_slides.pptx`

## Final Results

The current configured run uses `48` collocation points, `40` Adam epochs,
`31 x 31` evaluation grids, and `float64`.

Mean over three seeds:

| model | rel L2 temp | PDE RMS | active phase RMSE | victim phase RMSE | crosstalk abs error | params | runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| classical_matched | 0.2074 | 0.1989 | 0.0298 | 0.0129 | 0.0660 | 343 | 0.13 |
| classical_standard | 0.0868 | 0.1344 | 0.0141 | 0.0050 | 0.0192 | 2241 | 0.18 |
| q0_separable | 0.1235 | 0.1434 | 0.0179 | 0.0098 | 0.0357 | 343 | 23.15 |
| q1_entangled | 0.2137 | 0.1639 | 0.0291 | 0.0195 | 0.0354 | 343 | 24.52 |

Steady Poisson mean relative L2 values:

| model | steady rel L2 | params | runtime s |
|---|---:|---:|---:|
| classical_matched | 0.0981 | 341 | 0.12 |
| classical_standard | 0.0869 | 2209 | 0.10 |
| q0_separable | 0.1408 | 343 | 22.99 |
| q1_entangled | 0.0716 | 343 | 24.11 |

These values are fixed-protocol results over three seeds. Q0 improves over the
parameter-matched classical baseline on the transient PDE. The standard classical
PINN has the best transient mean relative L2. Q1 has the best steady mean
relative L2 under this small-budget protocol. The current experiment does not
establish quantum advantage.

## Scientific Caveats

- Do not claim quantum advantage from the current data.
- The Q0/Q1 comparison isolates entanglement at matched parameter count, but the
  training budget is intentionally small.
- The reference model is reduced-order and normalized; it is not a calibrated
  silicon photonics device simulator.
- Q2 data re-uploading remains future work.

## Final Submission Artifacts

- Technical report: `report/technical_report.pdf`
- Slides: `slides/qapinn_submission_slides.pptx`
- Final manifest: `configs/final_experiment_manifest.yaml`
- Interpretation: `docs/results_interpretation.md`
- Key findings and disclosure files: `submission/`
