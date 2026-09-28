# Explainable Quantum-Assisted PINNs for a Reduced Thermo-Optic Phase Shifter

Authors: Amoggh Bellad <amogghb@gmail.com> and Shahar Ankonina <shahar.ankonina05@gmail.com>

## Abstract

This study tests how a variational quantum feature layer changes physics-informed learning for a reduced thermo-optic silicon-photonic phase shifter. The learned component is a normalized one-dimensional thermal PDE. Optical phase and thermal crosstalk are computed after the thermal prediction with normalized mode-weighted temperature outputs. The study compares a parameter-matched classical PINN, a larger standard classical PINN, a separable two-qubit QAPINN (Q0), and an entangled two-qubit QAPINN (Q1). Q1 differs from Q0 by one CNOT gate. Under the fixed transient protocol, the standard classical PINN has the best mean relative L2 error. Q0 improves over the parameter-matched classical PINN, but Q1 does not improve transient physical accuracy. Under the fixed steady protocol, Q1 has the best mean steady relative L2 error. Explainability analysis shows that Q1 increases mixed space-time feature interaction compared with Q0. The study does not establish quantum advantage.

## Problem and Research Question

Thermo-optic phase shifters convert electrical heating into optical phase change. The physical chain is:

electrical heating -> temperature field -> refractive-index change -> optical phase shift -> thermal crosstalk.

This project models the normalized thermal part of that chain. The optical output is a reduced post-processing step. The research question is:

How does a shallow quantum feature layer change physics-informed learning of static and transient thermal PDEs?

The answer must separate three effects: physical accuracy, representation change, and runtime cost.

## Physical System and Reduced-Order Model

The model uses one normalized lateral coordinate, `xi`, and one normalized time coordinate, `tau`. A Gaussian heater source drives the thermal field. The active waveguide is centered at `xi = 0.0`. The victim waveguide is centered at `xi = 0.55`.

The model is a reduced-order thermo-optic model. It is not a complete electrothermal solver. It is not a Maxwell optical-mode solver. It is not a fabricated-device calibration model. It is not a full two-dimensional or three-dimensional silicon-photonic device simulator. This boundary lets the study test learning behavior under controlled conditions.

## Governing Equations

The transient equation is:

`theta_tau = theta_xixi + S(xi)u(tau)`.

The residual form is:

`theta_tau - theta_xixi - S(xi)u(tau) = 0`.

The steady equation is:

`-theta_xixi = S(xi)`.

Definitions:

- `xi` is normalized position in `[-1, 1]`.
- `tau` is normalized time in `[0, 1]`.
- `theta` is normalized temperature.
- `S(xi)` is a Gaussian source.
- Source amplitude is `1.0`.
- Source center is `0.0`.
- Source width is `0.2`.
- The heater waveform is a step waveform.

The transient model uses zero boundary conditions and a zero initial condition:

- `theta(-1, tau) = 0`;
- `theta(1, tau) = 0`;
- `theta(xi, 0) = 0`.

The steady model uses zero boundary conditions:

- `theta(-1) = 0`;
- `theta(1) = 0`.

The transient output transform is:

`theta_hat(xi, tau) = tau(1 - xi^2)N(xi, tau)`.

The steady output transform is:

`theta_hat(xi) = (1 - xi^2)N(xi)`.

These transforms enforce the boundary and initial conditions directly. The neural network only learns the unconstrained function `N`.

## Reference Solutions and Validation

The transient reference uses a Crank-Nicolson finite-difference solver. The steady reference uses a finite-difference Poisson solver. The reference grid uses 81 position points and 81 time points for the transient validation.

Stage 2-R reran the reference validation. The initial-condition error was `0.0`. The boundary-condition error was `0.0`. The 41-to-81 grid relative L2 difference was `0.0006412794460016723`. The peak normalized temperature was `0.19430987028320507`.

Stage 2-R also checked the hard output transforms, the quantum derivative path, the focused test suite, and two sentinel reproductions. The quantum derivative gate confirmed finite `dq/dxi`, `dq/dtau`, `d2q/dxi2`, and quantum-parameter gradients. The numerical derivative error was `3.181912511251994e-10`. The focused tests passed with `8 passed`.

## Classical PINN Models

The parameter-matched classical model is `classical_matched`. It uses a Tanh MLP with transient architecture `2 -> 2 -> 16 -> 16 -> 1`. It has 343 trainable parameters in the transient experiment.

The standard classical model is `classical_standard`. It uses a Tanh MLP with transient architecture `2 -> 32 -> 32 -> 32 -> 1`. It has 2241 trainable parameters in the transient experiment. It is not parameter matched. It is an accuracy-oriented conventional baseline.

## QAPINN Models

Q0 and Q1 use two qubits. Both circuits encode `xi` and `tau` with RY gates. Both circuits use one trainable layer of local Rot gates. Both circuits measure `<Z0>` and `<Z1>`. Both circuits feed a classical tail with layers `[16, 16]`.

Q0 is separable. It has no entangling gate. It has 343 total trainable parameters, 6 quantum parameters, gate count 4, and circuit depth 2.

Q1 is entangled. It adds one `CNOT(0,1)` gate after the local trainable rotations. It has 343 total trainable parameters, 6 quantum parameters, gate count 5, and circuit depth 3.

The Q0/Q1 comparison is controlled. The intended circuit difference is the Q1 CNOT gate.

## Experimental Protocol

All final models use the same fixed protocol:

- seeds `11`, `23`, and `37`;
- `float64` arithmetic;
- Adam optimizer;
- learning rate `0.01`;
- 40 epochs;
- 48 collocation points;
- transient evaluation grid `31 x 31`;
- steady evaluation grid with 31 points;
- PennyLane `default.qubit`;
- `shots=None`;
- Torch interface;
- backprop differentiation.

The training loss uses the PDE residual. It does not use reference temperature data. The fixed 40-epoch budget is small. Therefore, the experiment compares models under this fixed optimization budget. The study does not establish asymptotic model performance.

## Transient Results

The report PDF generates the transient result table from `results/metrics/final_transient_metrics.csv`.

Interpretation:

- `classical_standard` has the best mean transient relative L2 error.
- Q0 has lower mean transient relative L2 error than `classical_matched`.
- Q1 does not improve transient relative L2 error.
- Entanglement is not automatically beneficial.
- Parameter reduction does not imply computational advantage.

## Steady Results

The report PDF generates the steady result table from `results/metrics/steady_metrics.csv`.

Under the fixed steady protocol, Q1 has the best mean steady relative L2 error. This result is specific to three seeds, the 40-epoch budget, the current steady PDE, and the current architectures. It does not show a general entanglement advantage.

## Explainability Analysis

Feature maps show the first representation that enters the final trainable tail. For the classical models, the feature maps are first-layer activations. For Q0 and Q1, the feature maps are quantum expectation values.

The mixed space-time interaction score is:

`I_j = mean(abs(d2 q_j / dxi dtau))`.

This score measures whether feature `q_j` depends jointly on space and time. Q0 has near-zero mixed interaction scores. Q1 has one feature with a mean interaction near `3.80`. This confirms that the CNOT changes the representation. The larger interaction does not automatically reduce physical error.

## Fourier Analysis

The Fourier analysis compares normalized spectra of predicted and reference fields. It measures spatial temperature spectra at sampled times and active-phase temporal spectra. Spectral error measures the difference between the normalized predicted spectrum and the normalized reference spectrum.

The valid stored Fourier values are in `results/metrics/fourier_spectral_errors.csv` and `results/metrics/fourier_spectral_summary.csv`. The quantum-feature spectrum rows contain blank values. Therefore, this report does not claim numerical quantum-feature spectral errors.

## Gradient and Residual Analysis

The gradient history records the norm of the representational gradient during training. For quantum models, this is the quantum-parameter gradient norm. For classical models, this is the first classical layer gradient norm. The gradient data support a trainability check for the fixed protocol. They do not establish hardware trainability or barren plateau behavior.

The residual localization figure shows where the PDE residual remains large on the evaluation grid. This spatial view is important because a low global residual can hide local error near sharp source regions or boundaries.

## Feature-Diversity Analysis

Feature diversity is measured with pairwise feature correlation, singular values, and effective rank. These values are computed from saved feature maps. The standard classical model has more first-layer features and higher effective rank. Q0 has two nearly uncorrelated quantum features. Q1 increases interaction but does not increase steady effective rank in the small steady experiment.

## Circuit-Design Methodology

The study supports this problem-specific circuit design process:

1. Identify the PDE input dimension.
2. Identify whether the solution needs coupled input dependence.
3. Start with the shallowest encoding.
4. Add entanglement only when the PDE structure requires joint features.
5. Compare the entangled circuit with a parameter-matched separable circuit.
6. Measure physical error, PDE residual, interaction, gradient behavior, feature diversity, and runtime.
7. Retain entanglement only when it adds useful representation and improves physical metrics.
8. Do not infer usefulness from expressivity alone.
9. Test extra encoding depth only when the reference solution needs frequency content that the shallow encoding does not represent.

The final point is future data re-uploading work. Q2 was not tested in the reportable core.

## Limitations

- The study uses a fixed 40-epoch optimizer budget. Therefore, the results do not determine asymptotic model accuracy.
- The study uses three random seeds. Therefore, the results give limited statistical evidence.
- The study does not include optimizer-budget sensitivity. Therefore, it does not separate architecture effects from optimizer-budget effects.
- Q2 data re-uploading is not part of the reportable core. Therefore, the study does not test extra encoding repetitions.
- The study does not include a classical Fourier-feature control. Therefore, it does not isolate quantum angle encoding from classical trigonometric features.
- The study does not include peak-memory profiling. Therefore, runtime is not a full resource analysis.
- The study does not test unseen heater width or heater amplitude. Therefore, it does not measure out-of-domain generalization.
- The study does not test pulse heating. Therefore, it does not measure cooling or turn-off dynamics.
- The study does not test heterogeneous thermal conductivity. Therefore, it does not measure material-interface effects.
- The study does not include measurement-operator or qubit-count ablations. Therefore, it does not optimize the quantum circuit family.
- The study does not include a Maxwell optical-mode solver. Therefore, it does not predict a fabricated device's absolute phase shift.
- The quantum implementation is simulator-only. Therefore, the study does not make hardware runtime claims.
- The photonic model is normalized and reduced-order. Therefore, it does not replace a calibrated device model.

## Future Work

The next work should follow this order:

1. Test optimizer-budget sensitivity.
2. Test Q2 data re-uploading.
3. Add a classical Fourier-feature control.
4. Add peak-memory profiling.
5. Test generalization over heater width or heater amplitude.
6. Add pulse heating.
7. Add heterogeneous thermal conductivity.
8. Study measurement operators.
9. Study qubit count.
10. Add higher-fidelity optical modeling.

Q2 should follow a stable core. The experiment should test whether additional encoding repetitions capture reference frequencies that the shallow circuit does not represent. The study should not assume that Q2 improves performance.

## Contributions

Amoggh Bellad led silicon-photonics modeling; heater and waveguide physical interpretation; optical phase and crosstalk models; photonics literature review; reduced-device assumptions; quantum-circuit design; explainability, Fourier, gradient, residual, and feature analyses; limitations; and future research recommendations.

Shahar Ankonina led numerical PDE implementation; classical PINN implementation; QAPINN software implementation; optimizer and training implementation; numerical reference computation; experiment infrastructure; reproducibility workflow; and software infrastructure.

Both authors contributed to experimental-design decisions, result interpretation, final review, the technical report, the presentation, reproducibility validation, and submission packaging.

## AI-Use Disclosure

ChatGPT and Codex assisted with code development, code inspection, numerical workflow development, validation infrastructure, analysis tooling, visualization, documentation, report preparation, slide preparation, and interpretation support. The submitted numerical results come from the repository's executed numerical workflows. The authors are responsible for the submitted work and conclusions.

## Recommendations for Future Research

1. Run optimizer-budget sensitivity. This separates architecture effects from short training-budget effects.
2. Test Q2 data re-uploading. This measures whether added frequency capacity improves physical error.
3. Add a classical Fourier-feature control. This separates quantum angle encoding from trigonometric inductive bias.
4. Add pulse heating and generalization tests. This tests cooling dynamics and robustness across heater settings.
5. Defer 2D/3D photonic simulation until the reduced workflow is stable.

## Conclusion

The final result is a controlled negative-and-positive finding. Q0 is a compact parameter-matched baseline that improves over the matched classical PINN on the transient task, but the larger standard classical PINN remains the strongest transient accuracy baseline. Q1 confirms that one CNOT creates a measurable mixed space-time feature, but that feature does not improve transient physical error under the fixed protocol. Future work should therefore test whether added frequency capacity improves physical metrics, not only whether it increases feature complexity.

## References

[1] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 2019.

[2] Jay Shah, Rut Lineswala, and Abhishek Chopra. Benchmarking Quantum-Assisted PINN (QA-PINN) for Computational Fluid Dynamics. IEEE International Conference on Quantum Computing and Engineering (QCE), 2024. DOI: 10.1109/QCE60285.2024.00199.

[3] Schuld, Sweke, and Meyer. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A, 2021. DOI: 10.1103/PhysRevA.103.032430.

[4] Parra, Navarro-Arenas, and Sanchis. Silicon thermo-optic phase shifters: a review of configurations and optimization strategies. Advanced Photonics Nexus, 2024. DOI: 10.1117/1.APN.3.4.044001.

[5] Coenen et al. Static and Dynamic Thermal Modelling of Si Photonic Thermo-Optic Phase Shifter. IEEE ITherm, 2024. DOI: 10.1109/ITHERM55375.2024.10709411.

[6] WISER and BQP. WISER <> BQP Quantum Assisted Physics-Informed Neural Networks for CFD / Summer Program 2026 BQP Challenge, 2026.
