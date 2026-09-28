# Key Findings

1. Q0 separable improved over the parameter-matched classical baseline on the
   transient heat equation under the frozen protocol.
   - Mean relative L2: Q0 `0.1235`, classical_matched `0.2074`.
   - Mean PDE residual RMS: Q0 `0.1434`, classical_matched `0.1989`.

2. Q0 did not outperform the conventional larger classical PINN.
   - Mean transient relative L2: classical_standard `0.0868`, Q0 `0.1235`.
   - Mean runtime: classical_standard `0.18 s`, Q0 `23.15 s`.

3. Entanglement changed the representation but did not improve transient
   physical accuracy.
   - Q0 mixed interaction scores were approximately zero.
   - Q1 produced one strongly nonseparable feature with interaction near `3.80`.
   - Q1 mean transient relative L2 was `0.2137`, worse than Q0 and the standard
     classical baseline.

4. Parameter reduction did not imply computational efficiency.
   - Q0/Q1 matched the `343`-parameter budget of `classical_matched`.
   - Simulator runtime was roughly two orders of magnitude higher than classical
     training for this small problem.

5. The steady Poisson experiment did not justify a broad entanglement claim.
   - Q1 had the lowest mean steady relative L2 (`0.0716`) in the small-budget
     protocol.
   - The standard classical model was close (`0.0869`) and far faster.
   - The result should be treated as suggestive, not conclusive.

6. The defensible design rule is conditional.
   - Start with the smallest separable quantum feature layer for parameter
     efficiency experiments.
   - Add entanglement only when representation, spectral, and physical-error
     evidence improve together.
   - Report accuracy, parameter count, trainability, and runtime separately.
