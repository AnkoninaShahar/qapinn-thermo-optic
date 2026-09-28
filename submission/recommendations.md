# Recommendations for Future Work

1. Add Q2 data re-uploading only after preserving the current core submission.
   Compare Q2 against Q1 with added parameters, gates, depth, runtime, gradients,
   Fourier spectra, and physical error all reported.

2. Add a classical Fourier-feature control. If it reproduces Q0's behavior, the
   likely mechanism is trigonometric inductive bias rather than a uniquely
   quantum effect.

3. Increase the training budget after freezing a stronger protocol. The current
   runs are intentionally small enough to be reproducible quickly.

4. Add pulse heating and cooling. This would test transient behavior with both
   turn-on and decay dynamics.

5. Test generalization across heater width, heater amplitude, and active-victim
   separation. This would strengthen the circuit-selection argument.

6. Add memory profiling and more rigorous runtime benchmarking. The current
   results support simulator-runtime caution, not hardware conclusions.

7. Avoid moving to 2D/3D photonics simulation until the reduced-order workflow is
   fully stable with Q2 and classical Fourier controls.
