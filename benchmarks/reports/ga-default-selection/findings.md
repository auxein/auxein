## Findings

Choosing the default configuration of `GeneticAlgorithm`: three candidates, 15 runs each, on all five problems at d = 10 and 30, with the full budget (2000 × d evaluations). The instances (1000 to 1014) and seeds are not those of `full.toml` (instances 0 to 24), so the default is judged on other instances than it was chosen on.

- **A**: tournament selection (size 2), intermediate recombination, self-adaptive mutation with one step size per individual, clipping. μ = λ = 50.
- **B**: as A, with one self-adaptive step size per gene.
- **C**: as A, with stochastic universal sampling with sigma scaling instead of the tournament.

**Mean rank of the median final error** (the table under "Mean rank"): A 1.80, B 2.10, C 2.10. The mean ranks are within 0.5 of each other, a near tie, so the simplest configuration is preferred; A is also the best by mean rank. **A is the default.**

- On the sphere and the noisy sphere all three reach errors around 10⁻²² at d = 10 and 30, except B, which is slower at d = 30 (5 × 10⁻¹⁷ on the sphere, 6 × 10⁻⁶ on the noisy sphere).
- B (per-gene step sizes) is the best on Rosenbrock and Rastrigin at both dimensions, by a small margin (for example Rastrigin d = 30: 32.8 against 36.8 for A).
- C is the best on the 10-D ellipsoid (850 against 1,020 for A) and the 10-D noisy sphere, and the worst on the 30-D ellipsoid.
- All three are far from solving the ellipsoid (errors of 10³ to 10⁴ for a condition number of 10⁶): isotropic step sizes cannot adapt to a rotated, ill-conditioned landscape.
