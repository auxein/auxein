## Findings

Choosing the default configuration of `GeneticAlgorithm`: three candidates, 15 runs each, on all five problems at d = 10 and 30, with the full budget (2000 × d evaluations). The instances (1000 to 1014) and seeds are not those of `full.toml` (instances 0 to 24), so the default is judged on other instances than it was chosen on.

- **A**: tournament selection (size 2), intermediate recombination, self-adaptive mutation with one step size per individual, clipping. μ = λ = 50.
- **B**: as A, with one self-adaptive step size per gene.
- **C**: as A, with stochastic universal sampling with sigma scaling instead of the tournament.

**Mean rank of the median final error** (the table under "Mean rank"): C 1.70, A 2.10, B 2.20. All three are within 0.5 of the best, a near tie, so the simplest configuration is chosen: **A is the default** (a tournament needs no weights, and one step size per individual is the simpler self-adaptation).

How far to trust the order: the same selection run before an internal optimisation of the strategy (which changes the order in which the population is stored, and with it the random draws, not the algorithm) gave A 1.80, B 2.10, C 2.10, a different order and the same near tie. With 15 runs per cell the three configurations cannot be told apart by their ranks; what the data do show is below.

- On the sphere and the noisy sphere all three reach errors around 10⁻²² at d = 10 and 30, except B, which is slower at d = 30 (2 × 10⁻¹⁵ on the sphere, 6 × 10⁻⁴ on the noisy sphere).
- B (per-gene step sizes) is the best on Rastrigin at both dimensions (d = 30: 28.9 against 39.8 for A and 46.8 for C), and by a small margin on Rosenbrock at d = 10.
- C is the best on the ellipsoid at both dimensions (d = 30: 6.1 × 10³ against 6.9 × 10³ for A), by a small margin.
- All three are far from solving the ellipsoid (errors of 10³ to 10⁴ for a condition number of 10⁶): isotropic step sizes cannot adapt to a rotated, ill-conditioned landscape.
