## Findings

The new `GeneticAlgorithm` (`auxein-core-ga`, its constructor defaults: μ = λ = 50, tournament selection of size 2, intermediate recombination, one self-adaptive step size per individual, clipping) against the 0.2.0 engine (`auxein-default` and the two other 0.2.0 configurations), random search and CMA-ES. All numbers come from this one run: the same machine, the same 25 instances and seeds per problem × dimension, and a budget of 2000 × d evaluations. The acceptance criteria are those of design doc §11.4, criterion 2.

**Criterion 1, quality: met.** `auxein-core-ga` is not worse than `auxein-default` (0.2.0) on any of the 15 problem × dimension cells: it is **better on all 15, with a large effect** (A₁₂ between 0.00 and 0.04 for 0.2.0, Holm-adjusted Mann–Whitney p ≤ 10⁻⁷ everywhere).

| Median final error | 0.2.0 (`auxein-default`) | `auxein-core-ga` | CMA-ES |
|---|---|---|---|
| sphere d = 10 | 2.19 | 1.2 × 10⁻²² | 1.7 × 10⁻¹⁴ |
| ellipsoid d = 10 | 5.5 × 10⁴ | 1.3 × 10³ | 1.5 × 10⁻¹⁴ |
| Rosenbrock d = 10 | 466 | 7.8 | 1.7 × 10⁻¹⁴ |
| Rastrigin d = 10 | 51.1 | 8.0 | 11.9 |
| sphere d = 30 | 7.58 | 1.3 × 10⁻²¹ | 2.8 × 10⁻¹⁴ |
| ellipsoid d = 30 | 1.8 × 10⁵ | 9.6 × 10³ | 2.3 × 10⁻¹⁴ |
| Rosenbrock d = 30 | 2,850 | 27.5 | 3.7 × 10⁻¹⁴ |
| Rastrigin d = 30 | 191 | 38.8 | 51.7 |

- The gains are largest where 0.2.0 was weakest. 0.2.0 reached an error of 10⁻³ in only 2 of its 375 runs and 10⁻⁶ in none; the new GA reaches 10⁻⁶ in 25/25 runs on the sphere and the noisy sphere at every dimension, in 23/25 on 2-D Rastrigin (the optimum, 0, exactly), in 9/25 on 2-D Rosenbrock and in 2/25 on the 2-D ellipsoid. It reaches 10⁻⁶ in no run on the ellipsoid, Rosenbrock or Rastrigin at d = 10 and 30.
- The mean rank of the median final error over the 15 cells is 1.40 for `auxein-core-ga`, 1.60 for CMA-ES, and 4.60 for `auxein-default`.

**Criterion 2, overhead: met after an optimisation.** Time per evaluation on a negligible-cost objective, at every population size and dimension of the overhead benchmark:

| μ (λ = 50) | d = 2 | d = 10 | d = 100 |
|---|---|---|---|
| 50: 0.2.0 → `auxein-core-ga` | 11.3 → 4.9 µs | 11.3 → 5.1 µs | 12.1 → 5.9 µs |
| 200 | 10.0 → 5.4 µs | 9.8 → 5.5 µs | 10.2 → 6.3 µs |
| 800 | 9.4 → 6.4 µs | 9.6 → 6.4 µs | 9.8 → 7.6 µs |

- **The first run of this comparison did not meet it.** Before any optimisation the new GA took 7.4 to 11.3 µs per evaluation, and at population 800 it was slower than 0.2.0 in two of three dimensions (10.0 against 9.4 µs at d = 10, 11.3 against 9.6 µs at d = 100) and equal at d = 2 (9.8 µs). Profiling showed the cost was in per-candidate `Evaluation` construction and validation, candidates rebuilt for the result check, a per-evaluation Python loop in the result tracker and, in the GA, Python work proportional to the population and two copies of every array on each `tell`. These were removed without changing any public behaviour; the existing tests pass unmodified. The driver's own cost per evaluation with `RandomSearch` fell from 5.8 to 3.7 µs at d = 10 (`auxein-core-random`) as a result.
- The new GA's cost per evaluation grows slowly with the population (ranking μ + λ members at each generation), where 0.2.0's falls, because its generation re-scores the whole population: 0.2.0 spends μ + 4 evaluations per generation (54, 204, 804), the new GA exactly λ = 50 at every μ, so 20,000 evaluations buy 399, 396 and 384 generations of the new GA against 369, 97 and 23 of 0.2.0.

**For information: the gap to CMA-ES.**

- CMA-ES stays better on the **ellipsoid** (condition number 10⁶, rotated) and on **Rosenbrock**, at every dimension, with a large effect: isotropic, per-individual step sizes cannot follow a rotated, ill-conditioned valley. On the 30-D ellipsoid the median error is 9.6 × 10³ against 2.3 × 10⁻¹⁴. The gap to CMA-ES on these problems shrank in ratio from 0.2.0's (for example on the 10-D ellipsoid, 5.5 × 10⁴ became 1.3 × 10³) but is still many orders of magnitude.
- The new GA is **better than CMA-ES on Rastrigin at every dimension** with a large effect (d = 2: 0 against 0.995; d = 10: 8.0 against 11.9; d = 30: 38.8 against 51.7), where 0.2.0 was behind CMA-ES (and not significantly different from it only at d = 2).
- On the sphere and the noisy sphere both solve the problem (10⁻⁶ in 25/25 runs); the GA's smaller final errors (10⁻²¹ to 10⁻²⁵ against 10⁻¹⁴ to 10⁻¹⁶) come from CMA-ES stopping at its own function-value tolerance, not from a better search, and are below every precision target.

**Checks of the comparison itself.** `auxein-core-random` is statistically indistinguishable from the harness's `random-search` (the cross-check test: A₁₂ = 0.48 on the 2-D sphere and 0.57 on the 10-D sphere, within [0.35, 0.65]). The harness sanity checks pass (CMA-ES reaches 10⁻⁶ on the 10-D sphere and ellipsoid in 25/25 runs; random search reaches 10⁻³ on the 10-D sphere in 0/25).
