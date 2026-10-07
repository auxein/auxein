# Auxein benchmark report: ga-default-selection

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

## 1. Convergence

Median best-so-far error against fitness evaluations, with the interquartile band over runs.

![sphere, d=10](convergence-sphere-d10.png)

![sphere, d=30](convergence-sphere-d30.png)

![ellipsoid, d=10](convergence-ellipsoid-d10.png)

![ellipsoid, d=30](convergence-ellipsoid-d30.png)

![rosenbrock, d=10](convergence-rosenbrock-d10.png)

![rosenbrock, d=30](convergence-rosenbrock-d30.png)

![rastrigin, d=10](convergence-rastrigin-d10.png)

![rastrigin, d=30](convergence-rastrigin-d30.png)

![noisy_sphere, d=10](convergence-noisy_sphere-d10.png)

![noisy_sphere, d=30](convergence-noisy_sphere-d30.png)

## 2. Summary

Final error, success rate per precision target (runs that reached it) and expected running time (ERT, in evaluations).

### sphere, d=10

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 1.10e-22 [9.34e-23, 1.19e-22] | 15/15 | 15/15 | 15/15 | 1,139 | 1,963 | 3,214 |
| ga-b | 1.83e-22 [1.69e-22, 1.97e-22] | 15/15 | 15/15 | 15/15 | 1,328 | 2,615 | 4,461 |
| ga-c | 1.10e-22 [8.40e-23, 1.22e-22] | 15/15 | 15/15 | 15/15 | 1,105 | 1,897 | 3,138 |

### sphere, d=30

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 1.19e-21 [1.09e-21, 1.25e-21] | 15/15 | 15/15 | 15/15 | 3,142 | 5,099 | 8,157 |
| ga-b | 5.15e-17 [5.63e-21, 2.53e-13] | 15/15 | 15/15 | 15/15 | 5,038 | 10,420 | 21,601 |
| ga-c | 1.23e-21 [1.14e-21, 1.26e-21] | 15/15 | 15/15 | 15/15 | 3,286 | 5,218 | 8,048 |

### ellipsoid, d=10

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 1.02e+03 [618, 1.39e+03] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-b | 1.41e+03 [700, 1.83e+03] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-c | 850 [634, 1.57e+03] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |

### ellipsoid, d=30

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 6.26e+03 [5.35e+03, 1.07e+04] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-b | 7.29e+03 [6.35e+03, 9.81e+03] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-c | 7.49e+03 [4.32e+03, 9.31e+03] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |

### rosenbrock, d=10

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 6.97 [6.44, 7.79] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-b | 5.91 [5.8, 6.1] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-c | 6.46 [5.61, 7.25] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |

### rosenbrock, d=30

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 29.4 [26.8, 88.9] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-b | 25.9 [24.7, 78.7] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-c | 29.4 [25.5, 89.4] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |

### rastrigin, d=10

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 9.95 [5.97, 15.9] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-b | 8.95 [5.47, 11.9] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-c | 9.95 [6.47, 11.9] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |

### rastrigin, d=30

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 36.8 [29.4, 41.3] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-b | 32.8 [24.4, 44.3] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |
| ga-c | 38.8 [35.8, 50.7] | 0/15 | 0/15 | 0/15 | ∞ | ∞ | ∞ |

### noisy_sphere, d=10

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 1.18e-22 [1.07e-22, 1.35e-22] | 15/15 | 15/15 | 15/15 | 1,108 | 2,006 | 3,257 |
| ga-b | 1.97e-22 [1.70e-22, 2.07e-22] | 15/15 | 15/15 | 15/15 | 1,344 | 2,506 | 4,281 |
| ga-c | 9.31e-23 [7.96e-23, 1.25e-22] | 15/15 | 15/15 | 15/15 | 1,107 | 1,947 | 3,222 |

### noisy_sphere, d=30

| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |
|---|---|---|---|---|---|---|---|
| ga-a | 1.25e-21 [1.15e-21, 1.37e-21] | 15/15 | 15/15 | 15/15 | 3,636 | 5,868 | 9,360 |
| ga-b | 5.94e-06 [1.60e-06, 1.35e-03] | 14/15 | 11/15 | 3/15 | 10,188 | 33,670 | 263,737 |
| ga-c | 1.23e-21 [1.13e-21, 1.31e-21] | 15/15 | 15/15 | 15/15 | 3,614 | 5,955 | 9,316 |

### Mean rank

Rank of each algorithm by the median final error in each problem × dimension (1 = best), and the mean rank.

| Algorithm | sphere d=10 | sphere d=30 | ellipsoid d=10 | ellipsoid d=30 | rosenbrock d=10 | rosenbrock d=30 | rastrigin d=10 | rastrigin d=30 | noisy_sphere d=10 | noisy_sphere d=30 | Mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ga-a | 1 | 1 | 2 | 1 | 3 | 2 | 2 | 2 | 2 | 2 | 1.80 |
| ga-b | 3 | 3 | 3 | 2 | 1 | 1 | 1 | 1 | 3 | 3 | 2.10 |
| ga-c | 2 | 2 | 1 | 3 | 2 | 3 | 3 | 3 | 1 | 1 | 2.10 |

## 3. Statistical comparison

Final error of `auxein-default` against every other algorithm: two-sided Mann-Whitney U with Holm correction within each table, and the Vargha-Delaney A12 effect size (the probability that `auxein-default` wins).

No `auxein-default` runs in this benchmark.

## 4. Overhead

The overhead benchmark was not run.

## 5. Sanity checks

- CMA-ES reaches 1e-6 on 10-D sphere: not run in this benchmark
- CMA-ES reaches 1e-6 on 10-D ellipsoid: not run in this benchmark
- random search does not reach 1e-3 on 10-D sphere: not run in this benchmark

## 6. Methodology

- **Config**: `ga-default-selection`, commit `6318268c2c`, run at 2026-10-07T21:38:30+00:00 on 11 worker processes.
- **Software**: Python 3.12.13, auxein 0.2.0, numpy 2.5.3, pycma 4.5.0, scipy 1.18.1.
- **Machine**: Apple M4 Pro (12 logical CPUs), macOS-15.6.1-arm64-arm-64bit.
- **Budget**: 2000 × d fitness evaluations for every algorithm, counted outside the algorithms by `CountingObjective`. The initial population counts, and a partial generation is fine. Progress is always against evaluations, never generations.
- **Domain and conventions**: search domain [-5, 5]^d for every problem, boundaries not enforced. Optimum value 0, results are errors f(x) − f*. Algorithms see the noisy value on the noisy problem, the trace records the noise-free error of the evaluated point.
- **Instances**: each instance has its own random shift x* ~ U[-4, 4]^d (and, for the ellipsoid, a uniform random rotation from a QR decomposition with sign correction), drawn from a generator keyed by the instance id and the dimension, separate from algorithm randomness. Run k of every algorithm uses instance 1000 + k, so comparisons are paired. 15 runs per combination.
- **Seeds**: run k uses seed 1000 + k. Auxein is seeded with `np.random.seed(seed)` (it uses global numpy randomness), random search with `np.random.default_rng(seed)`, CMA-ES with seed + 1.
- **Precision targets**: 0.1, 0.001, 1e-06.
- **Traces**: best-so-far error at about 20 log-spaced evaluation counts per decade, plus the first and last evaluation. Plots show the median and the interquartile band over runs at each checkpoint. Errors below 1e-10 are drawn at 1e-10; dotted lines mark the targets.
- **Statistics**: two-sided Mann-Whitney U test on the final error of `auxein-default` against every other algorithm (scipy), with Holm correction over the comparisons of each problem × dimension table. A12 is the probability that a `auxein-default` run ends with a lower error than a run of the other algorithm (ties count half); 0.5 is no difference. Effect size labels follow Vargha and Delaney: negligible below |A12 − 0.5| = 0.06, small below 0.14, medium below 0.21, large above. Significance level 0.05.
- **ERT (expected running time)**: total evaluations spent over all runs, divided by the number of successful runs, infinity if there is none. A successful run spends the evaluations it needed to first reach the target, an unsuccessful run spends everything it evaluated. It estimates the evaluations needed to reach the target if failed runs were restarted from scratch.
- **Algorithms**:
  - `ga-a` (adapter `auxein_core_ga`): `{"population_size": 50, "offspring_size": 50, "selection": {"type": "tournament", "size": 2}, "recombination": {"type": "intermediate"}, "mutation": {"type": "self_adaptive", "per_gene": false}, "repair": "clip"}`
  - `ga-b` (adapter `auxein_core_ga`): `{"population_size": 50, "offspring_size": 50, "selection": {"type": "tournament", "size": 2}, "recombination": {"type": "intermediate"}, "mutation": {"type": "self_adaptive", "per_gene": true}, "repair": "clip"}`
  - `ga-c` (adapter `auxein_core_ga`): `{"population_size": 50, "offspring_size": 50, "selection": {"type": "sus", "scaling": 2.0}, "recombination": {"type": "intermediate"}, "mutation": {"type": "self_adaptive", "per_gene": false}, "repair": "clip"}`
- **Overhead benchmark**: not run

### Algorithms and problems

| Problem | Definition (z = transformed x) | Tests |
|---|---|---|
| sphere | `Σ z_i²`, z = x − x* | sanity: everything should solve it |
| ellipsoid | `Σ 10^(6(i−1)/(d−1)) z_i²` (condition number 10⁶), z = R(x − x*) with a random rotation R | step-size adaptation across very different scales; rotation defeats per-axis tricks |
| rosenbrock | `Σ_{i<d} 100(z_{i+1} − z_i²)² + (1 − z_i)²`, z = x − x* + 1 | following a narrow curved valley |
| rastrigin | `10d + Σ (z_i² − 10 cos(2π z_i))`, z = x − x* | many regularly spaced local optima |
| noisy_sphere | sphere × `(1 + 0.1·ε)`, ε ~ N(0, 1) per call | robustness to noisy fitness (the trace records the noise-free error) |

### Reproducing

```
uv run --group bench python -m benchmarks run --config benchmarks/configs/ga-default-selection.toml
uv run --group bench python -m benchmarks report <results-dir>
```

The same config, commit and seeds give identical `runs.jsonl` contents, except for the `wall_time` fields.
