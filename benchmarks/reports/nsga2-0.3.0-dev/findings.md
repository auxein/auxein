## Findings

`NSGA2` (`auxein-nsga2`, its defaults: population 100, offspring 100, binary crowded tournament, simulated binary crossover with η = 15 and probability 0.9, polynomial mutation with η = 20) against pymoo's NSGA-II with the same population, operators and parameters, and against uniform random search. Four problems with known fronts: ZDT1, ZDT2 and ZDT3 (25,000 evaluations), DTLZ2 (30,000), 25 runs each with fixed seeds. The quality of a run is the **hypervolume** (against 1.1 × the nadir of the true front) and the **IGD+** of the non-dominated set of everything it evaluated; the statistics are Mann–Whitney with Holm correction and the Vargha–Delaney A₁₂. The acceptance criteria are those of design doc §11.4, criterion 7.

**Criterion 1: `auxein-nsga2` is not statistically worse than pymoo's NSGA-II on final hypervolume, on any problem: met.** It is **significantly better on all four**, by small margins:

| Problem | Final hypervolume, `auxein-nsga2` | pymoo | A₁₂ | Holm p | IGD+ (Auxein, pymoo) |
|---|---|---|---|---|---|
| ZDT1 | 0.8754 | 0.8748 | 1.00 | 2 × 10⁻⁹ | 6.9 × 10⁻⁴, 1.0 × 10⁻³ |
| ZDT2 | 0.5420 | 0.5410 | 1.00 | 1.4 × 10⁻⁹ | 6.7 × 10⁻⁴, 1.2 × 10⁻³ |
| ZDT3 | 1.0256 | 1.0249 | 0.95 | 5.5 × 10⁻⁸ | 3.5 × 10⁻⁴, 5.3 × 10⁻⁴ |
| DTLZ2 | 0.7941 | 0.7937 | 0.81 | 2 × 10⁻⁴ | 5.6 × 10⁻³, 5.8 × 10⁻³ |

- **Read the margins before the p-values.** Both implementations end at 99.7 to 99.9 % of the hypervolume of the true front on the ZDT problems; the gap is 0.05 to 0.2 % of it. It is consistent (a large effect in rank terms: Auxein wins nearly every pair of runs) and it is tiny. It reaches 95 % of the true front's hypervolume earlier as well (median 7,943 evaluations on ZDT1 against 8,913; 11,220 against 12,589 on ZDT2; 5,012 against 5,623 on DTLZ2; equal on ZDT3).
- **The cause was not isolated.** The two implementations are not identical, whatever the parameters say: Auxein's SBX is the *unbounded* form (children may leave the box and are clipped back, which is favourable on ZDT, where the optimum of 29 of the 30 variables lies on the bound 0), pymoo's is the *bounded* form of Deb's code; pymoo removes duplicate offspring and Auxein does not; Auxein makes one child per pair of parents and pymoo two. DTLZ2, whose optimum is in the interior, shows the smallest margin, which fits the first explanation but does not prove it. The comparison is exactly the one described: the same parameters on two implementations.
- **Nothing was tuned against these instances.** The defaults are the textbook ones from the design (§3.3), fixed before the benchmark existed; the config was run twice (once from a working tree with uncommitted changes to the report code, once on the clean commit recorded in `metadata.json`, which is the one committed here), and the 300 runs were identical (apart from wall time).

**Criterion 2: `auxein-nsga2` is significantly better than random search on every problem: met** (A₁₂ = 1.00, Holm p ≤ 3 × 10⁻⁹ everywhere). Random search never gets a point inside the reference box of a ZDT problem within the budget (its hypervolume is 0, so look at IGD+: 1.4 to 2.8 against 3.5 × 10⁻⁴ to 1.2 × 10⁻³), and reaches 42 % of the true front's hypervolume on DTLZ2 against 100 %.

**What the fronts look like.** On ZDT3 the final fronts of both algorithms cover the five disconnected pieces of the true front; on DTLZ2 the 7,000 or so non-dominated points of both cover the whole octant of the sphere (the plot shows 2,500 of them). The hypervolume of a front can exceed that of the sample of the true front by a hair (100.4 % on DTLZ2), because the sample is finite.

**Cost.** Median wall time of a run (it includes the harness's indicator computations, identical for both): `auxein-nsga2` 1.4 to 1.6 s on the ZDT problems and 7.9 s on DTLZ2, pymoo 1.0 to 1.1 s and 4.6 s, on 12 cores with one numpy thread per process. Auxein is 1.4 to 1.7 times slower here; the difference is the Python of the driver and of the strategy around each generation, not the sorting. The driver's Pareto archive, which grows to thousands of points on DTLZ2, was a Python loop per evaluation that took 53 s per DTLZ2 run before it was vectorised as part of this step (6 s after).
