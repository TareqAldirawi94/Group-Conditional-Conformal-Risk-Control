# Corrections applied to the first 10 scripts (external audit response)

My assessment: the audit is careful and correct on its two publication-critical
points. A1 is a real defect (the three-label slice union was calibrated on the
correction fold, not the full budget) and is in fact the same fairness fix already
present in the Hamming experiment. A2 is a valid labeling concern. The remaining
items are sound code-hygiene points. Details and what I changed below.

## Publication-critical (three_label_direct_sim.py)

**A1 — full data budget for the no-shape methods.** The slice union, direct
unshaped, and direct oracle now calibrate on the **full n** observations. Only
`direct_learned` splits (shape on the fit fold, CRC on the correction fold). A
diagnostic `slice_union_cor` (union on the correction fold only) is added to
isolate the cost of splitting. Fit-fold subsampling (<=2000) was added to the
shape search for runtime; the correction fold is untouched, so exact validity is
preserved.

Effect on results (`struct_raw.csv` regenerated, 600 reps/cell):
- **n=6400 (headline) barely moves**, all within Monte Carlo error:
  slice union 2.870 (was 2.872), direct unshaped 2.760 (2.763),
  direct learned 2.740 (2.744), direct oracle 2.687 (2.692),
  Kandinsky 2.699 at worst-group risk 0.131 (was 2.700, 0.130).
- **n=100 changes qualitatively.** With the full budget, the slice union
  (2.927) and the full-budget direct methods (unshaped 2.889, oracle 2.893) do
  **not** fall back. Only `direct_learned`, whose correction fold drops below
  B/alpha=40 per group, defaults (size 3.0). The correction-only union
  (`slice_union_cor`) does fall back (~88%), which is the "86%" figure the
  manuscript previously attributed to the slice union.

  >> MANUSCRIPT ACTION: revise the Section 5.3 three-label n=100 sentence. It
  currently says "all direct variants default ... against 86% for the slice
  union." The corrected statement: at n=100 the full-budget slice union and the
  full-budget direct methods do not default; only the learned variant, which must
  split its budget, falls below the support barrier and defaults. The global-
  threshold fragility is thus exhibited by the split (learned) variant here.

**A2 — Kandinsky / fitted_union labeling.**
- The old `fit_hinge_shape` computed per-label quantiles and was called
  "Kandinsky". In THIS model the group-indicator quantile regression Kandinsky
  uses is *saturated* over the three realized membership patterns
  phi(0)=(1,0), phi(1)=(0,1), phi(2)=(1,1), so its pinball minimizer equals the
  per-pattern (= per-label) quantile exactly. I verified this numerically (a full
  `sklearn` QuantileRegressor returns the same thresholds to ~1e-3). The function
  is renamed `kandinsky_quantile` with a docstring that states the equivalence, so
  the comparator is now correctly justified as Kandinsky's construction for this
  design (the observed risk above alpha reflects its coverage target, not a
  failure of its own guarantee).
- `fitted_union` was an inclusion-type weighted-hinge construction misapplied to a
  structured-loss experiment, and was not shown in the figure. It is **removed**.

**A3 — replication command.** The `full` default is now 600 reps/cell (the
published design). Run: `python three_label_direct_sim.py full 600`. The
regenerated `struct_raw.csv` is included.

## Code hygiene

- **make_ddr_scores_deduplicated.py** — the personal Windows default paths are
  replaced with `PATH/TO/DDR/...` placeholders that error unless `DDR_IMG_DIR`
  and `DDR_CSV` are set. The determinism comment is softened to state that
  bitwise reproducibility is not guaranteed across GPUs/driver/library versions
  (the code already uses `warn_only=True`).
- **two_group_efficiency.py** — the SLSQP solve now asserts `res.success` and
  checks the risk-constraint residual before returning. The objective is already
  documented as an artificial linear deployment cost, not set cardinality.
- **reconciliation_slack.py** — the SLSQP solve now checks success and feasibility.
  A docstring note states this is a Gaussian two-group diagnostic with a linear
  size-aware cost, distinct from the appendix's bounded-uniform illustration.
- **two_group_oracle.py / two_group_oracle_sweep.py** — the docstrings now state
  the "oracle" is a finite 400-point grid search (a numerical grid oracle), not a
  certified global optimum.
- **simulation_sandwich.R** — auto-installation of packages is removed (it now
  stops with an install instruction); the worker count is pinned (default 4,
  override with `SANDWICH_WORKERS`).

## Unchanged (no code change required)

- **ddr_crc_eval_integrity.py** — the numerical kernel passed the audit's
  independent 100-redraw reproduction. The provenance gate (hash checks against
  the scoring recipe and dedup manifests) is an integrity feature and is kept;
  running it requires the DDR data release listed in the main README.
- **three_label_direct_make_fig.py** — no code change; it plots the same six
  methods and now renders from the regenerated `struct_raw.csv`
  (`fig_direct_struct.pdf/png` included).
- **ddr_analysis_scores.csv** — unchanged; the audit confirmed 2,428 rows,
  grade counts 1186/125/889/47/181, and the classifier metrics (QWK 0.8507).
