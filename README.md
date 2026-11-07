# Group-Conditional Conformal Risk Control — Reproducibility Scripts

**Source compilation from both corrected user-provided batches.** This repository
contains scripts for the manuscript's synthetic experiments, calibration
comparators, supporting diagnostics and DDR clinical application. It is a
**release candidate**, not a certification of the definitive DDR results.

## Quick start

```bash
python -m venv .venv
# Activate your environment (platform-specific), then:
python -m pip install -r requirements.txt
python verify_package.py
python regenerate_saved_figures.py
```

Figure regeneration uses **saved numerical outputs**; it does not rerun the
expensive experiments. Each script runs from its own experiment directory,
where its relative output paths are resolved. Full simulations may take
substantially longer, particularly at 25,600 observations or 50,000 repetitions.

For DDR image training install `requirements-ddr.txt`, provide the original
dataset, and follow `09_ddr/training/README.md`.

## Manuscript experiment map

| Directory | Source files | Purpose | Full rerun from directory |
|---|---|---|---|
| `01_disjoint_ordinal/` | `example1.py`, `example1.R` | Ordinal structured loss; Mondrian vs marginal and HB-RCPS | `python example1.py` or `Rscript example1.R` |
| `02_sandwich_precision/` | `simulation_sandwich.R` | Two-sided precision and rank-grid witness | `Rscript simulation_sandwich.R` |
| `03_overlapping_miscoverage/` | `example2.py`, `example2.R` | Label-overlapping miscoverage, rare-group barrier | `python example2.py` or `Rscript example2.R` |
| `04_hamming_reconciliation/` | `example2b.py`, `example2b.R` | Hamming slice/deployed gap (feature vs label overlap) | `python example2b.py` or `Rscript example2b.R` |
| `05_local_support_barrier/` | `fig_s4_barrier.R` | Necessary support-count barrier | `Rscript fig_s4_barrier.R` |
| `06_fitted_hinge/` | `fitted_hinge_sim.py`, `fitted_hinge_analyze.py`, `fitted_hinge_make_fig.py` | Weighted-hinge alignment + union | `python fitted_hinge_sim.py full 200` |
| `07_three_label_direct/` | `three_label_direct_sim.py`, `three_label_direct_make_fig.py` | Structured ordinal direct CRC with oracle/learned comparison | `python three_label_direct_sim.py full 600` |
| `08_hamming_direct/` | `hamming_direct_sim.py`, `hamming_direct_make_fig.py` | 7-candidate label-defined Hamming direct CRC; 8-candidate feature-defined control | `python hamming_direct_sim.py full 100` |
| `09_ddr/` | `training/`, `evaluation/`, `direct/` | Clinical DDR: retraining, provenance-gated original CRC, and direct comparison | See DDR notes below |
| `10_theoretical_diagnostics/` | four Python scripts | Numerical two-group optimization checks (auxiliary; not new empirical evidence) | See individual scripts |

**Experimental budgets.** No-shape full-budget baselines use all `n`
calibration observations. Learned shapes use independent fit/correction folds.
The correction-only union is an explicit sample-splitting diagnostic. The
three-label result uses 600 repetitions per configuration; Hamming direct uses
100; fitted-hinge uses 200 except n=6,400 with 120. Historic DDR direct uses
200 redraws per fitting fraction. R and Python twins are independent runs and
need not generate byte-identical values.

## Regenerate figures from saved raw results

From the repository root:

```bash
python regenerate_saved_figures.py
```

This runs the four figure makers for `fitshape_raw.csv`, `struct_raw.csv`,
`ham_raw.csv`, and **historical** `ddr_raw.csv`. To regenerate the older disjoint,
miscoverage, Hamming-reconciliation and sandwich figures, rerun their source
experiments from the directories shown above.

## DDR workflow and caveats

**The archived direct DDR results are historical.** The user-provided original
2,428-image score pool has not been demonstrated to come from the final
pre-split deduplicated and near-duplicate-screened retraining pipeline.

1. Obtain the original images and grading table with appropriate permission.
2. In `09_ddr/training`, set `DDR_IMG_DIR`, `DDR_CSV`, `DDR_OUTDIR`, plus the
   required pHash review or conservative-exclusion options. Run the trainer.
3. In `09_ddr/evaluation`, set `DDR_SCORES` to the new
   `scores_ddr_pool.csv`, `DDR_CRC_OUT` to a fresh output directory and
   `DDR_U_MODE=redraw`, then run `python ddr_crc_eval_integrity.py`.
4. From `09_ddr/evaluation`, pass the provenance-gated output path to
   `ddr_make_figures_integrity.py` and
   `ddr_make_manuscript_tables_integrity.py`.
5. Run `09_ddr/direct/ddr_direct_sim.py` with `DDR_POOL_CSV` pointing to the
   verified evaluation score CSV (name the source/path in the audit log).
   Recalculate the direct-method figures and replace any historical manuscript
   values **only after checking the new outputs**.

Illustrative shell commands (paths are examples; update for your machine):

```bash
cd 09_ddr/evaluation
DDR_SCORES=/path/to/ddr_scores_out/scores_ddr_pool.csv \
DDR_CRC_OUT=/path/to/results DDR_U_MODE=redraw python ddr_crc_eval_integrity.py
python ddr_make_figures_integrity.py /path/to/results/REAL_DATA_DEDUP_RETRAINED_REVIEWED/u_redrawn_each_split
python ddr_make_manuscript_tables_integrity.py /path/to/results/REAL_DATA_DEDUP_RETRAINED_REVIEWED/u_redrawn_each_split
```

The precise integrity directory is determined by the evaluated screening tier;
conservative-exclusion outputs use `REAL_DATA_DEDUP_RETRAINED_CONSERVATIVE`.

**Do not publish the historical DDR source score table or model weights until
you have verified their redistribution rights and provenance.** The aggregate
`09_ddr/direct/ddr_raw.csv` is included for transparent audit of the historical
experiment, not as definitive clinical evidence. The study cannot establish
patient-level independence from image identifiers alone.

A release-packaging audit corrected **Kandinsky-only** values in the archived
three-label output using the original RNG seeds; see
`docs/THREE_LABEL_KANDINSKY_OUTPUT_CORRECTION.md`. All other method rows were
unchanged.

## Reproducibility and provenance

`MANIFEST.csv` includes SHA-256 of each uploaded source and packaged file.
The corrected source file selection excludes older superseded copies, and one
Hamming source file has a **documentation-only clarification** of the candidate
universes (all 8 in feature-defined mode; nonzero 7 in label-defined mode).

- Python dependencies: `requirements.txt`; optional training: `requirements-ddr.txt`.
- R packages: `R_PACKAGES.md`.
- Review and outstanding items: `RELEASE_STATUS.md`.
- Author correction notes: `docs/`.

**For publication:** pin package versions, rerun from a fresh environment,
provide a versioned GitHub release or DOI, and ensure the final manuscript
matches the actual final figures/data.

There is no software license included. The repository owner should choose
and add a license deliberately before public publication.
