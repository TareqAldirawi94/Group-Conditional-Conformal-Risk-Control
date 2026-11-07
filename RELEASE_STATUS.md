# Release status and scientific scope

## What is included

Corrected source files for the two uploaded batches; archived replication-level CSVs
for the three-label, fitted-hinge, Hamming direct, and historical DDR direct studies;
independent R/Python implementations of the disjoint ordinal and overlapping-group
examples; two-group analytic diagnostics; provenance-controlled DDR training and
CRC evaluation scripts.

## What is not yet independently certified

1. **Definitive DDR results.** The existing `ddr_raw.csv` is reproducible on the
   historical pool, but definitive real-data analysis requires newly scored images
   after duplicate-controlled classifier retraining, integrity evidence, and CRC
   reevaluation. Do not describe the historical 2,428-image results as definitive.
2. **DDR scoring inputs and images.** Deliberately not included. Obtain from an
   authorized source; check image/data license before sharing any derived score
   table or checkpoint.
3. **Full end-to-end replication.** Selected numerical records and methods were
   independently audited in the conversation, but this package has not yet been
   run through every full simulation from scratch on a clean machine. A clean
   environment, package versions, and final GitHub release/tag must be tested.
4. **R scripts.** Source reviewed, but R runtime was not available for execution
   in the audit environment.
5. **Kandinsky distinction.** The three-label indicator model admits the
   saturated-quantile implementation used in that experiment. The Hamming
   experiment uses `groupwise_quantile_union` solely as a coverage reference,
   not as a joint Kandinsky optimizer. No equal-guarantee assertion is made.
6. **No universal efficiency claim.** Direct CRC wins in some label-defined
   Hamming and three-label designs, but loses on the historical DDR pool;
   matched-eligibility feature-defined Hamming yields near equal unshaped
   performance. Manuscript wording must reflect these different regimes.

## Source-output consistency correction

The three-label archived Kandinsky rows were regenerated to match the final
order-statistic quantile code. The substantive direct/slice outcomes are unchanged.
See `docs/THREE_LABEL_KANDINSKY_OUTPUT_CORRECTION.md`.

## Deliberately omitted

- `*.pyc`, superseded duplicate copies of source files, temporary figures.
- Historical image-level DDR score CSV and all original clinical image files.
- An unprovided script named `ddr_crc_eval_u_marginalized.py` mentioned in the
  legacy table-generator docstring; the supplied independent evaluator is
  `ddr_crc_eval_integrity.py` and its documented U-redraw mode.

## Before public publication

Verify manuscript figures against these archived raw CSVs; rerun definitive DDR;
complete the software environment lockfile; confirm dataset/figure sharing
permissions; and cite a fixed Git commit/tag rather than a mutable branch.
