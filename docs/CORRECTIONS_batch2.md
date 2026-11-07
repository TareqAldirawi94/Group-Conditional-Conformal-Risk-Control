# Batch-2 corrections (second-batch audit response)

My assessment: the batch-2 audit is correct on all substantive points. The most
important is P0-1, which changes the interpretation of the feature-defined Hamming
result. The label-defined (positive) Hamming result and the DDR negative result
are unaffected. Details and what I changed below.

## P0-1 (publication-critical): feature-defined Hamming eligibility (`hamming_direct_sim.py`)

In mode A (feature-defined), the per-row union deploys the EMPTY set when a test
point belongs to no feature group (~1/8 of points under independent Bernoulli
indicators), while the global-threshold direct methods kept admitting candidates.
That gave the union a size advantage unrelated to threshold flexibility.

Fix: in mode A the direct family (`direct_unshaped`, `direct_learned`, `atomwise`)
now uses the SAME "no group => empty" eligibility gate as the union
(`gate = mte.any(axis=1)`), and the learned-shape fit-fold size objective uses the
same gate (P1-2). Conditional group risk is unchanged (any G_j implies the gate is
true); only overall size changes. Reran the full sweep.

**Corrected feature-defined result (n=25,600, 50/50):**

| Method | Before (artifact) | After (matched eligibility) |
|---|---:|---:|
| Slice union | 1.321 | 1.321 |
| Direct, unshaped | 1.513 | **1.325** (≈ union) |
| Direct, learned | 1.929 | 1.688 |

>> MANUSCRIPT ACTION (Section 5.3, feature-defined Hamming and the two-mechanism
framing). The earlier claim that the global threshold is "less flexible than the
union's per-candidate cutoffs when the slices are already tight" is NOT supported:
with matched eligibility the unshaped direct size (1.325) matches the union
(1.321), so the global threshold imposes essentially no penalty in this symmetric
zero-gap case. Only the LEARNED SHAPE inflates the set (1.688), the cost of fitting
a shape when there is no reconciliation slack to exploit. The global-threshold
COUPLING cost is demonstrated by DDR (heterogeneous group demands), where even
unshaped direct is larger (2.518 vs 2.322), not by the symmetric feature-defined
case. Reframe the "cost" mechanism accordingly: global-threshold coupling bites
under group heterogeneity (DDR); in the symmetric zero-gap case the cost is the
shape fit, not the single threshold.

The LABEL-DEFINED (mode B) result is unchanged: direct unshaped 1.313 and learned
1.157 versus union 1.579 at n=25,600 (16.8% and 26.7% reductions), all valid. The
positive result stands.

## P0-2: Hamming "kandinsky" is not the joint fit (`hamming_direct_sim.py`)

Unlike the three-label model, the three focal indicators are NOT saturated over the
seven nonzero label patterns, so three independent conditional quantiles are not
the joint group-indicator pinball optimizer. The method is renamed
`groupwise_quantile_union`, its numbers are no longer attributed to Kandinsky, and
the docstring states the distinction. (It is a coverage reference, omitted from the
figure.) It is not in the figure, but it is now correctly labeled in the raw CSV.
Its empirical quantile also now uses the largest-pinball-minimizer convention.

## P1-1: Hamming figure risk statistic (`hamming_direct_make_fig.py`)

The worst-group panel plotted `mean_rep(max_j risk)`, which is upward-biased by
selection. It now plots `max_j(mean_rep risk_j)`, matching the theorem and the
three-label plot. The y-axis is relabeled `max_j mean risk`. Consequence: the
learned-direct label-defined worst-group risk is 0.493 (not 0.496), consistent
with the earlier batch-1 note.

## P1-3: support-barrier figure (`fig_s4_barrier.R`)

`cnt >= 9` is a NECESSARY support count for a nonmaximal cutoff, not a guarantee of
a nontrivial set, and the mass is computed in-sample on the calibration draw, not
on an independent test set. Captions, legend, and console lines are relabeled
accordingly ("cells meeting support count", "in-sample mass in supported cells",
"below support count"). `install.packages` is replaced by a fail-fast stop.

## P1-4: DDR figure pooling (`ddr_direct_make_fig.py`)

Panels 1-2 now use the matched 50/50 redraw stream for the full-budget baselines
(they are split-independent by design) instead of pooling the three split-seeded
streams, with a disclosing comment; Monte Carlo error bars (+/-1 SE over redraws)
are added. Panel 3 keeps the split-sensitivity view.

## P1-5 / P0-3: DDR direct sim documentation and provenance (`ddr_direct_sim.py`)

The docstring now states the full learned-shape settings (LEVELS^5 = 3,125 library,
zero margin, argmin tie-break, fit subsample cap 600 with deterministic seed, the
25/75/50/50/75/25 allocation, 200 redraws), documents that under infeasibility the
fitter returns the all-ones shape (no separate None branch), and clarifies that
`full_universe`/`fallback` is a threshold decision, not a per-row full-set flag.
The "pre-registered" comment is softened to an author claim the script cannot
prove. The header now labels the 2,428-row pool as HISTORICAL and states that this
script does not establish a leakage-controlled classifier; the integrity pipeline
and its provenance package are required for definitive real-data results, and
`ddr_crc_eval_integrity.py` correctly refuses definitive-mode runs without them.
This is a data-pipeline action on the authors' side.

## P1-6: fitted-hinge replication count (`fitted_hinge_sim.py`)

The cell-specific rep count (200 for n in {100,400,1600}, 120 for n=6400) is now
documented in code, with a note that uncertainty estimates must use the per-cell
count.

## P1-7: raw outputs included

`ham_raw.csv`, `ddr_raw.csv`, and `fitshape_raw.csv` (plus the regenerated figures)
are included so the published aggregates can be checked against the manuscript.

## Not changed (confirmed sound by the audit)

`example1/2/2b` (.py and .R), `fitted_hinge_sim.py` kernel (genuine joint quantile
solver), `ddr_make_figures_integrity.py`, `ddr_make_manuscript_tables_integrity.py`,
and the exact CRC kernels (500/500 brute-force checks passed).
