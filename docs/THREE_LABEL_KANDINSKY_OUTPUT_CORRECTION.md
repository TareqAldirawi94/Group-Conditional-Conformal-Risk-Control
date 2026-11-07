# Three-label Kandinsky raw-output correction

During release packaging, a seeded source-to-data check found that `struct_raw.csv`
had been generated using the earlier NumPy-interpolated quantile for the
three-label Kandinsky comparator, while the corrected simulation source uses
an empirical order-statistic (the largest pinball minimizer).

Only the 21,600 Kandinsky rows (12 cells x 600 repetitions x three group views)
were recomputed using the original generator seeds and exact RNG sequence.
The non-Kandinsky rows remain unchanged, including direct-learned/direct-unshaped,
oracle, slice union, and atom-wise. Since only the Kandinsky comparator is
recomputed without re-running the expensive entire simulation,
`cell_runtime_s` remains the original full-cell runtime, not a new timing.

The updated `struct_raw.csv` and figure-generation script now agree with the
corrected order-statistic implementation. The original user-uploaded CSV was
not overwritten outside this release-candidate folder.
