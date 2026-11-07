#!/usr/bin/env python3
"""Reproduce fig_sandwich.pdf and fig_sandwich_witness.pdf (matplotlib) from the
simulation_sandwich.R summary CSVs (sim_sandwich_outputs/). Same content as the
ggplot originals: benign saturation band and rank-grid witness sawtooth, faceted
by jump budget M."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ALPHA=0.4; MV=[1,2,4]

# ---- benign (upper edge) ----
s=pd.read_csv("sandwich_summary.csv")
fig,ax=plt.subplots(1,3,figsize=(11,3.7),sharey=True)
for i,M in enumerate(MV):
    d=s[s.M==M].sort_values("n_k")
    ax[i].fill_between(d.n_k, d.band_lo, d.band_hi, color="steelblue", alpha=0.15, zorder=0)
    ax[i].axhline(ALPHA, ls="--", color="grey", lw=1)
    ax[i].plot(d.n_k, d.risk, "-", color="steelblue", lw=0.7)
    ax[i].errorbar(d.n_k, d.risk, yerr=2*d.se, fmt="o", ms=4, color="steelblue", capsize=2)
    ax[i].set_xscale("log"); ax[i].set_xticks(d.n_k); ax[i].set_xticklabels(d.n_k, fontsize=7)
    ax[i].set_title(f"M = {M}", fontweight="bold"); ax[i].set_xlabel(r"calibration count $n_k$")
ax[0].set_ylabel("empirical within-group risk")
fig.suptitle(r"Benign saturation: risk in $[\max(0,\alpha-(M+1)B/(n_k+1)),\,\alpha]$ ($\alpha=0.4$, 50000 reps)", fontsize=9)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("fig_sandwich.pdf", bbox_inches="tight"); fig.savefig("fig_sandwich.png", dpi=150, bbox_inches="tight")

# ---- witness (lower edge) ----
w=pd.read_csv("sandwich_witness_summary.csv")
fig,ax=plt.subplots(1,3,figsize=(11,3.7),sharey=True)
for i,M in enumerate(MV):
    d=w[w.M==M].sort_values("n_k")
    ax[i].fill_between(d.n_k, d.band_lo, ALPHA, color="grey", alpha=0.25, zorder=0)
    ax[i].axhline(ALPHA, ls="--", color="grey", lw=1)
    ax[i].plot(d.n_k, d.predicted, "-", color="firebrick", lw=0.8, label=r"$\alpha-\Delta_{n_k,M}$")
    ax[i].plot(d.n_k, d.risk, "o", ms=2.6, color="black", label="empirical")
    ax[i].set_title(f"M = {M}", fontweight="bold"); ax[i].set_xlabel(r"calibration count $n_k$")
ax[0].set_ylabel("empirical within-group risk (witness)"); ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle(r"Rank-grid witness sawtooth ($\alpha=0.4$, 20000 reps/cell, 168 configs)", fontsize=9)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("fig_sandwich_witness.pdf", bbox_inches="tight"); fig.savefig("fig_sandwich_witness.png", dpi=150, bbox_inches="tight")
print("wrote fig_sandwich.{pdf,png} and fig_sandwich_witness.{pdf,png}")
