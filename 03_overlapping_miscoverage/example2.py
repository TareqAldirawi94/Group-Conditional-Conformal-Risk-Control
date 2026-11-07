"""
Section 5.2, E4a: Overlapping-group miscoverage and local support.
Python twin of the audited example2.R; numbers match within Monte Carlo error.

Audited design (matches example2.R):
  - fresh held-out test sample of NEVAL=6000 per replication, so across-rep
    Monte Carlo SEs include test-sampling variation;
  - TWO fallback definitions reported, not conflated:
      weighted_default_rate  = P(a test label's cell/group uses the maximal cutoff)
                               (prevalence-weighted);
      fraction_groups_default = fraction of the 8 atom cells / 4 original groups
                               that select the maximal cutoff (unweighted);
  - background group G_0={Y=000} miscoverage reported alongside the focal groups.

Y in {0,1}^L (L=3), 8 candidate vectors. Candidate-label focal groups
G_j={y_j=1}, j=1..L, plus background G_0. Miscoverage loss 1{y notin C},
B=1, alpha=0.10 (support barrier B/alpha=10). Methods: marginal, atom-wise
(2^L cells), union (4 original groups). Label 3 is a hard, rare group, so pooled
CRC under-covers G_3; the message is groupwise validity WITHOUT calibrating all
2^L atoms, not a smallest-set claim.
Emits figures/fig_overlap_miscoverage.{pdf,png}, example2_summary.csv,
example2_atoms.csv.
"""
import os, time
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

L, B, ALPHA = 3, 1.0, 0.10
NV = 2**L
VEC = np.array([[(a >> j) & 1 for j in range(L)] for a in range(NV)], float)
PAT = ["".join(str(int(b)) for b in VEC[a]) for a in range(NV)]
P   = np.array([0.55, 0.40, 0.18])
GAM = np.array([1.70, 1.00, 0.40])
NOISE = 1.0; TAU = norm.ppf(1 - P)
BARRIER = B / ALPHA
NS = [100, 400, 1600, 6400]; REPS = 500; NEVAL = 6000     # fresh test per replicate
LMAX = 1e9
POW2 = 2 ** np.arange(L)
METHODS = {"marg": 0, "atom": 1, "union": 2}
MLAB = {"marg": "Marginal CRC", "atom": "Atom-wise Mondrian", "union": "Union (ours)"}

def gen(n, rng, rho):
    f = rng.standard_normal((n, 1)); e = rng.standard_normal((n, L))
    z = np.sqrt(rho) * f + np.sqrt(1 - rho) * e
    Y = (z > TAU).astype(float)
    eta = GAM * (z - TAU) + NOISE * rng.standard_normal((n, L))
    q = np.clip(1 / (1 + np.exp(-eta)), 1e-6, 1 - 1e-6)
    base = (-np.log1p(-q)).sum(1); delta = np.log1p(-q) - np.log(q)
    sc = base[:, None] + delta @ VEC.T
    yid = (Y @ POW2).astype(int)
    return Y, yid, sc

def crc_cut(strue, alpha=ALPHA):
    n = len(strue); k = int(np.floor((n + 1) * alpha))
    if n == 0 or k < 1: return LMAX, True
    return float(np.partition(strue, n - k)[n - k]), False

def mc_hw(x): return 1.96 * x.std(ddof=1) / np.sqrt(len(x))

def run(rho, rng):
    nN = len(NS); nm = len(METHODS)
    Rg = np.full((REPS, L + 1, nm, nN), np.nan)   # group 0 = G0, 1..L = focal
    Rp = np.full((REPS, nm, nN), np.nan)
    Sz = np.full((REPS, nm, nN), np.nan)
    Fb = np.full((REPS, 2, nN), np.nan)           # weighted: col0 atom, col1 union
    Fg = np.full((REPS, 2, nN), np.nan)           # unweighted fraction of cells/groups
    atoms = np.zeros((nN, NV)); adefr = np.zeros((nN, NV))
    tmr = {"marg": 0.0, "atom": 0.0, "union": 0.0}
    for ni, n in enumerate(NS):
        for r in range(REPS):
            Yc, yidc, scc = gen(n, rng, rho); strue_c = scc[np.arange(n), yidc]
            t0 = time.perf_counter(); tm, _ = crc_cut(strue_c); tmr["marg"] += time.perf_counter() - t0
            cut_marg = np.full(NV, tm)
            t0 = time.perf_counter()
            ta = np.empty(NV); adef = np.zeros(NV, bool)
            for a in range(NV):
                ta[a], adef[a] = crc_cut(strue_c[yidc == a])
            tmr["atom"] += time.perf_counter() - t0
            cut_atom = ta.copy()
            atoms[ni] += np.bincount(yidc, minlength=NV); adefr[ni] += adef
            t0 = time.perf_counter()
            tf = np.empty(L); fdef = np.zeros(L, bool)
            for j in range(L): tf[j], fdef[j] = crc_cut(strue_c[Yc[:, j] == 1])
            t0b, bdef = crc_cut(strue_c[yidc == 0])
            cut_un = np.array([max(tf[k] for k in range(L) if VEC[a, k] == 1) if VEC[a].sum() > 0 else t0b
                               for a in range(NV)])
            tmr["union"] += time.perf_counter() - t0
            Fg[r, 0, ni] = adef.mean()
            Fg[r, 1, ni] = np.concatenate(([bdef], fdef)).mean()
            # fresh test sample for this replication
            Yp, yidp, scp = gen(NEVAL, rng, rho)
            gm = [yidp == 0] + [Yp[:, j] == 1 for j in range(L)]
            Fb[r, 0, ni] = adef[yidp].mean()
            Fb[r, 1, ni] = (cut_un[yidp] >= LMAX).mean()
            for name, cut in (("marg", cut_marg), ("atom", cut_atom), ("union", cut_un)):
                mi = METHODS[name]
                adm = scp <= cut[None, :]
                covered = adm[np.arange(NEVAL), yidp]
                for g in range(L + 1): Rg[r, g, mi, ni] = (~covered[gm[g]]).mean()
                Rp[r, mi, ni] = (~covered).mean()
                Sz[r, mi, ni] = adm.sum(1).mean()
        atoms[ni] /= REPS; adefr[ni] /= REPS
    return dict(Rg=Rg, Rp=Rp, Sz=Sz, Fb=Fb, Fg=Fg, atoms=atoms, adefr=adefr,
                cal_ms={k: 1000 * tmr[k] / (REPS * nN) for k in tmr})

def mk_summary(res, cond):
    rows = []
    for name, mi in METHODS.items():
        defidx = 0 if name == "atom" else 1
        for ni, n in enumerate(NS):
            is_marg = name == "marg"
            rows.append(dict(
                condition=cond, method=MLAB[name], n=n,
                pooled_risk=res["Rp"][:, mi, ni].mean(),
                risk_G0=res["Rg"][:, 0, mi, ni].mean(),
                risk_G1=res["Rg"][:, 1, mi, ni].mean(),
                risk_G2=res["Rg"][:, 2, mi, ni].mean(),
                risk_G3=res["Rg"][:, 3, mi, ni].mean(),
                mean_size=res["Sz"][:, mi, ni].mean(),
                weighted_default_rate=np.nan if is_marg else res["Fb"][:, defidx, ni].mean(),
                fraction_groups_default=np.nan if is_marg else res["Fg"][:, defidx, ni].mean(),
                cal_ms_per_rep=res["cal_ms"][name],
                risk_G3_hw=mc_hw(res["Rg"][:, 3, mi, ni]),
                mean_size_hw=mc_hw(res["Sz"][:, mi, ni])))
    return pd.DataFrame(rows)

rng = np.random.default_rng(20261025)
res_ind = run(0.0, rng); res_cor = run(0.5, rng)
summary = pd.concat([mk_summary(res_ind, "independent"), mk_summary(res_cor, "correlated")],
                    ignore_index=True)

os.makedirs("figures", exist_ok=True)
summary.to_csv("figures/example2_summary.csv", index=False)
atom_tab = pd.DataFrame({"atom": PAT, "avg_count_n100": res_ind["atoms"][0],
                         "default_rate_n100": res_ind["adefr"][0]})
atom_tab.to_csv("figures/example2_atoms.csv", index=False)

pd.set_option("display.width", 180, "display.max_columns", 30)
print(f"Example 2: L={L}, alpha={ALPHA:.2f}, B/alpha={BARRIER:.0f}, {REPS} reps, "
      f"{NEVAL} fresh test samples per rep\n")
show = ["condition","method","n","risk_G0","risk_G1","risk_G2","risk_G3",
        "mean_size","weighted_default_rate","fraction_groups_default"]
print(summary[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nAtom counts and fallback rates at n=100 (independent labels):")
print(atom_tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---- figure (independent condition) ----
R = res_ind; col = {"marg": "#8c8c8c", "atom": "forestgreen", "union": "firebrick"}
def make_plot(path, dpi=130):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.1)); xs = np.array(NS); gx = np.arange(L); w = 0.26
    for i, m in enumerate(("marg", "atom", "union")):
        mu = np.array([R["Rg"][:, g + 1, METHODS[m], 0].mean() for g in range(L)])
        e  = np.array([mc_hw(R["Rg"][:, g + 1, METHODS[m], 0]) for g in range(L)])
        ax[0].bar(gx + (i - 1) * w, mu, w, yerr=e, capsize=2, color=col[m], label=MLAB[m])
    ax[0].axhline(ALPHA, ls="--", color="k", lw=1, label=r"$\alpha=0.10$")
    ax[0].set_xticks(gx); ax[0].set_xticklabels([r"$G_1$", r"$G_2$", r"$G_3$ (hard, rare)"])
    ax[0].set_ylabel("conditional miscoverage"); ax[0].set_title(r"Per-group miscoverage, $n=100$"); ax[0].legend(fontsize=7)
    for m in ("marg", "atom", "union"):
        ax[1].plot(xs, [R["Sz"][:, METHODS[m], ni].mean() for ni in range(len(NS))], "o-", color=col[m], label=MLAB[m])
    ax[1].set_xscale("log"); ax[1].set_xticks(xs); ax[1].set_xticklabels(NS)
    ax[1].set_xlabel("calibration size $n$"); ax[1].set_ylabel("mean set size")
    ax[1].set_title("Prediction-set size"); ax[1].legend(fontsize=7)
    ax[2].plot(xs, [R["Fb"][:, 0, ni].mean() for ni in range(len(NS))], "s-", color=col["atom"], label="atom-wise")
    ax[2].plot(xs, [R["Fb"][:, 1, ni].mean() for ni in range(len(NS))], "^-", color=col["union"], label="union (ours)")
    ax[2].set_xscale("log"); ax[2].set_xticks(xs); ax[2].set_xticklabels(NS)
    ax[2].set_xlabel("calibration size $n$"); ax[2].set_ylabel("test-weighted fallback rate")
    ax[2].set_title("Local-support cost: atoms vs focal groups"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()

make_plot("figures/fig_overlap_miscoverage.pdf"); make_plot("figures/fig_overlap_miscoverage.png")
print("\nSaved figures/fig_overlap_miscoverage.{pdf,png} and two CSV files.")
