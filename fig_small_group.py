#!/usr/bin/env python3
"""
fig_small_group.py  -- reconstructed generator for the support-barrier figure
(Appendix, app:reproducibility-barrier / fig:sim-small-group).

Empirical maximal-threshold return (default) rates against n+1 for three designs:
  (A, left)   ordinal absolute-error loss, marginal group g==1, B=4
  (B, middle) multilabel false-negative-rate loss, size 1..3, g==1, B=1
  (C, right)  ordinal absolute-error loss, classwise groups g(x,y)=y, B=4,
              plus the randomized rank-interpolation comparator.

Scores are iid Unif(0,1); alpha in {0.05, 0.10, 0.20}. For designs with a
zero-loss full set the deterministic Mondrian rule returns the maximal threshold
iff n+1 < B/alpha (data-independent), so its default rate is the step
1{n+1 < B/alpha}; this is confirmed here by genuine CRC event-sweeps rather than
asserted. The randomized comparator (design C) selects S_(kappa+1) w.p.
theta = N*alpha/B - floor(N*alpha/B) and S_(kappa) otherwise, with
S_(0)=lambda_max; in the sub-barrier regime (kappa=0) it returns lambda_max with
probability 1 - N*alpha/B.

500 replications per deterministic panel, 20000 for the randomized comparison.
Output: fig_small_group.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHAS   = [0.05, 0.10, 0.20]
REPS_DET = 500
REPS_RND = 20000
SEED     = 20261103

# ---------- genuine CRC event-sweep default test --------------------------------
def _default_from_events(locs, drops, n, alpha, B):
    """locs, drops: 1-D arrays of admission events (location, loss decrease).
    Returns True iff the corrected CRC criterion is infeasible at every threshold,
    i.e. the deployed threshold is the maximal one (full-set default)."""
    target = alpha * (n + 1) - B
    if target < -1e-12:                 # B/(n+1) > alpha : infeasible even at full set
        return True
    o = np.argsort(locs, kind="stable")
    total = n * B - np.cumsum(drops[o])  # running summed loss as candidates are admitted
    return bool(total.min() > target + 1e-9)

def default_ordinal(N, alpha, rng, K=5, B=4):
    """Design A/C-deterministic: ordinal distance-to-set, nested by candidate scores."""
    n = N - 1
    if n <= 0:
        return True
    s = rng.random((n, K)); Y = rng.integers(0, K, n)
    order = np.argsort(s, axis=1)
    ss = np.take_along_axis(s, order, axis=1)
    dd = np.take_along_axis(np.abs(np.arange(K)[None, :] - Y[:, None]), order, axis=1)
    runmin = np.minimum.accumulate(dd, axis=1)
    prev = np.concatenate([np.full((n, 1), B, float), runmin[:, :-1]], axis=1)
    drops = (prev - runmin).ravel()
    return _default_from_events(ss.ravel(), drops, n, alpha, B)

def default_fnr(N, alpha, rng, L=5, B=1):
    """Design B: multilabel false-negative rate for true-label sets of size 1..3."""
    n = N - 1
    if n <= 0:
        return True
    s = rng.random((n, L))
    sizes = rng.integers(1, 4, n)                       # 1..3 true labels
    r = rng.random((n, L))
    kth = np.sort(r, axis=1)[np.arange(n), sizes - 1][:, None]
    T = r <= kth                                        # exactly sizes[i] True per row
    drop = (1.0 / sizes)[:, None] * np.ones((n, L))
    locs = s[T]; drops = drop[T]                        # one event per true label
    return _default_from_events(locs, drops, n, alpha, B)

def default_classwise(N, alpha, rng, B=4):
    """Design C-deterministic: common-height one-jump slice losses (P1)-(P3)."""
    n = N - 1
    if n <= 0:
        return True
    S = rng.random(n)                                   # distinct one-jump thresholds
    drops = np.full(n, float(B))                        # each path drops B -> 0 once
    return _default_from_events(S, drops, n, alpha, B)

def det_rate(fn, N, alpha, reps, rng, **kw):
    return float(np.mean([fn(N, alpha, rng, **kw) for _ in range(reps)]))

# ---------- randomized rank-interpolation comparator (design C) -----------------
def rnd_rate(N, alpha, reps, rng, B=4):
    kappa = int(np.floor(N * alpha / B))
    if kappa >= 1:
        return 0.0                                      # selects S_(kappa)/S_(kappa+1) < lambda_max
    theta = N * alpha / B - kappa                       # kappa == 0
    U = rng.random(reps)
    return float(np.mean(U < (1.0 - theta)))            # returns S_(0)=lambda_max w.p. 1-theta

def rnd_pred(N, alpha, B=4):
    return max(0.0, 1.0 - N * alpha / B) if N * alpha / B < 1 else 0.0

# ---------- sweep ---------------------------------------------------------------
rng = np.random.default_rng(SEED)
NB4 = np.arange(4, 101, 2)      # designs A, C  (boundaries B/alpha = 80, 40, 20)
NB1 = np.arange(2, 27, 1)       # design B      (boundaries B/alpha = 20, 10, 5)
COL = {0.05: "#1f77b4", 0.10: "#ff7f0e", 0.20: "#2ca02c"}

detA = {a: np.array([det_rate(default_ordinal,   N, a, REPS_DET, rng)          for N in NB4]) for a in ALPHAS}
detB = {a: np.array([det_rate(default_fnr,       N, a, REPS_DET, rng, L=5, B=1) for N in NB1]) for a in ALPHAS}
detC = {a: np.array([det_rate(default_classwise, N, a, REPS_DET, rng)          for N in NB4]) for a in ALPHAS}
rndC = {a: np.array([rnd_rate(N, a, REPS_RND, rng)                              for N in NB4]) for a in ALPHAS}
prdC = {a: np.array([rnd_pred(N, a)                                            for N in NB4]) for a in ALPHAS}

# ---------- plot ----------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))

def verticals(a_axis, B):
    for a in ALPHAS:
        a_axis.axvline(B / a, ls="--", lw=0.8, color=COL[a], alpha=0.7)

# Panel A: ordinal, marginal, B=4
for a in ALPHAS:
    ax[0].plot(NB4, detA[a], "o-", ms=4, color=COL[a], label=fr"$\alpha={a:.2f}$")
verticals(ax[0], 4)
ax[0].set_title(r"(A) ordinal, marginal $g\equiv1$ ($B=4$)")
ax[0].set_xlabel(r"$n+1$"); ax[0].set_ylabel("maximal-threshold return rate")
ax[0].set_ylim(-0.05, 1.08); ax[0].legend(fontsize=8, loc="upper right")

# Panel B: multilabel FNR, marginal, B=1
for a in ALPHAS:
    ax[1].plot(NB1, detB[a], "o-", ms=4, color=COL[a], label=fr"$\alpha={a:.2f}$")
verticals(ax[1], 1)
ax[1].set_title(r"(B) multilabel FNR, marginal ($B=1$)")
ax[1].set_xlabel(r"$n+1$"); ax[1].set_ylim(-0.05, 1.08); ax[1].legend(fontsize=8, loc="upper right")

# Panel C: ordinal, classwise, B=4, deterministic + randomized comparator
for a in ALPHAS:
    ax[2].plot(NB4, detC[a], "o-", ms=4, color=COL[a], label=fr"deterministic $\alpha={a:.2f}$")
    ax[2].plot(NB4, prdC[a], "-", lw=0.8, color=COL[a], alpha=0.5)
    ax[2].plot(NB4, rndC[a], "o", mfc="none", ms=5, color=COL[a], label=fr"randomized $\alpha={a:.2f}$")
verticals(ax[2], 4)
ax[2].set_title(r"(C) ordinal, classwise $g(x,y)=y$ ($B=4$)")
ax[2].set_xlabel(r"$n+1$"); ax[2].set_ylim(-0.05, 1.08); ax[2].legend(fontsize=7, loc="upper right", ncol=1)

plt.tight_layout()
plt.savefig("fig_small_group.pdf", bbox_inches="tight")
plt.savefig("fig_small_group.png", bbox_inches="tight", dpi=150)
print("wrote fig_small_group.{pdf,png}")

# ---------- console sanity checks ----------------------------------------------
print("\nSanity: deterministic default rate should be the step 1{n+1 < B/alpha}.")
for a in ALPHAS:
    bnd = 4 / a
    below = detA[a][NB4 <  bnd]; above = detA[a][NB4 >= bnd]
    print(f"  A alpha={a:.2f}: boundary N={bnd:.0f} | below max={below.max() if below.size else 0:.3f} "
          f"above max={above.max() if above.size else 0:.3f}")
print("Randomized comparator (C): empirical vs predicted 1 - N*alpha/B at a few N (alpha=0.10):")
for N in [10, 20, 30, 40, 50]:
    print(f"  N={N}: emp={rnd_rate(N,0.10,REPS_RND,rng):.3f}  pred={rnd_pred(N,0.10):.3f}")
