"""
Section 5.1  Group-conditional ordinal risk control (Python twin of example1.R).
Three feature-defined groups; iid observations.
  Y in {0..4},  Yhat = Y + sigma_g U,  U ~ Unif(-sqrt3, sqrt3).
  C_t = {j : |j-Yhat| <= t};  loss = min_{j in C_t}|j-Y|, loss(empty,Y)=B=4.
Methods: pooled CRC, coordinatewise Mondrian CRC, groupwise Hoeffding-Bentkus RCPS
(per-group delta=0.10). Emits the SAME artifacts as the R script:
  figures/fig_general.pdf, figures/fig_general.png,
  figures/example1_summary.csv, figures/example1_per_group.csv
Adds per-method calibration timing and an empirical M_g no-ties check next to the
theoretical M=1 used in the random-count sandwich floor. Numbers match the R run
within Monte Carlo error (independent RNG streams).
"""
import os, time
import numpy as np, pandas as pd
from scipy.stats import binom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

K, B, ALPHA = 5, 4.0, 0.5
SIG = np.array([0.7, 1.3, 2.0]); NG = len(SIG)
NCAL, NTEST, REPS = 1500, 3000, 500
DELTA = 0.10
GR = np.arange(K, dtype=float); sqrt3 = np.sqrt(3.0)
TMIN, TMAX, M_THEORY = 0.0, 12.0, 1
assert TMAX > (K-1) + SIG.max()*sqrt3
rng = np.random.default_rng(20261024)

def draw(n):
    g = rng.integers(0, NG, n)
    Y = rng.integers(0, K, n).astype(float)
    U = rng.uniform(-sqrt3, sqrt3, n)
    return g, Y, Y + SIG[g]*U

def loss_at(t, Yhat, Y):
    if len(Y) == 0: return np.empty(0)
    inc = np.abs(GR[None,:]-Yhat[:,None]) <= t
    d = np.where(inc, np.abs(GR[None,:]-Y[:,None]), np.inf)
    l = d.min(1); l[~np.isfinite(l)] = B
    return l

def size_at(t, Yhat):
    return (np.abs(GR[None,:]-Yhat[:,None]) <= t).sum(1)

def jump_mult(Yhat):
    if len(Yhat) == 0: return 0
    bp = np.round(np.abs(GR[None,:]-Yhat[:,None]).ravel(), 9)
    return int(np.unique(bp, return_counts=True)[1].max())

def crc_threshold(Yhat, Y, alpha=ALPHA, lo=TMIN, hi=TMAX):
    n = len(Y)
    if n == 0: return hi
    crit = lambda t: (loss_at(t,Yhat,Y).sum()+B)/(n+1)
    if crit(lo) <= alpha: return lo
    if crit(hi) > alpha: return hi
    for _ in range(50):
        m = 0.5*(lo+hi)
        if crit(m) <= alpha: hi = m
        else: lo = m
    return hi

def hb_pvalue(R, muhat, n):
    if muhat >= R: return 1.0
    kl = -np.log1p(-R) if muhat <= 0 else muhat*np.log(muhat/R)+(1-muhat)*np.log((1-muhat)/(1-R))
    p_b = np.e*binom.cdf(np.ceil(n*muhat), n, R)
    return min(1.0, p_b, np.exp(-n*kl))

def rcps_threshold(Yhat, Y, alpha=ALPHA, delta=DELTA, bound=B, lo=TMIN, hi=TMAX):
    n = len(Y)
    if n == 0: return hi
    tgt = alpha/bound
    pv = lambda t: hb_pvalue(tgt, loss_at(t,Yhat,Y).mean()/bound, n)
    if pv(lo) <= delta: return lo
    if pv(hi) > delta: return hi
    for _ in range(50):
        m = 0.5*(lo+hi)
        if pv(m) <= delta: hi = m
        else: lo = m
    return hi

methods = ["marg","mond","rcps"]
labels = {"marg":"Marginal CRC","mond":"Mondrian CRC (ours)","rcps":"Groupwise RCPS (HB)"}
Rg = np.full((REPS,NG,3), np.nan); Sg = np.full((REPS,NG,3), np.nan)
Rp = np.full((REPS,3), np.nan); Sz = np.full((REPS,3), np.nan)
counts = np.zeros((REPS,NG)); floorv = np.full((REPS,NG), np.nan)
fall = np.zeros((REPS,NG), bool); Memp = np.zeros((REPS,NG), int)
tmr = {"marg":0.0,"mond":0.0,"rcps":0.0}
mi = {"marg":0,"mond":1,"rcps":2}

for r in range(REPS):
    gc,Yc,Yhc = draw(NCAL); gt,Yt,Yht = draw(NTEST)
    t0=time.perf_counter(); t_marg=crc_threshold(Yhc,Yc); tmr["marg"]+=time.perf_counter()-t0
    t_grp=np.empty(NG); t_rcps=np.empty(NG)
    for g in range(NG):
        m = gc==g; ng=int(m.sum()); counts[r,g]=ng; Memp[r,g]=jump_mult(Yhc[m])
        t0=time.perf_counter(); t_grp[g]=crc_threshold(Yhc[m],Yc[m]); tmr["mond"]+=time.perf_counter()-t0
        t0=time.perf_counter(); t_rcps[g]=rcps_threshold(Yhc[m],Yc[m]); tmr["rcps"]+=time.perf_counter()-t0
        fall[r,g] = t_grp[g] >= TMAX-1e-9
        floorv[r,g] = ALPHA-(M_THEORY+1)*B/(ng+1)
    thr = {"marg": lambda g: t_marg, "mond": lambda g: t_grp[g], "rcps": lambda g: t_rcps[g]}
    for nm in methods:
        lt=np.empty(NTEST); sz=np.empty(NTEST)
        for g in range(NG):
            m=gt==g
            if not m.any(): raise RuntimeError("Empty evaluation group: raise NTEST.")
            t=thr[nm](g); lt[m]=loss_at(t,Yht[m],Yt[m]); sz[m]=size_at(t,Yht[m])
            Rg[r,g,mi[nm]]=lt[m].mean(); Sg[r,g,mi[nm]]=sz[m].mean()
        Rp[r,mi[nm]]=lt.mean(); Sz[r,mi[nm]]=sz.mean()

def hw(x): return 1.96*x.std(0,ddof=1)/np.sqrt(x.shape[0])
cal_ms = {nm: 1000*tmr[nm]/REPS for nm in methods}

summary = pd.DataFrame([dict(
    method=labels[nm], pooled_risk=Rp[:,mi[nm]].mean(),
    pooled_risk_halfwidth=hw(Rp[:,mi[nm]]),
    worst_group_risk=Rg[:,:,mi[nm]].mean(0).max(),
    max_abs_group_deviation=np.abs(Rg[:,:,mi[nm]].mean(0)-ALPHA).max(),
    pooled_mean_set_size=Sz[:,mi[nm]].mean(),
    cal_ms_per_rep=cal_ms[nm]) for nm in methods])
per_group = pd.DataFrame([dict(
    method=labels[nm], group=g+1, sigma=SIG[g],
    risk=Rg[:,g,mi[nm]].mean(), risk_halfwidth=hw(Rg[:,g,mi[nm]]),
    set_size=Sg[:,g,mi[nm]].mean(), size_halfwidth=hw(Sg[:,g,mi[nm]]),
    avg_cal_count=counts[:,g].mean(), mondrian_fallback_rate=fall[:,g].mean())
    for nm in methods for g in range(NG)])

print(f"Section 5.1: {REPS} reps, n_cal={NCAL}, n_test={NTEST}, target={ALPHA:.2f}")
print(f"RCPS: per-group delta={DELTA:.2f} (NOT a simultaneous group certificate)")
print(f"M_g: theoretical={M_THEORY} used in floor; empirical max={Memp.max()}, "
      f"mean={Memp.mean():.3f} (checks no-ties)")
print(f"mean random-count lower bound alpha-(M+1)B/(n_g+1)={floorv.mean():.4f}\n")
with pd.option_context("display.width",160,"display.max_columns",20):
    print(summary.to_string(index=False)); print(); print(per_group.to_string(index=False))

os.makedirs("figures", exist_ok=True)
summary.to_csv("figures/example1_summary.csv", index=False)
per_group.to_csv("figures/example1_per_group.csv", index=False)

def make_plot(path, dpi=125):
    cols={"marg":"#8c8c8c","mond":"firebrick","rcps":"steelblue"}
    fig,ax=plt.subplots(1,2,figsize=(10.4,4.4)); gx=np.arange(NG); w=0.24
    for panel,is_risk in (("risk",True),("size",False)):
        a=ax[0] if is_risk else ax[1]
        if is_risk:
            a.axhspan(floorv.mean(),ALPHA,color="0.91",zorder=0)
            a.axhline(ALPHA,ls="--",color="k",lw=1.1)
        for i,nm in enumerate(methods):
            x=gx+(i-1)*w
            mu=(Rg if is_risk else Sg)[:,:,mi[nm]].mean(0)
            e =hw((Rg if is_risk else Sg)[:,:,mi[nm]])
            a.bar(x,mu,w,yerr=e,capsize=2,color=cols[nm],label=labels[nm])
        a.set_xticks(gx); a.set_xticklabels([f"sigma={s}" for s in SIG])
        a.set_xlabel("Noise scale by group")
        a.set_ylabel("Group-conditional ordinal risk" if is_risk else "Mean set size")
        a.set_title("Conditional risk" if is_risk else "Prediction-set size")
        a.legend(fontsize=7)
    ax[0].set_ylim(0,1.22); ax[1].set_ylim(0,3.65)
    plt.tight_layout(); plt.savefig(path,dpi=dpi); plt.close()

make_plot("figures/fig_general.pdf"); make_plot("figures/fig_general.png")
print("\nSaved figures/fig_general.pdf, figures/fig_general.png, and two CSV files.")
