#!/usr/bin/env python3
"""Fitted-shape overlapping-group CRC experiment (Section 5.2).

Model (manuscript 5.2): Y in {0,1}^3; focal groups G_j={y_j=1}, j=1,2,3;
background G_0={y=(0,0,0)}. Z_j=sqrt(rho)F+sqrt(1-rho)E_j, Y_j=1{Z_j>tau_j},
eta_j=gamma_j(Z_j-tau_j)+xi_j, q_j=logit^{-1}(eta_j) clipped.
Score s(x,v)=-sum_j[v_j log q_j+(1-v_j) log(1-q_j)], v in {0,1}^3.
Miscoverage loss (B=1), target alpha=0.10.

Methods:
  union_full   : unshaped union (b=0), full calibration budget
  union_split  : unshaped union (b=0), correction fold only (split-cost control)
  fitted_union : weighted-hinge shape on fit fold + CRC correction on correction fold
  kandinsky    : joint pinball (1-alpha) quantile over group indicators, deployed directly
  atomwise     : Mondrian on the 8 label-vector atoms, full calibration budget
"""
import numpy as np, itertools, sys
from scipy.stats import norm
from sklearn.linear_model import QuantileRegressor

VECS = np.array(list(itertools.product([0,1],repeat=3)))   # 8 x 3
NV = len(VECS)
P  = np.array([0.55,0.40,0.18]); GAM=np.array([1.70,1.00,0.40])
TAU= norm.ppf(1-P)
ALPHA=0.10
# group membership of each candidate vector: columns G0,G1,G2,G3
PHI = np.zeros((NV,4))
PHI[:,0]=(VECS.sum(1)==0).astype(float)           # G0 = (0,0,0)
PHI[:,1:]=VECS.astype(float)                       # G_j = {v_j=1}
GROUPS=[0,1,2,3]

def gen(N, rho, rng):
    F=rng.standard_normal((N,1)); E=rng.standard_normal((N,3)); xi=rng.standard_normal((N,3))
    Z=np.sqrt(rho)*F+np.sqrt(1-rho)*E
    Y=(Z>TAU).astype(int)
    eta=GAM*(Z-TAU)+xi
    q=np.clip(1/(1+np.exp(-eta)),1e-6,1-1e-6)
    # score for every candidate vector: s(x,v) = -sum_j[v_j log q_j+(1-v_j)log(1-q_j)]
    lq=np.log(q); l1q=np.log(1-q)                   # N x 3
    S = -(VECS[None,:,:]*lq[:,None,:] + (1-VECS)[None,:,:]*l1q[:,None,:]).sum(2)  # N x 8
    yidx=(Y*np.array([4,2,1])).sum(1)               # index of true vector in VECS
    return S, Y, yidx

def crc_threshold(u, alpha):
    """miscoverage CRC threshold for shaped true-label scores u (1d). +inf = fallback."""
    n=len(u)
    if n==0: return np.inf, True
    k=int(np.ceil((1-alpha)*(n+1)))
    if k>n: return np.inf, True
    return np.sort(u)[k-1], False

def membership_idx(yidx):
    """for each candidate vector, which groups it belongs to (list of group lists)."""
    return [np.where(PHI[v]>0)[0] for v in range(NV)]
CAND_GROUPS=membership_idx(None)   # precompute: candidate v -> array of group ids

def group_of_point(yidx):
    """groups that true vector yidx belongs to."""
    return [np.where(PHI[v]>0)[0] for v in yidx]

def eval_union(S_te, yidx_te, tj, b):
    """deployed union with per-group thresholds tj (len4) and candidate shape b (len8).
       returns per-group deployed risk, slice risk, size; plus pooled."""
    shaped = S_te - b[None,:]                       # Nte x 8
    Tmax = np.full(NV,-np.inf)
    for v in range(NV):
        gs=CAND_GROUPS[v]
        Tmax[v]=max(tj[g] for g in gs) if len(gs) else -np.inf
    admit = shaped <= Tmax[None,:]                  # Nte x 8  (candidate admitted)
    size = admit.sum(1)
    # true label admitted?
    true_shaped = shaped[np.arange(len(yidx_te)),yidx_te]
    true_Tmax   = Tmax[yidx_te]
    dep_cover = true_shaped <= true_Tmax            # covered by union
    # per group metrics (conditional on true label in group)
    res={}
    for g in GROUPS:
        inG = PHI[yidx_te,g]>0
        ng=inG.sum()
        if ng==0:
            res[g]=dict(dep=np.nan, slice=np.nan, size=np.nan, n=0); continue
        dep_risk = 1 - dep_cover[inG].mean()
        slice_cover = true_shaped[inG] <= tj[g]
        slice_risk = 1 - slice_cover.mean()
        res[g]=dict(dep=dep_risk, slice=slice_risk, size=size[inG].mean(), n=ng)
    pooled_risk = 1-dep_cover.mean(); pooled_size=size.mean()
    return res, pooled_risk, pooled_size

def fit_shape(S_fit, yidx_fit, alpha):
    """Weighted-hinge shape = pinball (1-alpha) quantile of the true-label score over the
    4 group indicators (no intercept; the indicators span, with G0 for the all-zero vector).
    The manuscript hinge equals this pinball objective up to an additive constant, so the
    minimizer is the same. Returns the candidate-wise shape PHI@beta."""
    sy = S_fit[np.arange(len(yidx_fit)),yidx_fit]
    Phi_y = PHI[yidx_fit]
    qr=QuantileRegressor(quantile=1-alpha, alpha=0.0, fit_intercept=False, solver="highs")
    qr.fit(Phi_y, sy)
    beta=qr.coef_
    return PHI@beta, beta

def eval_atomwise(S_cal,yidx_cal,S_te,yidx_te,alpha):
    tv=np.full(NV,np.inf); fb=np.zeros(NV)
    for v in range(NV):
        u=S_cal[yidx_cal==v, v]
        tv[v],f=crc_threshold(u,alpha); fb[v]=f
    admit = S_te <= tv[None,:]
    size=admit.sum(1)
    cover = S_te[np.arange(len(yidx_te)),yidx_te] <= tv[yidx_te]
    res={}
    for g in GROUPS:
        inG=PHI[yidx_te,g]>0; ng=inG.sum()
        res[g]=dict(dep=(1-cover[inG].mean()) if ng else np.nan,
                    slice=np.nan, size=size[inG].mean() if ng else np.nan, n=ng)
    return res,1-cover.mean(),size.mean(), fb.mean()

def union_thresholds(S_cal,yidx_cal,b,alpha):
    tj=np.full(4,np.inf); fb=np.zeros(4)
    for g in GROUPS:
        inG=PHI[yidx_cal,g]>0
        u=(S_cal[inG, :][np.arange(inG.sum()), yidx_cal[inG]]) - b[yidx_cal[inG]]
        tj[g],f=crc_threshold(u,alpha); fb[g]=f
    return tj,fb

def run_cell(N,rho,split,reps,seed):
    rng=np.random.default_rng(seed)
    acc=[]  # rows
    for r in range(reps):
        S,Y,yidx=gen(N,rho,rng)
        Ste,Yte,yte=gen(6000,rho,rng)
        # split indices
        nfit=int(round(split*N))
        perm=rng.permutation(N); fit=perm[:nfit]; cor=perm[nfit:]
        b0=np.zeros(NV)
        # union_full (b=0, all N)
        tjF,fbF=union_thresholds(S,yidx,b0,ALPHA)
        rF,pR,pS=eval_union(Ste,yte,tjF,b0)
        # union_split (b=0, correction fold only)
        tjS,fbS=union_thresholds(S[cor],yidx[cor],b0,ALPHA)
        rS,pRs,pSs=eval_union(Ste,yte,tjS,b0)
        # fitted_union (fit shape on fit fold, CRC-correct on cor fold)
        bhat,beta=fit_shape(S[fit],yidx[fit],ALPHA)
        tjH,fbH=union_thresholds(S[cor],yidx[cor],bhat,ALPHA)
        rH,pRh,pSh=eval_union(Ste,yte,tjH,bhat)
        # kandinsky (fit shape on full N, deploy the fitted quantile directly, no CRC)
        qK,_=fit_shape(S,yidx,ALPHA)
        admitK=Ste<=qK[None,:]; sizeK=admitK.sum(1)
        coverK=Ste[np.arange(len(yte)),yte]<=qK[yte]
        rK={}
        for g in GROUPS:
            inG=PHI[yte,g]>0; ng=inG.sum()
            rK[g]=dict(dep=(1-coverK[inG].mean()) if ng else np.nan, slice=np.nan,
                       size=sizeK[inG].mean() if ng else np.nan,n=ng)
        pRk=1-coverK.mean(); pSk=sizeK.mean()
        # atomwise (full N)
        rA,pRa,pSa,fbA=eval_atomwise(S,yidx,Ste,yte,ALPHA)
        # dispersion of thresholds (alignment): sd over finite tj
        def disp(tj):
            f=tj[np.isfinite(tj)]; return f.std() if len(f)>1 else 0.0
        for meth,(res,pr,ps,tj,fb) in {
            "union_full":(rF,pR,pS,tjF,fbF),
            "union_split":(rS,pRs,pSs,tjS,fbS),
            "fitted_union":(rH,pRh,pSh,tjH,fbH),
            "kandinsky":(rK,pRk,pSk,np.array([np.nan]),np.array([np.nan])),
            "atomwise":(rA,pRa,pSa,np.array([np.nan]),np.array([fbA])),
        }.items():
            for g in GROUPS:
                acc.append(dict(N=N,rho=rho,split=split,rep=r,method=meth,group=g,
                    dep=res[g]["dep"],slice=res[g]["slice"],size=res[g]["size"],
                    gap=(res[g]["slice"]-res[g]["dep"]) if np.isfinite(res[g]["slice"]) else np.nan,
                    fallback=(fb[g] if (meth in ("union_full","union_split","fitted_union") and g<len(fb)) else np.nan),
                    pooled_risk=pr,pooled_size=ps,thr_disp=disp(tj)))
    return acc

def full_sweep(reps=200, seed0=4300):
    import pandas as pd, os
    out="fitshape_raw.csv"
    if os.path.exists(out): os.remove(out)
    sd=seed0; first=True
    for N in [100,400,1600,6400]:
        # NOTE: cell-specific replication count. n in {100,400,1600} use `reps`
        # (default 200); n=6400 uses 120 to lighten the slow asymptotic block. Any
        # uncertainty estimate must use the actual per-cell rep count, not one N.
        rr = 120 if N==6400 else reps
        for rho in [0.0,0.5]:
            for split in [0.25,0.50,0.75]:
                sd+=1
                rows=run_cell(N,rho,split,rr,seed=sd)
                pd.DataFrame(rows).to_csv(out,mode="a",header=first,index=False); first=False
                print(f"done N={N} rho={rho} split={split} reps={rr}",flush=True)

if __name__=="__main__":
    import pandas as pd
    mode=sys.argv[1] if len(sys.argv)>1 else "validate"
    if mode=="full":
        reps=int(sys.argv[2]) if len(sys.argv)>2 else 200
        full_sweep(reps=reps)
        df=pd.read_csv("fitshape_raw.csv")
        # per-cell/method/group summary
        g=df.groupby(["N","rho","split","method","group"]).agg(
            dep=("dep","mean"),dep_se=("dep",lambda x:x.std()/np.sqrt(x.notna().sum())),
            slice=("slice","mean"),gap=("gap","mean"),size=("size","mean"),
            fb=("fallback","mean"),disp=("thr_disp","mean")).reset_index()
        g.to_csv("fitshape_summary.csv",index=False)
        pooled=df.drop_duplicates(["N","rho","split","method","rep"]).groupby(
            ["N","rho","split","method"]).agg(pooled_risk=("pooled_risk","mean"),
            pooled_size=("pooled_size","mean")).reset_index()
        pooled.to_csv("fitshape_pooled.csv",index=False)
        print("WROTE fitshape_raw/summary/pooled.csv; rows:",len(df))
        sys.exit(0)
    if mode=="validate":
        acc=run_cell(1600,0.0,0.5,40,seed=11)
        df=pd.DataFrame(acc)
        g=df.groupby(["method","group"]).agg(dep=("dep","mean"),slice=("slice","mean"),
              gap=("gap","mean"),size=("size","mean"),fb=("fallback","mean"),
              disp=("thr_disp","mean")).round(3)
        print(g.to_string())
        print("\npooled by method:")
        print(df.groupby("method").agg(pooled_risk=("pooled_risk","mean"),
              pooled_size=("pooled_size","mean")).round(3).to_string())
