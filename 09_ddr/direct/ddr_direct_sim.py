#!/usr/bin/env python3
"""Deployment-aware CRC on the DDR ordinal grading pool (manuscript Section 5.3).

Extends the existing overlapping-union DDR evaluation (ddr_crc_eval_integrity.py):
identical pool, randomized APS score, ordinal loss, groups, and event-sweep CRC
primitive (copied verbatim), now adding deployment-aware (direct) CRC.

PROVENANCE / SCOPE: this script runs the direct-method comparison on the supplied
2,428-row score table only. Opening that CSV does NOT by itself establish a
leakage-controlled classifier: the dedup-before-split training and the SHA/manifest
checks live in ddr_crc_eval_integrity.py. Treat these results as HISTORICAL /
sensitivity on the supplied scores until regenerated on the canonical
dedup-before-split output with its provenance package (scoring recipe, content
audit, split manifest, pHash report). Patient-level independence is unverified, and
deployed risk <= alpha on this fixed pool is behavior on that pool, not a proof of
distribution-free validity.

Pool : 2,428 historical held-out scored images, counts 1186/125/889/47/181.
Score: randomized APS  s(x,k)=sum_{p_j>p_k}p_j + U_x p_k ; nested C_t={k:s<=t}.
Loss : ordinal |k-y|, empty-set penalty B=4; alpha=0.10.
Groups: {0},{1,2,3},{2,3,4},{3,4}  (universe = full 5-grade space for ALL methods).

Methods: slice_union (full cal), slice_union_cor (correction fold, split-cost
diagnostic), direct_unshaped (full cal), direct_learned (fit shape + CRC on
correction fold). 200 paired redraws; fit/correction splits 25/75, 50/50, 75/25.

Learned shape: finite library LEVELS^5 = 5^5 = 3,125 per-grade offsets with
LEVELS={0,0.15,0.35,0.65,1.0}; zero empirical feasibility margin; fit-fold size
surrogate minimized with np.argmin tie-break; fit fold subsampled to at most 600
rows (deterministic seed = fit-fold length), correction fold untouched. When no
library shape meets the fit-fold constraint the fitter returns the all-ones shape
(which admits everything), not a separate None/fallback branch. The recorded
`full_universe` field is a threshold decision for the direct deploy, not a per-row
full-set indicator.
"""
import numpy as np, pandas as pd, itertools, sys, time, os
K=5; B=4.0; ALPHA=0.10; LAMBDA_MAX=1.0+1e-9
DIST=np.abs(np.arange(K)[:,None]-np.arange(K)[None,:]).astype(float)   # |k-y|
GROUPS=[[0],[1,2,3],[2,3,4],[3,4]]
GLAB=["{0}","{1,2,3}","{2,3,4}","{3,4}"]
CSV_POOL=os.environ.get("DDR_POOL_CSV",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),"ddr_analysis_scores.csv"))

# ---- primitives copied verbatim from ddr_crc_eval_integrity.py ----------------
def aps_scores(P,U):
    N=P.shape[0]; S=np.empty_like(P)
    for i in range(N):
        p=P[i]; greater=(p[None,:]>p[:,None])
        S[i]=(greater*p[None,:]).sum(1)+U[i]*p
    return S
def losses_of_sets(member,y):
    d=DIST[:,y].T; big=np.where(member,d,np.inf); mn=big.min(1)
    mn[~np.isfinite(mn)]=B; return mn
def fit_threshold(S,y,cal_idx,group):
    group=np.asarray(group); in_g=np.isin(y[cal_idx],group); rows=cal_idx[in_g]; n_j=len(rows)
    target=ALPHA*(n_j+1)-B
    if n_j==0 or target<0: return LAMBDA_MAX,True,n_j
    cur=np.full(n_j,B); ssum=cur.sum()
    if ssum<=target: return LAMBDA_MAX,False,n_j
    ev_s,ev_i,ev_d=[],[],[]; yy=y[rows]
    for col,k in enumerate(group):
        ev_s.append(S[rows,k]); ev_i.append(np.arange(n_j)); ev_d.append(np.abs(k-yy).astype(float))
    ev_s=np.concatenate(ev_s); ev_i=np.concatenate(ev_i); ev_d=np.concatenate(ev_d)
    order=np.argsort(ev_s,kind="mergesort"); ev_s,ev_i,ev_d=ev_s[order],ev_i[order],ev_d[order]
    m=len(ev_s); p=0
    while p<m:
        s0=ev_s[p]
        while p<m and ev_s[p]==s0:
            i=ev_i[p]
            if ev_d[p]<cur[i]: ssum+=ev_d[p]-cur[i]; cur[i]=ev_d[p]
            p+=1
        if ssum<=target: return float(s0),False,n_j
    return LAMBDA_MAX,False,n_j
def deploy_union(S,groups,thr):
    Tmax=np.full(K,-np.inf)
    for g,t in zip(groups,thr):
        for k in g:
            if t>Tmax[k]: Tmax[k]=t
    return S<=Tmax[None,:]
def slice_membership(S,group,t):
    member=np.zeros_like(S,dtype=bool)
    for k in group: member[:,k]=S[:,k]<=t
    return member

# ---- deployment-aware (direct) threshold: select points y in cond_group, sweep
#      the FULL universe on shifted scores s-h, control the deployed ordinal loss.
def fit_threshold_direct(S,y,cal_idx,cond_group,h):
    in_g=np.isin(y[cal_idx],np.asarray(cond_group)); rows=cal_idx[in_g]; n_j=len(rows)
    target=ALPHA*(n_j+1)-B
    if n_j==0 or target<0: return LAMBDA_MAX,True,n_j
    cur=np.full(n_j,B); ssum=cur.sum(); yy=y[rows]
    ev_s,ev_i,ev_d=[],[],[]
    for k in range(K):
        ev_s.append(S[rows,k]-h[k]); ev_i.append(np.arange(n_j)); ev_d.append(np.abs(k-yy).astype(float))
    ev_s=np.concatenate(ev_s); ev_i=np.concatenate(ev_i); ev_d=np.concatenate(ev_d)
    order=np.argsort(ev_s,kind="mergesort"); ev_s,ev_i,ev_d=ev_s[order],ev_i[order],ev_d[order]
    m=len(ev_s); p=0
    while p<m:
        s0=ev_s[p]
        while p<m and ev_s[p]==s0:
            i=ev_i[p]
            if ev_d[p]<cur[i]: ssum+=ev_d[p]-cur[i]; cur[i]=ev_d[p]
            p+=1
        if ssum<=target: return float(s0),False,n_j
    return LAMBDA_MAX,False,n_j

# ---- learned shape: fixed finite per-grade offset library (LEVELS^5 = 3125) ------
#   The library and fitting criterion are defined here in code; "fixed before DDR"
#   is an author claim, not something this script can prove.
LEVELS=np.array([0.0,0.15,0.35,0.65,1.0])
LIB=np.array(list(itertools.product(LEVELS,repeat=K)))        # 5^5 = 3125 offsets
def fit_learned_shape(S,y,fit_idx):
    idx=fit_idx
    if len(idx)>600:
        idx=idx[np.random.default_rng(len(idx)).permutation(len(idx))[:600]]
    scU=S[idx]; dU=DIST[:,y[idx]].T                          # (nf,5)
    masks=[np.isin(y[idx],np.asarray(g)) for g in GROUPS]
    if any(mk.sum()==0 for mk in masks): return np.ones(K)
    # vectorized over library
    admit=scU[None,:,:]<=LIB[:,None,:]                        # (L,nf,5)
    loss=np.where(admit,dU[None,:,:],B).min(2)                # (L,nf)
    size=admit.sum(2).mean(1)                                 # (L,)
    feas=np.ones(len(LIB),bool)
    for mk in masks:
        feas &= (loss[:,mk].mean(1)<=ALPHA+1e-12)
    if not feas.any(): return np.ones(K)
    cand=np.where(feas)[0]
    return LIB[cand[np.argmin(size[cand])]]

def per_group_eval(member_te,y_te):
    loss=losses_of_sets(member_te,y_te); size=member_te.sum(1)
    out={}
    for gi,g in enumerate(GROUPS):
        in_g=np.isin(y_te,np.asarray(g))
        out[gi]=dict(risk=float(loss[in_g].mean()),size=float(size[in_g].mean()),n_te=int(in_g.sum()))
    out["overall_size"]=float(size.mean())
    return out

def run_redraw(P,y,split,rng):
    N=len(y); U=rng.random(N); S=aps_scores(P,U)
    perm=rng.permutation(N); cal=perm[:N//2]; te=perm[N//2:]   # random (NOT sorted: pool is grade-ordered)
    nf=int(round(split*len(cal))); fit=cal[:nf]; cor=cal[nf:]  # random fit/correction sub-split
    rows=[]
    # slice union (full cal)
    thr=[fit_threshold(S,y,cal,g) for g in GROUPS]
    dep=deploy_union(S,GROUPS,[t for t,_,_ in thr])
    ev=per_group_eval(dep[te],y[te])
    for gi,g in enumerate(GROUPS):
        sl=slice_membership(S[te],np.asarray(g),thr[gi][0])
        slr=losses_of_sets(sl,y[te])[np.isin(y[te],np.asarray(g))].mean()
        rows.append(dict(method="slice_union",split=split,group=gi,glab=GLAB[gi],
            risk=ev[gi]["risk"],size=ev[gi]["size"],n_te=ev[gi]["n_te"],
            overall_size=ev["overall_size"],fallback=float(thr[gi][1]),
            slice_risk=float(slr),gap=float(slr-ev[gi]["risk"]),full_universe=0.0))
    # slice union (correction fold only) -- split-cost diagnostic
    thrc=[fit_threshold(S,y,cor,g) for g in GROUPS]
    depc=deploy_union(S,GROUPS,[t for t,_,_ in thrc]); evc=per_group_eval(depc[te],y[te])
    for gi,g in enumerate(GROUPS):
        rows.append(dict(method="slice_union_cor",split=split,group=gi,glab=GLAB[gi],
            risk=evc[gi]["risk"],size=evc[gi]["size"],n_te=evc[gi]["n_te"],
            overall_size=evc["overall_size"],fallback=float(thrc[gi][1]),
            slice_risk=np.nan,gap=np.nan,full_universe=0.0))
    # direct unshaped (full cal)
    td=[fit_threshold_direct(S,y,cal,g,np.zeros(K)) for g in GROUPS]
    tdir=max(t for t,_,_ in td); depd=(S<=tdir); evd=per_group_eval(depd[te],y[te])
    fu=float(tdir>=LAMBDA_MAX)
    for gi,g in enumerate(GROUPS):
        rows.append(dict(method="direct_unshaped",split=split,group=gi,glab=GLAB[gi],
            risk=evd[gi]["risk"],size=evd[gi]["size"],n_te=evd[gi]["n_te"],
            overall_size=evd["overall_size"],fallback=float(td[gi][1]),
            slice_risk=np.nan,gap=np.nan,full_universe=fu))
    # direct learned shape (fit on fit fold, CRC on correction fold)
    h=fit_learned_shape(S,y,fit)
    tl=[fit_threshold_direct(S,y,cor,g,h) for g in GROUPS]
    tld=max(t for t,_,_ in tl); depl=((S-h[None,:])<=tld); evl=per_group_eval(depl[te],y[te])
    ful=float(tld>=LAMBDA_MAX)
    for gi,g in enumerate(GROUPS):
        rows.append(dict(method="direct_learned",split=split,group=gi,glab=GLAB[gi],
            risk=evl[gi]["risk"],size=evl[gi]["size"],n_te=evl[gi]["n_te"],
            overall_size=evl["overall_size"],fallback=float(tl[gi][1]),
            slice_risk=np.nan,gap=np.nan,full_universe=ful))
    return rows

def load_pool():
    d=pd.read_csv(CSV_POOL)
    y=d["grade"].to_numpy(int); P=d[[f"p{k}" for k in range(K)]].to_numpy(float)
    P=P/P.sum(1,keepdims=True)
    return P,y

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "validate"
    P,y=load_pool()
    if mode=="validate":
        rng=np.random.default_rng(7); acc=[]
        for r in range(100): acc+=run_redraw(P,y,0.5,rng)
        df=pd.DataFrame([a for a in acc if a["method"]=="slice_union"])
        g=df.groupby("glab").agg(risk=("risk","mean"),size=("size","mean")).round(4)
        print("slice_union deployed risk/size (100 redraws) -- target risk [.058,.056,.038,.036]:")
        print(g.to_string())
        sys.exit(0)
    if mode=="full":
        reps=int(sys.argv[2]) if len(sys.argv)>2 else 200
        out="ddr_raw.csv"
        if os.path.exists(out): os.remove(out)
        first=True
        for split in [0.25,0.50,0.75]:
            rng=np.random.default_rng(2026+int(split*100)); t0=time.time(); acc=[]
            for r in range(reps):
                rr=run_redraw(P,y,split,rng)
                for a in rr: a["redraw"]=r
                acc+=rr
            rt=round(time.time()-t0,2)
            for a in acc: a["split_runtime_s"]=rt
            pd.DataFrame(acc).to_csv(out,mode="a",header=first,index=False); first=False
            print(f"split={split} ({rt}s, {reps} redraws)",flush=True)
        print("WROTE ddr_raw.csv")
