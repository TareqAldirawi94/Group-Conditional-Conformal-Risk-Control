#!/usr/bin/env python3
"""Structured-loss deployment-aware CRC experiment (three-label ordinal model).

Model (Prop joint-direct-size-separation): Y~Unif{0,1,2} independent of candidate
scores s(x,v)=U_v, U_v~Unif[0,1] iid. Groups G1={0,2}, G2={1,2}, universe {0,1,2}.
Loss ell(C,y)=min_{v in C}|v-y|^r, ell(empty)=B=2^r (r=2 -> B=4). alpha=0.10.
Theory benchmarks: S_slice*=2.750, S_direct=2.675.

Budget policy (common total budget n):
  - Methods that learn no shape -- slice_union, direct_unshaped, direct_oracle,
    atomwise, kandinsky -- calibrate on the FULL n observations.
  - direct_learned fits its shape on an independent fit fold and runs the CRC
    correction on the remaining correction fold (total budget n).
  - slice_union_cor is a diagnostic: the slice union restricted to the correction
    fold, reported to isolate the cost of sample splitting.

Methods: slice_union, slice_union_cor (diagnostic), direct_unshaped,
direct_oracle, direct_learned, atomwise, kandinsky.
"""
import numpy as np, itertools, sys, time
R=2; B=2**R; ALPHA=0.10
CANDS=np.array([0,1,2])
DIST=np.abs(CANDS[:,None]-CANDS[None,:]).astype(float)**R     # d[v,y]=|v-y|^r , 3x3
GC={1:np.array([0,2]), 2:np.array([1,2])}          # candidates in each group
CAND_GROUPS={0:[1],1:[2],2:[1,2]}                  # groups each candidate belongs to
def CAND_GROUPS_pt(y):
    return (1,2) if y==2 else ((1,) if y==0 else (2,))

def gen(n,rng):
    Y=rng.integers(0,3,size=n)
    U=rng.random((n,3))
    return Y,U

# ---- event-sweep CRC threshold for a structured nested family -----------------
def crc_threshold(theta, y, cand_set, alpha, B):
    """Smallest t s.t. (sum_i ell(D_t,y_i)+B)/(n+1)<=alpha, where
    D_t(i)={v in cand_set: theta[i,v]<=t}, ell=min dist or B if empty.
    theta: n x 3 (score - shape). Returns (t_hat, fallback_bool)."""
    n=len(y)
    if n==0: return np.inf, True
    target=(n+1)*alpha - B
    if target < -1e-12: return np.inf, True       # barrier n+1<B/alpha
    cur=np.full(n, B, float); total=n*B
    ev_t=[]; ev_i=[]; ev_d=[]
    for v in cand_set:
        ev_t.append(theta[:,v]); ev_i.append(np.arange(n)); ev_d.append(DIST[v,y])
    ev_t=np.concatenate(ev_t); ev_i=np.concatenate(ev_i); ev_d=np.concatenate(ev_d)
    order=np.argsort(ev_t,kind="stable")
    ev_t=ev_t[order]; ev_i=ev_i[order]; ev_d=ev_d[order]
    k=0; m=len(ev_t)
    while k<m:
        t=ev_t[k]
        while k<m and ev_t[k]==t:
            i=ev_i[k]; d=ev_d[k]
            if d<cur[i]: total+=d-cur[i]; cur[i]=d
            k+=1
        if total<=target+1e-9:
            return t, False
    return np.inf, True

# ---- shape fitting ------------------------------------------------------------
LIB=np.array(list(itertools.product(np.linspace(0,1,11),repeat=3)))  # 1331 offsets h(v)
def fit_learned_shape(Ufit,yfit,alpha,eta=0.0):
    """min sum_v h_v s.t. fitting direct-risk_j(h) <= alpha-eta (practical, eta=0).
       D_0^h admits v iff U<=h[v]. Returns h (3,) or None if infeasible."""
    # subsample the fit fold for the coarse 1331-point library search (the correction
    # fold is untouched, so exact finite-sample validity is preserved)
    if len(yfit)>2000:
        sub=np.random.default_rng(len(yfit)).permutation(len(yfit))[:2000]
        Ufit=Ufit[sub]; yfit=yfit[sub]
    m1=np.array([y in (0,2) for y in yfit]); m2=np.array([y in (1,2) for y in yfit])
    if m1.sum()==0 or m2.sum()==0: return None
    best=None; bestsize=9e9
    for h in LIB:
        admit=Ufit<=h[None,:]
        loss=np.full(len(yfit),B,float)
        for v in range(3):
            dv=DIST[v,yfit]; loss=np.where(admit[:,v]&(dv<loss),dv,loss)
        r1=loss[m1].mean(); r2=loss[m2].mean()
        if r1<=alpha-eta+1e-12 and r2<=alpha-eta+1e-12:
            s=h.sum()
            if s<bestsize: bestsize=s; best=h
    return best

def kandinsky_quantile(Y,U):
    """Kandinsky deploys the (1-alpha) quantile of the conformal (true-label) score
    as a function of the group-membership indicators, WITHOUT a CRC correction.
    The indicator model q(phi)=b0+b1*1{G1}+b2*1{G2} is SATURATED over the three
    realized membership patterns phi(0)=(1,0), phi(1)=(0,1), phi(2)=(1,1) (the
    candidate- and label-membership maps coincide here), so its joint pinball
    minimizer equals the per-pattern empirical (1-alpha) quantile. For candidate v
    this is the (1-alpha) quantile of U_v among points whose true label is v, which
    is what is computed below; a full group-indicator quantile regression returns
    the same thresholds to within sampling error in this design. The empirical
    quantile uses the LARGEST pinball minimizer (0-based index floor(tau*n_v) into
    the ascending-sorted scores, i.e. the (floor(tau*n_v)+1)-th smallest score),
    which is the conformal/Kandinsky convention, rather than NumPy's interpolated
    default."""
    h=np.zeros(3); tau=1-ALPHA
    for v in range(3):
        sv=np.sort(U[Y==v, v])
        if len(sv)==0:
            h[v]=1.0
        else:
            h[v]=sv[min(len(sv)-1, int(np.floor(tau*len(sv))))]
    return h

ORACLE_H=np.array([1-2*ALPHA/(1+2*ALPHA*(B-1)), 1-2*ALPHA, 1.0])  # (0.875,0.8,1.0)

def _slice_cuts(idx,Y,U):
    """Per-group slice CRC on calibration index set idx; return per-candidate union cut."""
    tsl={}
    for g in (1,2):
        gi=idx[np.array([g in CAND_GROUPS_pt(Y[i]) for i in idx])]
        tsl[g],_=crc_threshold(U[gi], Y[gi], GC[g], ALPHA, B)
    return np.array([max(tsl[g] for g in CAND_GROUPS[v]) for v in range(3)])

def _direct_cut(idx,Y,U,h):
    """Deployment-aware (direct) per-group CRC over the full universe with shape h;
    return the global max correction threshold."""
    td=[]
    for g in (1,2):
        gi=idx[np.array([g in CAND_GROUPS_pt(Y[i]) for i in idx])]
        t,_=crc_threshold(U[gi]-h[None,:], Y[gi], CANDS, ALPHA, B); td.append(t)
    return max(td)

def run_cell(n,split,reps,seed):
    rng=np.random.default_rng(seed); rows=[]; z3=np.zeros(3)
    for rep in range(reps):
        Y,U=gen(n,rng); Yte,Ute=gen(6000,rng)
        nf=int(round(split*n)); perm=rng.permutation(n); fit=perm[:nf]; cor=perm[nf:]
        allidx=np.arange(n)
        methods={}
        # ---- full-budget methods (no learned shape) ----
        methods["slice_union"]   =(z3, _slice_cuts(allidx,Y,U),          "percand")
        methods["direct_unshaped"]=(z3, _direct_cut(allidx,Y,U,z3),      "scalar")
        methods["direct_oracle"] =(ORACLE_H, _direct_cut(allidx,Y,U,ORACLE_H), "scalar")
        # split-cost diagnostic: slice union on the correction fold only
        methods["slice_union_cor"]=(z3, _slice_cuts(cor,Y,U),            "percand")
        # ---- direct learned shape: fit on fit fold, correct on correction fold ----
        hl=fit_learned_shape(U[fit],Y[fit],ALPHA)
        if hl is None:
            methods["direct_learned"]=(z3, np.inf, "scalar")
        else:
            methods["direct_learned"]=(hl, _direct_cut(cor,Y,U,hl), "scalar")
        # ---- atomwise (singletons), full budget ----
        ta=np.full(3,np.inf)
        for v in range(3):
            pts=(Y==v)
            ta[v],_=crc_threshold(U[pts], Y[pts], np.array([v]), ALPHA, B)
        methods["atomwise"]=(z3, ta, "percand")
        # ---- kandinsky (group-indicator quantile, deployed without CRC), full budget ----
        hk=kandinsky_quantile(Y,U)
        methods["kandinsky"]=(hk, hk.copy(), "percand_rawfixed")

        # ---- evaluate all ----
        for name,(shape,thr,mode) in methods.items():
            if mode=="scalar":
                theta=Ute-shape[None,:]; admit=theta<=thr
            else:                               # percand / percand_rawfixed
                admit=Ute<=np.asarray(thr)[None,:]
            size=admit.sum(1); loss=np.full(6000,B,float)
            for v in range(3):
                dv=DIST[v,Yte]; loss=np.where(admit[:,v]&(dv<loss),dv,loss)
            if mode=="scalar": fb=float(np.isinf(thr))
            else:              fb=float(np.all(np.isinf(np.asarray(thr,float))))
            for g in (1,2):
                inG=np.array([g in CAND_GROUPS_pt(y) for y in Yte])
                rows.append(dict(n=n,split=split,rep=rep,method=name,group=g,
                    risk=loss[inG].mean(), size=size[inG].mean(), fallback=fb))
            rows.append(dict(n=n,split=split,rep=rep,method=name,group=0,
                risk=loss.mean(), size=size.mean(), fallback=fb))
    return rows

if __name__=="__main__":
    import pandas as pd, os
    mode=sys.argv[1] if len(sys.argv)>1 else "validate"
    if mode=="full":
        reps=int(sys.argv[2]) if len(sys.argv)>2 else 600   # published design: 600 reps/cell
        out="struct_raw.csv"
        if os.path.exists(out): os.remove(out)
        first=True; sd=9100
        for n in [100,400,1600,6400]:
            for split in [0.25,0.50,0.75]:
                sd+=1; t0=time.time()
                rows=run_cell(n,split,reps,seed=sd)
                for r in rows: r["cell_runtime_s"]=round(time.time()-t0,2)
                pd.DataFrame(rows).to_csv(out,mode="a",header=first,index=False); first=False
                print(f"done n={n} split={split} ({time.time()-t0:.1f}s)",flush=True)
        print("WROTE struct_raw.csv")
        sys.exit(0)
    if mode=="timing":
        # method-level calibration timing (supports the per-method runtime claim;
        # cell_runtime_s in struct_raw.csv is a whole-cell duration and cannot).
        rng=np.random.default_rng(3); n=1600; REPS=30; tf=[]; ts=[]
        for _ in range(REPS):
            Y,U=gen(n,rng); perm=rng.permutation(n); fit=perm[:n//2]
            t0=time.perf_counter(); fit_learned_shape(U[fit],Y[fit],ALPHA); tf.append(time.perf_counter()-t0)
            t0=time.perf_counter(); _slice_cuts(np.arange(n),Y,U);          ts.append(time.perf_counter()-t0)
        import numpy as _np
        print(f"n={n} ({REPS} reps): learned-shape library fit "
              f"{1000*_np.mean(tf):.1f} ms vs slice-union CRC {1000*_np.mean(ts):.1f} ms")
        sys.exit(0)
    if mode=="validate":
        t0=time.time()
        rows=run_cell(6400,0.5,30,seed=7)
        df=pd.DataFrame(rows)
        g=df[df.group==0].groupby("method").agg(pooled_risk=("risk","mean"),
             pooled_size=("size","mean")).round(4)
        print(g.to_string()); print("time %.1fs"%(time.time()-t0))
        print("\nbenchmarks: S_slice*=2.750  S_direct=2.675")
        for gg in (1,2):
            sub=df[df.group==gg].groupby("method").agg(risk=("risk","mean"),size=("size","mean")).round(4)
            print(f"\n-- group {gg} --"); print(sub.to_string())
