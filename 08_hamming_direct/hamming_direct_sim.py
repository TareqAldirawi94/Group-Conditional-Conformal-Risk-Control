#!/usr/bin/env python3
"""Deployment-aware CRC under Hamming-to-set loss (eight label-vector model).

Extends the existing Section 5.2 reconciliation experiment (example2b.py):
same generative model, score, groups, and event-sweep CRC primitive, now
comparing the slice union against deployment-aware (direct) CRC.

Model:  Y in {0,1}^3, Z_j=sqrt(rho)F+sqrt(1-rho)E_j, Y_j=1{Z_j>0},
        eta_j=gam*Z_j+xi_j, q_j=sigmoid(eta_j),
        s(x,v)=-sum_j [v_j log q_j + (1-v_j) log(1-q_j)],  v in {0,1}^3.
Loss:   Hamming-to-set, ell(C,y)=min_{v in C} d_H(v,y), ell(empty)=B=3; alpha=0.5.
Groups: label-defined G_j={y_j=1} (mode B) and feature-defined (mode A).
Universe: mode B uses 7 nonzero candidates (000 excluded); mode A uses all 8.
Candidate universe is identical across union and direct within each mode.

Methods: slice_union, slice_union_cor (diagnostic), direct_unshaped,
direct_learned, atomwise, groupwise_quantile_union (coverage reference; NOT the
joint Kandinsky fit -- see note in run_cell). In mode A the direct methods use the
same no-group => empty eligibility as the union, so size comparisons isolate the
global-threshold effect from candidate eligibility.
"""
import numpy as np, pandas as pd, itertools, sys, time, os
from scipy.special import expit

L=3; B=3; ALPHA=.50; RHO=.4; GAM=1.3; NOISE=1.; LMAX=42.
V=((np.arange(2**L)[:,None]>>np.arange(L))&1).astype(np.int8)     # 8x3 label-vectors
DH=np.abs(V[:,None,:]-V[None,:,:]).sum(axis=2).astype(np.int8)    # 8x8 Hamming
CARD=V.sum(1).astype(int)                                         # cardinality 0..3
NONZERO=np.where(CARD>0)[0]                                       # the 7 deployable candidates
UNIV_B=(CARD>0)                                                   # mode B universe mask (len 8)
UNIV_A=np.ones(8,bool)                                            # mode A universe mask

def gen(n,rng):
    z=np.sqrt(RHO)*rng.standard_normal((n,1))+np.sqrt(1-RHO)*rng.standard_normal((n,L))
    Y=(z>0).astype(np.int8)
    eta=GAM*z+NOISE*rng.standard_normal((n,L))
    q=np.clip(expit(eta),1e-6,1-1e-6)
    sc=(-np.log1p(-q)).sum(axis=1)[:,None]+(np.log1p(-q)-np.log(q))@V.T   # n x 8
    yid=(Y@np.array([1,2,4])).astype(int)
    m=rng.integers(0,2,(n,L),dtype=np.int8)                               # feature membership
    return Y,m,sc,yid

# ---- event-sweep CRC threshold (as in example2b.exact_crc) --------------------
def exact_crc(sc,dcol,eligible):
    """Smallest cut t s.t. (sum_i min_{v in eligible, sc<=t} d(v,y_i) + ...) controls
    group risk at alpha. Returns LMAX on barrier/empty."""
    n=len(sc)
    if n==0 or B*n < (n+1)*(B-ALPHA): return LMAX
    s=sc[:,eligible]; d=dcol[:,eligible]
    order=np.argsort(s,axis=1)
    scores=np.take_along_axis(s,order,axis=1)
    distances=np.take_along_axis(d,order,axis=1)
    after=np.minimum.accumulate(distances,axis=1)
    before=np.concatenate((np.full((n,1),B,dtype=np.int8),after[:,:-1]),axis=1)
    jumps=before-after; mask=jumps>0
    seq=np.argsort(scores[mask]); cum=np.cumsum(jumps[mask][seq])
    needed=(n+1)*(B-ALPHA); k=np.searchsorted(cum,needed,side='left')
    if k>=len(cum): return LMAX
    return float(scores[mask][seq][k])

def group_mask(Y,m,j,mode): return (m[:,j]==1) if mode=='A' else (Y[:,j]==1)

# ---- shape fitting (symmetric per-cardinality offsets) ------------------------
def fit_learned_shape(sc_fit,yid_fit,Y_fit,m_fit,mode,univ,grid):
    """Per-cardinality offsets H=(H1,H2,H3). D_0^h admits nonzero v iff sc_v<=H[card_v].
    Minimize fit-fold mean deployed size s.t. every group's fit Hamming risk<=alpha."""
    # subsample fit fold for the coarse shape search (correction fold stays full -> validity intact)
    if len(sc_fit)>4000:
        sub=np.random.default_rng(len(sc_fit)).permutation(len(sc_fit))[:4000]
        sc_fit,yid_fit,Y_fit,m_fit=sc_fit[sub],yid_fit[sub],Y_fit[sub],m_fit[sub]
    cols=np.where(univ)[0]
    scU=sc_fit[:,cols]; dU=DH[yid_fit][:,cols]; cardU=CARD[cols]
    masks=[group_mask(Y_fit,m_fit,j,mode) for j in range(L)]
    if any(mk.sum()==0 for mk in masks): return None
    best=None; bestsize=9e18
    for H in itertools.product(grid,repeat=3):
        thr=np.array([ (0 if c==0 else H[c-1]) for c in cardU ])   # per-column cut
        admit=scU<=thr[None,:]
        loss=np.where(admit,dU,B).min(axis=1)
        if all(loss[mk].mean()<=ALPHA+1e-12 for mk in masks):
            szrow=admit.sum(1)
            if mode=='A':                      # match deployment eligibility: empty when no group applies
                szrow=np.where(m_fit.any(axis=1),szrow,0)
            size=szrow.mean()
            if size<bestsize: bestsize=size; best=np.array(H,float)
    if best is None: return None
    h=np.zeros(8)
    for v in range(8):
        if CARD[v]>0: h[v]=best[CARD[v]-1]
    return h

# ---- deploy + evaluate --------------------------------------------------------
def evaluate(admit, Yte, mte, yidte, mode):
    """admit: NEVAL x 8 boolean deployed set. Returns per-group & pooled risk/size."""
    dc=DH[yidte]
    loss=np.where(admit,dc,B).min(axis=1)
    size=admit.sum(1)
    res={}
    for j in range(L):
        mk=group_mask(Yte,mte,j,mode)
        res[f'risk{j+1}']=float(loss[mk].mean()); res[f'size{j+1}']=float(size[mk].mean())
    res['worst_risk']=max(res[f'risk{j+1}'] for j in range(L))
    res['pooled_size']=float(size.mean()); res['pooled_risk']=float(loss.mean())
    return res

def lab_candidate_cuts(tf):   # per-candidate union cut = max tf over labels v asserts
    return np.array([max((tf[j] for j in range(L) if V[v,j]), default=-np.inf) for v in range(8)])

def _slice_union_admit(Yc,mc,scc,dcc,scte,mte,mode,univ):
    tf=np.array([exact_crc(scc[group_mask(Yc,mc,j,mode)],dcc[group_mask(Yc,mc,j,mode)],
                           (V[:,j]==1) if mode=='B' else univ) for j in range(L)])
    if mode=='B':
        uc=lab_candidate_cuts(tf); admit=scte<=uc[None,:]
    else:
        ucrow=np.where(mte.astype(bool),tf[None,:],-np.inf).max(axis=1)[:,None]; admit=scte<=ucrow
    return admit, bool(np.all(tf>=LMAX))

def _direct_unshaped_admit(Yc,mc,scc,dcc,scte,mode,univ):
    td=np.array([exact_crc(scc[group_mask(Yc,mc,j,mode)],dcc[group_mask(Yc,mc,j,mode)],univ)
                 for j in range(L)])
    tdir=td.max(); admit=(scte<=tdir)&univ[None,:]
    return admit, bool(tdir>=LMAX)

def run_cell(n,split,reps,seed,mode):
    rng=np.random.default_rng(seed); rows=[]
    univ=UNIV_B if mode=='B' else UNIV_A
    for rep in range(reps):
        Y,m,sc,yid=gen(n,rng); dc=DH[yid]
        nf=int(round(split*n)); perm=rng.permutation(n); fit=perm[:nf]; cor=perm[nf:]
        Yc,mc,scc,yidc,dcc=Y[cor],m[cor],sc[cor],yid[cor],dc[cor]      # correction fold
        Yf,mf,scf,yidf=Y[fit],m[fit],sc[fit],yid[fit]                  # fit fold
        Yte,mte,scte,yidte=gen(NEVAL,rng)
        meth={}  # name -> (admit matrix NEVAL x 8, fallback bool)
        # In mode A (feature-defined) a test point in NO feature group gets the empty
        # set under the per-row union. The global-threshold direct methods must use the
        # SAME deployment eligibility so the size comparison isolates threshold coupling
        # from candidate eligibility (audit P0-1). Groupwise/union methods already gate
        # via the per-row max (-inf when no group applies).
        gate = mte.any(axis=1)[:,None] if mode=='A' else np.ones((NEVAL,1),bool)

        # no-shape methods calibrate on the FULL budget (all n)
        meth['slice_union']=_slice_union_admit(Y,m,sc,dc,scte,mte,mode,univ)
        du,dufb=_direct_unshaped_admit(Y,m,sc,dc,scte,mode,univ)
        meth['direct_unshaped']=(du&gate, dufb)
        # split-cost diagnostic: union restricted to the correction fold
        meth['slice_union_cor']=_slice_union_admit(Yc,mc,scc,dcc,scte,mte,mode,univ)

        # direct learned shape: shape on fit fold, CRC on correction fold (total n)
        h=fit_learned_shape(scf,yidf,Yf,mf,mode,univ,GRID)
        if h is None:
            meth['direct_learned']=(np.tile(univ,(NEVAL,1))&gate, True)
        else:
            scs=scc-h[None,:]
            th=np.array([exact_crc(scs[group_mask(Yc,mc,j,mode)],dcc[group_mask(Yc,mc,j,mode)],univ)
                         for j in range(L)])
            tdh=th.max()
            admit=((scte-h[None,:])<=tdh)&univ[None,:]&gate
            meth['direct_learned']=(admit, bool(tdh>=LMAX))

        # atomwise: per nonzero candidate, calibrate on points whose TRUE label == v (full n)
        ta=np.full(8,LMAX)
        for v in NONZERO:
            pts=(yid==v); elig=np.zeros(8,bool); elig[v]=True
            ta[v]=exact_crc(sc[pts],dc[pts],elig) if pts.sum()>0 else LMAX
        admit=(scte<=ta[None,:])&univ[None,:]&gate
        meth['atomwise']=(admit, bool(np.all(ta[NONZERO]>=LMAX)))

        # groupwise_quantile_union (coverage reference, no CRC): per-group (1-alpha) score
        # quantile at the TRUE label among group points, deployed as the per-candidate /
        # per-row union (full n). NOTE: these are THREE independent conditional quantiles,
        # NOT the joint group-indicator pinball fit; the three focal indicators are not
        # saturated over the seven nonzero label patterns, so this is not the Kandinsky
        # joint optimizer and its numbers are not attributed to Kandinsky. It targets
        # label coverage, not Hamming risk, and is a reference only.
        kk=np.full(L,LMAX)
        for j in range(L):
            mk=group_mask(Y,m,j,mode)
            sj=np.sort(sc[mk, yid[mk]])
            kk[j]=sj[min(len(sj)-1,int(np.floor((1-ALPHA)*len(sj))))] if len(sj)>0 else LMAX
        if mode=='B':
            kc=lab_candidate_cuts(kk); admit=scte<=kc[None,:]
        else:
            kcrow=np.where(mte.astype(bool),kk[None,:],-np.inf).max(axis=1)[:,None]
            admit=scte<=kcrow
        meth['groupwise_quantile_union']=(admit, False)

        for name,(admit,fb) in meth.items():
            r=evaluate(admit,Yte,mte,yidte,mode)
            r.update(mode=mode,n=n,split=split,rep=rep,method=name,fallback=float(fb))
            rows.append(r)
    return rows

if __name__=='__main__':
    mode_arg=sys.argv[1] if len(sys.argv)>1 else 'validate'
    NEVAL=15000
    # score-scale grid for the shape library: spans nonzero-candidate score range
    _rng=np.random.default_rng(1); _,_,_sc,_=gen(4000,_rng)
    _lo,_hi=np.percentile(_sc[:,NONZERO],[1,99])
    GRID=np.linspace(_lo-0.1,_hi+0.1,9)     # 9^3=729 symmetric shapes
    if mode_arg=='validate':
        t0=time.time()
        rows=run_cell(6400,0.5,20,seed=11,mode='B')
        df=pd.DataFrame(rows)
        g=df.groupby('method').agg(pooled_size=('pooled_size','mean'),
            worst_risk=('worst_risk','mean'),r1=('risk1','mean'),r2=('risk2','mean'),
            r3=('risk3','mean'),fb=('fallback','mean')).round(4)
        print("mode B, n=6400, 20 reps, 50/50 split  (alpha=0.5, B=3)")
        print(g.to_string()); print("time %.1fs"%(time.time()-t0))
        sys.exit(0)
    if mode_arg=='full':
        reps=int(sys.argv[2]) if len(sys.argv)>2 else 100
        out='ham_raw.csv'
        if os.path.exists(out): os.remove(out)
        first=True; sd=4200
        for mode in ['B','A']:
            for n in [400,1600,6400,25600]:
                for split in [0.25,0.50,0.75]:
                    sd+=1; t0=time.time()
                    rows=run_cell(n,split,reps,seed=sd,mode=mode)
                    rt=round(time.time()-t0,2)
                    for r in rows: r['cell_runtime_s']=rt
                    pd.DataFrame(rows).to_csv(out,mode='a',header=first,index=False); first=False
                    print(f"mode {mode} n={n} split={split} ({rt}s)",flush=True)
        print("WROTE ham_raw.csv")
