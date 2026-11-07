"""Independent corrected replication of Section 5.2 E4b (NumPy/SciPy).
A: feature-defined group union contains only groups whose membership is on.
B: focal-label groups only, candidate 000 remains ineligible.
Exact event-sweep CRC cutoff instead of bisection. Population reference via bisection.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from pathlib import Path

L=3; B=3; ALPHA=.50; RHO=.4; GAM=1.3; NOISE=1.
NS=(400,1600,6400,25600)
REPS=100
NEVAL=15000
POP=400000
SEED=20261025
LMAX=42.
V=((np.arange(2**L)[:,None]>>np.arange(L))&1).astype(np.int8)
DH=np.abs(V[:,None,:]-V[None,:,:]).sum(axis=2).astype(np.int8)
RNG=np.random.default_rng(SEED)
OUT=Path('figures');OUT.mkdir(exist_ok=True,parents=True)

def gen(n):
    z=np.sqrt(RHO)*RNG.standard_normal((n,1))+np.sqrt(1-RHO)*RNG.standard_normal((n,L))
    Y=(z>0).astype(np.int8)
    eta=GAM*z+NOISE*RNG.standard_normal((n,L))
    q=np.clip(expit(eta),1e-6,1-1e-6)
    sc=(-np.log1p(-q)).sum(axis=1)[:,None]+(np.log1p(-q)-np.log(q))@V.T
    yid=(Y@np.array([1,2,4])).astype(int)
    m=RNG.integers(0,2,(n,L),dtype=np.int8)
    return Y,m,sc,yid

def risk_for_cut(sc,dcol,cut):
    # scalar, length-8 candidate cutoff, or n-row cutoff
    eligible=sc<=cut
    return np.where(eligible,dcol,B).min(axis=1)

def size_for_cut(sc,cut): return (sc<=cut).sum(axis=1)

def exact_crc(sc,dcol,eligible):
    n=len(sc)
    if n==0 or B*n < (n+1)*(B-ALPHA): return LMAX
    s=sc[:,eligible]; d=dcol[:,eligible]
    order=np.argsort(s,axis=1)
    scores=np.take_along_axis(s,order,axis=1)
    distances=np.take_along_axis(d,order,axis=1)
    after=np.minimum.accumulate(distances,axis=1)
    before=np.concatenate((np.full((n,1),B,dtype=np.int8),after[:,:-1]),axis=1)
    jumps=before-after
    mask=jumps>0
    seq=np.argsort(scores[mask])
    cum=np.cumsum(jumps[mask][seq])
    needed=(n+1)*(B-ALPHA)
    k=np.searchsorted(cum,needed,side='left')
    if k>=len(cum): return LMAX
    return float(scores[mask][seq][k])

def lab_candidate_cuts(tf):
    return np.array([max((tf[j] for j in range(L) if V[v,j]), default=-np.inf) for v in range(8)])

def feature_union_cuts(m,tf):
    # n x 1. -inf if no G_j applies (not used for conditional G_j risks)
    return np.where(m.astype(bool),tf[None,:],-np.inf).max(axis=1)[:,None]

def population_reference(mode, ep):
    Y,m,sc,yid=ep
    mask=(m[:,0]==1) if mode=='A' else (Y[:,0]==1)
    sel=(np.ones(8,dtype=bool) if mode=='A' else V[:,0]==1)
    s=sc[mask];d=DH[yid[mask]]
    def sr(t): return float(np.where(s[:,sel]<=t,d[:,sel],B).min(axis=1).mean())
    lo=-2.;hi=LMAX
    for i in range(32):
        mid=.5*(lo+hi)
        if sr(mid)>ALPHA:lo=mid
        else: hi=mid
    K=hi
    Rs=sr(K)
    if mode=='A':Rd=Rs
    else:
        U=lab_candidate_cuts(np.repeat(K,L))
        Rd=float(risk_for_cut(s,d,U).mean())
    return {'scenario':mode,'K':K,'R_slice_inf':Rs,'R_dep_inf':Rd,'Delta_star':Rs-Rd}

def one_rep(n,mode):
    Y,m,sc,yid=gen(n);dc=DH[yid]
    tf=[]
    for j in range(L):
        mask=(m[:,j]==1) if mode=='A' else (Y[:,j]==1)
        eligible=(np.ones(8,dtype=bool) if mode=='A' else V[:,j]==1)
        tf.append(exact_crc(sc[mask],dc[mask],eligible))
    tf=np.array(tf)
    Y,m,sc,yid=gen(NEVAL);dc=DH[yid]
    uc=feature_union_cuts(m,tf) if mode=='A' else lab_candidate_cuts(tf)
    rj=[];dj=[];sj=[];sn=[]
    for j in range(L):
        mask=(m[:,j]==1) if mode=='A' else (Y[:,j]==1)
        csc=sc[mask]; dis=dc[mask]
        slcut=tf[j] if mode=='A' else np.where(V[:,j]==1,tf[j],-np.inf)
        depcut=uc[mask] if mode=='A' else uc
        lrs=risk_for_cut(csc,dis,slcut)
        lrd=risk_for_cut(csc,dis,depcut)
        srs=size_for_cut(csc,slcut)
        srd=size_for_cut(csc,depcut)
        rj.append(float(lrs.mean()));dj.append(float(lrd.mean()))
        sj.append(float(srs.mean()));sn.append(float(srd.mean()))
        if np.any(lrd>lrs) or np.any(srd<srs) or np.any((lrs-lrd)>B*(srd-srs)):
            raise AssertionError('pointwise bound failed')
    return {'R_slice':np.mean(rj),'R_dep':np.mean(dj),'H':np.mean(np.array(rj)-np.array(dj)),
        'B_dS': B*np.mean(np.array(sn)-np.array(sj)),'spread':tf.max()-tf.min(),
        'group_max_violation':max(np.array(rj)-np.array(dj)-B*(np.array(sn)-np.array(sj)))}

def run():
    ep=gen(POP)
    pop=[population_reference(mode,ep) for mode in 'AB']
    pd.DataFrame(pop).to_csv(OUT/'example2b_population.csv',index=False)
    rows=[];raw=[]
    for mode in 'AB':
        for n in NS:
            vals=[]
            for r in range(REPS):
                a=one_rep(n,mode);a.update(scenario=mode,n=n,rep=r+1); vals.append(a)
            raw+=vals
            out={'scenario':mode,'n':n}
            for key in ('R_slice','R_dep','H','B_dS','spread'):
                z=np.array([v[key] for v in vals]);out[key]=z.mean();out[key+'_hw']=1.96*z.std(ddof=1)/np.sqrt(REPS)
            out['inequality_all_reps']=all(v['group_max_violation'] <=1e-12 for v in vals)
            rows.append(out)
            print(mode,n, {k:round(out[k],5) for k in ('R_slice','R_dep','H','B_dS','spread')},flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'example2b_summary.csv',index=False)
    pd.DataFrame(raw).to_csv(OUT/'example2b_replicates.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    a=axes[0];b=df[df.scenario=='B'];d=next(x for x in pop if x['scenario']=='B')
    a.plot(NS,b.R_slice,marker='o',label='Slice risk')
    a.plot(NS,b.R_dep,marker='s',label='Deployed risk')
    a.axhline(ALPHA,ls='--',lw=1.1,label='Nominal / population slice risk')
    a.axhline(d['R_dep_inf'],ls=':',lw=1.2,label='Population deployed risk')
    a.set_xscale('log');a.set_xticks(NS,[str(x) for x in NS]);a.set_ylim(.25,.56)
    a.set_title('Label-defined focal groups');a.set_xlabel('Calibration size');a.set_ylabel('Conditional Hamming risk');a.legend(fontsize=8)
    a=axes[1]
    for mode,label in [('A','Feature-defined'),('B','Label-defined')]:
        dat=df[df.scenario==mode]
        a.plot(NS,dat.H,marker='s',label=f'{label}: risk reduction')
        a.plot(NS,dat.B_dS,marker='^',ls='--',label=f'{label}: B × size inflation')
    a.set_xscale('log');a.set_xticks(NS,[str(x) for x in NS]);a.set_xlabel('Calibration size')
    a.set_ylabel('Risk reduction and bound');a.set_title('Risk–size inequality');a.legend(fontsize=7)
    for fmt in ('pdf','png'): fig.savefig(OUT/('fig_overlap_hamming.'+fmt),dpi=175)
    plt.close(fig)
    print('POP: ',pop,flush=True)

if __name__=='__main__':run()
