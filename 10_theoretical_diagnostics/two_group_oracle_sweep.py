"""Robustness sweep of the two-group population-optimum diagnostic across
several (correlation, difficulty-contrast, base-rate, alpha) settings.
Like two_group_oracle.py, the "oracle" is a NUMERICAL finite-grid search over the
overlap cutoff, not a certified global optimum; reported sizes are grid-accurate."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
VEC=np.array([[0,0],[1,0],[0,1],[1,1]],float)

def run(R_CORR,BASE,GAMMA,BIAS,ALPHA,seed):
    TAU=norm.ppf(1-np.array(BASE))
    GAMMA=np.array(GAMMA); BIAS=np.array(BIAS)
    def gen(n,rng):
        f=rng.standard_normal((n,1)); e=rng.standard_normal((n,2))
        z=np.sqrt(R_CORR)*f+np.sqrt(1-R_CORR)*e
        Y=(z>TAU[None,:]).astype(float)
        eta=GAMMA[None,:]*(z-TAU[None,:])-BIAS[None,:]+NOISE*rng.standard_normal((n,2))
        q=np.clip(1/(1+np.exp(-eta)),1e-9,1-1e-9); c1=-np.log(q); c0=-np.log(1-q)
        S=(Y*c1+(1-Y)*c0).sum(1); base=c0.sum(1); delta=c1-c0
        reg=np.where((Y[:,0]==1)&(Y[:,1]==1),3,np.where(Y[:,0]==1,1,np.where(Y[:,1]==1,2,0)))
        return Y,S,base,delta,reg
    NOISE=1.0
    rng=np.random.default_rng(seed)
    Yc,Sc,basec,deltac,regc=gen(600000,rng)
    m10=(regc==1).mean();m01=(regc==2).mean();m11=(regc==3).mean();w1=m10+m11;w2=m01+m11
    lam1=np.quantile(Sc[Yc[:,0]==1],1-ALPHA); lam2=np.quantile(Sc[Yc[:,1]==1],1-ALPHA)
    Scm=basec[:200000,None]+deltac[:200000]@VEC.T
    cols=[np.sort(Scm[:,v]) for v in range(4)]
    Fv=lambda v,t: np.searchsorted(cols[v],t,side='right')/len(cols[v])
    Sreg={r:np.sort(Sc[regc==r]) for r in [1,2,3]}
    def rho(r,t): s=Sreg[r]; return 1.0-np.searchsorted(s,t,side='right')/len(s)
    def rho_inv(r,tg):
        s=Sreg[r]
        if tg<=0:return s[-1]+1
        if tg>=1:return s[0]-1
        k=min(max(int(np.ceil((1-tg)*len(s)))-1,0),len(s)-1); return s[k]
    # union
    u=(lam1,lam2,max(lam1,lam2))
    # shape
    def sq(b,g): U=Sc-b*Yc[:,1]; return np.quantile(U[Yc[:,g]==1],1-ALPHA)
    b2=brentq(lambda b:sq(b,0)-sq(b,1),-15,15); tS=sq(b2,0)
    s=(tS+VEC[1,1]*b2, tS+VEC[2,1]*b2, tS+VEC[3,1]*b2)
    # oracle (1-D over t11)
    best=None
    for t11 in np.linspace(np.quantile(Sc,0.4),np.quantile(Sc,0.999),400):
        r1=(w1*ALPHA-m11*rho(3,t11))/m10; r2=(w2*ALPHA-m11*rho(3,t11))/m01
        if r1<0 or r2<0: continue
        t10=rho_inv(1,r1); t01=rho_inv(2,r2); sz=Fv(1,t10)+Fv(2,t01)+Fv(3,t11)
        if best is None or sz<best[0]: best=(sz,t10,t01,t11)
    o=best[1:]
    # eval on separate sample
    Ye,Se,basee,deltae,rege=gen(300000,np.random.default_rng(seed+1000))
    Sce=basee[:,None]+deltae@VEC.T
    def ev(t):
        size=(Sce[:,1]<=t[0]).mean()+(Sce[:,2]<=t[1]).mean()+(Sce[:,3]<=t[2]).mean()
        cut=np.where(rege==1,t[0],np.where(rege==2,t[1],np.where(rege==3,t[2],np.inf)))
        R1=(Se[Ye[:,0]==1]>cut[Ye[:,0]==1]).mean(); R2=(Se[Ye[:,1]==1]>cut[Ye[:,1]==1]).mean()
        return size,R1,R2
    Su,_,_=ev(u); Ss,Rs1,Rs2=ev(s); So,Ro1,Ro2=ev(o)
    valid=(Rs1<=ALPHA+2e-3 and Rs2<=ALPHA+2e-3 and Ro1<=ALPHA+2e-3 and Ro2<=ALPHA+2e-3)
    frac=(Su-Ss)/(Su-So) if Su>So+1e-6 else float('nan')
    return abs(lam1-lam2),Su,Ss,So,frac,valid

NOISE=1.0
cfgs=[
 dict(R_CORR=0.3,BASE=[0.5,0.3],GAMMA=[3.0,0.4],BIAS=[0,2.0],ALPHA=0.10,seed=3),
 dict(R_CORR=0.2,BASE=[0.5,0.25],GAMMA=[3.2,0.5],BIAS=[0,2.5],ALPHA=0.10,seed=11),
 dict(R_CORR=0.4,BASE=[0.45,0.35],GAMMA=[2.8,0.6],BIAS=[0,1.5],ALPHA=0.10,seed=21),
 dict(R_CORR=0.3,BASE=[0.5,0.3],GAMMA=[3.0,0.4],BIAS=[0,2.0],ALPHA=0.20,seed=31),
 dict(R_CORR=0.3,BASE=[0.5,0.3],GAMMA=[3.5,0.3],BIAS=[0,3.0],ALPHA=0.10,seed=41),
]
print(f"{'disagree':>9}{'S_union':>9}{'S_shape':>9}{'S_oracle':>9}{'frac_recov':>11}  valid")
for c in cfgs:
    d,Su,Ss,So,fr,val=run(**c)
    print(f"{d:9.3f}{Su:9.3f}{Ss:9.3f}{So:9.3f}{fr:11.2f}  {val}")
