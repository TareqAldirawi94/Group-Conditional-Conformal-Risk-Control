"""
Verifies the reconciliation-slack theory for overlapping-group CRC.

Decomposition (any bounded set-monotone loss):
    alpha_j - R_j^dep = (alpha_j - R_j^slice) + (R_j^slice - R_j^dep),  both >= 0.
Inclusion-type identity (miscoverage, c=1), for the true label's shaped score U=S-b:
    R_j^slice - R_j^dep = E[ 1{ t_j < U <= max_{k: Z in G_k} t_k } | Z in G_j ].
Quantitative bound (bounded conditional score density f<=fbar):
    R_j^slice - R_j^dep <= fbar * E[ max_{k: Z in G_k} (t_k - t_j)_+ | Z in G_j ].
Two-group inclusion model; G1={10,11}, G2={01,11}.

This is an analytic diagnostic that verifies the slack identities in a Gaussian
two-group score model with a linear size-aware cost. It is NOT the bounded-uniform
numerical illustration of the appendix; it checks the decomposition and the
inclusion-type gap formula, not manuscript figure values.
"""
import numpy as np
from math import erf, sqrt, pi as PI
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

def Phi(z): return 0.5*(1+erf(z/sqrt(2)))
phi=norm.pdf
FBAR=1/sqrt(2*PI)                         # max standard-normal density

alpha=0.20
pi_={'10':0.40,'01':0.40,'11':0.20}
mu ={'10':0.0 ,'01':1.6 ,'11':0.8 }
regions=['10','01','11']
G={'1':['10','11'],'2':['01','11']}
def rho(r,t): return 1-Phi(t-mu[r])

# population raw union thresholds (b=0): R_j(lambda)=alpha
lam1=brentq(lambda L:(pi_['10']*rho('10',L)+pi_['11']*rho('11',L))/(pi_['10']+pi_['11'])-alpha,-30,30)
lam2=brentq(lambda L:(pi_['01']*rho('01',L)+pi_['11']*rho('11',L))/(pi_['01']+pi_['11'])-alpha,-30,30)
print(f"population union thresholds: lam1={lam1:.4f}  lam2={lam2:.4f}  (disagreement {abs(lam2-lam1):.4f})")

def group_terms(j, t):
    """t: dict of per-group raw thresholds; deployed overlap cutoff = max over containing groups."""
    Gj=G[j]; w=sum(pi_[r] for r in Gj)
    tj=t[j]
    # slice risk: every group-j candidate at t_j
    Rslice=sum(pi_[r]*rho(r,tj) for r in Gj)/w
    # deployed risk: region r in G_j gets max over groups k with r in G_k
    def dep_cut(r):
        ks=[k for k in G if r in G[k]]
        return max(t[k] for k in ks)
    Rdep=sum(pi_[r]*rho(r,dep_cut(r)) for r in Gj)/w
    term2=Rslice-Rdep
    # identity RHS: P(t_j < S <= dep_cut(r)) over regions, weighted
    ident=sum(pi_[r]*(rho(r,tj)-rho(r,dep_cut(r))) for r in Gj)/w
    # bound: fbar * E[max_k (t_k - t_j)_+ | G_j]
    disag=sum(pi_[r]*max((t[k]-tj) for k in G if r in G[k]) for r in Gj)/w
    bound=FBAR*max(disag,0.0)
    return Rslice,Rdep,term2,ident,bound

t_union={'1':lam1,'2':lam2}
print("\n=== UNSHAPED UNION (large threshold disagreement) ===")
for j in ['1','2']:
    Rs,Rd,t2,idn,bd=group_terms(j,t_union)
    print(f" group {j}: R_slice={Rs:.4f}  R_dep={Rd:.4f}  alpha-R_dep={alpha-Rd:+.4f}")
    print(f"          reconciliation term2={t2:.4f}  identity={idn:.4f}  (match={abs(t2-idn)<1e-9})")
    print(f"          bound={bd:.4f}  ratio term2/bound={ (t2/bd if bd>0 else float('nan')):.3f}")

# aligned shape: size-aware optimum makes both groups' crossings coincide -> t1=t2=K, zero disagreement
# at the optimum the deployed region cutoffs are t*=(1.04,2.64,1.32); per-group shifts equal.
def R1(x): return (pi_['10']*rho('10',x[0])+pi_['11']*rho('11',x[2]))/(pi_['10']+pi_['11'])
def R2(x): return (pi_['01']*rho('01',x[1])+pi_['11']*rho('11',x[2]))/(pi_['01']+pi_['11'])
from scipy.optimize import minimize
a={'10':1.,'01':1.,'11':3.}
cons=[{'type':'ineq','fun':lambda x:alpha-R1(x)},{'type':'ineq','fun':lambda x:alpha-R2(x)}]
res=minimize(lambda x:sum(pi_[r]*a[r]*(x[i]+6) for i,r in enumerate(regions)),
             [1,2.5,1.3],constraints=cons,method='SLSQP',bounds=[(-5,8)]*3,options={'ftol':1e-14})
if not res.success or max(R1(res.x),R2(res.x))>alpha+1e-6:
    raise RuntimeError("SLSQP failed or violates constraints: "+str(res.message))
tstar={r:res.x[i] for i,r in enumerate(regions)}
print(f"\naligned size-aware optimum t* = ({tstar['10']:.3f},{tstar['01']:.3f},{tstar['11']:.3f})")
print(" aligned: both group constraints bind, overlap cutoff = t*_11 for both groups => zero disagreement")
for j in ['1','2']:
    Gj=G[j]; w=sum(pi_[r] for r in Gj)
    # deployed = t* ; slice at aligned shift = same for true label -> term2=0
    Rdep=sum(pi_[r]*rho(r,tstar[r]) for r in Gj)/w
    print(f" group {j}: R_dep={Rdep:.4f}  (alpha-R_dep={alpha-Rdep:+.4f}; reconciliation term2=0 by alignment)")

# ---- finite-sample: learned aligning shape drives D_j->0 and R_dep->alpha ----
from scipy.stats import norm as NN
K=2.0
def fit_smooth(Df):
    muhat={r:(np.mean(Df[r]) if len(Df[r]) else mu[r]) for r in regions}
    cnt={r:len(Df[r]) for r in regions}; tot=sum(cnt.values()) or 1
    ph={r:cnt[r]/tot for r in regions}
    def t_of_u(u):
        r10=((ph['10']+ph['11'])*alpha-ph['11']*u)/ph['10']
        r01=((ph['01']+ph['11'])*alpha-ph['11']*u)/ph['01']
        return {'10':muhat['10']+NN.ppf(1-r10),'01':muhat['01']+NN.ppf(1-r01),'11':muhat['11']+NN.ppf(1-u)}
    uhi=min(0.999,(ph['10']+ph['11'])*alpha/ph['11'],(ph['01']+ph['11'])*alpha/ph['11'])
    rr=minimize_scalar(lambda u:sum(ph[r]*a[r]*(t_of_u(u)[r]+6) for r in regions),
                       bounds=(1e-4,max(2e-4,uhi-1e-4)),method='bounded',options={'xatol':1e-9})
    tau=t_of_u(rr.x); return {r:tau[r]-K for r in regions}
def crc_shift(S,grp,base):
    m=[(r,s) for r in G[grp] for s in S[r]]; nj=len(m)
    if nj==0: return 30.0
    k=alpha*(nj+1)-1.0
    if k<0: return 30.0
    js=sorted(s-base[r] for r,s in m); idx=nj-int(np.floor(k))-1
    return js[idx] if 0<=idx<nj else -30.0
def crc_raw(S,grp):
    m=[s for r in G[grp] for s in S[r]]; nj=len(m)
    if nj==0: return 30.0
    k=alpha*(nj+1)-1.0
    if k<0: return 30.0
    js=sorted(m); idx=nj-int(np.floor(k))-1
    return js[idx] if 0<=idx<nj else -30.0
def dep_risk(j, cut):   # cut: dict region->deployed cutoff
    Gj=G[j]; w=sum(pi_[r] for r in Gj)
    return sum(pi_[r]*rho(r,cut[r]) for r in Gj)/w
def disagreement1(t1,t2):   # D_1 = (pi11/(pi10+pi11)) (t2-t1)_+
    return pi_['11']/(pi_['10']+pi_['11'])*max(t2-t1,0.0)
def sim_learned(n,reps=3000,seed=0):
    rng=np.random.default_rng(seed); RU=[]; RL=[]; DU=[]; DL=[]
    for _ in range(reps):
        cf=rng.multinomial(n,[pi_[r] for r in regions]); Sf={r:list(mu[r]+rng.standard_normal(k)) for r,k in zip(regions,cf)}
        cc=rng.multinomial(n,[pi_[r] for r in regions]); Sc={r:list(mu[r]+rng.standard_normal(k)) for r,k in zip(regions,cc)}
        # union (raw), pooled 2n for fairness
        Sp={r:Sf[r]+Sc[r] for r in regions}
        u1=crc_raw(Sp,'1'); u2=crc_raw(Sp,'2')
        RU.append(dep_risk('1',{'10':u1,'01':u2,'11':max(u1,u2)})); DU.append(disagreement1(u1,u2))
        # learned shape + per-group CRC on shaped slice
        b=fit_smooth(Sf); t1=crc_shift(Sc,'1',b); t2=crc_shift(Sc,'2',b)
        cut={'10':b['10']+t1,'01':b['01']+t2,'11':b['11']+max(t1,t2)}
        RL.append(dep_risk('1',cut)); DL.append(disagreement1(t1,t2))
    return np.mean(RU),np.mean(DU),np.mean(RL),np.mean(DL)
print("\n=== finite-sample: deployed group-1 risk & disagreement D_1 (union vs learned shape) ===")
print("          union R1_dep   union D_1 |  learned R1_dep   learned D_1")
for n in [1000,4000,16000]:
    ru,du,rl,dl=sim_learned(n)
    print(f" n={n:6d}   {ru:.4f}      {du:.4f}   |   {rl:.4f}        {dl:.4f}")
print(" (target alpha=0.20; union stays ~0.106 with D_1~0.36; learned -> 0.20 with D_1 -> 0)")

# ---- finite-sample E-check of the identity (union, miscoverage) ----
print("\n=== finite-sample check: E[term2] vs E[identity], union, group 1 ===")
def crc_raw(S, grp):
    m=[s for r in G[grp] for s in S[r]]; nj=len(m)
    if nj==0: return 30.0
    k=alpha*(nj+1)-1.0
    if k<0: return 30.0
    js=sorted(m); idx=nj-int(np.floor(k))-1
    return js[idx] if 0<=idx<nj else -30.0
def sim(n,reps=4000,seed=0):
    rng=np.random.default_rng(seed); T2=[]; IDN=[]; RD=[]; BD=[]
    for _ in range(reps):
        c=rng.multinomial(n,[pi_[r] for r in regions])
        S={r:list(mu[r]+rng.standard_normal(k)) for r,k in zip(regions,c)}
        t={'1':crc_raw(S,'1'),'2':crc_raw(S,'2')}
        Rs,Rd,t2,idn,bd=group_terms('1',t)   # population risks at the realized thresholds
        T2.append(t2); IDN.append(idn); RD.append(Rd); BD.append(bd)
    return np.mean(T2),np.mean(IDN),np.mean(RD),np.mean(BD)
for n in [500,2000]:
    mt2,midn,mrd,mbd=sim(n)
    print(f" n={n:5d}: E[term2]={mt2:.4f}  E[identity]={midn:.4f}  E[R_dep]={mrd:.4f}  E[bound]={mbd:.4f}  ratio={mt2/mbd:.3f}")
