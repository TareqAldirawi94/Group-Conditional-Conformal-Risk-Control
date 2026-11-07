"""
Reproduces every number in TwoGroupEfficiency.tex (interior linear-cost model).

Two overlapping groups G1={10,11}, G2={01,11}; inclusion-type (scalar-collapse) loss.
Region-r realized-label score  S|r ~ N(mu_r, 1);  region risk  rho_r(t)=1-Phi(t-mu_r).
Non-saturating linear deployment cost  q_r(t)=a_r*(t+6)_+   (so the optimum is interior).
B_j=1 (miscoverage).

Outputs:
  (a) population baselines + interior optimum  -> V(C*), V(union), Delta
  (b) finite-sample table for the LEARNED smooth plug-in shape + pointwise CRC, vs union
  (c) oracle shape and naive empirical constrained estimator as comparisons
"""
import numpy as np
from math import erf, sqrt
from scipy.optimize import brentq, minimize

def Phi(z): return 0.5*(1+erf(z/sqrt(2)))

alpha = 0.20
pi  = {'10':0.40, '01':0.40, '11':0.20}
mu  = {'10':0.0 , '01':1.6 , '11':0.8 }
a   = {'10':1.0 , '01':1.0 , '11':3.0 }      # linear-cost slopes
regions = ['10','01','11']
G = {'1':['10','11'], '2':['01','11']}
SHIFT = 6.0                                  # (t+6)_+ keeps cost positive on Theta

def rho(r,t):  return 1-Phi(t-mu[r])
def q_r(r,t):  return a[r]*max(t+SHIFT, 0.0)
def tfor(r,target): return brentq(lambda t: rho(r,t)-target, -30, 30)

def grisk(t):
    R1=(pi['10']*rho('10',t['10'])+pi['11']*rho('11',t['11']))/(pi['10']+pi['11'])
    R2=(pi['01']*rho('01',t['01'])+pi['11']*rho('11',t['11']))/(pi['01']+pi['11'])
    return R1,R2
def size(t): return sum(pi[r]*q_r(r,t[r]) for r in regions)

up = lambda x:{'10':x[0],'01':x[1],'11':x[2]}

def solve_sizeaware(rho_fn, pi_fn):
    """min sum pi_r q_r(tau_r) s.t. R_j(tau)<=alpha, using supplied rho/pi (smooth)."""
    def R1(x): return (pi_fn['10']*rho_fn('10',x[0])+pi_fn['11']*rho_fn('11',x[2]))/(pi_fn['10']+pi_fn['11'])
    def R2(x): return (pi_fn['01']*rho_fn('01',x[1])+pi_fn['11']*rho_fn('11',x[2]))/(pi_fn['01']+pi_fn['11'])
    cons=[{'type':'ineq','fun':lambda x: alpha-R1(x)},
          {'type':'ineq','fun':lambda x: alpha-R2(x)}]
    x0=[tfor(r,alpha) for r in regions]
    res=minimize(lambda x: sum(pi_fn[r]*a[r]*max(x[i]+SHIFT,0.0) for i,r in enumerate(regions)),
                 x0, constraints=cons, method='SLSQP',
                 bounds=[(-10,10)]*3, options={'ftol':1e-14,'maxiter':5000})
    if not res.success:
        raise RuntimeError("SLSQP did not converge: "+str(res.message))
    # constraint residual check: both group risks must be <= alpha at the solution
    if max(R1(res.x), R2(res.x)) > alpha + 1e-6:
        raise RuntimeError("SLSQP solution violates the risk constraints")
    return up(res.x)

# ---------------- population baselines & interior optimum ----------------
lam1=brentq(lambda L:(pi['10']*rho('10',L)+pi['11']*rho('11',L))/(pi['10']+pi['11'])-alpha,-30,30)
lam2=brentq(lambda L:(pi['01']*rho('01',L)+pi['11']*rho('11',L))/(pi['01']+pi['11'])-alpha,-30,30)
t_union={'10':lam1,'01':lam2,'11':max(lam1,lam2)}
t_star =solve_sizeaware(rho, pi)
Vstar, Vunion = size(t_star), size(t_union)
K=2.0

# ---- one-dimensional characterization (global optimality, interior model) ----
# parametrize by overlap risk u = rho_11(t_11); both constraints binding give
# rho_10 = rho_01 = 0.3 - u/2, hence closed-form t_r(u); V(u) is strictly convex.
from scipy.optimize import minimize_scalar
from scipy.stats import norm as _N
def _t10(u): return _N.ppf(0.7+u/2)
def _t01(u): return 1.6+_N.ppf(0.7+u/2)
def _t11(u): return 0.8+_N.ppf(1-u)
def _V1d(u): return 0.4*(_t10(u)+6)+0.4*(_t01(u)+6)+0.6*(_t11(u)+6)
_r=minimize_scalar(_V1d,bounds=(1e-6,0.599),method='bounded',options={'xatol':1e-10})
u_star=_r.x
_h=1e-4; _Vpp=lambda u:(_V1d(u+_h)-2*_V1d(u)+_V1d(u-_h))/_h**2
Vpp_min=min(_Vpp(u) for u in np.linspace(0.02,0.58,400))   # strict convexity check
b_oracle={r:t_star[r]-K for r in regions}       # oracle shape

# ---------------- estimators of the shape on D_fit ----------------
def fit_smooth(Df):
    """smooth plug-in via the REDUCED 1-D convex program (matches the proof exactly).
    Both constraints bind => rho_10,rho_01 are affine in the overlap risk u=rho_11(t_11),
    giving closed-form t_r(u); minimize the strictly convex V(u) in one variable."""
    muhat={r:(np.mean(Df[r]) if len(Df[r]) else mu[r]) for r in regions}
    cnt={r: len(Df[r]) for r in regions}; tot=sum(cnt.values()) or 1
    ph={r: cnt[r]/tot for r in regions}
    def t_of_u(u):
        rho10=((ph['10']+ph['11'])*alpha - ph['11']*u)/ph['10']
        rho01=((ph['01']+ph['11'])*alpha - ph['11']*u)/ph['01']
        return {'10':muhat['10']+_N.ppf(1-rho10),
                '01':muhat['01']+_N.ppf(1-rho01),
                '11':muhat['11']+_N.ppf(1-u)}
    def V1(u):
        t=t_of_u(u); return sum(ph[r]*a[r]*max(t[r]+SHIFT,0.0) for r in regions)
    # feasible u: rho_10,rho_01 in (0,1) and u in (0,1); upper bound from rho>=0
    uhi=min(0.999,
            (ph['10']+ph['11'])*alpha/ph['11'] if ph['11']>0 else 0.999,
            (ph['01']+ph['11'])*alpha/ph['11'] if ph['11']>0 else 0.999)
    r=minimize_scalar(V1, bounds=(1e-4, max(2e-4, uhi-1e-4)),
                      method='bounded', options={'xatol':1e-9})
    tau=t_of_u(r.x)
    return {rr:tau[rr]-K for rr in regions}

def fit_naive(Df):
    """naive empirical constrained ERM: step-function risks, SLSQP on a nonsmooth program."""
    S={r:np.sort(np.asarray(Df[r])) for r in regions}
    pihat={r: len(Df[r]) for r in regions}; tot=sum(pihat.values())
    pihat={r: (pihat[r]/tot if tot else pi[r]) for r in regions}
    def rh(r,t):
        s=S[r]
        return (np.sum(s>t)/len(s)) if len(s) else 0.0
    def R1(x): return (pihat['10']*rh('10',x[0])+pihat['11']*rh('11',x[2]))/(pihat['10']+pihat['11'])
    def R2(x): return (pihat['01']*rh('01',x[1])+pihat['11']*rh('11',x[2]))/(pihat['01']+pihat['11'])
    cons=[{'type':'ineq','fun':lambda x: alpha-R1(x)},
          {'type':'ineq','fun':lambda x: alpha-R2(x)}]
    x0=[tfor(r,alpha) for r in regions]
    res=minimize(lambda x: sum(pihat[r]*a[r]*max(x[i]+SHIFT,0.0) for i,r in enumerate(regions)),
                 x0, constraints=cons, method='SLSQP',
                 bounds=[(-10,10)]*3, options={'ftol':1e-12,'maxiter':5000})
    return {r:res.x[i]-K for i,r in enumerate(regions)}

# ---------------- pointwise CRC on a shaped slice ----------------
def crc_shift(Sd, grp, base):
    members=[(r,s) for r in G[grp] for s in Sd[r]]
    nj=len(members)
    if nj==0: return 30.0
    kmax=alpha*(nj+1)-1.0
    if kmax<0: return 30.0
    js=sorted(s-base[r] for r,s in members)
    idx=nj-int(np.floor(kmax))-1
    return js[idx] if 0<=idx<nj else -30.0

def deploy_pw(Scal, base):
    t1=crc_shift(Scal,'1',base); t2=crc_shift(Scal,'2',base)
    return {'10':base['10']+t1,'01':base['01']+t2,'11':base['11']+max(t1,t2)}

def crc_raw(Scal, grp):                          # union baseline: no shape
    members=[s for r in G[grp] for s in Scal[r]]
    nj=len(members)
    if nj==0: return 30.0
    kmax=alpha*(nj+1)-1.0
    if kmax<0: return 30.0
    js=sorted(members); idx=nj-int(np.floor(kmax))-1
    return js[idx] if 0<=idx<nj else -30.0

def deploy_union(Scal):
    u1=crc_raw(Scal,'1'); u2=crc_raw(Scal,'2')
    return {'10':u1,'01':u2,'11':max(u1,u2)}

def draw(n, rng):
    c=rng.multinomial(n,[pi[r] for r in regions])
    return {r:list(mu[r]+rng.standard_normal(k)) for r,k in zip(regions,c)}, c

def simulate(n, reps=None, naive_reps=80, seed=0):
    # fast estimators (smooth/oracle/union) at full reps; naive (slow SLSQP) at naive_reps
    if reps is None: reps=max(1200, min(4000, int(1.2e6/n)))
    rng=np.random.default_rng(seed)
    szL=[]; szO=[]; szN=[]; szU=[]; R1L=[]; R2L=[]
    for i in range(reps):
        Sfit,_=draw(n, rng); Scal,_=draw(n, rng)        # 50/50 fit/cal
        bL=fit_smooth(Sfit)
        depL=deploy_pw(Scal, bL);   szL.append(size(depL))
        r1,r2=grisk(depL); R1L.append(r1); R2L.append(r2)
        szO.append(size(deploy_pw(Scal, b_oracle)))
        # union benchmark gets the SAME total budget (pooled 2n), needing no fit fold
        Spool={r:Sfit[r]+Scal[r] for r in regions}
        szU.append(size(deploy_union(Spool)))
        if i<naive_reps:
            bN=fit_naive(Sfit); szN.append(size(deploy_pw(Scal, bN)))
    return (np.mean(szL),np.median(szL),np.mean(szU),np.mean(szO),np.mean(szN),
            max(np.mean(R1L),np.mean(R2L)))

if __name__=="__main__":
    print("=== population ===")
    print(f"lambda1={lam1:.3f}  lambda2={lam2:.3f}")
    print(f"t*=({t_star['10']:.2f},{t_star['01']:.2f},{t_star['11']:.2f})  "
          f"R1,R2 at t*=({grisk(t_star)[0]:.3f},{grisk(t_star)[1]:.3f})")
    print(f"V(C*)={Vstar:.3f}  V(union)={Vunion:.3f}  Delta={Vunion-Vstar:.3f}")
    print(f"1-D reduction: u*={u_star:.6f}  V(u*)={_V1d(u_star):.6f}  "
          f"min V''={Vpp_min:.3f} (>0 => strictly convex => unique global)")

    print("\n=== finite sample (learned smooth plug-in + pointwise CRC) ===")
    print(f"oracle shape deploys at ~V(C*).  union benchmark ~V(union)={Vunion:.3f}")
    print("\n   n   smooth+CRC  median   gain_vs_union   oracle   naive    worst E[Rj]")
    for n in [1000, 3000, 10000]:
        mL,medL,mU,mO,mN,worst=simulate(n)
        print(f"{n:6d}   {mL:7.3f}  {medL:7.3f}   {mU-mL:+7.3f}      {mO:7.3f} {mN:7.3f}   {worst:.3f}")

    # naive instability at large n
    mL,medL,mU,mO,mN,worst=simulate(40000, reps=200, naive_reps=150, seed=1)
    print(f"\nnaive at n=40000: deploys ~{mN:.2f} (vs union {mU:.2f}); smooth plug-in ~{mL:.2f}")
