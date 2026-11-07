"""
Population joint-optimum diagnostic in the TWO-GROUP, THREE-REGION model
(the advisor's exact program: theta=(t10,t01,t11)).

Regions well-populated, so plug-in risks are stable (no rare-atom noise). NOTE: the
"oracle" here is a NUMERICAL grid oracle -- a finite 400-point search over the overlap
cutoff t11 with the remaining thresholds at the risk boundary -- not a certified
global optimum; reported sizes are grid-accurate.
Candidates v in {00,10,01,11}; focal groups G1={Y1=1}={10,11}, G2={Y2=1}={01,11}.
Inclusion-type miscoverage, B=1, loss 1{Y not covered}.
Prediction set C_t(x)={v: s(x,v) <= t_{region(v)}}; size compared over the three
focal candidates {10,01,11} (the empty-label candidate 00 is handled identically
by all methods and excluded from the objective).

Three region-specific threshold rules under one distribution / loss / size objective,
constructed on one large sample and evaluated on a separate one:
  union  : t10=lam1, t01=lam2, t11=max(lam1,lam2)   (slice-alpha quantiles)
  shape  : additive b(v)=beta2*v2 aligning the two group crossings, then union of shaped slices
  oracle : min_t (F10+F01+F11) s.t. R1,R2<=alpha  (1-D search over t11, t10/t01 at the risk boundary)

Reports fraction of available improvement recovered = (S_union-S_shape)/(S_union-S_oracle).
"""
import numpy as np
from scipy.stats import norm

ALPHA=0.10
R_CORR=0.3
BASE=np.array([0.50,0.30])            # P(Y1=1), P(Y2=1)
TAU=norm.ppf(1-BASE)
GAMMA=np.array([3.0,0.4])             # label 1 easy, label 2 hard
BIAS=np.array([0.0,2.0])             # extra penalty on hard label 2 positives -> lam2 >> lam1
NOISE=1.0
VEC=np.array([[0,0],[1,0],[0,1],[1,1]],float)   # 00,10,01,11
FOCAL=[1,2,3]                          # indices of 10,01,11

def gen(n,rng):
    f=rng.standard_normal((n,1)); e=rng.standard_normal((n,2))
    z=np.sqrt(R_CORR)*f+np.sqrt(1-R_CORR)*e
    Y=(z>TAU[None,:]).astype(float)
    eta=GAMMA[None,:]*(z-TAU[None,:])-BIAS[None,:]+NOISE*rng.standard_normal((n,2))
    q=np.clip(1/(1+np.exp(-eta)),1e-9,1-1e-9)
    c1=-np.log(q); c0=-np.log(1-q)
    S=(Y*c1+(1-Y)*c0).sum(1)
    base=c0.sum(1); delta=c1-c0
    reg=(Y[:,0]*1+Y[:,1]*2).astype(int)    # 0:00 1:10 2:01 3:11
    reg=np.where((Y[:,0]==1)&(Y[:,1]==1),3,np.where(Y[:,0]==1,1,np.where(Y[:,1]==1,2,0)))
    return Y,S,base,delta,reg

def cand_scores(base,delta):           # n x 4 (00,10,01,11)
    return base[:,None]+delta@VEC.T

rng=np.random.default_rng(3)
Yc,Sc,basec,deltac,regc=gen(600000,rng)
# region masses among groups
m10=(regc==1).mean(); m01=(regc==2).mean(); m11=(regc==3).mean()
w1=m10+m11; w2=m01+m11
# group slice thresholds (alpha-upper-quantile of S on the group)
lam1=np.quantile(Sc[(Yc[:,0]==1)],1-ALPHA)
lam2=np.quantile(Sc[(Yc[:,1]==1)],1-ALPHA)
print(f"region mass 10={m10:.3f} 01={m01:.3f} 11={m11:.3f};  lam1={lam1:.3f} lam2={lam2:.3f}  (disagreement {abs(lam1-lam2):.3f})")

# survivals rho_r(t)=P(S>t | region r) and candidate CDFs F_v(t)
Scm=cand_scores(basec[:200000],deltac[:200000])     # candidate scores for size
def Fv(vidx,t):
    col=np.sort(Scm[:,vidx]); return np.searchsorted(col,t,side='right')/len(col)
S_by_reg={r:np.sort(Sc[regc==r]) for r in [1,2,3]}
def rho(r,t):
    s=S_by_reg[r]; return 1.0-np.searchsorted(s,t,side='right')/len(s)
def rho_inv(r,target):          # smallest t with rho(r,t) <= target (boundary)
    s=S_by_reg[r]
    if target<=0: return s[-1]+1.0
    if target>=1: return s[0]-1.0
    k=int(np.ceil((1-target)*len(s)))-1; k=min(max(k,0),len(s)-1)
    return s[k]

def size_focal(t10,t01,t11):
    return Fv(1,t10)+Fv(2,t01)+Fv(3,t11)
def risks(t10,t01,t11):
    R1=(m10*rho(1,t10)+m11*rho(3,t11))/w1
    R2=(m01*rho(2,t01)+m11*rho(3,t11))/w2
    return R1,R2

# ---- union ----
u10,u01,u11=lam1,lam2,max(lam1,lam2)
# ---- shape: align crossings; b(v)=beta2*v2, choose beta2 so shaped group quantiles match ----
# shaped score U=S-beta2*Y2. group1 slice quantile q1(beta2), group2 q2(beta2); solve q1=q2.
from scipy.optimize import brentq
def shaped_q(beta2,gcol):
    U=Sc-beta2*Yc[:,1]
    return np.quantile(U[(Yc[:,gcol]==1)],1-ALPHA)
beta2=brentq(lambda b: shaped_q(b,0)-shaped_q(b,1), -10,10)
tS1=shaped_q(beta2,0)   # == tS2
# shaped deployed cutoffs per region: s <= t_group_shaped + b(v)
bvec=VEC[:,1]*beta2
s10=tS1+bvec[1]; s01=tS1+bvec[2]; s11=max(tS1,tS1)+bvec[3]
# ---- oracle: 1-D search over t11 ----
t11_grid=np.linspace(np.quantile(Sc,0.50),np.quantile(Sc,0.999),400)
best=None
for t11 in t11_grid:
    # boundary t10: rho(1,t10) = ((w1*alpha)-m11*rho(3,t11))/m10
    rhs1=(w1*ALPHA-m11*rho(3,t11))/m10
    rhs2=(w2*ALPHA-m11*rho(3,t11))/m01
    if rhs1<0 or rhs2<0:      # overlap alone already violates -> infeasible t11 too small
        continue
    t10=rho_inv(1,rhs1); t01=rho_inv(2,rhs2)
    sz=size_focal(t10,t01,t11)
    if best is None or sz<best[0]:
        best=(sz,t10,t01,t11)
_,o10,o01,o11=best

# ---- evaluate all three on a SEPARATE sample ----
Ye,Se,basee,deltae,rege=gen(300000,np.random.default_rng(101))
me10=(rege==1).mean(); me01=(rege==2).mean(); me11=(rege==3).mean()
we1=me10+me11; we2=me01+me11
Sce=cand_scores(basee,deltae)
def eval_rule(t10,t01,t11):
    size=((Sce[:,1]<=t10).mean()+(Sce[:,2]<=t01).mean()+(Sce[:,3]<=t11).mean())
    cut=np.where(rege==1,t10,np.where(rege==2,t01,np.where(rege==3,t11,np.inf)))
    R1=(Se[(Ye[:,0]==1)]>cut[(Ye[:,0]==1)]).mean()
    R2=(Se[(Ye[:,1]==1)]>cut[(Ye[:,1]==1)]).mean()
    return size,R1,R2

Su,Ru1,Ru2=eval_rule(u10,u01,u11)
Ss,Rs1,Rs2=eval_rule(s10,s01,s11)
So,Ro1,Ro2=eval_rule(o10,o01,o11)
print(f"\n{'rule':8}{'size(3 focal)':>14}{'R1':>8}{'R2':>8}  valid")
print(f"{'union':8}{Su:14.3f}{Ru1:8.3f}{Ru2:8.3f}  {Ru1<=ALPHA+1e-3 and Ru2<=ALPHA+1e-3}")
print(f"{'shape':8}{Ss:14.3f}{Rs1:8.3f}{Rs2:8.3f}  {Rs1<=ALPHA+1e-3 and Rs2<=ALPHA+1e-3}")
print(f"{'oracle':8}{So:14.3f}{Ro1:8.3f}{Ro2:8.3f}  {Ro1<=ALPHA+1e-3 and Ro2<=ALPHA+1e-3}")
if Su>So:
    frac=(Su-Ss)/(Su-So)
    print(f"\navailable room  S_union-S_oracle = {Su-So:.3f}")
    print(f"shape gain      S_union-S_shape  = {Su-Ss:.3f}")
    print(f"fraction of available improvement recovered by learned shape = {frac:.2f}")
print(f"beta2 (shape) = {beta2:.3f}")
