#!/usr/bin/env python3
import pandas as pd, numpy as np
s=pd.read_csv("fitshape_summary.csv"); p=pd.read_csv("fitshape_pooled.csv")
GN={0:"{0}",1:"{1,2,3->y1}",2:"y2",3:"y3(rare)"}

print("="*70)
print("ALIGNMENT: slice-to-deployed gap by method (rho=0, split=0.5)")
sub=s[(s.rho==0)&(s.split==0.5)]
piv=sub.pivot_table(index="N",columns="method",values="gap")
print(piv.round(4).to_string())

print("\nSET SIZE (pooled) and POOLED RISK by method (rho=0, split=0.5):")
pp=p[(p.rho==0)&(p.split==0.5)].pivot_table(index="N",columns="method",values=["pooled_risk","pooled_size"])
print(pp.round(3).to_string())

print("\nRARE GROUP (group 3) deployed risk & size by method, rho=0, split=0.5:")
g3=s[(s.rho==0)&(s.split==0.5)&(s.group==3)]
print(g3.pivot_table(index="N",columns="method",values=["dep","size"]).round(3).to_string())

print("\nSPLIT SENSITIVITY (fitted_union vs union_full), N=1600 rho=0, pooled:")
ss=p[(p.N==1600)&(p.rho==0)&(p.method.isin(["fitted_union","union_full","union_split"]))]
print(ss.pivot_table(index="split",columns="method",values=["pooled_risk","pooled_size"]).round(3).to_string())

print("\nTHRESHOLD DISPERSION (alignment proxy), rho=0 split=0.5:")
d=s[(s.rho==0)&(s.split==0.5)&(s.method.isin(["union_full","fitted_union"]))&(s.group==0)]
print(d.pivot_table(index="N",columns="method",values="disp").round(4).to_string())

# headline deltas for the text
print("\n--- HEADLINE (rho=0, split=0.5) ---")
for N in [100,400,1600,6400]:
    uf=p[(p.N==N)&(p.rho==0)&(p.split==0.5)&(p.method=="union_full")]
    fu=p[(p.N==N)&(p.rho==0)&(p.split==0.5)&(p.method=="fitted_union")]
    ka=p[(p.N==N)&(p.rho==0)&(p.split==0.5)&(p.method=="kandinsky")]
    if len(uf) and len(fu):
        print(f"N={N}: union(size={uf.pooled_size.iloc[0]:.2f},risk={uf.pooled_risk.iloc[0]:.3f})  "
              f"fitted(size={fu.pooled_size.iloc[0]:.2f},risk={fu.pooled_risk.iloc[0]:.3f})  "
              f"kand(size={ka.pooled_size.iloc[0]:.2f},risk={ka.pooled_risk.iloc[0]:.3f})")
