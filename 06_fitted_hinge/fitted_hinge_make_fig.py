#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df=pd.read_csv("fitshape_raw.csv")
RHO,SPLIT=0.0,0.5
po=df.drop_duplicates(["N","rho","split","method","rep"])
meths=["union_full","fitted_union","kandinsky","atomwise"]
lab={"union_full":"Unshaped union","fitted_union":"Fitted-hinge union",
     "kandinsky":"Kandinsky","atomwise":"Atom-wise Mondrian","union_split":"Union (split control)"}
col={"union_full":"#1f77b4","fitted_union":"#ff7f0e","kandinsky":"#2ca02c",
     "atomwise":"#9467bd","union_split":"#7f7f7f"}
mk ={"union_full":"o","fitted_union":"s","kandinsky":"^","atomwise":"D","union_split":"x"}
Ns=sorted(df.N.unique())

fig,ax=plt.subplots(1,3,figsize=(13.5,4.0))

# Panel 1: rare-group (y3) deployed risk vs n  -> validity story
g3=df[(df.group==3)&(df.rho==RHO)&(df.split==SPLIT)]
for m in meths:
    s=g3[g3.method==m].groupby("N").dep.mean().reindex(Ns)
    ax[0].plot(Ns,s.values,marker=mk[m],color=col[m],label=lab[m])
ax[0].axhline(0.10,ls="--",c="k",lw=1); ax[0].set_xscale("log")
ax[0].set_title("Rare group $\\{y_3=1\\}$: deployed risk"); ax[0].set_xlabel("total $n$")
ax[0].set_ylabel("conditional miscoverage"); ax[0].set_xticks(Ns); ax[0].set_xticklabels(Ns)

# Panel 2: pooled set size vs n -> efficiency story
for m in meths:
    s=po[(po.rho==RHO)&(po.split==SPLIT)&(po.method==m)].groupby("N").pooled_size.mean().reindex(Ns)
    ax[1].plot(Ns,s.values,marker=mk[m],color=col[m],label=lab[m])
ax[1].set_xscale("log"); ax[1].set_title("Mean prediction-set size")
ax[1].set_xlabel("total $n$"); ax[1].set_ylabel("mean set size")
ax[1].set_xticks(Ns); ax[1].set_xticklabels(Ns)

# Panel 3: reconciliation gap vs n (union vs fitted) -> alignment story
for m in ["union_full","fitted_union"]:
    s=df[(df.rho==RHO)&(df.split==SPLIT)&(df.method==m)].groupby("N").gap.mean().reindex(Ns)
    ax[2].plot(Ns,s.values,marker=mk[m],color=col[m],label=lab[m])
ax[2].axhline(0,ls=":",c="k",lw=0.8); ax[2].set_xscale("log")
ax[2].set_title("Slice-to-deployed gap $\\widehat R^{\\rm slice}-\\widehat R^{\\rm dep}$")
ax[2].set_xlabel("total $n$"); ax[2].set_ylabel("mean gap (averaged over groups)")
ax[2].set_xticks(Ns); ax[2].set_xticklabels(Ns)

h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=4,bbox_to_anchor=(0.5,1.06),frameon=False)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig_fitted_shape.pdf",bbox_inches="tight",dpi=200)
fig.savefig("fig_fitted_shape.png",bbox_inches="tight",dpi=150)
print("wrote fig_fitted_shape.pdf/png ; Ns=",Ns)
