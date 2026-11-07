#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_csv("struct_raw.csv"); SPLIT=0.5
Ns=sorted(df.n.unique())
order=["slice_union","direct_unshaped","direct_learned","direct_oracle","atomwise","kandinsky","fitted_union"]
lab={"slice_union":"Slice union","direct_unshaped":"Direct, unshaped","direct_learned":"Direct, learned shape",
 "direct_oracle":"Direct, oracle shape","atomwise":"Atom-wise","kandinsky":"Kandinsky","fitted_union":"Fitted-hinge union"}
col={"slice_union":"#1f77b4","direct_unshaped":"#ff7f0e","direct_learned":"#d62728",
 "direct_oracle":"#2ca02c","atomwise":"#9467bd","kandinsky":"#8c564b","fitted_union":"#17becf"}
mk={"slice_union":"o","direct_unshaped":"v","direct_learned":"s","direct_oracle":"^",
 "atomwise":"D","kandinsky":"X","fitted_union":"P"}
show=["slice_union","direct_unshaped","direct_learned","direct_oracle","atomwise","kandinsky"]

fig,ax=plt.subplots(1,3,figsize=(14,4.1))
# P1: set size vs n (pooled)
po=df[df.group==0]
for m in show:
    s=po[(po.split==SPLIT)&(po.method==m)].groupby("n")["size"].mean().reindex(Ns)
    ax[0].plot(Ns,s.values,marker=mk[m],color=col[m],label=lab[m])
ax[0].axhline(2.750,ls="--",c="#1f77b4",lw=1,alpha=.6)
ax[0].axhline(2.675,ls="--",c="#2ca02c",lw=1,alpha=.6)
ax[0].text(Ns[0],2.752,"$S_{\\rm slice}^\\star$",fontsize=8,color="#1f77b4")
ax[0].text(Ns[0],2.648,"$S_{\\rm direct}$",fontsize=8,color="#2ca02c")
ax[0].set_xscale("log"); ax[0].set_title("Mean prediction-set size"); ax[0].set_xlabel("total $n$")
ax[0].set_ylabel("mean set size"); ax[0].set_xticks(Ns); ax[0].set_xticklabels(Ns)
# P2: worst-group deployed risk vs n
g12=df[df.group.isin([1,2])]
for m in show:
    s=g12[(g12.split==SPLIT)&(g12.method==m)].groupby(["n","group"])["risk"].mean().groupby("n").max().reindex(Ns)
    ax[1].plot(Ns,s.values,marker=mk[m],color=col[m],label=lab[m])
ax[1].axhline(0.10,ls="--",c="k",lw=1)
ax[1].set_xscale("log"); ax[1].set_title("Worst-group deployed risk"); ax[1].set_xlabel("total $n$")
ax[1].set_ylabel("max over groups"); ax[1].set_xticks(Ns); ax[1].set_xticklabels(Ns)
# P3: fallback frequency (direct methods)
for m in ["slice_union","direct_unshaped","direct_learned","direct_oracle"]:
    s=po[(po.split==SPLIT)&(po.method==m)].groupby("n")["fallback"].mean().reindex(Ns)
    ax[2].plot(Ns,s.values,marker=mk[m],color=col[m],label=lab[m])
ax[2].set_xscale("log"); ax[2].set_title("Maximal-threshold fallback frequency"); ax[2].set_xlabel("total $n$")
ax[2].set_ylabel("P(procedure uses max-threshold fallback)"); ax[2].set_xticks(Ns); ax[2].set_xticklabels(Ns)
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=6,bbox_to_anchor=(0.5,1.07),frameon=False,fontsize=9)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig_direct_struct.pdf",bbox_inches="tight"); fig.savefig("fig_direct_struct.png",dpi=150,bbox_inches="tight")
print("wrote fig_direct_struct.pdf/png; Ns=",Ns)
