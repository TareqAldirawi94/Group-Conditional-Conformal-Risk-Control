#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_csv("ham_raw.csv"); SPLIT=0.5; ALPHA=0.5
Ns=sorted(df.n.unique())
show=["slice_union","direct_unshaped","direct_learned","atomwise"]
lab={"slice_union":"Slice union","direct_unshaped":"Direct, unshaped",
 "direct_learned":"Direct, learned shape","atomwise":"Atom-wise"}
col={"slice_union":"#1f77b4","direct_unshaped":"#ff7f0e","direct_learned":"#d62728","atomwise":"#9467bd"}
mk={"slice_union":"o","direct_unshaped":"v","direct_learned":"s","atomwise":"D"}

def series(mode,method,field,agg="mean"):
    s=df[(df["mode"]==mode)&(df.split==SPLIT)&(df.method==method)].groupby("n")
    return s[field].mean().reindex(Ns)

fig,ax=plt.subplots(1,3,figsize=(14,4.1))
# P1: label-defined set size
for m in show:
    ax[0].plot(Ns,series("B",m,"pooled_size").values,marker=mk[m],color=col[m],label=lab[m])
ax[0].set_xscale("log"); ax[0].set_title("Label-defined groups: mean set size")
ax[0].set_xlabel("total $n$"); ax[0].set_ylabel("mean set size")
ax[0].set_xticks(Ns); ax[0].set_xticklabels(Ns)
# P2: label-defined worst-group deployed risk = max_j ( mean_rep risk_j ),
#     i.e. the max over groups of the group-mean risk (NOT the mean of per-rep
#     maxima, which is upward-biased by selection).
for m in show:
    sub=df[(df["mode"]=="B")&(df.split==SPLIT)&(df.method==m)]
    vals=[sub[sub.n==n][["risk1","risk2","risk3"]].mean().max() for n in Ns]
    ax[1].plot(Ns,vals,marker=mk[m],color=col[m],label=lab[m])
ax[1].axhline(ALPHA,ls="--",c="k",lw=1)
ax[1].set_xscale("log"); ax[1].set_title("Label-defined groups: worst-group risk")
ax[1].set_xlabel("total $n$"); ax[1].set_ylabel(r"$\max_j$ mean risk")
ax[1].set_xticks(Ns); ax[1].set_xticklabels(Ns)
# P3: feature-defined (zero-gap benchmark) set size
for m in ["slice_union","direct_unshaped","direct_learned"]:
    ax[2].plot(Ns,series("A",m,"pooled_size").values,marker=mk[m],color=col[m],label=lab[m])
ax[2].set_xscale("log"); ax[2].set_title("Feature-defined groups: mean set size")
ax[2].set_xlabel("total $n$"); ax[2].set_ylabel("mean set size")
ax[2].set_xticks(Ns); ax[2].set_xticklabels(Ns)
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=4,bbox_to_anchor=(0.5,1.07),frameon=False,fontsize=9)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig_hamming_direct.pdf",bbox_inches="tight"); fig.savefig("fig_hamming_direct.png",dpi=150,bbox_inches="tight")
print("wrote fig_hamming_direct.pdf/png; Ns=",Ns)
