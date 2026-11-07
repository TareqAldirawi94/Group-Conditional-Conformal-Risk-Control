#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_csv("ddr_raw.csv"); GL=["{0}","{1,2,3}","{2,3,4}","{3,4}"]; ALPHA=0.10
col={"slice_union":"#1f77b4","direct_unshaped":"#ff7f0e","direct_learned":"#d62728","slice_union_cor":"#17becf"}
def se(x): x=np.asarray(x,float); return x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.0
# Full-budget baselines (slice_union, direct_unshaped) are split-independent by
# design; to avoid mixing the three split-seeded redraw streams, panels 1-2 use the
# 50/50 stream only. Panel 3 shows split sensitivity. Error bars are +/-1 Monte
# Carlo SE over redraws.
d05=df[df.split==0.5]
fig,ax=plt.subplots(1,3,figsize=(14,4.1))
# P1: per-group deployed risk, union vs direct_unshaped (full cal, 50/50 stream)
x=np.arange(4); w=0.35
su=[d05[(d05.method=="slice_union")&(d05.group==i)]["risk"] for i in range(4)]
du=[d05[(d05.method=="direct_unshaped")&(d05.group==i)]["risk"] for i in range(4)]
ax[0].bar(x-w/2,[s.mean() for s in su],w,yerr=[se(s) for s in su],capsize=2,color=col["slice_union"],label="Slice union")
ax[0].bar(x+w/2,[s.mean() for s in du],w,yerr=[se(s) for s in du],capsize=2,color=col["direct_unshaped"],label="Direct, unshaped")
ax[0].axhline(ALPHA,ls="--",c="k",lw=1,label=r"$\alpha=0.10$")
ax[0].set_xticks(x); ax[0].set_xticklabels(GL); ax[0].set_ylabel("deployed risk")
ax[0].set_title("Group deployed risk (full cal, 50/50 stream)"); ax[0].legend(fontsize=8)
# P2: overall set size, union vs direct_unshaped (50/50 stream) + direct_learned by split
labels=["Slice\nunion","Direct\nunshaped","Direct\nlearned\n25/75","Direct\nlearned\n50/50","Direct\nlearned\n75/25"]
base=lambda m: d05[d05.method==m].drop_duplicates(['redraw'])["overall_size"]
vals=[base("slice_union").mean(), base("direct_unshaped").mean()]; ses=[se(base("slice_union")),se(base("direct_unshaped"))]
for s in [0.25,0.5,0.75]:
    col_s=df[(df.method=="direct_learned")&(df.split==s)].drop_duplicates(['redraw'])["overall_size"]
    vals.append(col_s.mean()); ses.append(se(col_s))
colors=[col["slice_union"],col["direct_unshaped"]]+[col["direct_learned"]]*3
ax[1].bar(range(5),vals,yerr=ses,capsize=2,color=colors)
ax[1].axhline(vals[0],ls=":",c=col["slice_union"],lw=1)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(labels,fontsize=8); ax[1].set_ylabel("mean set size")
ax[1].set_title("Overall set size (lower is better)")
# P3: 75/25 fallback contrast -- union localizes vs direct global
sp=[0.25,0.5,0.75]
uc=[df[(df.method=="slice_union_cor")&(df.split==s)].drop_duplicates(['redraw'])["overall_size"].mean() for s in sp]
dl=[df[(df.method=="direct_learned")&(df.split==s)].drop_duplicates(['redraw'])["overall_size"].mean() for s in sp]
ax[2].plot(sp,uc,marker="o",color=col["slice_union_cor"],label="Slice union (corr. fold): localizes")
ax[2].plot(sp,dl,marker="s",color=col["direct_learned"],label="Direct learned: global fallback")
ax[2].set_xticks(sp); ax[2].set_xlabel("fit fraction (correction = 1 - fit)")
ax[2].set_ylabel("mean set size"); ax[2].set_title("Severe-group under-support (75/25)")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig_ddr_direct.pdf",bbox_inches="tight"); fig.savefig("fig_ddr_direct.png",dpi=150,bbox_inches="tight")
print("wrote fig_ddr_direct.pdf/png")
