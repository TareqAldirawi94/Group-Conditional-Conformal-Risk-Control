#!/usr/bin/env python3
"""Generate publication figures from a provenance-gated DDR CRC summary.

USAGE: python ddr_make_figures_integrity.py <.../REAL_DATA_DEDUP_RETRAINED_REVIEWED/u_redrawn_each_split>
The output is a visualization of the persisted CSV, not a new CRC analysis.
"""
import csv, hashlib, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv)!=2:
    raise SystemExit('Usage: python ddr_make_figures_integrity.py <integrity-eval-output-dir>')
root=pathlib.Path(sys.argv[1]).resolve()
if root.name!='u_redrawn_each_split' or root.parent.name not in ('REAL_DATA_DEDUP_RETRAINED_REVIEWED','REAL_DATA_DEDUP_RETRAINED_CONSERVATIVE'):
    raise SystemExit('Publication figures require REAL_DATA_DEDUP_RETRAINED_REVIEWED/u_redrawn_each_split.')
prov=json.loads((root/'ddr_integrity_provenance.json').read_text(encoding='utf8'))
if prov.get('integrity_tier')!='DEDUP_RETRAINED' or prov.get('near_duplicate_review_status') not in ('PHASH_CANDIDATES_REVIEWED','CONSERVATIVE_PHASH_CANDIDATES_EXCLUDED'):
    raise SystemExit('Duplicate-aware retraining and pHash review or conservative exclusion required for manuscript figures.')
path=root/'ddr_analysis_scores.csv'
if hashlib.sha256(path.read_bytes()).hexdigest()!=prov.get('analysis_scores_sha256'):
    raise SystemExit('Analysis-score hash mismatch.')
checks=(root/'ddr_crc_checks.txt').read_text(encoding='utf8')
if 'source=DEDUP_RETRAINED' not in checks or 'U protocol=redraw' not in checks:
    raise SystemExit('Missing definitive U-redraw evidence markers.')
with (root/'ddr_crc_summary.csv').open(newline='',encoding='utf8') as f:
    data=list(csv.DictReader(f))
if not data or set(z['source'] for z in data)!={'DEDUP_RETRAINED'} or set(z['u_protocol'] for z in data)!={'redraw'}:
    raise SystemExit('Summary source does not match integrity tier.')
d={(r['arm'],r['group']):r for r in data}

def get(arm,g,metric):
    z=d.get((arm,g));
    if not z: raise ValueError(f'Missing {arm}/{g} for figure')
    return float(z[metric])

def one_figure(groups,labels,arms,title,outname):
    x=np.arange(len(groups)); w=.23 if len(arms)==3 else .32
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.6),constrained_layout=True)
    for i,(arm,label) in enumerate(arms):
        pos=x+(i-(len(arms)-1)/2)*w
        for a,field,err,ylabel in ((axs[0],'risk','risk_se','Conditional ordinal loss'),
                                   (axs[1],'size','size_se','Mean prediction-set size')):
            vals=[get(arm,g,field) for g in groups]
            ses=[get(arm,g,err) for g in groups]
            a.bar(pos,vals,width=w*.96,label=label)
            a.errorbar(pos,vals,yerr=np.asarray(ses)*2,fmt='none',ecolor='#333333',capsize=2.4,linewidth=.8)
            a.set_ylabel(ylabel)
    axs[0].axhline(.10,color='#333333',ls='--',linewidth=1)
    axs[0].text(.99,.98,r'$\alpha=0.10$',ha='right',va='top',transform=axs[0].transAxes,fontsize=9)
    axs[0].set_ylim(bottom=0);axs[1].set_ylim(bottom=0)
    for a in axs:
        a.set_xticks(x,labels)
        a.tick_params(axis='x',labelsize=8)
        a.spines[['top','right']].set_visible(False)
    h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,1.055),ncol=len(arms),frameon=False)
    fig.suptitle(title,fontsize=12,y=1.11)
    for ext in ('.pdf','.png'):
        fig.savefig(root/(outname+ext),dpi=250,bbox_inches='tight')
    plt.close(fig)

one_figure(['{0}','{1,2,3}','{2,3,4}','{3,4}','{3}'],
           ['No DR\n{0}','NPDR\n{1,2,3}','Referable\n{2,3,4}','Severe/PDR\n{3,4}','Grade 3\n{3}'],
           [('marginal','Marginal CRC'),('exact_grade','Exact-grade refinement'),('overlap_union','Overlapping union')],
           'DDR: overlapping-group risk and set size','fig_ddr_crc')
one_figure(['{0}','{1,2}','{3,4}'],
           ['No DR\n{0}','Mild/mod.\n{1,2}','Severe/PDR\n{3,4}'],
           [('marginal','Marginal CRC'),('disjoint_sev','Disjoint Mondrian CRC')],
           'DDR: disjoint severity-conditional risk','fig_ddr_disjoint')
print(f'Wrote {root / "fig_ddr_crc.pdf"}, fig_ddr_disjoint.pdf and 250-dpi PNGs')
