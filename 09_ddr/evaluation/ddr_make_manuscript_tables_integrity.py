#!/usr/bin/env python3
"""Generate Section 5.3 TeX tables ONLY from duplicate-aware, retrained and reviewed DDR U-redraw outputs.

  DDR_CRC_OUT=<directory> DDR_U_MODE=redraw python ddr_crc_eval_integrity.py
  python ddr_make_manuscript_tables_integrity.py <.../REAL_DATA_DEDUP_RETRAINED_REVIEWED/u_redrawn_each_split>

The generator refuses the synthetic-mode outputs. It writes TeX tables, not
manuscript narrative, to avoid freezing unsupported headline numbers.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

if len(sys.argv) != 2:
    raise SystemExit('Usage: python ddr_make_manuscript_tables.py /path/to/REAL_DATA/u_redrawn_each_split')
root = Path(sys.argv[1]).resolve()
if root.name != 'u_redrawn_each_split' or root.parent.name not in ('REAL_DATA_DEDUP_RETRAINED_REVIEWED', 'REAL_DATA_DEDUP_RETRAINED_CONSERVATIVE'):
    raise ValueError('Publication table input requires retrained, exact-deduplicated and conservatively screened or manually reviewed U-redraw data')
import json, hashlib
provenance=json.loads((root/'ddr_integrity_provenance.json').read_text())
if provenance.get('integrity_tier')!='DEDUP_RETRAINED' or provenance.get('near_duplicate_review_status') not in ('PHASH_CANDIDATES_REVIEWED','CONSERVATIVE_PHASH_CANDIDATES_EXCLUDED'):
    raise ValueError('Missing duplicate-aware retraining provenance and pHash review or conservative-exclusion record')
analysis_csv=root/'ddr_analysis_scores.csv'
hash_csv=hashlib.sha256(analysis_csv.read_bytes()).hexdigest()
if hash_csv != provenance.get('analysis_scores_sha256'):
    raise ValueError('Analysis score file hash disagrees with evaluator provenance')
checks=(root/'ddr_crc_checks.txt').read_text()
if 'source=DEDUP_RETRAINED' not in checks or 'U protocol=redraw' not in checks:
    raise ValueError('Missing REAL DATA / U-redraw provenance markers')
d=pd.read_csv(root/'ddr_crc_summary.csv')
if set(d['source']) != {'DEDUP_RETRAINED'} or set(d['u_protocol']) != {'redraw'}:
    raise ValueError('Summary source/mode does not match real U-redraw analysis')
assert not d.duplicated(['arm','group']).any()

# Caption precision is computed from the currently displayed method/group rows.
def ceil_precision(x):
    return np.ceil(float(x)*1000)/1000

def max_finite(arms_groups, fields):
    vals=[]
    for arm,groups in arms_groups:
        for gg in groups:
            for field in fields:
                rows=d[(d.arm==arm)&(d.group==gg)]
                if len(rows)!=1: raise ValueError(f'Missing {arm}/{gg}')
                x=float(rows.iloc[0][field])
                if np.isfinite(x): vals.append(x)
    return ceil_precision(max(vals))

disjoint_groups=['{0}','{1,2}','{3,4}']
overlap_groups=['{0}','{1,2,3}','{2,3,4}','{3,4}','{3}']
se_dis_risk=max_finite([('marginal',disjoint_groups),('disjoint_sev',disjoint_groups)],
                       ['risk_se','slice_risk_se'])
se_dis_size=max_finite([('marginal',disjoint_groups),('disjoint_sev',disjoint_groups)], ['size_se'])
se_ov_risk=max_finite([('marginal',overlap_groups),('exact_grade',overlap_groups),('overlap_union',overlap_groups)],
                      ['risk_se','slice_risk_se'])
se_ov_size=max_finite([('marginal',overlap_groups),('exact_grade',overlap_groups),('overlap_union',overlap_groups)],
                      ['size_se'])


def val(arm, group, key):
    s=d[(d.arm==arm)&(d.group==group)]
    if len(s)!=1: raise ValueError(f'Missing {arm} group={group}')
    return float(s.iloc[0][key])

def v(arm,group,key,prec=3):
    x=val(arm,group,key)
    return f'{x:.{prec}f}' if np.isfinite(x) else '--'

for gg in ('{0}','{1,2,3}','{2,3,4}','{3,4}'):
    sr=val('overlap_union',gg,'slice_risk'); r=val('overlap_union',gg,'risk')
    delta=val('overlap_union',gg,'delta')
    if not np.isclose(sr-r,delta,atol=1e-10):
        raise ValueError(f'Reconciliation identity fails for {gg}')

lines=[r'% Automatically generated from the real DDR U-redraw results. Do not hand-edit numbers.',
       r'% Across-redraw SEs are conditional on the fixed scored pool, not clinical population SEs.',
       r'\begin{table}[t]',r'\centering\small',
       rf'\caption{{DDR disjoint severity groups at $\alpha=0.10$. Values average independent uniform splits and fresh APS score randomizations. Risk and slice-risk SEs are $\le {se_dis_risk:.3f}$ and set-size SEs $\le {se_dis_size:.3f}$, conditional on the fixed held-out pool.}}',
       r'\label{tab:ddr-disjoint}',
       r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lrrrrrrr}',r'\toprule',
       r'Severity group & $\bar n_{\rm cal}$ & Marg. risk & Mond. risk & Slice risk & Marg. size & Mond. size & Fallback\\',
       r'\midrule']
for gg,lab in (('{0}',r'No DR $\{0\}$'),('{1,2}',r'Mild/mod. $\{1,2\}$'),('{3,4}',r'Severe/PDR $\{3,4\}$')):
    lines.append(f'{lab} & {v("disjoint_sev",gg,"n_cal",1)} & {v("marginal",gg,"risk")} & '
                 f'{v("disjoint_sev",gg,"risk")} & {v("disjoint_sev",gg,"slice_risk")} & '
                 f'{v("marginal",gg,"size",2)} & {v("disjoint_sev",gg,"size",2)} & '
                 f'{v("disjoint_sev",gg,"fallback_freq",2)}'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}', '',
          r'\begin{table}[t]',r'\centering\small',
          rf'\caption{{DDR overlapping groups under the same randomizations. $\widehat\Delta_j=\widehat R_j^{{\rm slice}}-\widehat R_j^{{\rm dep}}$ is descriptive, not necessarily the aligned population residual gap. The finest disjoint refinement is the five-grade partition. Risk and slice-risk SEs are $\le {se_ov_risk:.3f}$ and set-size SEs $\le {se_ov_size:.3f}$; gaps are computed before rounding.}}',
          r'\label{tab:ddr-overlap}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lrrrrrrr}',r'\toprule',
          r'Clinical group & $\bar n_{\rm cal}$ & Marg. risk & Grade risk & Union risk & Slice risk & $\widehat\Delta$ & Union size\\',
          r'\midrule']
for gg,lab in (('{0}',r'No DR $\{0\}$'),('{1,2,3}',r'NPDR $\{1,2,3\}$'),('{2,3,4}',r'Referable $\{2,3,4\}$'),('{3,4}',r'Severe/PDR $\{3,4\}$'),('{3}',r'Grade $\{3\}$')):
    has_slice= gg!='{3}'
    lines.append(f'{lab} & {v("overlap_union",gg,"n_cal",1)} & {v("marginal",gg,"risk")} & '
                 f'{v("exact_grade",gg,"risk")} & {v("overlap_union",gg,"risk")} & '
                 f'{v("overlap_union",gg,"slice_risk") if has_slice else "--"} & '
                 f'{v("overlap_union",gg,"delta") if has_slice else "--"} & '
                 f'{v("overlap_union",gg,"size",2)}'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}', '']
out=root/'ddr_crc_manuscript_tables.tex';out.write_text('\n'.join(lines))
print('Wrote',out)
print('Grade-3 marginal risk:',v('marginal','{3}','risk'), 'SE:', v('marginal','{3}','risk_se'))
print('Severe-stratum marginal risk:',v('marginal','{3,4}','risk'), 'SE:',v('marginal','{3,4}','risk_se'))
print('Pooled risk is not calculated from stratum means; use pooled per-redraw records.')
