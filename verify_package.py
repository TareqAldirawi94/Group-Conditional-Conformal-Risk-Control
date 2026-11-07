#!/usr/bin/env python3
"""Fast, source-only and archived-result checks; not full simulation replication."""
from pathlib import Path
import ast, csv, hashlib
import pandas as pd
root=Path(__file__).resolve().parent
with (root/'MANIFEST.csv').open(newline='') as f: manifest=list(csv.DictReader(f))
for row in manifest:
 p=root/row['repo_path']
 assert p.is_file(), str(p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['release_sha256'],str(p)
 if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
print(f'SHA-256 + syntax passed: {len(manifest)} archived files')

def means(filepath, filters, col):
 df=pd.read_csv(root/filepath)
 for k,v in filters.items():df=df[df[k]==v]
 return df[col].mean()
# These numbers are rounded summaries of the author-supplied archived results,
# not exact-equality mathematical guarantees.
f='07_three_label_direct/struct_raw.csv'
for meth,target in [('slice_union',2.870),('direct_learned',2.740),('direct_unshaped',2.760)]:
 z=means(f,dict(n=6400,split=0.5,group=0,method=meth),'size')
 assert abs(z-target)<0.015,(meth,z)
f='08_hamming_direct/ham_raw.csv'
for mode,meth,target in [('B','slice_union',1.579),('B','direct_learned',1.157),('A','slice_union',1.321),('A','direct_unshaped',1.325),('A','direct_learned',1.688)]:
 z=means(f,dict(n=25600,split=0.5,mode=mode,method=meth),'pooled_size')
 assert abs(z-target)<0.02,(mode,meth,z)
f='09_ddr/direct/ddr_raw.csv'
for meth,target in [('slice_union',2.322),('direct_unshaped',2.518)]:
 z=means(f,dict(split=0.5,method=meth,group=0),'overall_size')
 assert abs(z-target)<0.06,(meth,z)
print('Saved-CSV summary spot checks passed. DDR data are HISTORICAL, not definitive.')
