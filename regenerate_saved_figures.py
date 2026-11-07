#!/usr/bin/env python3
"""Regenerate four saved-result figures, without running simulations or GPU tasks."""
from pathlib import Path
import subprocess, sys
root=Path(__file__).resolve().parent
jobs=[
 ('06_fitted_hinge','fitted_hinge_make_fig.py','fitshape_raw.csv'),
 ('07_three_label_direct','three_label_direct_make_fig.py','struct_raw.csv'),
 ('08_hamming_direct','hamming_direct_make_fig.py','ham_raw.csv'),
 ('09_ddr/direct','ddr_direct_make_fig.py','ddr_raw.csv'),
]
for dirname,script,csv in jobs:
 folder=root/dirname
 if not (folder/csv).exists(): raise FileNotFoundError(folder/csv)
 print('Regenerating',dirname,flush=True)
 subprocess.run([sys.executable,script],cwd=folder,check=True)
print('Completed all four saved-output figures. DDR figure is historical.')
