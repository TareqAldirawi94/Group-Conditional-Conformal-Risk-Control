# Historical DDR direct experiment — NOT definitive real-data evidence

`ddr_raw.csv` was produced from the historical 2,428-image score table.
Its historical provenance is insufficient to certify leakage-controlled retraining.
The image-level score table (`ddr_analysis_scores.csv`) is deliberately omitted
from this public-ready package pending data-sharing rights and provenance checks.

Regenerate the supplied historical figure from the aggregate output:

```bash
python ddr_direct_make_fig.py
```

Run the method with a legally distributable input score file:

```bash
DDR_POOL_CSV=/absolute/path/to/ddr_analysis_scores.csv python ddr_direct_sim.py full 200
```

This direct evaluator itself does not verify training provenance. For definitive
results, regenerate scores with the controlled training pipeline, pass the
independent integrity evaluation, and rerun direct CRC on that verified pool.
Do not silently reuse the historical numbers in a final manuscript.
