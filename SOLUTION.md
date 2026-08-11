# Approach

Detect-and-link baseline for the Biohub cell tracking competition, scored
locally against the official metric.

## Reading the metric

The metric is public ([royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition),
BSD-3), cloned to `reference/` and used directly for offline scoring so local
numbers match the leaderboard definition rather than a reimplementation.

```
score = adj_edge_jaccard + 0.1 · division_jaccard
adj_edge_jaccard = max(0, J · (1 − 0.1 · (T_pred − T_true) / T_true))
```

Three properties drive every design decision:

**Over-prediction is cheap.** In `metrics.py`, a predicted edge counts as FP
only when at least one endpoint matched a GT node that itself has a GT edge
(`pred_valid = out_valid | in_valid`, then `edge_fp = edge_valid_pred − edge_tp`).
Edges between two unmatched detections are dropped entirely. The only cost of
extra detections is the node-count penalty, and at α = 0.1 predicting 2× the
true count costs just 10%. Predicting *fewer* than `T_true` yields a factor
above 1.0 — which is why published scores can exceed 1.0.

**The ground truth is ~0.66% labelled.** Mean 253 annotated nodes per sample
against `estimated_number_of_nodes` ≈ 38,438. The penalty compares against that
estimate, not the labelled count, so the target is dense prediction: roughly
`T_true / T` ≈ 250–590 detections per frame.

**Edges are ~91% of the score.** Divisions carry a 0.1 weight and are rare
(0–1 per sample in the 30 surveyed), so linking quality dominates. Divisions are
a late refinement, not a starting point.

Two structural constraints the metric enforces: edges must span exactly
`t → t+1` (anything else is filtered out), and out-degree is capped at 2.

## Pipeline

| Stage | Module | Notes |
| --- | --- | --- |
| Detection | `src/biohub/detect.py` | Difference-of-Gaussians local maxima |
| Linking | `src/biohub/track.py` | Hungarian assignment on µm centroid distance |
| Scoring | `src/biohub/evaluate.py` | Wraps the official `tracking_cellmot.metrics` |
| Export | `src/biohub/submit.py` | Competition CSV writer |

### Detection uses DoG, not raw intensity

Absolute brightness is not comparable across samples. Sample `44b6_0b24845f`
has background median 1114 and 80th percentile 1606, while its annotated cell
centres sit at ~1408 — *below* a global 80th-percentile threshold that worked
fine elsewhere. Thresholding raw intensity gave node recall **0.176** on that
sample versus 1.000 on others.

Switching to a difference-of-Gaussians response (cell-scale detail minus a
4 µm background estimate) measures local blob contrast instead, and lifted that
sample to **0.875** while holding 1.000 on the rest.

The grid is strongly anisotropic (z = 1.625 µm, y = x = 0.40625 µm/voxel), so
smoothing widths and the peak-separation footprint are specified in µm and
converted per axis — the footprint is a flat ellipsoid, wide in x/y and only a
few voxels tall in z.

### Linking gate

Ground-truth cells move **1.9 µm per frame median, p95 3.8, p99 5.3** (measured
over 10,572 GT edges). A 6 µm gate covers essentially all real motion while
staying inside the metric's 7 µm matching threshold.

## Reproducing

```bash
pip install zarr numcodecs polars tracksdata
git clone https://github.com/royerlab/kaggle-cell-tracking-competition reference

python scripts/run_baseline.py --limit 8      # score on train
python scripts/predict.py --out submission.csv # write a submission
```

## Environment note

The local GPU is a Quadro P1000 (4 GB), too small for 3D deep learning on
`(100, 64, 256, 256)` uint16 volumes. This machine is for baselines and offline
scoring; any learned model trains on Kaggle (T4/P100, 12 h cap, internet
disabled).
