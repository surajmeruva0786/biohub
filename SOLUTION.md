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
| Detection (CUDA) | `src/biohub/detect_gpu.py` | Same algorithm in PyTorch, 6.8× faster |
| Linking | `src/biohub/track.py` | Hungarian assignment, distance or learned cost |
| Pruning | `src/biohub/prune.py` | Drops nodes that carry penalty without earning TPs |
| Divisions | `src/biohub/divide.py` | Adds second children to a one-to-one graph |
| Orchestration | `src/biohub/pipeline.py` | `Config` + `run()` — one frozen setting per variant |
| Scoring | `src/biohub/evaluate.py` | Wraps the official `tracking_cellmot.metrics` |
| Export | `src/biohub/submit.py` | Competition CSV writer |

### Detection runs on the GPU

Detection is ~85% of pipeline runtime, and all of it is two separable
stencils — a difference-of-Gaussians and a local-maximum filter. Both map
directly onto the GPU, so `detect_gpu.py` implements them in PyTorch: three 1D
convolutions per blur (the background sigma is ~10 voxels in x/y, so a 3D
kernel would be 80³ taps against 3 × 81 separable) and a strided `max_pool3d`
for the peak test.

| Backend | Per frame | Peak GPU memory | Detections |
| --- | --- | --- | --- |
| SciPy (CPU) | 1.04 s | — | 1483 |
| PyTorch (Quadro P1000) | 0.15 s | 189 MB | 1484 |

**6.8× faster, and the detections are identical** — `scripts/bench_gpu.py`
checks set equality per frame and reports 100% agreement. That parity is not a
nicety: every experiment below is scored against one shared detection cache, so
a GPU path that found even slightly different blobs would invalidate
comparisons between runs cached on different backends. The one-frame difference
in the totals above is a tie at the percentile threshold, not a disagreement on
any peak that both backends kept.

Sizing is comfortable rather than tight — one `(64, 256, 256)` float32
timepoint is 16 MB and the separable form keeps the working set a small
multiple of that, so 189 MB of a 4 GB card is the whole footprint and frames
stream one at a time rather than scaling with movie length. `detect_volume`
takes `device="auto"`, using CUDA when present and falling back to SciPy
otherwise, so nothing depends on the GPU being free.

### Experiments run against a detection cache

Detection is expensive and independent of every linking decision, so it is paid
once: `scripts/cache_detections.py` writes detections per sample and
`scripts/experiment.py` A/Bs named `Config` variants against them in parallel.
A variant costs seconds instead of the hours a re-detection would take, which
is what makes the sweeps below affordable.

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

### Detection recall is the binding constraint

The baseline's error budget points away from linking. Over 8 samples it scores
1311 edge TP against 185 FP and 153 FN — and the weak samples are the
low-recall ones, not the dense ones:

| Sample | Node recall | Edge Jaccard |
| --- | --- | --- |
| `44b6_0db75fae` | 1.000 | 0.993 |
| `44b6_0113de3b` | 1.000 | 0.923 |
| `44b6_0b24845f` | 0.902 | 0.556 |
| `44b6_0c582fdc` | 0.901 | 0.612 |

A missed cell is expensive twice over. The metric only credits `t → t+1` edges,
so a ground-truth cell with no detection within 7 µm loses both the edge
entering it *and* the edge leaving it. Worse, the detection that should have
linked to it does not simply stop — it links to whatever else is in gate, so
each FN tends to purchase an FP as well. That is why FP and FN are of similar
size here despite linking being one-to-one.

`scripts/sweep_recall.py` measures recall against detection parameters
directly. It is cheap for the same reason the ground truth is hard to learn
from: with ~0.66% of cells annotated, only a handful of frames per sample
contain any ground truth at all, and recall can only be measured on those — so
only those are detected. Matching uses the metric's own one-to-one gated
assignment rather than nearest-neighbour counting, which would overstate recall
wherever two ground-truth cells share their closest detection.

| DoG sigma (µm) | Node recall | Detections/frame |
| --- | --- | --- |
| 0.6 | **0.968** | 590 |
| 1.0 (baseline) | 0.948 | 448 |
| 1.5 | 0.897 | 367 |
| 2.0 | 0.897 | 310 |
| 2.5 | 0.819 | 261 |

Recall falls monotonically as sigma grows. But a smaller sigma also detects
more per frame, and every extra node feeds the over-prediction penalty, so this
is a trade rather than a free win and has to be scored end to end.

### Linking gate and drift compensation

Ground-truth cells move **1.9 µm per frame median, p95 3.8, p99 5.3** (measured
over 10,572 GT edges).

Much of that motion is embryo-wide drift rather than independent cell movement —
typically ~1.6/1.2/0.8 µm in (z, y, x), comparable to the median displacement
itself. Linking therefore runs two passes: a first assignment estimates the
frame's median displacement, and the second re-assigns with that global motion
removed, so the distance budget discriminates between neighbouring cells instead
of being spent on drift.

Gate width and drift compensation are partly redundant, and tuning them together
matters:

| gate (µm) | drift off | drift on |
| --- | --- | --- |
| 5.0 | 0.7490 | 0.7595 |
| 6.0 | 0.7584 | **0.7658** |
| 7.0 | 0.7623 | 0.7642 |
| 8.0 | 0.7311 | 0.7412 |

Without drift compensation the optimum is a loose 7 µm gate; with it the optimum
tightens to 6 µm and scores higher. Beyond 8 µm the score collapses (0.59 at
10 µm) as links jump between neighbouring cells.

### Gap closing

The metric only credits edges spanning exactly `t → t+1`, so a cell missed in a
single frame costs *two* edges and cannot be bridged directly. Instead, a track
ending at `t` and one resuming at `t+2` within twice the gate are joined by a
synthetic node interpolated at their midpoint in frame `t+1`, producing two
well-formed edges.

This is worth most exactly where the baseline is weakest — sample
`44b6_0c582fdc` went from 0.476 to 0.562 edge Jaccard — and is roughly neutral
on samples that already track well.

## Reproducing

```bash
pip install zarr numcodecs polars tracksdata torch xgboost
git clone https://github.com/royerlab/kaggle-cell-tracking-competition reference

python scripts/bench_gpu.py                     # verify CUDA parity with SciPy
python scripts/cache_detections.py --limit 140  # pay for detection once
python scripts/experiment.py --limit 40         # A/B pipeline variants
python scripts/predict.py --out submission.csv  # write a submission
```

## Environment note

The local GPU is a Quadro P1000 (4 GB). That is too small to *train* a 3D
segmentation network on `(100, 64, 256, 256)` uint16 volumes, but it is ample
for the classical stages: DoG detection peaks at 189 MB and runs 6.8× faster
there than on CPU, and the learned linker is a gradient-boosted model over ten
geometric features, which trains in seconds.

Backends are chosen so the GPU is never a bottleneck for someone else's work.
`--device cpu` fans detection out over processes instead, and the trained
linker is saved with `device="cpu"` because inference happens inside parallel
scoring workers where several processes contending for 4 GB of VRAM is slower
than plain CPU on batches this small.
