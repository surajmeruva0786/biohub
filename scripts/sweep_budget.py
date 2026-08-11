#!/usr/bin/env python
"""Sweep the per-frame detection budget.

``detect_timepoint`` returns detections ordered strongest-response first, so a
budget is applied by truncating each frame -- no re-detection needed. The
budget trades two effects against each other:

* fewer detections lower node recall (fewer TP, more FN)
* fewer detections raise the node-count penalty factor, and shrink the pool of
  spurious edges that can attach to a GT-matched node and become FP

Usage:
    python scripts/sweep_budget.py --limit 12 --budgets 150 250 350 500 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub.detect import detect_volume
from biohub.evaluate import score_sample, summarise_rows
from biohub.io import estimated_n_nodes, list_samples, open_volume
from biohub.track import build_graph, close_gaps

DATA = Path("biohub-cell-tracking-during-development")
CACHE = Path("cache/detections")


def cached_detections(sample, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{sample.name}.npz"
    if f.exists():
        d = np.load(f)
        return d["coords"], d["times"]
    vol = open_volume(sample.zarr_path)
    coords, times = detect_volume(vol, vol.shape[0], **kwargs)
    np.savez_compressed(f, coords=coords, times=times)
    return coords, times


def apply_budget(
    coords: np.ndarray, times: np.ndarray, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the *budget* strongest detections per frame (0 = keep all)."""
    if budget <= 0:
        return coords, times
    keep = []
    for t in np.unique(times):
        idx = np.flatnonzero(times == t)  # already strongest-first within a frame
        keep.append(idx[:budget])
    keep = np.concatenate(keep)
    return coords[keep], times[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument(
        "--budgets", type=int, nargs="+", default=[150, 200, 250, 350, 500, 0]
    )
    ap.add_argument("--max-link-um", type=float, default=6.0)
    args = ap.parse_args()

    samples = list_samples(DATA / "train", require_gt=True)[: args.limit]

    det, ests = {}, {}
    for i, s in enumerate(samples, 1):
        t0 = time.time()
        det[s.name] = cached_detections(s)
        ests[s.name] = estimated_n_nodes(s.geff_path)
        print(
            f"detect [{i}/{len(samples)}] {s.name} n={len(det[s.name][0])} "
            f"est={ests[s.name]:.0f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    print(f"\n{'budget':>8}{'edge_J':>10}{'adj_J':>10}{'recall':>9}{'mean_ratio':>12}")
    for budget in args.budgets:
        rows = []
        for s in samples:
            coords, times = apply_budget(*det[s.name], budget)
            _, edges = build_graph(coords, times, max_link_um=args.max_link_um)
            coords, times, edges = close_gaps(
                coords, times, edges, max_link_um=args.max_link_um
            )
            rows.append(
                score_sample(
                    coords, times, edges, s.geff_path, n_total=ests[s.name]
                )
            )
        summ = summarise_rows(rows)
        ratio = float(np.mean([r["total_node_ratio"] for r in rows]))
        print(
            f"{budget if budget else 'all':>8}{summ['edge_jaccard']:>10.4f}"
            f"{summ['adj_edge_jaccard']:>10.4f}{summ['node_recall']:>9.3f}"
            f"{ratio:>+12.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
