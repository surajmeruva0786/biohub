#!/usr/bin/env python
"""Sweep detection density (DoG percentile and peak separation).

The budget sweep showed score rising monotonically with detection count up to
the densest setting tested, so this probes below that threshold. Each setting
gets its own detection cache, since changing the parameters changes the
detections themselves.

Usage:
    python scripts/sweep_detection.py --limit 6 --percentiles 90 80 70
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


def detections(sample, pct: float, sep: float) -> tuple[np.ndarray, np.ndarray]:
    cache = Path(f"cache/det_p{pct:g}_s{sep:g}")
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{sample.name}.npz"
    if f.exists():
        d = np.load(f)
        return d["coords"], d["times"]
    vol = open_volume(sample.zarr_path)
    coords, times = detect_volume(
        vol, vol.shape[0], intensity_percentile=pct, min_separation_um=sep
    )
    np.savez_compressed(f, coords=coords, times=times)
    return coords, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--percentiles", type=float, nargs="+", default=[90.0, 80.0, 70.0])
    ap.add_argument("--separations", type=float, nargs="+", default=[2.5])
    ap.add_argument("--max-link-um", type=float, default=6.0)
    args = ap.parse_args()

    samples = list_samples(DATA / "train", require_gt=True)[: args.limit]
    ests = {s.name: estimated_n_nodes(s.geff_path) for s in samples}

    print(f"{'pct':>6}{'sep':>6}{'det/fr':>9}{'edge_J':>10}{'adj_J':>10}{'recall':>9}{'ratio':>9}")
    for sep in args.separations:
        for pct in args.percentiles:
            t0 = time.time()
            rows, per_frame = [], []
            for s in samples:
                coords, times = detections(s, pct, sep)
                per_frame.append(len(coords) / (times.max() + 1))
                _, edges = build_graph(coords, times, max_link_um=args.max_link_um)
                coords, times, edges = close_gaps(
                    coords, times, edges, max_link_um=args.max_link_um
                )
                rows.append(
                    score_sample(coords, times, edges, s.geff_path, n_total=ests[s.name])
                )
            summ = summarise_rows(rows)
            ratio = float(np.mean([r["total_node_ratio"] for r in rows]))
            print(
                f"{pct:>6.0f}{sep:>6.1f}{np.mean(per_frame):>9.0f}"
                f"{summ['edge_jaccard']:>10.4f}{summ['adj_edge_jaccard']:>10.4f}"
                f"{summ['node_recall']:>9.3f}{ratio:>+9.2f}   ({time.time() - t0:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
