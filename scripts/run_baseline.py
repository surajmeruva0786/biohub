#!/usr/bin/env python
"""Run the detect-and-link baseline over train samples and report the official score.

Usage:
    python scripts/run_baseline.py --limit 8
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="number of train samples")
    ap.add_argument("--max-link-um", type=float, default=6.0)
    ap.add_argument("--no-gap-closing", action="store_true")
    ap.add_argument("--sigma-um", type=float, default=1.0)
    ap.add_argument("--separation-um", type=float, default=2.5)
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--background-um", type=float, default=4.0)
    args = ap.parse_args()

    samples = list_samples(DATA / "train", require_gt=True)[: args.limit]
    rows = []
    for i, s in enumerate(samples, 1):
        t0 = time.time()
        vol = open_volume(s.zarr_path)
        n_t = vol.shape[0]

        coords, times = detect_volume(
            vol,
            n_t,
            sigma_um=args.sigma_um,
            min_separation_um=args.separation_um,
            intensity_percentile=args.percentile,
            background_um=args.background_um,
        )
        _, edges = build_graph(coords, times, max_link_um=args.max_link_um)
        if not args.no_gap_closing:
            coords, times, edges = close_gaps(
                coords, times, edges, max_link_um=args.max_link_um
            )

        row = score_sample(
            coords, times, edges, s.geff_path, n_total=estimated_n_nodes(s.geff_path)
        )
        rows.append(row)
        print(
            f"[{i}/{len(samples)}] {s.name:20} "
            f"nodes={row['num_pred_nodes']:6d} edges_tp={row['edge_tp']:5d} "
            f"J={row['edge_jaccard']:.3f} Jadj={row['adj_edge_jaccard']:.3f} "
            f"recall={row['node_recall']:.3f} ratio={row['total_node_ratio']:+.2f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    summary = summarise_rows(rows)
    print("\n=== run summary ===")
    for k in ("n", "edge_jaccard", "adj_edge_jaccard", "node_recall", "score"):
        print(f"  {k:18} {summary[k]}")


if __name__ == "__main__":
    main()
