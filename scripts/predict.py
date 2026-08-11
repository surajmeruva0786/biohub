#!/usr/bin/env python
"""Run the pipeline over every test sample and write ``submission.csv``.

Usage:
    python scripts/predict.py --data-dir <root>/test --out submission.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub.detect import detect_volume
from biohub.io import list_samples, open_volume
from biohub.submit import SubmissionWriter
from biohub.track import build_graph, close_gaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="biohub-cell-tracking-during-development/test",
        help="directory of .zarr samples to predict",
    )
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--max-link-um", type=float, default=6.0)
    ap.add_argument("--no-gap-closing", action="store_true")
    ap.add_argument("--sigma-um", type=float, default=1.0)
    ap.add_argument("--separation-um", type=float, default=2.5)
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--background-um", type=float, default=4.0)
    args = ap.parse_args()

    samples = list_samples(args.data_dir)
    if not samples:
        raise SystemExit(f"no .zarr samples found in {args.data_dir}")

    with SubmissionWriter(args.out) as w:
        for i, s in enumerate(samples, 1):
            t0 = time.time()
            vol = open_volume(s.zarr_path)
            coords, times = detect_volume(
                vol,
                vol.shape[0],
                sigma_um=args.sigma_um,
                min_separation_um=args.separation_um,
                intensity_percentile=args.percentile,
                background_um=args.background_um,
            )
            node_ids, edges = build_graph(coords, times, max_link_um=args.max_link_um)
            if not args.no_gap_closing:
                coords, times, edges = close_gaps(
                    coords, times, edges, max_link_um=args.max_link_um
                )
                node_ids = np.arange(1, len(coords) + 1, dtype=np.int64)
            w.add_sample(s.name, coords, times, node_ids, edges)
            print(
                f"[{i}/{len(samples)}] {s.name:20} nodes={len(coords):6d} "
                f"edges={len(edges):6d} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
