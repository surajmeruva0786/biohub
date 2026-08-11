#!/usr/bin/env python
"""Sweep linking parameters over cached detections.

Detection dominates runtime (~85 s/sample) while linking is seconds, so
detections are computed once per sample and reused across the sweep.

Usage:
    python scripts/sweep_linking.py --limit 6
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
from biohub.track import build_graph

DATA = Path("biohub-cell-tracking-during-development")
CACHE = Path("cache/detections")


def cached_detections(sample, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Detect once per sample, then reuse from disk on later sweeps."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{sample.name}.npz"
    if f.exists():
        d = np.load(f)
        return d["coords"], d["times"]

    vol = open_volume(sample.zarr_path)
    coords, times = detect_volume(vol, vol.shape[0], **kwargs)
    np.savez_compressed(f, coords=coords, times=times)
    return coords, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument(
        "--gates", type=float, nargs="+", default=[3.0, 4.0, 5.0, 6.0, 7.0],
    )
    args = ap.parse_args()

    samples = list_samples(DATA / "train", require_gt=True)[: args.limit]

    det = {}
    for i, s in enumerate(samples, 1):
        t0 = time.time()
        det[s.name] = cached_detections(s)
        print(
            f"detect [{i}/{len(samples)}] {s.name} "
            f"n={len(det[s.name][0])} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    print(f"\n{'gate_um':>8}{'drift':>7}{'edge_J':>10}{'adj_J':>10}{'recall':>9}")
    for gate in args.gates:
        for drift in (False, True):
            rows = []
            for s in samples:
                coords, times = det[s.name]
                _, edges = build_graph(
                    coords, times, max_link_um=gate, compensate_drift=drift
                )
                rows.append(
                    score_sample(
                        coords, times, edges, s.geff_path,
                        n_total=estimated_n_nodes(s.geff_path),
                    )
                )
            summ = summarise_rows(rows)
            print(
                f"{gate:>8.1f}{str(drift):>7}{summ['edge_jaccard']:>10.4f}"
                f"{summ['adj_edge_jaccard']:>10.4f}{summ['node_recall']:>9.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
