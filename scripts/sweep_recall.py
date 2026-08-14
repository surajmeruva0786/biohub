#!/usr/bin/env python
"""Sweep detection parameters against node recall, the pipeline's actual ceiling.

A ground-truth cell with no detection within the metric's 7 µm matching radius
costs *two* edges -- the one entering it and the one leaving -- and it costs
them twice over, because the detection that should have linked to it links to
whatever else is in range instead, turning a false negative into a false
positive as well. Measured on the baseline, edge FP and FN are 185 and 153
against 1311 TP, and the weakest samples are precisely the low-recall ones. So
recall, not linking cleverness, is what bounds the score.

The sweep is cheap for a reason worth stating: annotations cover ~0.66% of
cells, so only a couple of frames per sample contain any ground truth at all.
Recall can only be measured on those frames, so only those are detected --
turning a 100-frame pass into a handful of frames per sample.

Usage:
    python scripts/sweep_recall.py --limit 12 --workers 3
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub.io import list_samples, select_samples

DATA = Path("biohub-cell-tracking-during-development")
MATCH_UM = 7.0  # the metric's node-matching radius


def _recall_one(
    zarr_path: str, geff_path: str, params: dict, max_frames: int, device: str
) -> tuple[int, int, int, int]:
    """Worker: (matched, total, detections, frames) for one sample."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from scipy.optimize import linear_sum_assignment

    from biohub.io import SCALE, open_volume, read_gt_graph, read_timepoint

    gt = read_gt_graph(geff_path)
    gt_t = np.asarray(gt["t"])
    gt_pos = np.stack([gt["z"], gt["y"], gt["x"]], axis=1) * np.asarray(SCALE)

    frames = np.unique(gt_t)
    if len(frames) > max_frames:
        # Spread the sample across the movie rather than taking a prefix: density
        # and contrast both drift as the embryo develops.
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).astype(int)]

    vol = open_volume(zarr_path)
    if device == "cuda":
        from biohub.detect_gpu import detect_timepoint_gpu as detect
    else:
        from biohub.detect import detect_timepoint as detect

    matched = total = n_det = 0
    for t in frames:
        truth = gt_pos[gt_t == t]
        if len(truth) == 0:
            continue
        coords = detect(read_timepoint(vol, int(t)), **params)
        n_det += len(coords)
        total += len(truth)
        if len(coords) == 0:
            continue

        # Same one-to-one, distance-gated matching the metric uses -- a nearest
        # neighbour count would overstate recall wherever two GT cells share
        # their closest detection.
        pred = coords.astype(np.float64) * np.asarray(SCALE)
        cost = np.linalg.norm(truth[:, None, :] - pred[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(np.where(cost > MATCH_UM, 1e6, cost))
        matched += int((cost[rows, cols] <= MATCH_UM).sum())

    return matched, total, n_det, len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12, help="samples to measure on")
    ap.add_argument("--max-frames", type=int, default=8, help="GT frames per sample")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.6, 1.0, 1.5, 2.0])
    ap.add_argument("--percentiles", type=float, nargs="+", default=[90.0, 80.0])
    ap.add_argument("--separations", type=float, nargs="+", default=[2.5])
    ap.add_argument("--backgrounds", type=float, nargs="+", default=[4.0])
    args = ap.parse_args()

    samples = select_samples(
        list_samples(DATA / "train", require_gt=True), args.limit
    )
    print(f"{len(samples)} samples, <={args.max_frames} GT frames each, device={args.device}\n")

    grid = list(
        itertools.product(
            args.sigmas, args.percentiles, args.separations, args.backgrounds
        )
    )
    header = f"{'sigma':>7}{'pct':>6}{'sep':>6}{'bg':>6}{'recall':>9}{'det/frame':>12}{'missed':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for sigma, pct, sep, bg in grid:
        params = dict(
            sigma_um=sigma,
            intensity_percentile=pct,
            min_separation_um=sep,
            background_um=bg,
        )
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            out = list(
                pool.map(
                    _recall_one,
                    [str(s.zarr_path) for s in samples],
                    [str(s.geff_path) for s in samples],
                    [params] * len(samples),
                    [args.max_frames] * len(samples),
                    [args.device] * len(samples),
                )
            )
        matched = sum(o[0] for o in out)
        total = sum(o[1] for o in out)
        det = sum(o[2] for o in out)
        frames = sum(o[3] for o in out)
        recall = matched / max(1, total)
        results.append((recall, det / max(1, frames), sigma, pct, sep, bg))
        print(
            f"{sigma:>7.2f}{pct:>6.0f}{sep:>6.1f}{bg:>6.1f}{recall:>9.4f}"
            f"{det / max(1, frames):>12.0f}{total - matched:>9d}"
            f"   ({time.time() - t0:.0f}s)",
            flush=True,
        )

    best = max(results)
    print(
        f"\nbest recall {best[0]:.4f} at sigma={best[2]} pct={best[3]} "
        f"sep={best[4]} bg={best[5]} ({best[1]:.0f} det/frame)"
    )


if __name__ == "__main__":
    main()
