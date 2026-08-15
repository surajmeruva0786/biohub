#!/usr/bin/env python
"""Pre-compute and cache detections for train samples, in parallel.

Detection costs about a second per timepoint, so a 100-frame sample takes
~100 s single-threaded. Every linking, pruning and division experiment reuses
the *same* detections, so paying that cost once up front turns each subsequent
experiment from hours into seconds.

Usage:
    python scripts/cache_detections.py --limit 60 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub.io import list_samples, select_samples

DATA = Path("biohub-cell-tracking-during-development")
CACHE = Path("cache/detections")


def _detect_one(
    name: str,
    zarr_path: str,
    cache_dir: str,
    device: str = "cpu",
    params: dict | None = None,
) -> tuple[str, int, float]:
    """Worker: detect one sample and write its ``.npz``. Runs in a subprocess."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from biohub.detect import detect_volume
    from biohub.io import open_volume

    t0 = time.time()
    out = Path(cache_dir) / f"{name}.npz"
    if out.exists():
        d = np.load(out)
        return name, len(d["coords"]), 0.0

    vol = open_volume(zarr_path)
    coords, times = detect_volume(vol, vol.shape[0], device=device, **(params or {}))
    # Write via a temp file so an interrupted run never leaves a truncated cache.
    # The name must still end in .npz -- savez_compressed appends the suffix
    # itself otherwise, and the rename then targets a file that was never written.
    tmp = out.with_name(f"{name}.tmp.npz")
    np.savez_compressed(tmp, coords=coords, times=times)
    tmp.replace(out)
    return name, len(coords), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60, help="first N train samples")
    ap.add_argument("--workers", type=int, default=8, help="CPU workers (device=cpu)")
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="cuda runs one sample at a time on the GPU; cpu fans out over processes",
    )
    ap.add_argument("--out", default=str(CACHE), help="cache directory to fill")
    ap.add_argument("--sigma-um", type=float, default=1.0)
    ap.add_argument("--separation-um", type=float, default=2.5)
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--background-um", type=float, default=4.0)
    ap.add_argument(
        "--threshold",
        default="percentile",
        choices=["percentile", "otsu"],
        help="otsu adapts the count to the frame; percentile fixes a voxel fraction",
    )
    ap.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="0 keeps every peak, 1 is Otsu's split; below 1 trades nodes for recall",
    )
    ap.add_argument("--peak-percentile", type=float, default=0.0)
    args = ap.parse_args()

    # Detections depend on these, so a changed setting needs its own directory --
    # silently mixing two parameter sets in one cache would corrupt every
    # comparison drawn from it.
    params = dict(
        sigma_um=args.sigma_um,
        min_separation_um=args.separation_um,
        intensity_percentile=args.percentile,
        background_um=args.background_um,
        threshold=args.threshold,
        threshold_scale=args.threshold_scale,
        peak_percentile=args.peak_percentile,
    )

    cache_dir = Path(args.out)
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples = select_samples(
        list_samples(DATA / "train", require_gt=True), args.limit
    )
    todo = [s for s in samples if not (cache_dir / f"{s.name}.npz").exists()]
    print(
        f"{len(samples)} samples requested, {len(todo)} to detect "
        f"(device={args.device}, out={cache_dir}, {params})",
        flush=True,
    )

    t0 = time.time()
    if args.device == "cuda":
        # One GPU, so no fan-out: the device is already the parallel unit, and a
        # second process would only contend for its 4 GB.
        for i, s in enumerate(todo, 1):
            name, n, secs = _detect_one(s.name, str(s.zarr_path), str(cache_dir), "cuda", params)
            done, left = i, len(todo) - i
            eta = (time.time() - t0) / done * left
            print(
                f"[{i}/{len(todo)}] {name:22} n={n:7d} ({secs:.0f}s) "
                f"elapsed={time.time() - t0:.0f}s eta={eta:.0f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _detect_one, s.name, str(s.zarr_path), str(cache_dir), "cpu", params
                ): s.name
                for s in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                name, n, secs = fut.result()
                print(
                    f"[{i}/{len(todo)}] {name:22} n={n:7d} ({secs:.0f}s) "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

    print(f"\ncached {len(samples)} samples in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
