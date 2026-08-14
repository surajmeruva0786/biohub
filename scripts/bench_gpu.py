#!/usr/bin/env python
"""Verify the CUDA detection backend against SciPy, and time both.

Parity matters more than speed here: the whole point of caching detections is
that every downstream experiment shares them, so a GPU path that quietly finds
different blobs would invalidate every comparison made against a CPU-cached
run. This reports how many detections the two backends agree on exactly.

Usage:
    python scripts/bench_gpu.py --frames 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub.detect import detect_timepoint
from biohub.detect_gpu import cuda_available, detect_timepoint_gpu
from biohub.io import list_samples, open_volume, read_timepoint

DATA = Path("biohub-cell-tracking-during-development")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    if not cuda_available():
        raise SystemExit("no CUDA device available")

    import torch

    print(f"device: {torch.cuda.get_device_name(0)}")

    sample = list_samples(DATA / "train", require_gt=True)[0]
    vol = open_volume(sample.zarr_path)
    frames = [read_timepoint(vol, t) for t in range(args.frames)]
    print(f"sample: {sample.name}  frames: {args.frames}  shape: {frames[0].shape}\n")

    t0 = time.time()
    cpu = [detect_timepoint(f) for f in frames]
    cpu_s = time.time() - t0

    detect_timepoint_gpu(frames[0])  # warm up CUDA context and autotuner
    torch.cuda.synchronize()
    t0 = time.time()
    gpu = [detect_timepoint_gpu(f) for f in frames]
    torch.cuda.synchronize()
    gpu_s = time.time() - t0

    print(f"{'':>10}{'time':>10}{'per frame':>12}{'detections':>13}")
    print(f"{'scipy':>10}{cpu_s:>9.2f}s{cpu_s / args.frames:>11.2f}s{sum(map(len, cpu)):>13d}")
    print(f"{'cuda':>10}{gpu_s:>9.2f}s{gpu_s / args.frames:>11.2f}s{sum(map(len, gpu)):>13d}")
    print(f"\nspeedup: {cpu_s / gpu_s:.1f}x")

    print(f"\n{'frame':>7}{'cpu':>8}{'gpu':>8}{'shared':>9}{'agree':>9}")
    for i, (a, b) in enumerate(zip(cpu, gpu)):
        sa = {tuple(r) for r in a}
        sb = {tuple(r) for r in b}
        shared = len(sa & sb)
        print(f"{i:>7}{len(a):>8}{len(b):>8}{shared:>9}{shared / max(1, len(sa)):>8.1%}")

    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"\npeak GPU memory: {peak:.0f} MB")


if __name__ == "__main__":
    main()
