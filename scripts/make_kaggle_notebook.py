#!/usr/bin/env python
"""Bundle the pipeline into one self-contained script for a Kaggle notebook.

Submissions run as notebooks with internet disabled, so the code has to arrive
as a single file rather than an installed package. Writing that file by hand
would mean maintaining a second copy of the pipeline that silently drifts from
the one the offline experiments actually validated, so it is generated from the
real modules instead: the sources are concatenated in dependency order and
their intra-package imports stripped, since everything lands in one namespace.

Scoring modules are deliberately excluded -- ``evaluate.py`` pulls in
``tracksdata`` and the vendored reference metric, neither of which the
inference path needs.

Usage:
    python scripts/make_kaggle_notebook.py --out kaggle/submission_script.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "biohub"

# Dependency order: each module may only reference names defined above it.
MODULES = [
    "io",
    "detect_gpu",
    "detect",
    "features",
    "track",
    "prune",
    "divide",
    "pipeline",
    "submit",
]

HEADER = '''"""Biohub cell tracking -- inference for Kaggle submission.

GENERATED FILE. Do not edit here; edit src/biohub/*.py and regenerate with
    python scripts/make_kaggle_notebook.py

Detect cells per timepoint, link them frame to frame, repair one-frame gaps,
prune nodes that cannot earn a true positive, and write submission.csv.
Detection runs on CUDA when the notebook has a GPU and falls back to SciPy
otherwise, so the same file works on either accelerator setting.
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

'''

MAIN = '''

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

# The hidden test set is roughly the size of the training set (~200 samples of
# 100 timepoints), against a 12 hour notebook limit. Detection is the dominant
# cost, which is why the CUDA path matters here rather than being a local
# convenience.
DATA_DIR = Path("/kaggle/input/biohub-cell-tracking-during-development/test")
OUT = Path("submission.csv")

CONFIG = Config(
    max_link_um=MAX_LINK_UM,
    compensate_drift=True,
    gap_closing=True,
    prune_isolated=True,
    min_track_len=MIN_TRACK_LEN,
    divisions=DIVISIONS,
)

DETECT_KWARGS = dict(
    sigma_um=SIGMA_UM,
    min_separation_um=SEPARATION_UM,
    intensity_percentile=PERCENTILE,
    background_um=BACKGROUND_UM,
)


def main() -> None:
    data_dir = DATA_DIR if DATA_DIR.exists() else Path(sys.argv[1])
    samples = list_samples(data_dir)
    if not samples:
        raise SystemExit(f"no .zarr samples found in {data_dir}")

    device = "cuda" if cuda_available() else "cpu"
    print(f"{len(samples)} samples, device={device}", flush=True)
    print(f"{CONFIG}\\n{DETECT_KWARGS}\\n", flush=True)

    started = time.time()
    with SubmissionWriter(OUT) as w:
        for i, s in enumerate(samples, 1):
            t0 = time.time()
            vol = open_volume(s.zarr_path)
            coords, times = detect_volume(
                vol, vol.shape[0], device=device, **DETECT_KWARGS
            )
            n_det = len(coords)
            coords, times, edges = run(coords, times, CONFIG)
            node_ids = np.arange(1, len(coords) + 1, dtype=np.int64)
            w.add_sample(s.name, coords, times, node_ids, edges)
            print(
                f"[{i}/{len(samples)}] {s.name:22} det={n_det:7d} "
                f"nodes={len(coords):7d} edges={len(edges):7d} "
                f"({time.time() - t0:.0f}s, total {(time.time() - started) / 60:.1f}m)",
                flush=True,
            )

    print(f"\\nwrote {OUT} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
'''

# Lines to drop from each module: the package is flattened into one namespace,
# so intra-package imports have nothing to resolve against, and the shared
# stdlib/third-party imports are hoisted into HEADER.
DROP = re.compile(
    r"^\s*(from\s+\.\S*\s+import\s|from\s+__future__\s+import\s|import\s+numpy|"
    r"import\s+zarr|import\s+csv|import\s+sys|import\s+time|"
    r"from\s+scipy\b|from\s+dataclasses\s+import\s|from\s+pathlib\s+import\s|"
    r"from\s+concurrent\.futures\s+import\s|from\s+collections\s+import\s)"
)


def strip_module(text: str) -> str:
    """Drop hoisted imports, keeping everything else (docstrings included)."""
    kept = [ln for ln in text.splitlines() if not DROP.match(ln)]
    return "\n".join(kept).strip("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kaggle/submission_script.py")
    ap.add_argument("--sigma-um", type=float, default=0.6)
    ap.add_argument("--separation-um", type=float, default=2.5)
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--background-um", type=float, default=4.0)
    ap.add_argument("--max-link-um", type=float, default=5.0)
    ap.add_argument("--min-track-len", type=int, default=0)
    ap.add_argument("--divisions", action="store_true")
    args = ap.parse_args()

    parts = [HEADER]
    parts.append(
        "# --- tuned settings, from scripts/experiment.py and sweep_recall.py ---\n"
        f"SIGMA_UM = {args.sigma_um}\n"
        f"SEPARATION_UM = {args.separation_um}\n"
        f"PERCENTILE = {args.percentile}\n"
        f"BACKGROUND_UM = {args.background_um}\n"
        f"MAX_LINK_UM = {args.max_link_um}\n"
        f"MIN_TRACK_LEN = {args.min_track_len}\n"
        f"DIVISIONS = {args.divisions}\n"
    )

    for name in MODULES:
        body = strip_module((SRC / f"{name}.py").read_text(encoding="utf-8"))
        parts.append(
            f"\n# {'=' * 74}\n# {name}.py\n# {'=' * 74}\n\n{body}\n"
        )

    parts.append(MAIN)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(parts)
    out.write_text(text, encoding="utf-8")

    compile(text, str(out), "exec")  # fail loudly here rather than on Kaggle
    print(f"wrote {out} ({len(text.splitlines())} lines, compiles clean)")


if __name__ == "__main__":
    main()
