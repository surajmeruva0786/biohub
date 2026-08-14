#!/usr/bin/env python
"""A/B named pipeline variants on cached detections, scored by the official metric.

Every variant reuses the same cached detection set, so the only cost per
variant is linking and scoring -- seconds instead of the hours a re-detection
would take. Variants are scored in parallel across samples.

Usage:
    python scripts/experiment.py --limit 20 --variants baseline prune divisions
    python scripts/experiment.py --list
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub.pipeline import Config

DATA = Path("biohub-cell-tracking-during-development")
CACHE = Path("cache/detections")


# Detection parameters change the detections themselves, so each setting gets its
# own cache directory: comparing two of them means comparing two directories
# rather than two `Config`s. The directory is passed explicitly into workers --
# a module global set in the parent would not survive the fork to a subprocess.

BASE = Config()

# Each variant is one hypothesis, stated as a delta from the baseline so the
# table below reads as an ablation rather than a list of unrelated settings.
VARIANTS: dict[str, Config] = {
    "baseline": BASE,
    "prune_iso": BASE.with_(prune_isolated=True),
    "track2": BASE.with_(prune_isolated=True, min_track_len=2),
    "track3": BASE.with_(prune_isolated=True, min_track_len=3),
    "track5": BASE.with_(prune_isolated=True, min_track_len=5),
    "track8": BASE.with_(prune_isolated=True, min_track_len=8),
    "track12": BASE.with_(prune_isolated=True, min_track_len=12),
    "divisions": BASE.with_(prune_isolated=True, divisions=True),
    # Detection density: fewer nodes means a smaller over-prediction penalty but
    # lower recall. The two effects pull opposite ways, so the optimum is empirical.
    "budget200": BASE.with_(prune_isolated=True, budget=200),
    "budget300": BASE.with_(prune_isolated=True, budget=300),
    "budget400": BASE.with_(prune_isolated=True, budget=400),
    "budget500": BASE.with_(prune_isolated=True, budget=500),
    # Gate width, retuned now that pruning changed the node population.
    "gate5": BASE.with_(prune_isolated=True, max_link_um=5.0),
    "gate7": BASE.with_(prune_isolated=True, max_link_um=7.0),
}


def _score_one(name: str, geff: str, cfg: Config, cache: str) -> dict:
    """Worker: run one variant on one sample and return its metric row."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    warnings.filterwarnings("ignore")
    from biohub.evaluate import score_sample
    from biohub.io import estimated_n_nodes
    from biohub.pipeline import run

    d = np.load(Path(cache) / f"{name}.npz")
    coords, times, edges = run(d["coords"], d["times"], cfg)
    row = score_sample(coords, times, edges, geff, n_total=estimated_n_nodes(geff))
    row["sample"] = name
    return row


def evaluate(
    names, geffs, cfg: Config, workers: int, cache: str = str(CACHE)
) -> tuple[dict, list[dict]]:
    from biohub.evaluate import summarise_rows

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(
            pool.map(_score_one, names, geffs, [cfg] * len(names), [cache] * len(names))
        )
    return summarise_rows(rows), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--list", action="store_true", help="print variant names and exit")
    ap.add_argument("--per-sample", action="store_true")
    ap.add_argument("--cache", default=str(CACHE), help="detection cache directory")
    ap.add_argument(
        "--embryo",
        default=None,
        help="restrict to one embryo id, to check cross-embryo transfer",
    )
    args = ap.parse_args()

    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k:12} {v}")
        return

    from biohub.io import embryo_of, list_samples, select_samples

    cache = Path(args.cache)
    cached = [
        s
        for s in list_samples(DATA / "train", require_gt=True)
        if (cache / f"{s.name}.npz").exists()
    ]
    # Balanced across embryos: names sort by embryo, so a plain prefix would
    # measure a setting against one animal while the leaderboard scores it on an
    # unseen one.
    samples = select_samples(cached, args.limit)
    if args.embryo:
        samples = [s for s in samples if embryo_of(s.name) == args.embryo]
    if not samples:
        raise SystemExit(f"no cached detections in {cache} -- run cache_detections.py")

    names = [s.name for s in samples]
    geffs = [str(s.geff_path) for s in samples]
    counts = Counter(embryo_of(n) for n in names)
    print(f"{len(samples)} samples from {cache} {dict(counts)}, {args.workers} workers\n")

    header = f"{'variant':>12}{'edge_J':>10}{'adj_J':>10}{'div_J':>9}{'score':>10}{'recall':>9}{'ratio':>9}{'nodes':>10}"
    print(header)
    print("-" * len(header))

    best = None
    for name in args.variants:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; --list to see options")
        t0 = time.time()
        summ, rows = evaluate(names, geffs, VARIANTS[name], args.workers, str(cache))
        ratio = float(np.mean([r["total_node_ratio"] for r in rows]))
        nodes = int(np.mean([r["num_pred_nodes"] for r in rows]))
        print(
            f"{name:>12}{summ['edge_jaccard']:>10.4f}{summ['adj_edge_jaccard']:>10.4f}"
            f"{summ['division_jaccard']:>9.3f}{summ['score']:>10.4f}"
            f"{summ['node_recall']:>9.3f}{ratio:>+9.2f}{nodes:>10d}"
            f"   ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if best is None or summ["score"] > best[1]:
            best = (name, summ["score"], rows)

    if args.per_sample and best:
        print(f"\nper-sample for best variant ({best[0]}):")
        for r in sorted(best[2], key=lambda r: r["adj_edge_jaccard"]):
            print(
                f"  {r['sample']:22} J={r['edge_jaccard']:.3f} "
                f"adj={r['adj_edge_jaccard']:.3f} recall={r['node_recall']:.3f} "
                f"ratio={r['total_node_ratio']:+.2f}"
            )


if __name__ == "__main__":
    main()
