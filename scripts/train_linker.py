#!/usr/bin/env python
"""Train a learned edge scorer to replace the pure-distance linking cost.

Supervision is sparse, as in the reference solution: a candidate link is only
labelled when its *source* detection matches a ground-truth node that has an
outgoing GT edge. For those sources the correct partner is known, so the true
link is positive and its competing candidates are negative. Sources whose truth
is unknown contribute nothing, rather than being assumed negative.

Usage:
    python scripts/train_linker.py --limit 60 --out models/linker.joblib
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub.detect import detect_volume
from biohub.features import FEATURE_NAMES, build_candidates
from biohub.io import SCALE, list_samples, open_volume, read_gt_graph
from biohub.track import _assign, estimate_drift, to_physical

DATA = Path("biohub-cell-tracking-during-development")
CACHE = Path("cache/detections")
MATCH_UM = 7.0


def cached_detections(sample) -> tuple[np.ndarray, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{sample.name}.npz"
    if f.exists():
        d = np.load(f)
        return d["coords"], d["times"]
    vol = open_volume(sample.zarr_path)
    coords, times = detect_volume(vol, vol.shape[0])
    np.savez_compressed(f, coords=coords, times=times)
    return coords, times


def sample_training_rows(
    coords: np.ndarray,
    times: np.ndarray,
    gt: dict,
    max_link_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Featurise candidate links for one sample and label the supervised ones."""
    gt_pos = np.stack([gt["z"], gt["y"], gt["x"]], axis=1) * np.asarray(SCALE)
    gt_t = gt["t"]
    id_to_row = {int(i): r for r, i in enumerate(gt["ids"])}

    # GT edges grouped by the source node's timepoint.
    edges_by_t: dict[int, list[tuple[int, int]]] = {}
    for s, d in gt["edges"]:
        rs, rd = id_to_row.get(int(s)), id_to_row.get(int(d))
        if rs is None or rd is None:
            continue
        t = int(gt_t[rs])
        if int(gt_t[rd]) == t + 1:
            edges_by_t.setdefault(t, []).append((rs, rd))

    by_t = {int(t): np.flatnonzero(times == t) for t in np.unique(times)}
    X, y = [], []

    for t, gt_edges in edges_by_t.items():
        if t not in by_t or (t + 1) not in by_t:
            continue
        si, di = by_t[t], by_t[t + 1]
        src, dst = to_physical(coords[si]), to_physical(coords[di])

        drift = estimate_drift(src, dst, _assign(src, dst, max_link_um))
        src_c = src + drift

        pairs, feats = build_candidates(src_c, dst, max_link_um)
        if len(pairs) == 0:
            continue

        # Nearest detection to each GT endpoint, within the metric's threshold.
        src_tree, dst_tree = cKDTree(src), cKDTree(dst)
        truth: dict[int, int] = {}
        for rs, rd in gt_edges:
            ds, a = src_tree.query(gt_pos[rs])
            dd, b = dst_tree.query(gt_pos[rd])
            if ds <= MATCH_UM and dd <= MATCH_UM:
                truth[int(a)] = int(b)

        if not truth:
            continue
        # Keep only candidates whose source has known truth.
        keep = np.array([p[0] in truth for p in pairs])
        if not keep.any():
            continue
        X.append(feats[keep])
        y.append(
            np.array([1 if truth[p[0]] == p[1] else 0 for p in pairs[keep]])
        )

    if not X:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, dtype=int)
    return np.concatenate(X), np.concatenate(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-link-um", type=float, default=6.0)
    ap.add_argument("--out", default="models/linker.joblib")
    args = ap.parse_args()

    samples = list_samples(DATA / "train", require_gt=True)[: args.limit]
    X_all, y_all = [], []

    for i, s in enumerate(samples, 1):
        t0 = time.time()
        coords, times = cached_detections(s)
        gt = read_gt_graph(s.geff_path)
        X, y = sample_training_rows(coords, times, gt, args.max_link_um)
        if len(y):
            X_all.append(X)
            y_all.append(y)
        print(
            f"[{i}/{len(samples)}] {s.name:20} rows={len(y):6d} pos={int(y.sum()):5d} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    print(f"\ntotal rows={len(y)} positives={int(y.sum())} ({y.mean():.1%})")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=6, random_state=0
    )
    clf.fit(X_tr, y_tr)

    from sklearn.metrics import average_precision_score, roc_auc_score

    p = clf.predict_proba(X_va)[:, 1]
    print(f"val ROC-AUC={roc_auc_score(y_va, p):.4f}  AP={average_precision_score(y_va, p):.4f}")

    # Distance-only baseline for comparison: nearer is better, so negate.
    d_only = -X_va[:, 0]
    print(
        f"distance-only ROC-AUC={roc_auc_score(y_va, d_only):.4f}  "
        f"AP={average_precision_score(y_va, d_only):.4f}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({"model": clf, "features": FEATURE_NAMES}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
