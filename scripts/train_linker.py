#!/usr/bin/env python
"""Train a learned edge scorer to replace the pure-distance linking cost.

Supervision is sparse, by necessity. A candidate link is only labelled when its
*source* detection matches a ground-truth node that has an outgoing GT edge:
for those sources the correct partner is known, so the true link is positive
and its in-gate competitors are negative. Sources whose truth is unknown
contribute nothing rather than being assumed negative -- with under 1% of cells
annotated, treating unlabelled candidates as negatives would poison the target.

The split is by *sample*, not by row. Candidates from one frame are highly
correlated, so a random row split would leak the answer across the boundary and
report a validation number the linker cannot reproduce on unseen samples.

Usage:
    python scripts/train_linker.py --train 80 --workers 6 --out models/linker.joblib
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

from biohub.features import FEATURE_NAMES
from biohub.io import embryo_of, list_samples, select_samples

DATA = Path("biohub-cell-tracking-during-development")
CACHE = Path("cache/detections")
MATCH_UM = 7.0  # the metric's own node-matching threshold


def sample_training_rows(
    coords: np.ndarray,
    times: np.ndarray,
    gt: dict,
    max_link_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Featurise candidate links for one sample and label the supervised ones."""
    from scipy.spatial import cKDTree

    from biohub.features import build_candidates
    from biohub.io import SCALE
    from biohub.track import _assign, estimate_drift, to_physical

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

        # Drift removal must match inference exactly, or the features shift.
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
        keep = np.array([p[0] in truth for p in pairs])
        if not keep.any():
            continue
        X.append(feats[keep])
        y.append(np.array([1 if truth[p[0]] == p[1] else 0 for p in pairs[keep]]))

    if not X:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, dtype=int)
    return np.concatenate(X), np.concatenate(y)


def _rows_for(name: str, geff: str, max_link_um: float):
    """Worker: build the labelled rows for one sample."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    warnings.filterwarnings("ignore")
    from biohub.io import read_gt_graph

    d = np.load(CACHE / f"{name}.npz")
    return sample_training_rows(d["coords"], d["times"], read_gt_graph(geff), max_link_um)


def collect(samples, max_link_um: float, workers: int, label: str):
    """Featurise a list of samples in parallel and stack the results."""
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(
            pool.map(
                _rows_for,
                [s.name for s in samples],
                [str(s.geff_path) for s in samples],
                [max_link_um] * len(samples),
            )
        )
    X = np.concatenate([x for x, _ in out if len(x)]) if out else np.zeros((0, 10))
    y = np.concatenate([b for _, b in out if len(b)]) if out else np.zeros(0)
    print(
        f"{label:>6}: {len(samples):3d} samples -> {len(y):7d} rows, "
        f"{int(y.sum()):6d} positive ({y.mean():.1%})  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=80, help="training samples")
    ap.add_argument("--val", type=int, default=25, help="held-out samples after those")
    ap.add_argument("--max-link-um", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="models/linker.joblib")
    ap.add_argument(
        "--holdout-embryo",
        default=None,
        help="validate on this embryo only, training on the rest (true transfer test)",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    cached = select_samples(
        [
            s
            for s in list_samples(DATA / "train", require_gt=True)
            if (CACHE / f"{s.name}.npz").exists()
        ]
    )
    if args.holdout_embryo:
        # The real split is embryo-disjoint, so validating on a held-out embryo
        # measures the thing the leaderboard measures: transfer to an unseen
        # animal. Splitting by sample within shared embryos would not.
        train_s = [s for s in cached if embryo_of(s.name) != args.holdout_embryo][
            : args.train
        ]
        val_s = [s for s in cached if embryo_of(s.name) == args.holdout_embryo][
            : args.val
        ]
    else:
        train_s = cached[: args.train]
        val_s = cached[args.train : args.train + args.val]
    if not train_s or not val_s:
        raise SystemExit(
            f"need at least {args.train + args.val} cached samples, have {len(cached)}"
        )
    print(
        f"train embryos {dict(Counter(embryo_of(s.name) for s in train_s))}, "
        f"val embryos {dict(Counter(embryo_of(s.name) for s in val_s))}"
    )

    X_tr, y_tr = collect(train_s, args.max_link_um, args.workers, "train")
    X_va, y_va = collect(val_s, args.max_link_um, args.workers, "val")

    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    device = args.device
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    # Positives are ~1 in 5 candidates; scale_pos_weight balances the gradient
    # so the model does not simply learn to reject everything.
    pos = max(1.0, float(y_tr.sum()))
    clf = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,
        reg_lambda=1.0,
        scale_pos_weight=(len(y_tr) - pos) / pos,
        eval_metric="aucpr",
        early_stopping_rounds=40,
        device=device,
        tree_method="hist",
        n_jobs=args.workers,
    )
    t0 = time.time()
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(
        f"\ntrained on {device} in {time.time() - t0:.1f}s "
        f"(best iteration {clf.best_iteration})"
    )

    p = clf.predict_proba(X_va)[:, 1]
    d_only = -X_va[:, 0]  # distance-only baseline; nearer is better, so negate
    print(f"\n{'':>16}{'ROC-AUC':>10}{'AP':>10}")
    print(f"{'learned':>16}{roc_auc_score(y_va, p):>10.4f}{average_precision_score(y_va, p):>10.4f}")
    print(f"{'distance only':>16}{roc_auc_score(y_va, d_only):>10.4f}{average_precision_score(y_va, d_only):>10.4f}")

    gain = clf.get_booster().get_score(importance_type="gain")
    print("\nfeature gain:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:12} {gain.get(f'f{i}', 0.0):>10.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    # Force CPU prediction: inference runs inside parallel scoring workers, and
    # several processes contending for 4 GB of VRAM is slower than plain CPU on
    # batches this small.
    clf.set_params(device="cpu")
    joblib.dump({"model": clf, "features": FEATURE_NAMES}, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
