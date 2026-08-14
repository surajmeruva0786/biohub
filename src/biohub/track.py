"""Frame-to-frame linking by optimal assignment on physical centroid distance.

The metric only credits edges that span exactly ``t -> t+1`` and caps a node's
out-degree at 2, so links are formed strictly between consecutive frames.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .io import SCALE


def to_physical(coords: np.ndarray) -> np.ndarray:
    """Scale ``(N, 3)`` voxel ``(z, y, x)`` coordinates to µm."""
    return coords.astype(np.float64) * np.asarray(SCALE)


def _assign(src: np.ndarray, dst: np.ndarray, max_link_um: float) -> np.ndarray:
    """Optimal one-to-one assignment between two µm point sets, gated by distance."""
    cost = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=-1)
    gated = np.where(cost > max_link_um, 1e6, cost)
    rows, cols = linear_sum_assignment(gated)
    keep = cost[rows, cols] <= max_link_um
    return np.stack([rows[keep], cols[keep]], axis=1).astype(np.int64)


def estimate_drift(
    src: np.ndarray, dst: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    """Median displacement of matched pairs — a robust global motion estimate."""
    if len(pairs) < 3:
        return np.zeros(3)
    return np.median(dst[pairs[:, 1]] - src[pairs[:, 0]], axis=0)


def _model_cost(
    src: np.ndarray, dst: np.ndarray, max_link_um: float, model, weight: float
) -> np.ndarray:
    """Dense assignment cost from a learned link probability.

    The classifier scores only in-gate candidate pairs, so pairs it never saw
    keep the sentinel cost and stay unlinkable. Cost is
    ``-log(p)`` blended with the raw distance: probability decides which of two
    plausible partners wins, distance still breaks ties among pairs the model
    finds equally likely.
    """
    from .features import build_candidates

    pairs, feats = build_candidates(src, dst, max_link_um)
    cost = np.full((len(src), len(dst)), np.inf)
    if len(pairs) == 0:
        return cost

    p = np.clip(model["model"].predict_proba(feats)[:, 1], 1e-6, 1.0)
    d = np.linalg.norm(src[pairs[:, 0]] - dst[pairs[:, 1]], axis=1)
    cost[pairs[:, 0], pairs[:, 1]] = weight * -np.log(p) + (1.0 - weight) * d
    return cost


def _assign_cost(cost: np.ndarray) -> np.ndarray:
    """Optimal assignment over a cost matrix where ``inf`` marks a forbidden pair."""
    finite = np.isfinite(cost)
    if not finite.any():
        return np.zeros((0, 2), dtype=np.int64)

    gated = np.where(finite, cost, 1e6)
    rows, cols = linear_sum_assignment(gated)
    keep = finite[rows, cols]
    return np.stack([rows[keep], cols[keep]], axis=1).astype(np.int64)


def link_consecutive(
    src_coords: np.ndarray,
    dst_coords: np.ndarray,
    max_link_um: float = 6.0,
    compensate_drift: bool = True,
    model=None,
    model_weight: float = 1.0,
) -> np.ndarray:
    """Match detections in one frame to the next by minimum total distance.

    Candidate pairs beyond *max_link_um* are never linked. Solves a rectangular
    assignment restricted to in-gate pairs, so every match is one-to-one.

    When *compensate_drift* is set, a first assignment estimates the frame's
    median displacement and the second runs with that global motion removed.
    Embryo-wide drift otherwise consumes the distance budget that should be
    discriminating between neighbouring cells.

    Returns
    -------
    np.ndarray
        ``(M, 2)`` array of ``(src_index, dst_index)`` pairs.
    """
    if len(src_coords) == 0 or len(dst_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    src = to_physical(src_coords)
    dst = to_physical(dst_coords)

    pairs = _assign(src, dst, max_link_um)
    if compensate_drift:
        drift = estimate_drift(src, dst, pairs)
        if np.any(drift):
            src = src + drift
            pairs = _assign(src, dst, max_link_um)

    if model is not None:
        pairs = _assign_cost(_model_cost(src, dst, max_link_um, model, model_weight))
    return pairs


def build_graph(
    coords: np.ndarray,
    times: np.ndarray,
    max_link_um: float = 6.0,
    compensate_drift: bool = True,
    model=None,
    model_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Link a full detection set into a tracking graph.

    Parameters
    ----------
    coords
        ``(N, 3)`` voxel ``(z, y, x)`` centroids.
    times
        ``(N,)`` timepoint per detection.

    Returns
    -------
    (node_ids, edges)
        ``node_ids`` is ``(N,)`` of 1-based ids aligned with *coords*; ``edges``
        is ``(M, 2)`` of ``(source_id, target_id)``.
    """
    node_ids = np.arange(1, len(coords) + 1, dtype=np.int64)
    if len(coords) == 0:
        return node_ids, np.zeros((0, 2), dtype=np.int64)

    by_t: dict[int, np.ndarray] = {
        t: np.flatnonzero(times == t) for t in np.unique(times)
    }

    edges = []
    for t in sorted(by_t):
        if t + 1 not in by_t:
            continue
        src_idx, dst_idx = by_t[t], by_t[t + 1]
        pairs = link_consecutive(
            coords[src_idx], coords[dst_idx], max_link_um, compensate_drift,
            model=model, model_weight=model_weight,
        )
        if len(pairs):
            edges.append(
                np.stack(
                    [node_ids[src_idx[pairs[:, 0]]], node_ids[dst_idx[pairs[:, 1]]]],
                    axis=1,
                )
            )

    if not edges:
        return node_ids, np.zeros((0, 2), dtype=np.int64)
    return node_ids, np.concatenate(edges)


def close_gaps(
    coords: np.ndarray,
    times: np.ndarray,
    edges: np.ndarray,
    max_link_um: float = 7.0,
    gap_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repair one-frame detection gaps by inserting interpolated nodes.

    The metric only credits edges spanning exactly ``t -> t+1``, so a cell missed
    in a single frame costs *two* edges rather than one -- a break cannot be
    bridged directly. Instead, a track ending at ``t`` and one resuming at
    ``t+2`` within ``gap_factor * max_link_um`` are joined by a synthetic node at
    their midpoint in frame ``t+1``, yielding two well-formed edges.

    Returns
    -------
    (coords, times, edges)
        Extended detection set and edge list, with ids renumbered 1-based over
        the combined nodes.
    """
    if len(coords) == 0:
        return coords, times, edges

    n = len(coords)
    has_out = np.zeros(n, dtype=bool)
    has_in = np.zeros(n, dtype=bool)
    if len(edges):
        has_out[edges[:, 0] - 1] = True
        has_in[edges[:, 1] - 1] = True

    by_t: dict[int, np.ndarray] = {
        int(t): np.flatnonzero(times == t) for t in np.unique(times)
    }

    new_coords, new_times, new_edges = [], [], [edges] if len(edges) else []
    next_id = n + 1

    for t in sorted(by_t):
        if t + 2 not in by_t:
            continue
        # Track ends at t (no successor) and resumes at t+2 (no predecessor).
        tail = by_t[t][~has_out[by_t[t]]]
        head = by_t[t + 2][~has_in[by_t[t + 2]]]
        if len(tail) == 0 or len(head) == 0:
            continue

        pairs = _assign(
            to_physical(coords[tail]),
            to_physical(coords[head]),
            max_link_um * gap_factor,
        )
        for si, di in pairs:
            a, b = tail[si], head[di]
            mid = ((coords[a].astype(np.float64) + coords[b]) / 2.0).round()
            new_coords.append(mid)
            new_times.append(t + 1)
            new_edges.append(
                np.array([[a + 1, next_id], [next_id, b + 1]], dtype=np.int64)
            )
            next_id += 1

    if not new_coords:
        return coords, times, edges

    coords = np.concatenate([coords, np.asarray(new_coords, dtype=np.int64)])
    times = np.concatenate([times, np.asarray(new_times, dtype=np.int64)])
    edges = np.concatenate(new_edges)
    return coords, times, edges


def gt_displacement_stats(gt: dict) -> dict[str, float]:
    """Measure how far ground-truth cells travel per frame, in µm.

    Used to calibrate ``max_link_um``: the gate must cover real motion without
    admitting links to neighbouring cells.
    """
    id_to_idx = {int(i): k for k, i in enumerate(gt["ids"])}
    pos = np.stack(
        [gt["z"], gt["y"], gt["x"]], axis=1
    ).astype(np.float64) * np.asarray(SCALE)

    d = []
    for s, tgt in gt["edges"]:
        si, ti = id_to_idx.get(int(s)), id_to_idx.get(int(tgt))
        if si is None or ti is None:
            continue
        d.append(np.linalg.norm(pos[ti] - pos[si]))

    if not d:
        return {}
    d = np.asarray(d)
    return {
        "mean": float(d.mean()),
        "p50": float(np.percentile(d, 50)),
        "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
        "n": int(len(d)),
    }
