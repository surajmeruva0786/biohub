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


def link_consecutive(
    src_coords: np.ndarray,
    dst_coords: np.ndarray,
    max_link_um: float = 6.0,
) -> np.ndarray:
    """Match detections in one frame to the next by minimum total distance.

    Candidate pairs beyond *max_link_um* are never linked. Solves a rectangular
    assignment restricted to in-gate pairs, so every match is one-to-one.

    Returns
    -------
    np.ndarray
        ``(M, 2)`` array of ``(src_index, dst_index)`` pairs.
    """
    if len(src_coords) == 0 or len(dst_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    src = to_physical(src_coords)
    dst = to_physical(dst_coords)

    # Restrict the cost matrix to in-gate pairs; a dense N x N over ~400
    # detections is cheap, but the gate is what keeps assignments sensible.
    cost = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=-1)
    gated = cost.copy()
    gated[gated > max_link_um] = 1e6

    rows, cols = linear_sum_assignment(gated)
    keep = cost[rows, cols] <= max_link_um
    return np.stack([rows[keep], cols[keep]], axis=1).astype(np.int64)


def build_graph(
    coords: np.ndarray,
    times: np.ndarray,
    max_link_um: float = 6.0,
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
        pairs = link_consecutive(coords[src_idx], coords[dst_idx], max_link_um)
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
