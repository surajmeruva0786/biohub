"""Removing predicted nodes that cost more than they earn.

The adjusted edge Jaccard scales the raw Jaccard by
``1 - 0.1 * (T_pred - T_true) / T_true``, so every predicted node carries a
small fixed cost regardless of whether it participates in a correct link. A
node with no incident edges can never contribute a true positive, so its cost
is pure loss -- and worse, per-timepoint node matching is one-to-one, so an
isolated detection sitting near a ground-truth cell can absorb that cell's
match and leave the *linked* detection unmatched, converting a would-be true
positive into a false negative.

Both pruners here therefore drop nodes and any edges incident on them.
"""

from __future__ import annotations

import numpy as np


def _renumber(
    coords: np.ndarray, times: np.ndarray, edges: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict to *keep* (a boolean mask over nodes) and renumber ids 1-based."""
    new_index = np.full(len(coords), -1, dtype=np.int64)
    new_index[keep] = np.arange(keep.sum(), dtype=np.int64)

    if len(edges):
        src, dst = edges[:, 0] - 1, edges[:, 1] - 1
        alive = keep[src] & keep[dst]
        edges = np.stack([new_index[src[alive]], new_index[dst[alive]]], axis=1) + 1
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    return coords[keep], times[keep], edges.astype(np.int64)


def _components(n: int, edges: np.ndarray) -> np.ndarray:
    """Weakly connected component label per node, via union-find."""
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in edges:
        ra, rb = find(int(s) - 1), find(int(d) - 1)
        if ra != rb:
            parent[rb] = ra

    return np.array([find(i) for i in range(n)], dtype=np.int64)


def prune_isolated(
    coords: np.ndarray, times: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop nodes with no incident edge in either direction."""
    if len(coords) == 0:
        return coords, times, edges

    keep = np.zeros(len(coords), dtype=bool)
    if len(edges):
        keep[edges[:, 0] - 1] = True
        keep[edges[:, 1] - 1] = True
    return _renumber(coords, times, edges, keep)


def prune_short_tracks(
    coords: np.ndarray,
    times: np.ndarray,
    edges: np.ndarray,
    min_len: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop whole tracks shorter than *min_len* nodes.

    A track that survives only two or three frames is far more likely to be a
    noise blob that happened to sit near another noise blob than a real cell,
    which persists for the length of the movie. Pruning by component keeps
    lineages intact: a dividing cell's parent and both daughters form one
    component and are kept or dropped together.
    """
    if len(coords) == 0 or min_len <= 1:
        return coords, times, edges

    labels = _components(len(coords), edges)
    sizes = np.bincount(labels, minlength=len(coords))
    keep = sizes[labels] >= min_len
    return _renumber(coords, times, edges, keep)
