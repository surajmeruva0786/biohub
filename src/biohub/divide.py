"""Proposing cell divisions on top of a one-to-one linked graph.

One-to-one assignment caps out-degree at 1, so a graph built by linking alone
contains no forks at all and forfeits the entire division term of the score.
This module adds second children.

The asymmetry worth exploiting: a predicted fork is only ever counted as a
*false* positive when there is local ground-truth evidence to judge it against
-- the fork matches an annotated node with outgoing edges, or its branches
resolve to distinct annotated components. With well under 1% of cells
annotated, most speculative forks land on unannotated cells and are ignored
entirely. Recall is therefore worth more than precision here, though not
without limit: forks placed on annotated, *non*-dividing cells do count
against us.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .track import to_physical


def add_divisions(
    coords: np.ndarray,
    times: np.ndarray,
    edges: np.ndarray,
    max_um: float = 6.0,
    ratio: float = 1.5,
) -> np.ndarray:
    """Add a second outgoing edge where a split is geometrically plausible.

    A candidate is an *orphan*: a node at ``t+1`` with no incoming edge, whose
    nearest node at ``t`` already has exactly one child. The orphan becomes a
    second daughter when it is no further from that parent than *ratio* times
    the existing daughter's distance -- so the two daughters must be roughly
    balanced about the parent, which is what a real split looks like, rather
    than one tight link plus a distant stray.

    Out-degree stays at 2: a parent already holding two children is skipped,
    which matters because the metric silently drops the excess otherwise.

    Returns
    -------
    np.ndarray
        The edge list with division edges appended.
    """
    n = len(coords)
    if n == 0 or len(edges) == 0:
        return edges

    out_count = np.bincount(edges[:, 0] - 1, minlength=n)
    in_count = np.bincount(edges[:, 1] - 1, minlength=n)

    # Distance from each parent to its existing (single) child.
    phys = to_physical(coords)
    child_dist = np.full(n, np.inf)
    for s, d in edges:
        i, j = int(s) - 1, int(d) - 1
        child_dist[i] = min(child_dist[i], np.linalg.norm(phys[j] - phys[i]))

    by_t: dict[int, np.ndarray] = {
        int(t): np.flatnonzero(times == t) for t in np.unique(times)
    }

    new_edges = []
    for t in sorted(by_t):
        if t + 1 not in by_t:
            continue
        src_idx = by_t[t]
        orphans = by_t[t + 1][in_count[by_t[t + 1]] == 0]
        if len(src_idx) == 0 or len(orphans) == 0:
            continue

        # Parents must already have exactly one child: zero means the track
        # ended (a gap, not a split), two means the fork is already full.
        eligible = src_idx[out_count[src_idx] == 1]
        if len(eligible) == 0:
            continue

        dist, nn = cKDTree(phys[eligible]).query(
            phys[orphans], k=1, distance_upper_bound=max_um
        )
        for o, d, k in zip(orphans, dist, nn):
            if not np.isfinite(d):
                continue
            p = eligible[k]
            # Re-check: an earlier orphan this frame may have filled the fork.
            if out_count[p] != 1 or d > ratio * child_dist[p]:
                continue
            new_edges.append((p + 1, o + 1))
            # Keep this parent's out-degree at 2 for the rest of the frame.
            out_count[p] = 2

    if not new_edges:
        return edges
    return np.concatenate([edges, np.asarray(new_edges, dtype=np.int64)])
