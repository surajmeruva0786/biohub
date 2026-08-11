"""Candidate-link features for learned edge scoring.

Pure distance is a weak linking cost: in dense regions the correct partner and a
neighbouring cell sit at similar range. These features add the context that
disambiguates them -- how much better the candidate is than the runner-up,
whether the preference is mutual, and whether the implied displacement agrees
with what nearby cells are doing.

All features are geometric, computed from detection coordinates alone, so they
need no image access at scoring time.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

FEATURE_NAMES = [
    "dist",
    "dist_frac",
    "rank_fwd",
    "rank_bwd",
    "ratio_fwd",
    "ratio_bwd",
    "mutual",
    "n_cand_fwd",
    "n_cand_bwd",
    "coherence",
]


def _neighbour_table(
    src: np.ndarray, dst: np.ndarray, max_link_um: float, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """k nearest *dst* points for each *src* point, as (distances, indices).

    Missing neighbours come back as ``inf`` distance and an out-of-range index,
    which is how scipy reports fewer than *k* hits.
    """
    k = min(k, len(dst))
    tree = cKDTree(dst)
    d, idx = tree.query(src, k=k, distance_upper_bound=max_link_um)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    return d, idx


def _local_displacement(src: np.ndarray, dst: np.ndarray, radius_um: float = 12.0) -> np.ndarray:
    """Median displacement of each source point's spatial neighbourhood.

    Each source point is provisionally matched to its nearest target; the median
    of those provisional displacements over nearby sources is a local motion
    estimate that a correct link should broadly agree with.
    """
    if len(dst) == 0:
        return np.zeros_like(src)

    nn_d, nn_i = cKDTree(dst).query(src, k=1)
    valid = np.isfinite(nn_d)
    disp = np.zeros_like(src)
    disp[valid] = dst[nn_i[valid]] - src[valid]

    # Average each point's provisional displacement over its spatial neighbours.
    local = np.zeros_like(src)
    tree = cKDTree(src)
    for a, nbrs in enumerate(tree.query_ball_point(src, r=radius_um)):
        nbrs = [n for n in nbrs if valid[n]]
        local[a] = np.median(disp[nbrs], axis=0) if nbrs else 0.0
    return local


def build_candidates(
    src: np.ndarray,
    dst: np.ndarray,
    max_link_um: float = 6.0,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate candidate links between two frames and featurise them.

    Parameters
    ----------
    src, dst
        ``(N, 3)`` and ``(M, 3)`` centroid arrays in µm, drift already removed.

    Returns
    -------
    (pairs, features)
        ``pairs`` is ``(P, 2)`` of ``(src_index, dst_index)``; ``features`` is
        ``(P, len(FEATURE_NAMES))``.
    """
    if len(src) == 0 or len(dst) == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0, len(FEATURE_NAMES)))

    d_fwd, i_fwd = _neighbour_table(src, dst, max_link_um, k)
    d_bwd, i_bwd = _neighbour_table(dst, src, max_link_um, k)

    # Rank of each source among a target's neighbours, for the reverse view.
    bwd_rank: dict[tuple[int, int], int] = {}
    bwd_second = np.full(len(dst), np.inf)
    for j in range(len(dst)):
        row_d, row_i = d_bwd[j], i_bwd[j]
        finite = np.flatnonzero(np.isfinite(row_d))
        for r, c in enumerate(finite):
            bwd_rank[(int(row_i[c]), j)] = r
        if len(finite) > 1:
            bwd_second[j] = row_d[finite[1]]

    n_cand_bwd = np.isfinite(d_bwd).sum(axis=1)
    local = _local_displacement(src, dst)

    pairs, feats = [], []
    for i in range(len(src)):
        row_d, row_i = d_fwd[i], i_fwd[i]
        finite = np.flatnonzero(np.isfinite(row_d))
        if len(finite) == 0:
            continue
        second = row_d[finite[1]] if len(finite) > 1 else np.inf
        n_fwd = len(finite)

        for r, c in enumerate(finite):
            j = int(row_i[c])
            d = float(row_d[c])
            rb = bwd_rank.get((i, j), k)
            coherence = float(np.linalg.norm((dst[j] - src[i]) - local[i]))

            pairs.append((i, j))
            feats.append(
                [
                    d,
                    d / max_link_um,
                    float(r),
                    float(rb),
                    d / second if np.isfinite(second) else 0.0,
                    d / bwd_second[j] if np.isfinite(bwd_second[j]) else 0.0,
                    1.0 if (r == 0 and rb == 0) else 0.0,
                    float(n_fwd),
                    float(n_cand_bwd[j]),
                    coherence,
                ]
            )

    return (
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        np.asarray(feats, dtype=np.float64).reshape(-1, len(FEATURE_NAMES)),
    )
