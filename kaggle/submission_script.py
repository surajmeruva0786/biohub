"""Biohub cell tracking -- inference for Kaggle submission.

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


# --- tuned settings, from scripts/experiment.py and sweep_recall.py ---
SIGMA_UM = 0.6
SEPARATION_UM = 2.5
PERCENTILE = 90.0
BACKGROUND_UM = 4.0
MAX_LINK_UM = 5.0
MIN_TRACK_LEN = 0
DIVISIONS = False


# ==========================================================================
# io.py
# ==========================================================================

"""Loading competition volumes and ground-truth graphs.

Volumes are Zarr v3 arrays of shape ``(T, Z, Y, X)`` chunked one timepoint per
chunk, so timepoints are read individually rather than materialising the whole
81 GB tree. See ``DATASET.md`` for the verified layout.
"""




# Physical voxel size in µm, in (z, y, x) order. The metric scales centroid
# distances by this before applying its 7 µm matching threshold.
SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class Sample:
    """One dataset: an image volume and, for train, its ground-truth graph."""

    name: str
    zarr_path: Path
    geff_path: Path | None

    @property
    def has_gt(self) -> bool:
        return self.geff_path is not None


def list_samples(data_dir: Path | str, require_gt: bool = False) -> list[Sample]:
    """List every ``.zarr`` sample in *data_dir*, pairing each with its ``.geff``."""
    data_dir = Path(data_dir)
    samples = []
    for zp in sorted(data_dir.glob("*.zarr")):
        gp = data_dir / f"{zp.stem}.geff"
        if not gp.exists():
            gp = None
        if require_gt and gp is None:
            continue
        samples.append(Sample(name=zp.stem, zarr_path=zp, geff_path=gp))
    return samples


def embryo_of(name: str) -> str:
    """Embryo id from a sample name: ``44b6_0113de3b`` -> ``44b6``."""
    return name.split("_", 1)[0]


def select_samples(samples: list[Sample], limit: int | None = None) -> list[Sample]:
    """Take *limit* samples spread evenly across embryos, not the sorted prefix.

    Sample names sort by embryo, so a plain prefix of the training set is drawn
    entirely from one embryo -- 71 ``44b6`` samples come before any of the 128
    ``6bba`` ones. Train and test are embryo-disjoint in the real split, so a
    single-embryo evaluation measures exactly the thing that does not transfer:
    it would report how well a setting fits one animal, while the leaderboard
    asks how well it fits an unseen one.

    Round-robin over embryos instead, so any prefix of the result is balanced.
    """
    by_embryo: dict[str, list[Sample]] = {}
    for s in samples:
        by_embryo.setdefault(embryo_of(s.name), []).append(s)

    out: list[Sample] = []
    for row in zip(*(by_embryo[k] for k in sorted(by_embryo))):
        out.extend(row)
    # Embryos with more samples than the smallest have a tail zip() drops.
    seen = {s.name for s in out}
    out.extend(s for s in samples if s.name not in seen)

    return out[:limit] if limit else out


def open_volume(zarr_path: Path | str) -> zarr.Array:
    """Open the ``0/`` array of a sample volume without reading any chunks."""
    return zarr.open_group(str(zarr_path), mode="r")["0"]


def read_timepoint(volume: zarr.Array, t: int) -> np.ndarray:
    """Read a single ``(Z, Y, X)`` timepoint as float32."""
    return np.asarray(volume[t], dtype=np.float32)


def read_gt_graph(geff_path: Path | str) -> dict[str, np.ndarray]:
    """Read a ``.geff`` into flat arrays: ``ids``, ``t``, ``z``, ``y``, ``x``, ``edges``."""
    g = zarr.open(str(geff_path), mode="r")
    return {
        "ids": g["nodes/ids"][:],
        "t": g["nodes/props/t/values"][:],
        "z": g["nodes/props/z/values"][:],
        "y": g["nodes/props/y/values"][:],
        "x": g["nodes/props/x/values"][:],
        "edges": g["edges/ids"][:],
    }


def estimated_n_nodes(geff_path: Path | str) -> float:
    """Read ``estimated_number_of_nodes`` — the metric's ``T_true`` penalty target.

    The ground truth annotates well under 1% of cells, so this estimate (not the
    labelled node count) is what the over-prediction penalty compares against.
    """
    g = zarr.open(str(geff_path), mode="r")
    extra = g.attrs.get("geff", {}).get("extra", {}) or {}
    val = extra.get("estimated_number_of_nodes")
    return float(val) if val is not None else float("nan")


# ==========================================================================
# detect_gpu.py
# ==========================================================================

"""CUDA backend for detection, via PyTorch.

Detection dominates the pipeline's runtime -- roughly a second per timepoint on
CPU, so ~100 s per sample and hours across the training set -- and all of that
time goes into two separable, embarrassingly parallel stencils: a
difference-of-Gaussians and a local-maximum filter. Both are a direct fit for
the GPU.

Sized for the 4 GB Quadro P1000 this runs on: one ``(64, 256, 256)`` float32
timepoint is 16 MB, and separable 1D convolutions mean the working set stays a
small multiple of that rather than materialising a 3D kernel. Frames are
streamed one at a time, so peak memory does not grow with the movie length.

Numerically this matches the SciPy path closely (see ``scripts/bench_gpu.py``);
it is the same algorithm, not an approximation.
"""




_TRUNCATE = 4.0  # scipy.ndimage.gaussian_filter's default kernel extent


def cuda_available() -> bool:
    """True when a usable CUDA device is present."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _gaussian_kernel1d(sigma: float, device, dtype):
    """1D Gaussian matching scipy's radius convention (``truncate=4.0``)."""
    import torch

    radius = max(1, int(_TRUNCATE * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur3d(vol, sigmas):
    """Separable 3D Gaussian with edge replication, matching ``mode="nearest"``.

    Applied as three 1D passes rather than one 3D kernel: for the background
    sigma (~10 voxels in x/y) the 3D kernel would be over 80^3 taps, while the
    separable form is 3 x 81.
    """
    import torch
    import torch.nn.functional as F

    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        k = _gaussian_kernel1d(sigma, vol.device, vol.dtype)
        r = (len(k) - 1) // 2

        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = len(k)
        weight = k.view(shape)

        pad = [0, 0, 0, 0, 0, 0]  # (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        pad[2 * (2 - axis)] = r
        pad[2 * (2 - axis) + 1] = r
        vol = F.pad(vol, pad, mode="replicate")
        vol = F.conv3d(vol, weight)
    return vol


def detect_timepoint_gpu(
    vol: np.ndarray,
    sigma_um: float = 1.0,
    min_separation_um: float = 2.5,
    intensity_percentile: float = 90.0,
    background_um: float = 4.0,
    max_cells: int | None = None,
    device: str = "cuda",
) -> np.ndarray:
    """GPU equivalent of :func:`biohub.detect.detect_timepoint`.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` integer ``(z, y, x)`` voxel coordinates, strongest first.
    """
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(np.ascontiguousarray(vol), dtype=torch.float32, device=device)
    t = t[None, None]

    fine = _gaussian_blur3d(t, [sigma_um / s for s in SCALE])
    coarse = _gaussian_blur3d(t, [background_um / s for s in SCALE])
    dog = (fine - coarse)[0, 0]
    del t, fine, coarse

    # Local-maximum test. Same flat-ellipsoid footprint as the CPU path: wide in
    # x/y, only a few voxels tall in the 4x coarser z.
    size = tuple(max(3, int(round(2 * min_separation_um / s)) | 1) for s in SCALE)
    pad = tuple(s // 2 for s in size)
    padded = F.pad(
        dog[None, None], (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]), mode="replicate"
    )
    maxima = F.max_pool3d(padded, kernel_size=size, stride=1)[0, 0]

    # torch.quantile caps out around 16M elements; a 64x256x256 frame is 4.2M,
    # but fall back to sorting if a larger volume ever arrives.
    flat = dog.reshape(-1)
    q = intensity_percentile / 100.0
    if flat.numel() <= 16_000_000:
        thresh = torch.quantile(flat, q)
    else:
        thresh = torch.kthvalue(flat, max(1, int(q * flat.numel()))).values

    is_peak = (dog == maxima) & (dog > thresh)
    idx = torch.nonzero(is_peak, as_tuple=False)
    if idx.numel() == 0:
        return np.zeros((0, 3), dtype=np.int64)

    vals = dog[is_peak]
    order = torch.argsort(vals, descending=True)
    if max_cells is not None:
        order = order[:max_cells]
    return idx[order].to(torch.int64).cpu().numpy()


def detect_volume_gpu(
    volume,
    n_timepoints: int,
    per_frame_budget: int | None = None,
    device: str = "cuda",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Run :func:`detect_timepoint_gpu` over every timepoint of a sample.

    Timepoints are zstd-compressed one chunk each, so decompression is a real
    cost next to a 0.15 s GPU pass. A single prefetch thread reads frame
    ``t + 1`` while the GPU works on ``t``, which hides it -- the two stages use
    different resources and the CPU side releases the GIL inside blosc.
    """


    all_coords, all_t = [], []
    with ThreadPoolExecutor(max_workers=1) as io:
        pending = io.submit(read_timepoint, volume, 0)
        for t in range(n_timepoints):
            vol = pending.result()
            if t + 1 < n_timepoints:
                pending = io.submit(read_timepoint, volume, t + 1)
            c = detect_timepoint_gpu(
                vol, max_cells=per_frame_budget, device=device, **kwargs
            )
            all_coords.append(c)
            all_t.append(np.full(len(c), t, dtype=np.int64))

    if not all_coords:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(all_coords), np.concatenate(all_t)


# ==========================================================================
# detect.py
# ==========================================================================

"""Cell detection: bright-blob local maxima in anisotropic 3D volumes.

The voxel grid is strongly anisotropic (z is 4x coarser than x/y), so the
smoothing kernel and the peak-separation footprint are both defined in µm and
converted to voxels per axis.
"""





def _sigma_voxels(sigma_um: float) -> tuple[float, float, float]:
    """Convert an isotropic µm smoothing width into per-axis voxel sigmas."""
    return tuple(sigma_um / s for s in SCALE)


def _dog(vol: np.ndarray, sigma_um: float, background_um: float) -> np.ndarray:
    """Difference-of-Gaussians blob response: cell-scale detail minus background."""
    fine = ndi.gaussian_filter(vol, sigma=_sigma_voxels(sigma_um), mode="nearest")
    coarse = ndi.gaussian_filter(vol, sigma=_sigma_voxels(background_um), mode="nearest")
    return fine - coarse


def detect_timepoint(
    vol: np.ndarray,
    sigma_um: float = 1.0,
    min_separation_um: float = 2.5,
    intensity_percentile: float = 90.0,
    background_um: float = 4.0,
    max_cells: int | None = None,
) -> np.ndarray:
    """Detect cell centroids in one ``(Z, Y, X)`` volume.

    Uses a difference-of-Gaussians response rather than raw intensity. Samples
    differ hugely in background level -- one volume's cell centres can sit below
    another's 80th intensity percentile -- so thresholding absolute brightness
    silently drops real cells. DoG measures local blob contrast instead, which
    is comparable across samples.

    Keeps voxels that are the maximum in a *min_separation_um* neighbourhood and
    above *intensity_percentile* of the response, then returns the strongest
    *max_cells* of them.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(N, 3)`` with ``(z, y, x)`` voxel coordinates,
        ordered strongest-response first.
    """
    smooth = _dog(vol, sigma_um, background_um)

    # Peak separation footprint, sized in voxels per axis. Anisotropy makes this
    # a flat ellipsoid: wide in x/y, only a couple of voxels tall in z.
    size = tuple(max(3, int(round(2 * min_separation_um / s)) | 1) for s in SCALE)
    maxima = ndi.maximum_filter(smooth, size=size, mode="nearest")
    is_peak = smooth == maxima

    thresh = np.percentile(smooth, intensity_percentile)
    is_peak &= smooth > thresh

    coords = np.argwhere(is_peak)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    order = np.argsort(smooth[tuple(coords.T)])[::-1]
    coords = coords[order]
    if max_cells is not None:
        coords = coords[:max_cells]
    return coords.astype(np.int64)


def detect_volume(
    volume,
    n_timepoints: int,
    per_frame_budget: int | None = None,
    device: str = "auto",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Run :func:`detect_timepoint` over every timepoint of a sample.

    Parameters
    ----------
    device
        ``"auto"`` uses CUDA when a device is present and falls back to SciPy
        otherwise; ``"cpu"`` forces SciPy. The two backends are verified to
        produce identical detections (``scripts/bench_gpu.py``), so cached
        results from either are interchangeable.

    Returns
    -------
    (coords, times)
        ``coords`` is ``(N, 3)`` of ``(z, y, x)``; ``times`` is ``(N,)`` of ``t``.
    """

    if device != "cpu":

        if cuda_available():
            return detect_volume_gpu(
                volume, n_timepoints, per_frame_budget=per_frame_budget, **kwargs
            )
        if device != "auto":
            raise RuntimeError(f"device={device!r} requested but CUDA is unavailable")

    all_coords, all_t = [], []
    for t in range(n_timepoints):
        vol = read_timepoint(volume, t)
        c = detect_timepoint(vol, max_cells=per_frame_budget, **kwargs)
        all_coords.append(c)
        all_t.append(np.full(len(c), t, dtype=np.int64))

    if not all_coords:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(all_coords), np.concatenate(all_t)


# ==========================================================================
# features.py
# ==========================================================================

"""Candidate-link features for learned edge scoring.

Pure distance is a weak linking cost: in dense regions the correct partner and a
neighbouring cell sit at similar range. These features add the context that
disambiguates them -- how much better the candidate is than the runner-up,
whether the preference is mutual, and whether the implied displacement agrees
with what nearby cells are doing.

All features are geometric, computed from detection coordinates alone, so they
need no image access at scoring time.
"""



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


# ==========================================================================
# track.py
# ==========================================================================

"""Frame-to-frame linking by optimal assignment on physical centroid distance.

The metric only credits edges that span exactly ``t -> t+1`` and caps a node's
out-degree at 2, so links are formed strictly between consecutive frames.
"""





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


# ==========================================================================
# prune.py
# ==========================================================================

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


# ==========================================================================
# divide.py
# ==========================================================================

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


# ==========================================================================
# pipeline.py
# ==========================================================================

"""The end-to-end graph-building pipeline, as one configurable function.

Detection is expensive and configuration-independent once its own parameters
are fixed, so it stays outside: everything here takes a detection set and turns
it into a tracking graph. That split is what lets the experiment harness try
dozens of linking and pruning settings against one cached detection run.
"""






@dataclass(frozen=True)
class Config:
    """One pipeline setting. Frozen so variants are built with :meth:`with_`."""

    budget: int = 0  # detections kept per frame, strongest first; 0 = keep all
    max_link_um: float = 6.0
    compensate_drift: bool = True
    gap_closing: bool = True
    gap_factor: float = 2.0
    prune_isolated: bool = False
    min_track_len: int = 0
    divisions: bool = False
    division_max_um: float = 6.0
    division_ratio: float = 1.0
    model_path: str | None = None
    model_weight: float = 1.0

    def with_(self, **kwargs) -> "Config":
        return replace(self, **kwargs)


def apply_budget(
    coords: np.ndarray, times: np.ndarray, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the *budget* strongest detections per frame.

    ``detect_timepoint`` already returns each frame strongest-response first, so
    a density setting is a slice rather than a re-detection -- which is what
    makes sweeping it cheap against a fixed cache.
    """
    keep = []
    for t in np.unique(times):
        idx = np.flatnonzero(times == t)  # already strongest-first within a frame
        keep.append(idx[:budget])
    keep = np.concatenate(keep)
    return coords[keep], times[keep]


_MODEL_CACHE: dict[str, object] = {}


def load_model(path: str | Path):
    """Load and memoise a learned linker, so sweeps don't re-read it per sample."""
    key = str(path)
    if key not in _MODEL_CACHE:
        import joblib

        _MODEL_CACHE[key] = joblib.load(key)
    return _MODEL_CACHE[key]


def run(
    coords: np.ndarray, times: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a tracking graph from a detection set.

    Stage order is deliberate. Gap closing runs before pruning so that a track
    broken by one missed frame is repaired into a single long track rather than
    two short ones that the length filter would then discard. Divisions are
    added last, on the pruned graph, so forks are only proposed on tracks that
    survived.

    Returns
    -------
    (coords, times, edges)
        Node ids are implicitly ``1..len(coords)`` aligned with *coords*.
    """
    model = load_model(cfg.model_path) if cfg.model_path else None

    if cfg.budget > 0:
        coords, times = apply_budget(coords, times, cfg.budget)

    _, edges = build_graph(
        coords,
        times,
        max_link_um=cfg.max_link_um,
        compensate_drift=cfg.compensate_drift,
        model=model,
        model_weight=cfg.model_weight,
    )

    if cfg.gap_closing:
        coords, times, edges = close_gaps(
            coords, times, edges,
            max_link_um=cfg.max_link_um,
            gap_factor=cfg.gap_factor,
        )

    if cfg.prune_isolated:
        coords, times, edges = prune_isolated(coords, times, edges)

    if cfg.min_track_len > 1:
        coords, times, edges = prune_short_tracks(
            coords, times, edges, min_len=cfg.min_track_len
        )

    if cfg.divisions:
        edges = add_divisions(
            coords, times, edges,
            max_um=cfg.division_max_um,
            ratio=cfg.division_ratio,
        )

    return coords, times, edges


# ==========================================================================
# submit.py
# ==========================================================================

"""Submission CSV writer.

The required format interleaves two row types per dataset:

* ``node`` rows carry ``node_id, t, z, y, x`` with ``source_id = target_id = -1``
* ``edge`` rows carry ``source_id, target_id`` with ``node_id, t, z, y, x = -1``

``id`` is a throwaway consecutive index. Every dataset in ``test/`` must appear.
"""




COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]


class SubmissionWriter:
    """Streams submission rows to disk, keeping the ``id`` column consecutive."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._fh = None
        self._writer = None
        self._next_id = 0

    def __enter__(self) -> "SubmissionWriter":
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(COLUMNS)
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()

    def add_sample(
        self,
        dataset: str,
        coords: np.ndarray,
        times: np.ndarray,
        node_ids: np.ndarray,
        edges: np.ndarray,
    ) -> None:
        """Write all node and edge rows for one dataset.

        Node ids only need to be unique within a dataset, since the metric
        rebuilds each dataset's graph independently.
        """
        w = self._writer
        for nid, t, (z, y, x) in zip(node_ids, times, coords):
            w.writerow(
                [self._next_id, dataset, "node", int(nid), int(t),
                 int(z), int(y), int(x), -1, -1]
            )
            self._next_id += 1

        for s, t in edges:
            w.writerow(
                [self._next_id, dataset, "edge", -1, -1, -1, -1, -1,
                 int(s), int(t)]
            )
            self._next_id += 1



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
    print(f"{CONFIG}\n{DETECT_KWARGS}\n", flush=True)

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

    print(f"\nwrote {OUT} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
