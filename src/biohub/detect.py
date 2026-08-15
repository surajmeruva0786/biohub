"""Cell detection: bright-blob local maxima in anisotropic 3D volumes.

The voxel grid is strongly anisotropic (z is 4x coarser than x/y), so the
smoothing kernel and the peak-separation footprint are both defined in µm and
converted to voxels per axis.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .io import SCALE


def _sigma_voxels(sigma_um: float) -> tuple[float, float, float]:
    """Convert an isotropic µm smoothing width into per-axis voxel sigmas."""
    return tuple(sigma_um / s for s in SCALE)


def _dog(vol: np.ndarray, sigma_um: float, background_um: float) -> np.ndarray:
    """Difference-of-Gaussians blob response: cell-scale detail minus background."""
    fine = ndi.gaussian_filter(vol, sigma=_sigma_voxels(sigma_um), mode="nearest")
    coarse = ndi.gaussian_filter(vol, sigma=_sigma_voxels(background_um), mode="nearest")
    return fine - coarse


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: the split maximising between-class variance.

    Applied to *peak* responses rather than to every voxel. The voxel histogram
    is overwhelmingly background and its optimal split lands far below any cell,
    but local maxima are already a mix of two populations -- real cells and
    noise peaks -- which is exactly the shape Otsu assumes.
    """
    if len(values) < 2:
        return -np.inf
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0

    w = np.cumsum(hist)
    w_bg, w_fg = w[:-1], w[-1] - w[:-1]
    valid = (w_bg > 0) & (w_fg > 0)
    if not valid.any():
        return -np.inf

    cum = np.cumsum(hist * centres)
    mean_bg = np.where(w_bg > 0, cum[:-1] / np.maximum(w_bg, 1), 0.0)
    mean_fg = np.where(w_fg > 0, (cum[-1] - cum[:-1]) / np.maximum(w_fg, 1), 0.0)

    between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -np.inf
    return float(centres[int(np.argmax(between))])


def detect_timepoint(
    vol: np.ndarray,
    sigma_um: float = 1.0,
    min_separation_um: float = 2.5,
    intensity_percentile: float = 90.0,
    background_um: float = 4.0,
    max_cells: int | None = None,
    threshold: str = "percentile",
    peak_percentile: float = 0.0,
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

    if threshold == "percentile":
        is_peak &= smooth > np.percentile(smooth, intensity_percentile)

    coords = np.argwhere(is_peak)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    response = smooth[tuple(coords.T)]

    if threshold == "otsu":
        # Split the peak population, not the voxel population. A percentile over
        # voxels keeps a fixed *fraction* of the volume, so it returns roughly
        # the same count whatever the true cell count is -- which is how one
        # embryo ends up over-predicted 5.5x while another sits at 0.5x. Otsu
        # over peak responses adapts to how many peaks actually look like cells.
        keep = response > _otsu(response)
        coords, response = coords[keep], response[keep]
        if peak_percentile > 0 and len(response):
            keep = response > np.percentile(response, peak_percentile)
            coords, response = coords[keep], response[keep]
        if len(coords) == 0:
            return np.zeros((0, 3), dtype=np.int64)

    order = np.argsort(response)[::-1]
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
    from .io import read_timepoint

    if device != "cpu":
        from .detect_gpu import cuda_available, detect_volume_gpu

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
