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

from __future__ import annotations

import numpy as np

from .io import SCALE

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
    from concurrent.futures import ThreadPoolExecutor

    from .io import read_timepoint

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
