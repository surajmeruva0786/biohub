"""Loading competition volumes and ground-truth graphs.

Volumes are Zarr v3 arrays of shape ``(T, Z, Y, X)`` chunked one timepoint per
chunk, so timepoints are read individually rather than materialising the whole
81 GB tree. See ``DATASET.md`` for the verified layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

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
