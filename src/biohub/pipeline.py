"""The end-to-end graph-building pipeline, as one configurable function.

Detection is expensive and configuration-independent once its own parameters
are fixed, so it stays outside: everything here takes a detection set and turns
it into a tracking graph. That split is what lets the experiment harness try
dozens of linking and pruning settings against one cached detection run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .divide import add_divisions
from .prune import prune_isolated, prune_short_tracks
from .track import build_graph, close_gaps


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
