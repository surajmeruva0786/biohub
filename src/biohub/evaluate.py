"""Local scoring against the official competition metric.

Wraps ``tracking_cellmot.metrics`` (vendored under ``reference/``, BSD-3) so
offline scores match the leaderboard definition exactly rather than a
reimplementation of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REFERENCE_SRC = Path(__file__).resolve().parents[2] / "reference" / "src"
if str(REFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(REFERENCE_SRC))


def build_td_graph(coords: np.ndarray, times: np.ndarray, edges: np.ndarray):
    """Build a tracksdata graph from detections and 1-based ``(source, target)`` edges."""
    import tracksdata as td

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    assigned = graph.bulk_add_nodes(
        [
            {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
            for t, (z, y, x) in zip(times, coords)
        ]
    )
    # Our node ids are 1-based and dense, so index i maps to id i+1.
    if len(edges):
        graph.bulk_add_edges(
            [
                {"source_id": assigned[int(s) - 1], "target_id": assigned[int(t) - 1]}
                for s, t in edges
            ]
        )
    return graph


def score_sample(
    coords: np.ndarray,
    times: np.ndarray,
    edges: np.ndarray,
    geff_path: Path | str,
    n_total: float,
) -> dict:
    """Score one prediction against its ground-truth ``.geff``.

    Returns the official per-sample metric row, including ``adj_edge_jaccard``.
    """
    import tracksdata as td
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    from .io import SCALE

    gt = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    gt = gt[0] if isinstance(gt, tuple) else gt

    pred = build_td_graph(coords, times, edges)
    result = evaluate(pred, gt, scale=SCALE, max_distance=7.0)
    recall = node_recall(pred, gt)
    return per_sample_metrics(result, n_total=n_total, node_recall=recall)


def summarise_rows(rows: list[dict]) -> dict:
    """Aggregate per-sample rows into the run-level score."""
    from tracking_cellmot.metrics import summarise

    return summarise(rows)
