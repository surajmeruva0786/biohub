"""Submission CSV writer.

The required format interleaves two row types per dataset:

* ``node`` rows carry ``node_id, t, z, y, x`` with ``source_id = target_id = -1``
* ``edge`` rows carry ``source_id, target_id`` with ``node_id, t, z, y, x = -1``

``id`` is a throwaway consecutive index. Every dataset in ``test/`` must appear.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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
