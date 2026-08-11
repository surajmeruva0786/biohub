"""Detect-and-link cell tracking for the Biohub competition.

Pipeline: :mod:`biohub.detect` finds cell centroids per timepoint,
:mod:`biohub.track` links them across consecutive frames,
:mod:`biohub.evaluate` scores against the official metric, and
:mod:`biohub.submit` writes the competition CSV.
"""

from .io import SCALE

__all__ = ["SCALE"]
