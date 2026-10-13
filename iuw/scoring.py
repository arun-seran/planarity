"""Normalised score S = (J - J_random) / (J_reference - J_random), per profile and world."""

from __future__ import annotations

import math
from typing import Mapping, Optional

UNSCORED_MARGIN = 1.0  # a reference within 1 reward unit of random is unscored


def normalised_score(j: float, j_random: float, j_reference: float) -> Optional[float]:
    """Scores above 1 are kept. Returns None when the cell is unscored."""
    denom = j_reference - j_random
    if denom <= UNSCORED_MARGIN:
        return None
    return (j - j_random) / denom


class Scorer:
    """Looks up frozen references by "world/body" and normalises returns."""

    def __init__(self, references: Mapping[str, Mapping[str, float]]):
        self.references = references

    def reference(self, world: str, body: str) -> Mapping[str, float]:
        key = f"{world}/{body}"
        if key not in self.references:
            raise KeyError(f"no frozen reference for {key}; run `iuw certify --freeze`")
        return self.references[key]

    def score(self, world: str, body: str, j: float) -> Optional[float]:
        r = self.reference(world, body)
        return normalised_score(j, r["J_random"], r["J_reference"])

    def regret(self, world: str, body: str, j: float) -> float:
        """Per-episode regret against the reference return (learning cost unit)."""
        return self.reference(world, body)["J_reference"] - j


def mean_score(scores: list[Optional[float]]) -> float:
    vals = [s for s in scores if s is not None]
    return float(sum(vals) / len(vals)) if vals else math.nan
