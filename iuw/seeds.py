"""Seed splits: disjoint, fixed, with a manifest hash and a one-time seal on the test split.

Training 128, development 16, test 32, all disjoint; a separate 200-seed pool
for notebook selection; the outer loop's discovery pool and replication
panel; and a pilot pool kept out of every result.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

SPLITS: dict[str, range] = {
    "train": range(0, 128),
    "dev": range(100_000, 100_016),
    "test": range(200_000, 200_032),
    "notebook": range(300_000, 300_200),
    "discovery": range(400_000, 401_000),
    "replication": range(500_000, 500_100),
    "pilot": range(900_000, 900_100),
}


def split(name: str) -> range:
    return SPLITS[name]


def manifest() -> dict:
    body = {k: [r.start, r.stop] for k, r in SPLITS.items()}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return {"splits": body, "sha256_16": digest}


def assert_disjoint() -> None:
    seen: set[int] = set()
    for name, r in SPLITS.items():
        s = set(r)
        if s & seen:
            raise AssertionError(f"split {name} overlaps another split")
        seen |= s


def training_batch(update: int, seeds_per_update: int = 8) -> list[tuple[int, int]]:
    """(seed, episode) pairs for one update; seeds cycle through the training split.

    ``update`` counts from 1. When a seed comes round again its episode index
    increases, so repeated seeds still get fresh disturbances.
    """
    train = SPLITS["train"]
    out = []
    for i in range(seeds_per_update):
        k = (update - 1) * seeds_per_update + i
        out.append((train[k % len(train)], k // len(train)))
    return out


class SealError(RuntimeError):
    pass


@dataclass
class Seal:
    """The sealed test is opened once per run; a second opening is refused."""

    path: Path

    def open(self, note: str = "") -> None:
        if self.path.exists():
            raise SealError(f"sealed test already opened: {self.path.read_text().strip()}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"opened_at": time.time(), "note": note}))

    @property
    def opened(self) -> bool:
        return self.path.exists()
