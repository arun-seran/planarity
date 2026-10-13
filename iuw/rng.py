"""Keyed, counter-based randomness.

Every random draw in the simulator is a pure function of
(stream, seed, episode, cycle, event type, extra). Two runs that share a
seed and episode therefore share every disturbance, whatever policy or body
they use, which is what lets A and B, Named and Masked, and successive
checkpoints be compared on paired episodes.

Streams keep world randomness (where the target is) apart from execution
randomness (whether a body carries out a command) and from policy sampling,
so changing one never shifts another.
"""

from __future__ import annotations

import hashlib
from enum import IntEnum

MASK64 = (1 << 64) - 1


class Stream(IntEnum):
    WORLD = 1
    EXECUTION = 2
    POLICY = 3
    MAPPING = 4
    SHUFFLE = 5
    SELECTION = 6


class Event(IntEnum):
    TARGET_INIT = 1
    TARGET_STAY = 2
    TARGET_DRAW = 3
    EXECUTE = 4
    PROBE_SHUFFLE = 5
    ACTION = 6
    MISC = 7


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def namespace_key(name: str) -> int:
    """Stable 64-bit key for a string namespace (independent of PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")


def keyed_bits(*keys: int) -> int:
    h = 0x6A09E667F3BCC908
    for k in keys:
        h = _splitmix64(h ^ (int(k) & MASK64))
    return h


def keyed_uniform(*keys: int) -> float:
    """Uniform float in [0, 1) determined entirely by the keys."""
    return (keyed_bits(*keys) >> 11) * (1.0 / (1 << 53))


DEFAULT_NAMESPACE = namespace_key("iuw/water-v1")


def draw(stream: Stream, seed: int, episode: int, cycle: int, event: Event, extra: int = 0,
         namespace: int = DEFAULT_NAMESPACE) -> float:
    return keyed_uniform(namespace, int(stream), seed, episode, cycle, int(event), extra)


def draw_index(n: int, stream: Stream, seed: int, episode: int, cycle: int, event: Event,
               extra: int = 0, namespace: int = DEFAULT_NAMESPACE) -> int:
    return min(n - 1, int(draw(stream, seed, episode, cycle, event, extra, namespace) * n))
