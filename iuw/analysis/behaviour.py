"""Behavioural measures: probe cadence, repeat runs, switching, rest, probe-following."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import PROBE, REST


def behaviour_profile(actions: Sequence[int], decisions_phase: Sequence[str] | None = None) -> dict:
    """Summarise one episode's actions (probe = 0, valves 1-4, rest = 5)."""
    cycles = sum(1 for a in actions if a != PROBE)
    probes = sum(1 for a in actions if a == PROBE)
    rests = sum(1 for a in actions if a == REST)
    acts = [a for a in actions if a not in (PROBE, REST)]
    repeats = sum(1 for x, y in zip(acts, acts[1:]) if x == y)
    runs, cur = [], 1
    for x, y in zip(acts, acts[1:]):
        if x == y:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    if acts:
        runs.append(cur)
    follow = total_after_probe = 0
    for i, a in enumerate(actions[:-1]):
        if a == PROBE:
            total_after_probe += 1
    return {
        "cycles": cycles,
        "probe_rate": probes / cycles if cycles else 0.0,
        "rest_rate": rests / cycles if cycles else 0.0,
        "repeat_rate": repeats / (len(acts) - 1) if len(acts) > 1 else 0.0,
        "switch_rate": 1 - repeats / (len(acts) - 1) if len(acts) > 1 else 0.0,
        "mean_run_length": float(np.mean(runs)) if runs else 0.0,
        "first_probe_cycles": _probe_cycles(actions)[:5],
    }


def follow_probe_rate(observed_probe_results: Sequence[int | None], actions: Sequence[int]) -> float:
    """Fraction of post-probe commits that actuate the reported valve."""
    hits = n = 0
    for pr, a in zip(observed_probe_results, actions):
        if pr is not None:
            n += 1
            hits += a == pr
    return hits / n if n else 0.0


def _probe_cycles(actions: Sequence[int]) -> list[int]:
    out, cycle = [], 1
    for a in actions:
        if a == PROBE:
            out.append(cycle)
        else:
            cycle += 1
    return out


def aggregate(profiles: Sequence[dict]) -> dict:
    keys = [k for k in profiles[0] if isinstance(profiles[0][k], (int, float))] if profiles else []
    return {k: float(np.mean([p[k] for p in profiles])) for k in keys}


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (nats) between two distributions over the same support."""
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
