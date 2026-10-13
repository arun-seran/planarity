"""Statistics: interquartile means, bootstrap intervals clustered by run, adaptation speed.

Episodes from one run are not independent agents, so uncertainty is
clustered by training run (and by mechanism family when there are several).
Report interquartile means with bootstrap intervals, all seeds and all
failures (Agarwal et al., 2021). Welfare violations stay in the denominator.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np


def iqm(x: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% (by sorted position, as in rliable)."""
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    if hi <= lo:
        return float(a.mean())
    return float(a[lo:hi].mean())


def bootstrap_ci(x: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    stats = np.array([stat(a[i]) for i in idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap(diff: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float]:
    """Percentile interval for the mean paired difference."""
    return bootstrap_ci(diff, np.mean, n_boot, alpha, seed)


def cluster_bootstrap(values: Mapping[str, Sequence[float]], stat: Callable[[np.ndarray], float] = np.mean,
                      n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Resample whole clusters (training runs), then apply ``stat`` to the pooled values."""
    keys = list(values)
    if not keys:
        return (float("nan"), float("nan"))
    arrays = [np.asarray(values[k], dtype=float) for k in keys]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        stats.append(stat(np.concatenate([arrays[i] for i in pick])))
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def summarise(x: Sequence[float], seed: int = 0) -> dict:
    a = np.asarray(x, dtype=float)
    lo, hi = bootstrap_ci(a, iqm, seed=seed) if len(a) else (float("nan"), float("nan"))
    return {"n": int(len(a)), "mean": float(a.mean()) if len(a) else float("nan"),
            "iqm": iqm(a), "iqm_ci95": [lo, hi]}


def adaptation_speed(curve: Sequence[tuple[float, float]], threshold: float = 0.9,
                     consecutive: int = 3) -> dict:
    """First interaction count at which S >= threshold holds on ``consecutive`` scheduled
    evaluations in a row. Runs that never reach it are censored."""
    run = 0
    for i, (x, s) in enumerate(curve):
        run = run + 1 if (s is not None and not np.isnan(s) and s >= threshold) else 0
        if run >= consecutive:
            return {"interactions": curve[i - consecutive + 1][0], "censored": False}
    return {"interactions": curve[-1][0] if curve else None, "censored": True}


def fully_adapted(curve: Sequence[tuple[float, float]], welfare_clean: bool, threshold: float = 0.9,
                  window: int = 5, min_gain: float = 0.01) -> bool:
    """Adapted, plus less than 0.01 improvement over five evaluations and a clean welfare record.
    A poor plateau does not qualify."""
    if adaptation_speed(curve, threshold)["censored"] or not welfare_clean or len(curve) < window:
        return False
    last = [s for _, s in curve[-window:]]
    return (max(last) - last[0]) < min_gain and min(last) >= threshold


def normal_interval(mean: float, se: float, z: float = 1.959964) -> tuple[float, float]:
    return mean - z * se, mean + z * se
