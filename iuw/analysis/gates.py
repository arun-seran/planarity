"""Success gates (illustrative until calibrated on pilot worlds, then preregistered)."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .stats import iqm, paired_bootstrap

LEARNING_S = 0.9
LEARNING_MARGIN = 0.15


def _key(e: Mapping) -> tuple:
    return (e["world"], e["body"], e["seed"])


def learning_gate(trained: Sequence[Mapping], frozen_baselines: Mapping[str, Sequence[Mapping]],
                  world: str = "counterfactual") -> dict:
    """S >= 0.9 on held-out seeds, at least 0.15 above the best frozen baseline, with a
    positive 95% paired interval. Episodes are dicts with world/body/seed/score."""
    t = {_key(e): (e["score"] or 0.0) for e in trained if e["world"] == world}
    s_trained = float(np.mean(list(t.values()))) if t else float("nan")
    best_name, best_mean, best_diff = None, -np.inf, None
    for name, eps in frozen_baselines.items():
        b = {_key(e): (e["score"] or 0.0) for e in eps if e["world"] == world}
        keys = sorted(set(t) & set(b))
        if not keys:
            continue
        m = float(np.mean([b[k] for k in keys]))
        if m > best_mean:
            best_name, best_mean = name, m
            best_diff = np.array([t[k] - b[k] for k in keys])
    if best_diff is None:
        return {"passes": False, "reason": "no paired frozen baseline"}
    lo, hi = paired_bootstrap(best_diff)
    passes = s_trained >= LEARNING_S and s_trained - best_mean >= LEARNING_MARGIN and lo > 0
    return {"passes": bool(passes), "S": s_trained, "S_iqm": iqm(list(t.values())),
            "best_frozen": best_name, "best_frozen_S": best_mean,
            "paired_diff_mean": float(best_diff.mean()), "paired_ci95": [lo, hi]}


def zero_shot_check(frozen_named_cf_S: float) -> dict:
    """If the frozen model already reaches S >= 0.9 in the Named counterfactual cell, there was
    no prior to unlearn there, and the claim is unsupported in that world."""
    unsupported = frozen_named_cf_S >= 0.9
    return {"S": frozen_named_cf_S, "claim_supported": not unsupported,
            "note": "frozen model already competent; such worlds are witnesses, not learning results"
            if unsupported else ""}


def causality_gate(with_law: float, without_law: float, predicted_sign: int = 1) -> dict:
    """Removing the proposed law should remove the predicted advantage."""
    removed = (with_law - without_law) * predicted_sign > 0
    return {"passes": bool(removed), "with_law": with_law, "without_law": without_law}
