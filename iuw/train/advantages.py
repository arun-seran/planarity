"""Advantages: return-to-go, minus the batch mean, divided by the batch std.

No value head. Gamma is 1. Bootstrap only on external truncation, never on
true termination. A reference-KL penalty, when scheduled, is folded into
per-decision rewards as beta * (log pi_old(a) - log pi_ref(a)), using the
legal-renormalised log-probabilities of the sampled action.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..llm.policy import Trajectory


def returns_to_go(rewards: Sequence[float], bootstrap: float = 0.0, gamma: float = 1.0) -> np.ndarray:
    out = np.zeros(len(rewards))
    acc = bootstrap
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out


@dataclass(frozen=True)
class KLSchedule:
    kind: str = "none"  # "none" | "fixed" | "anneal"
    coef: float = 0.0
    total_updates: int = 50

    def at(self, update: int) -> float:
        if self.kind == "none" or self.coef == 0.0:
            return 0.0
        if self.kind == "fixed":
            return self.coef
        if self.kind == "anneal":
            return self.coef * max(0.0, 1.0 - (update - 1) / max(1, self.total_updates - 1))
        raise ValueError(self.kind)


def shaped_rewards(traj: Trajectory, kl_coef: float) -> np.ndarray:
    r = traj.rewards.copy()
    if kl_coef and len(r):
        if traj.ref_lp_legal is None:
            raise ValueError("reference log-probs missing; call attach_reference_logprobs")
        old = np.array([d.lp_legal for d in traj.decisions])
        r = r - kl_coef * (old - traj.ref_lp_legal)
    return r


BootstrapFn = Callable[[Trajectory, Sequence[Trajectory]], float]


def batch_mean_bootstrap(traj: Trajectory, batch: Sequence[Trajectory]) -> float:
    """Value estimate for a truncated episode: mean remaining return of untruncated peers
    from the same world, body and cycle. Truncation is never expected (context test)."""
    if not traj.decisions:
        return 0.0
    cycle = traj.decisions[-1].cycle + 1
    vals = []
    for t in batch:
        if t.truncated or t.world != traj.world or t.body != traj.body:
            continue
        rtg = returns_to_go(t.rewards)
        idx = [i for i, d in enumerate(t.decisions) if d.cycle >= cycle]
        if idx:
            vals.append(rtg[idx[0]])
    return float(np.mean(vals)) if vals else 0.0


def compute_advantages(trajs: Sequence[Trajectory], kl_coef: float = 0.0,
                       bootstrap_fn: Optional[BootstrapFn] = batch_mean_bootstrap,
                       baseline: str = "batch", eps: float = 1e-8) -> list[np.ndarray]:
    """Spec default ``baseline="batch"``: (G - batch mean) / batch std.

    ``baseline="per-cycle"`` is an optional variance reduction, off by default:
    it subtracts the mean return-to-go of decisions at the same cycle and
    phase across the batch before dividing by the batch std. Both baselines
    are independent of the action taken, so the gradient stays unbiased.
    """
    rtgs = []
    for t in trajs:
        boot = bootstrap_fn(t, trajs) if (t.truncated and not t.terminated and bootstrap_fn) else 0.0
        rtgs.append(returns_to_go(shaped_rewards(t, kl_coef), bootstrap=boot))
    flat = np.concatenate(rtgs) if rtgs else np.zeros(0)
    if len(flat) == 0:
        return rtgs
    if baseline == "batch":
        mu, sd = flat.mean(), flat.std()
        return [(g - mu) / (sd + eps) for g in rtgs]
    if baseline != "per-cycle":
        raise ValueError(baseline)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for t, g in zip(trajs, rtgs):
        for d, v in zip(t.decisions, g):
            groups[(d.cycle, d.phase)].append(v)
    means = {k: float(np.mean(v)) for k, v in groups.items()}
    centred = [np.array([v - means[(d.cycle, d.phase)] for d, v in zip(t.decisions, g)])
               for t, g in zip(trajs, rtgs)]
    sd = np.concatenate(centred).std()
    return [c / (sd + eps) for c in centred]


def clipped_surrogate(lp_new: np.ndarray, lp_old: np.ndarray, adv: np.ndarray, clip: float = 0.2) -> dict:
    """Reference implementation of the clipped importance-weighted loss (mean over decisions)."""
    ratio = np.exp(lp_new - lp_old)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv
    loss = -np.minimum(unclipped, clipped)
    clipfrac = np.mean(((adv >= 0) & (ratio > 1 + clip)) | ((adv < 0) & (ratio < 1 - clip)))
    return {"loss": float(loss.mean()) if len(loss) else 0.0, "clipfrac": float(clipfrac) if len(loss) else 0.0,
            "ratio_mean": float(ratio.mean()) if len(loss) else 1.0}
