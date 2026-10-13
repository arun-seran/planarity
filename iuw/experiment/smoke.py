"""Smoke test (build stage 5): 100 episodes to measure throughput, prompt-cache hit rate,
illegal-token mass and real cost, then project the MVP's cost from the measurements."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from ..config import PROFILES, Prices, get_world
from ..llm.backend import Backend
from ..llm.policy import EpisodeSpec, SamplingConfig, rollout
from ..render.rendering import make_rendering
from ..seeds import SPLITS

# MVP episode counts (spec v3, "Cost, roughly").
MVP_EPISODES = {
    "training grid (12 x 800)": 9_600,
    "middle-loop proposals (8 x 100)": 800,
    "checkpoint evaluations (12 x 6 x 64)": 4_608,
    "middle-loop evaluations (8 x 64)": 512,
    "sealed tests (12 x 128)": 1_536,
    "frozen controls (4 x 128)": 512,
    "notebook climbs and tests (2 x 728)": 1_456,
    "smoke test": 100,
}
MVP_TRAINING_EPISODES = 10_400


def smoke_test(backend: Backend, n_episodes: int = 100, batch: int = 16, prices: Prices = Prices(),
               progress=None) -> dict:
    specs = []
    pilot = SPLITS["pilot"]
    i = 0
    while len(specs) < n_episodes:
        for kind in ("named", "masked"):
            for w in ("reference", "counterfactual"):
                for b in ("A", "B"):
                    if len(specs) < n_episodes:
                        specs.append(EpisodeSpec(get_world(w), PROFILES[b], pilot[i % len(pilot)],
                                                 make_rendering(kind, i % 3), 0, split="pilot"))
                        i += 1
    usage0 = asdict(backend.usage)
    t0 = time.time()
    trajs = []
    for k in range(0, len(specs), batch):
        trajs.extend(rollout(specs[k:k + batch], backend.scorer(), backend.tokenizer,
                             SamplingConfig(salt=4242)))
        if progress:
            progress(f"[smoke] {len(trajs)}/{len(specs)} episodes")
    secs = time.time() - t0
    u = backend.usage
    prompt = u.prompt_tokens - usage0["prompt_tokens"]
    cached = u.cached_prompt_tokens - usage0["cached_prompt_tokens"]
    decisions = sum(len(t.decisions) for t in trajs)
    hit = cached / prompt if prompt else 0.0
    per_ep_prompt = prompt / len(trajs)
    per_ep_tokens = float(np.mean([len(t.tokens) for t in trajs]))
    cost_per_ep = (per_ep_prompt * (1 - hit) * prices.prompt + per_ep_prompt * hit * prices.cached_prompt) / 1e6
    total_eps = sum(MVP_EPISODES.values())
    sampling_cost = total_eps * cost_per_ep
    train_cost_per_pass = MVP_TRAINING_EPISODES * per_ep_tokens * prices.training / 1e6
    return {
        "episodes": len(trajs), "decisions": decisions, "seconds": secs,
        "episodes_per_hour": len(trajs) / secs * 3600 if secs else None,
        "decisions_per_second": decisions / secs if secs else None,
        "prompt_tokens_per_episode": per_ep_prompt, "transcript_tokens_per_episode": per_ep_tokens,
        "cache_hit_rate": hit, "illegal_mass_mean": float(np.mean([t.illegal_mass for t in trajs])),
        "illegal_mass_max": float(np.max([t.illegal_mass for t in trajs])),
        "truncated": sum(t.truncated for t in trajs),
        "unclean_audits": sum(bool(t.violations) for t in trajs),
        "measured_cost_usd": {"sampling": prompt and (prompt - cached) * prices.prompt / 1e6
                              + cached * prices.cached_prompt / 1e6},
        "projected_mvp_usd": {"sampling": sampling_cost, "training_per_pass": train_cost_per_pass,
                              "total_one_pass": sampling_cost + train_cost_per_pass,
                              "episodes": total_eps},
        "switch_to_renormalised_loss": bool(np.mean([t.illegal_mass for t in trajs]) > 0.01),
    }
