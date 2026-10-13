"""Matched-history replay.

Record a history once (observations and actions from a fixed policy and
seed), then render it in any rendering, with any prompt options and any
weights, and read the model's legal-action distribution at every decision,
with no sampling. Identical held-out histories are what the change ledger,
the "words evoke priors" check and the return-trip tests compare on.
Replaying keeps the information a partially observed task needs, so
deleting the context is never mistaken for deleting learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..config import Profile, WorldConfig
from ..env.runner import Policy, run_episode
from ..env.water import Observation
from ..llm.backend import Scorer, legal_log_softmax
from ..llm.tokenizer import action_token_ids
from ..render.rendering import PromptOptions, Rendering
from ..render.transcript import ChatFrame, Transcript


@dataclass
class History:
    world: str
    body: str
    seed: int
    noise: float
    observations: list[Observation]
    actions: list[int]


def record_history(world: WorldConfig, profile: Profile, seed: int, policy: Policy,
                   max_decisions: Optional[int] = None) -> History:
    res = run_episode(world, profile, seed, policy, keep_observations=True)
    obs, acts = res.observations, res.actions
    if max_decisions is not None:
        obs, acts = obs[:max_decisions], acts[:max_decisions]
    return History(world.name, profile.name, seed, world.noise, obs, acts)


def build_transcript(history: History, rendering: Rendering, tokenizer,
                     options: PromptOptions = PromptOptions(),
                     frame: Optional[ChatFrame] = None) -> Transcript:
    tr = Transcript(tokenizer, rendering, frame or ChatFrame(), options)
    tr.start(history.body, history.noise)
    for obs, a in zip(history.observations, history.actions):
        tr.observe(obs)
        tr.act(a)
    return tr


def replay_distributions(history: History, scorer: Scorer, tokenizer, rendering: Rendering,
                         options: PromptOptions = PromptOptions(),
                         frame: Optional[ChatFrame] = None) -> np.ndarray:
    """[n_decisions, 6] legal-renormalised action probabilities along the history."""
    tr = build_transcript(history, rendering, tokenizer, options, frame)
    ids = action_token_ids(tokenizer)
    rows = scorer.score_positions(tr.tokens, tr.action_positions, ids)
    out = np.zeros((len(rows), len(ids)))
    for i, (row, obs) in enumerate(zip(rows, history.observations)):
        lp = legal_log_softmax(np.asarray(row), obs.legal)
        out[i] = np.where(np.isfinite(lp), np.exp(lp), 0.0)
    return out


def action_log_likelihood(history: History, scorer: Scorer, tokenizer, rendering: Rendering,
                          options: PromptOptions = PromptOptions()) -> float:
    """Mean log-probability the model assigns to the recorded actions."""
    probs = replay_distributions(history, scorer, tokenizer, rendering, options)
    return float(np.mean([np.log(max(probs[i, a], 1e-12)) for i, a in enumerate(history.actions)]))
