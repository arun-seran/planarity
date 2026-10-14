"""Backend interfaces shared by Tinker, the local Hugging Face backend and the mock.

A *scorer* returns full-softmax log-probabilities of chosen candidate tokens
at chosen positions: the policy computes the distribution over legal action
tokens itself and samples there (Tinker has no option to restrict sampling
to chosen tokens). A *backend* adds training, checkpointing and generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from ..config import Prices


@dataclass
class ScoreRequest:
    tokens: list[int]  # the prefix; candidates are scored as the next token
    candidates: list[int]


@dataclass
class ScoreResult:
    logprobs: np.ndarray  # full-softmax log-probability of each candidate
    prompt_tokens: int = 0
    cached_tokens: int = 0


class Scorer(Protocol):
    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]: ...

    def score_positions(self, tokens: Sequence[int], positions: Sequence[int],
                        candidates: Sequence[int]) -> np.ndarray:
        """Row i: log-probs of the candidates at index positions[i], given tokens[:positions[i]]."""
        ...


@dataclass
class UsageStats:
    """Token accounting for cost estimates and cache hit rates."""

    score_calls: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    generated_tokens: int = 0
    training_tokens: int = 0
    train_calls: int = 0

    def add_score(self, prompt: int, cached: int, generated: int = 1) -> None:
        self.score_calls += 1
        self.prompt_tokens += prompt
        self.cached_prompt_tokens += cached
        self.generated_tokens += generated

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_prompt_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def cost(self, prices: Prices = Prices()) -> dict[str, float]:
        uncached = self.prompt_tokens - self.cached_prompt_tokens
        prompt_cost = (uncached * prices.prompt + self.cached_prompt_tokens * prices.cached_prompt) / 1e6
        train_cost = self.training_tokens * prices.training / 1e6
        return {"prompt": prompt_cost, "training": train_cost, "total": prompt_cost + train_cost}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cache_hit_rate"] = self.cache_hit_rate
        d["cost_usd"] = self.cost()
        return d


@dataclass
class LossConfig:
    clip: float = 0.2
    loss: str = "builtin-ppo"  # "builtin-ppo" (full softmax) | "renormalised" (legal tokens only)
    learning_rate: float = 1e-5
    passes: int = 1  # one to four passes per batch


@dataclass
class TrainItem:
    """One episode prepared for an update: tokens plus per-decision training targets."""

    tokens: list[int]
    positions: list[int]  # index in tokens of each action token
    actions: list[int]  # action IDs (0..5)
    legal: list[tuple[int, ...]]
    old_logprob_full: np.ndarray  # rollout policy, full softmax, sampled token
    old_logprob_legal: np.ndarray  # rollout policy, renormalised over legal tokens
    advantages: np.ndarray


class Backend(Protocol):
    name: str
    tokenizer: object
    usage: UsageStats

    def scorer(self) -> Scorer: ...

    def reference_scorer(self) -> Scorer: ...

    def train(self, items: Sequence[TrainItem], cfg: LossConfig) -> dict: ...

    def save(self, name: str) -> str: ...

    def load(self, path: str) -> None:
        """Restore adapters and optimiser state from a path returned by save()."""
        ...

    def generate(self, tokens: Sequence[int], max_tokens: int, temperature: float = 1.0,
                 stop: Optional[Sequence[str]] = None, seed: Optional[int] = None) -> list[int]: ...

    def supervised(self, examples: Sequence[tuple[list[int], list[int]]], learning_rate: float) -> dict:
        """Cross-entropy on (prompt tokens, completion tokens) pairs (codebook familiarisation)."""
        ...

    def snapshot_reference(self) -> None:
        """Make the current weights the fixed initial reference policy (familiarised arm)."""
        ...


def legal_log_softmax(logprobs_all: np.ndarray, legal: Sequence[int], temperature: float = 1.0) -> np.ndarray:
    """Renormalise full-softmax action log-probs over the legal subset (illegal -> -inf)."""
    out = np.full(len(logprobs_all), -np.inf)
    z = np.asarray([logprobs_all[a] for a in legal], dtype=float) / temperature
    z = z - (np.max(z) + np.log(np.sum(np.exp(z - np.max(z)))))
    for a, v in zip(legal, z):
        out[a] = v
    return out
