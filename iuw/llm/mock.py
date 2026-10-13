"""A small numpy policy that stands in for the language model in tests.

It scores action tokens with a linear softmax over hashed (offset, token)
features of the most recent tokens, plus an "other tokens" bucket so that
illegal-token mass is meaningful. Training applies the same clipped
importance-weighted objective as the real arms, with Adam. It exists so that
the whole pipeline (rollouts, advantages, losses, evaluation schedules,
checkpoints, analysis) runs and is tested without a GPU or network. It is
not a baseline and its results are not evidence about language models.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .backend import LossConfig, ScoreRequest, ScoreResult, TrainItem, UsageStats, legal_log_softmax
from .tokenizer import ApproxQwenTokenizer, action_token_ids


@dataclass
class MockParams:
    W: np.ndarray
    b: np.ndarray
    other_logit: float


class MockModel:
    def __init__(self, action_ids: Sequence[int], n_features: int = 8192, window: int = 24,
                 other_logit: float = -1.0, init_scale: float = 0.0, seed: int = 0):
        self.action_ids = list(action_ids)
        self.n_features = n_features
        self.window = window
        rng = np.random.default_rng(seed)
        self.params = MockParams(rng.normal(0, init_scale, (n_features, len(action_ids))),
                                 np.zeros(len(action_ids)), other_logit)
        self._index = {t: i for i, t in enumerate(self.action_ids)}
        self.newline_ids: set[int] = set()

    def features(self, prefix: Sequence[int]) -> np.ndarray:
        """Hashed (offset, token) pairs over the window, plus a bag of the current line's tokens."""
        tail = prefix[-self.window:]
        if not tail:
            return np.zeros(0, dtype=np.int64)
        idx = [((off + 1) * 1_000_003 + int(tok) * 7_919) % self.n_features
               for off, tok in enumerate(reversed(tail))]
        line = []
        for tok in reversed(tail):
            if tok in self.newline_ids:
                break
            line.append(tok)
        idx.extend((int(tok) * 104_729 + 17) % self.n_features for tok in line)
        return np.unique(np.asarray(idx, dtype=np.int64))

    def logits(self, prefix: Sequence[int], params: Optional[MockParams] = None) -> np.ndarray:
        p = params or self.params
        phi = self.features(prefix)
        return p.W[phi].sum(axis=0) + p.b

    def logprobs_all(self, prefix: Sequence[int], params: Optional[MockParams] = None) -> np.ndarray:
        """Full-softmax log-probs of the six action tokens (plus an implicit 'other' bucket)."""
        p = params or self.params
        z = self.logits(prefix, p)
        m = max(z.max(), p.other_logit)
        lse = m + np.log(np.exp(z - m).sum() + np.exp(p.other_logit - m))
        return z - lse

    def candidate_logprobs(self, prefix, candidates, params=None) -> np.ndarray:
        lp = self.logprobs_all(prefix, params)
        other = np.log(1e-6)
        return np.array([lp[self._index[c]] if c in self._index else other for c in candidates])


class MockScorer:
    def __init__(self, model: MockModel, params: MockParams, usage: UsageStats):
        self.model = model
        self.params = params
        self.usage = usage

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        out = []
        for r in requests:
            self.usage.add_score(len(r.tokens), 0)
            out.append(ScoreResult(self.model.candidate_logprobs(r.tokens, r.candidates, self.params),
                                   len(r.tokens)))
        return out

    def score_positions(self, tokens, positions, candidates) -> np.ndarray:
        self.usage.add_score(len(tokens), 0, 0)
        return np.stack([self.model.candidate_logprobs(tokens[:p], candidates, self.params)
                         for p in positions])


class MockBackend:
    name = "mock"

    def __init__(self, seed: int = 0, n_features: int = 8192, other_logit: float = -4.0,
                 beta1: float = 0.9, beta2: float = 0.999):
        self.tokenizer = ApproxQwenTokenizer()
        self.action_ids = action_token_ids(self.tokenizer)
        self.model = MockModel(self.action_ids, n_features=n_features, other_logit=other_logit, seed=seed)
        self.model.newline_ids = set(self.tokenizer.encode("\n"))
        self.reference = copy.deepcopy(self.model.params)
        self.usage = UsageStats()
        self.checkpoints: dict[str, MockParams] = {}
        self._m = [np.zeros_like(self.model.params.W), np.zeros_like(self.model.params.b)]
        self._v = [np.zeros_like(self.model.params.W), np.zeros_like(self.model.params.b)]
        self._t = 0
        self.beta1, self.beta2 = beta1, beta2
        self.generated_notebooks: list[str] = []

    # ----- scoring ---------------------------------------------------------------------------
    def scorer(self) -> MockScorer:
        return MockScorer(self.model, copy.deepcopy(self.model.params), self.usage)

    def reference_scorer(self) -> MockScorer:
        return MockScorer(self.model, self.reference, self.usage)

    def snapshot_reference(self) -> None:
        self.reference = copy.deepcopy(self.model.params)

    # ----- training --------------------------------------------------------------------------
    def train(self, items: Sequence[TrainItem], cfg: LossConfig) -> dict:
        stats = {"loss": [], "clipfrac": [], "ratio": []}
        first_ratio = None
        for _ in range(cfg.passes):
            gW = np.zeros_like(self.model.params.W)
            gb = np.zeros_like(self.model.params.b)
            n = sum(len(it.actions) for it in items)
            losses, clipped, ratios = [], 0, []
            for it in items:
                for j, (pos, a, legal) in enumerate(zip(it.positions, it.actions, it.legal)):
                    prefix = it.tokens[:pos]
                    phi = self.model.features(prefix)
                    z = self.model.params.W[phi].sum(axis=0) + self.model.params.b
                    if cfg.loss == "renormalised":
                        lp = legal_log_softmax(z, legal)  # z are logits: softmax over legal only
                        probs = np.where(np.isfinite(lp), np.exp(lp), 0.0)
                        old = it.old_logprob_legal[j]
                    else:
                        o = self.model.params.other_logit
                        m = max(z.max(), o)
                        lse = m + np.log(np.exp(z - m).sum() + np.exp(o - m))
                        lp = z - lse
                        probs = np.exp(lp)
                        old = it.old_logprob_full[j]
                    adv = it.advantages[j]
                    r = float(np.exp(lp[a] - old))
                    ratios.append(r)
                    unclipped, clipped_v = r * adv, float(np.clip(r, 1 - cfg.clip, 1 + cfg.clip)) * adv
                    losses.append(-min(unclipped, clipped_v))
                    active = not ((adv >= 0 and r > 1 + cfg.clip) or (adv < 0 and r < 1 - cfg.clip))
                    if not active:
                        clipped += 1
                        continue
                    dz = -adv * r * (np.eye(len(z))[a] - probs) / max(n, 1)
                    gW[phi] += dz
                    gb += dz
            self._adam([gW, gb], cfg.learning_rate)
            stats["loss"].append(float(np.mean(losses)) if losses else 0.0)
            stats["clipfrac"].append(clipped / max(n, 1))
            stats["ratio"].append(float(np.mean(ratios)) if ratios else 1.0)
            if first_ratio is None:
                first_ratio = stats["ratio"][-1]
            self.usage.train_calls += 1
            self.usage.training_tokens += sum(len(it.tokens) for it in items)
        out = {k: v[-1] for k, v in stats.items()}
        out["ratio_first_pass"] = first_ratio  # must be 1: rollout and training score alike
        return out

    def _adam(self, grads, lr):
        self._t += 1
        params = [self.model.params.W, self.model.params.b]
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
            mh = self._m[i] / (1 - self.beta1 ** self._t)
            vh = self._v[i] / (1 - self.beta2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

    def supervised(self, examples, learning_rate: float) -> dict:
        self.usage.train_calls += 1
        self.usage.training_tokens += sum(len(p) + len(c) for p, c in examples)
        return {"examples": len(examples), "note": "mock backend ignores supervised data"}

    # ----- checkpoints and generation --------------------------------------------------------
    def save(self, name: str) -> str:
        self.checkpoints[name] = copy.deepcopy(self.model.params)
        return f"mock://{name}"

    def load(self, path: str) -> None:
        self.model.params = copy.deepcopy(self.checkpoints[path.removeprefix("mock://")])

    def generate(self, tokens, max_tokens, temperature=1.0, stop=None, seed=None) -> list[int]:
        text = ("Strategy: inspect at the start. When the last command was ok and water rose, "
                "repeat it. Otherwise inspect and open the valve shown.")
        self.generated_notebooks.append(text)
        return self.tokenizer.encode(text)[:max_tokens]
