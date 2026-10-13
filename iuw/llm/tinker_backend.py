"""Tinker backend: LoRA training on Tinker's GPUs, driven by this CPU-side loop.

Written against the tinker SDK 0.33 (``pip install tinker``; set
``TINKER_API_KEY``). Re-check the model list, prices and loss options before
the smoke test (build stage 5).

* Legal-action scoring: one prefill per decision. The prompt is the
  transcript plus one placeholder token, and ``target_prompt_logprobs`` asks
  for the six action tokens at that last position only, as a sparse row, so
  the server can serve the prefix from its cache. ``max_tokens=1`` makes the
  call a single prefill; the generated token is ignored.
* Reference log-probs for KL: one call per episode, with sparse rows at every
  action position.
* Built-in loss ("builtin-ppo"): Tinker's ``ppo`` loss over the full softmax,
  with sampler log-probs and advantages on action positions only.
* Renormalised loss ("renormalised"): ``forward_backward_custom`` with all
  six action tokens as targets at each action position; the clipped loss is
  computed over the legal tokens only. Needs torch locally. Switch to it if
  illegal-token mass passes about 1%.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Optional, Sequence

import numpy as np

from .backend import LossConfig, ScoreRequest, ScoreResult, TrainItem, UsageStats
from .tokenizer import HFTokenizer, action_token_ids


def _tinker():
    try:
        import tinker
        from tinker import types
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install tinker (and set TINKER_API_KEY)") from e
    return tinker, types


def sparse_rows(n_rows: int, rows: Sequence[int], candidates: Sequence[int]):
    """CSR int64 TensorData of shape [n_rows, K] naming ``candidates`` on the given rows."""
    _, types = _tinker()
    k = len(candidates)
    rows = sorted(set(rows))
    if rows and (rows[0] < 0 or rows[-1] >= n_rows):
        raise ValueError("row out of range")
    counts = [0] * n_rows
    for r in rows:
        counts[r] = k
    crow = [0]
    for c in counts:
        crow.append(crow[-1] + c)
    data = np.tile(np.asarray(candidates, dtype=np.int64), len(rows))
    cols = list(range(k)) * len(rows)
    return types.TensorData(data=data, dtype="int64", shape=[n_rows, k],
                            sparse_crow_indices=crow, sparse_col_indices=cols)


def dense(td, pad: float = 0.0) -> np.ndarray:
    """Densify a (possibly sparse CSR) TensorData without needing torch."""
    if getattr(td, "sparse_crow_indices", None) is None:
        return np.asarray(td.data, dtype=float).reshape(td.shape)
    n_rows, k = td.shape
    out = np.full((n_rows, k), pad, dtype=float)
    vals = np.asarray(td.data, dtype=float)
    crow, cols = td.sparse_crow_indices, td.sparse_col_indices
    for r in range(n_rows):
        for j in range(crow[r], crow[r + 1]):
            out[r, cols[j]] = vals[j]
    return out


class TinkerScorer:
    def __init__(self, sampling_client, usage: UsageStats):
        self.client = sampling_client
        self.usage = usage

    def _submit(self, tokens: Sequence[int], rows: Sequence[int], candidates: Sequence[int]) -> Future:
        _, types = _tinker()
        tokens = list(tokens)
        target = sparse_rows(len(tokens) - 1, rows, candidates)
        return self.client.sample(
            prompt=types.ModelInput.from_ints(tokens), num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1, temperature=1.0),
            include_prompt_logprobs=True, prompt_logprobs_last_n=1,
            target_prompt_logprobs=target)

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        futs = []
        for r in requests:
            toks = list(r.tokens) + [r.candidates[0]]  # placeholder at the scored position
            futs.append((len(toks), self._submit(toks, [len(toks) - 2], r.candidates)))
        out = []
        for n, f in futs:
            res = f.result()
            row = dense(res.target_prompt_logprobs)[n - 2]
            self.usage.add_score(n, int(getattr(res, "prompt_cache_hit_tokens", 0)))
            out.append(ScoreResult(row, n, int(getattr(res, "prompt_cache_hit_tokens", 0))))
        return out

    def score_positions(self, tokens, positions, candidates) -> np.ndarray:
        if not positions:
            return np.zeros((0, len(candidates)))
        res = self._submit(tokens, [p - 1 for p in positions], candidates).result()
        self.usage.add_score(len(tokens), int(getattr(res, "prompt_cache_hit_tokens", 0)))
        mat = dense(res.target_prompt_logprobs)
        return mat[[p - 1 for p in positions]]


class TinkerBackend:
    name = "tinker"

    def __init__(self, base_model: str, rank: int = 32, seed: Optional[int] = None,
                 init_checkpoint: Optional[str] = None):
        tinker, types = _tinker()
        self.types = types
        self.base_model = base_model
        self.service = tinker.ServiceClient()
        if init_checkpoint:
            self.training = self.service.create_training_client_from_state(init_checkpoint)
        else:
            self.training = self.service.create_lora_training_client(base_model=base_model, rank=rank,
                                                                     seed=seed)
        self.tokenizer = HFTokenizer(self.training.get_tokenizer(), base_model)
        self.action_ids = action_token_ids(self.tokenizer)
        self.usage = UsageStats()
        self._sampler = self.training.save_weights_and_get_sampling_client()
        # The fixed initial reference policy: the starting weights, never updated.
        self._reference = self._sampler

    # ----- scoring ---------------------------------------------------------------------------
    def scorer(self) -> TinkerScorer:
        return TinkerScorer(self._sampler, self.usage)

    def reference_scorer(self) -> TinkerScorer:
        return TinkerScorer(self._reference, self.usage)

    def snapshot_reference(self) -> None:
        self._reference = self.training.save_weights_and_get_sampling_client()

    # ----- training --------------------------------------------------------------------------
    def _datum_builtin(self, it: TrainItem):
        t = self.types
        toks = np.asarray(it.tokens, dtype=np.int64)
        n = len(toks) - 1
        lp = np.zeros(n, dtype=np.float32)
        adv = np.zeros(n, dtype=np.float32)
        for j, p in enumerate(it.positions):
            lp[p - 1] = it.old_logprob_full[j]
            adv[p - 1] = it.advantages[j]
        return t.Datum(model_input=t.ModelInput.from_ints(toks[:-1].tolist()),
                       loss_fn_inputs={"target_tokens": t.TensorData.from_numpy(toks[1:]),
                                       "logprobs": t.TensorData.from_numpy(lp),
                                       "advantages": t.TensorData.from_numpy(adv)})

    def _datum_renorm(self, it: TrainItem):
        """Targets: all six action tokens at every position (weights zero off-action)."""
        t = self.types
        toks = np.asarray(it.tokens, dtype=np.int64)
        n = len(toks) - 1
        targets = np.tile(np.asarray(self.action_ids, dtype=np.int64), (n, 1))
        return t.Datum(model_input=t.ModelInput.from_ints(toks[:-1].tolist()),
                       loss_fn_inputs={"target_tokens": t.TensorData.from_numpy(targets)})

    def train(self, items: Sequence[TrainItem], cfg: LossConfig) -> dict:
        t = self.types
        metrics: dict = {}
        if cfg.loss == "builtin-ppo":
            data = [self._datum_builtin(it) for it in items]
        elif cfg.loss == "renormalised":
            data = [self._datum_renorm(it) for it in items]
        else:
            raise ValueError(cfg.loss)
        for _ in range(cfg.passes):
            if cfg.loss == "builtin-ppo":
                fb = self.training.forward_backward(
                    data, "ppo", loss_fn_config={"clip_low_threshold": 1 - cfg.clip,
                                                 "clip_high_threshold": 1 + cfg.clip})
            else:
                fb = self.training.forward_backward_custom(data, _renormalised_loss(items, cfg.clip))
            opt = self.training.optim_step(t.AdamParams(learning_rate=cfg.learning_rate))
            out = fb.result()
            opt.result()
            metrics = dict(getattr(out, "metrics", {}) or {})
            self.usage.train_calls += 1
            self.usage.training_tokens += sum(len(it.tokens) for it in items)
        self._sampler = self.training.save_weights_and_get_sampling_client()
        return metrics

    def supervised(self, examples, learning_rate: float) -> dict:
        t = self.types
        data = []
        for prompt, completion in examples:
            toks = list(prompt) + list(completion)
            w = np.zeros(len(toks) - 1, dtype=np.float32)
            w[len(prompt) - 1:] = 1.0
            data.append(t.Datum(model_input=t.ModelInput.from_ints(toks[:-1]),
                                loss_fn_inputs={"target_tokens": t.TensorData.from_numpy(np.asarray(toks[1:], dtype=np.int64)),
                                                "weights": t.TensorData.from_numpy(w)}))
        fb = self.training.forward_backward(data, "cross_entropy")
        opt = self.training.optim_step(t.AdamParams(learning_rate=learning_rate))
        out = fb.result()
        opt.result()
        self.usage.train_calls += 1
        self._sampler = self.training.save_weights_and_get_sampling_client()
        return dict(getattr(out, "metrics", {}) or {})

    # ----- checkpoints and generation --------------------------------------------------------
    def save(self, name: str) -> str:
        """Saves full state (adapters + optimiser) and sampler weights; returns the state path."""
        state = self.training.save_state(name).result()
        self.training.save_weights_for_sampler(name).result()
        return state.path

    def generate(self, tokens, max_tokens, temperature=1.0, stop=None, seed=None) -> list[int]:
        t = self.types
        res = self._sampler.sample(
            prompt=t.ModelInput.from_ints(list(tokens)), num_samples=1,
            sampling_params=t.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                             stop=list(stop) if stop else None, seed=seed)).result()
        self.usage.add_score(len(tokens), int(getattr(res, "prompt_cache_hit_tokens", 0)),
                             len(res.sequences[0].tokens))
        return list(res.sequences[0].tokens)


def _renormalised_loss(items: Sequence[TrainItem], clip: float):
    """Clipped surrogate over legal action tokens only; logprobs arrive as [T, 6] per datum."""
    import torch

    def loss_fn(data, logprobs_list):
        losses, clipped, n = [], 0, 0
        for it, lp in zip(items, logprobs_list):
            rows = torch.as_tensor([p - 1 for p in it.positions], dtype=torch.long)
            lp_rows = lp[rows]  # [n_dec, 6], full-softmax log-probs of the action tokens
            mask = torch.full_like(lp_rows, float("-inf"))
            for j, legal in enumerate(it.legal):
                mask[j, list(legal)] = 0.0
            lp_legal = torch.log_softmax(lp_rows + mask, dim=-1)
            acts = torch.as_tensor(it.actions, dtype=torch.long)
            new = lp_legal.gather(1, acts[:, None]).squeeze(1)
            old = torch.as_tensor(it.old_logprob_legal, dtype=new.dtype)
            adv = torch.as_tensor(it.advantages, dtype=new.dtype)
            ratio = torch.exp(new - old)
            surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            losses.append(-surr)
            clipped += int((((adv >= 0) & (ratio > 1 + clip)) | ((adv < 0) & (ratio < 1 - clip))).sum())
            n += len(it.actions)
        loss = torch.cat(losses).mean()
        return loss, {"clipfrac": clipped / max(n, 1)}

    return loss_fn
