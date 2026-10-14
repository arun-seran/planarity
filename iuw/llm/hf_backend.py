"""Local Hugging Face + PEFT backend (dense twin, inspection, small local runs).

Mirrors the Tinker backend: LoRA adapters on a frozen base, Adam with
Tinker's defaults (beta2 0.95, eps 1e-12, no weight decay), the clipped
importance-weighted loss over either the full softmax or the legal action
tokens. The initial reference policy is the base model with adapters
disabled, unless :meth:`snapshot_reference` fixes another one.

Requires ``torch``, ``transformers`` and ``peft``. Downloaded Tinker adapters
can be loaded with ``init_adapter`` for inspection.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .backend import LossConfig, ScoreRequest, ScoreResult, TrainItem, UsageStats
from .tokenizer import HFTokenizer, action_token_ids


class _CacheEntry:
    __slots__ = ("tokens", "kv", "last_call")

    def __init__(self, tokens: list[int], kv, last_call: int):
        self.tokens, self.kv, self.last_call = tokens, kv, last_call


class HFScorer:
    """Scores next-token candidates for rollouts.

    Transcripts are append-only, so each request usually extends the previous request of
    the same episode by a few tokens. With ``cache_episodes`` > 0 the scorer keeps a
    key/value cache per episode and runs only the new tokens, instead of the whole prefix
    at every decision. Episodes are matched by prefix; an entry not used in the previous
    call (a finished episode) is dropped, and episodes beyond the limit fall back to the
    uncached batched path. A scorer is tied to the weights it was created under: it clears
    its cache when the backend reports a training step.
    """

    def __init__(self, backend: "HFBackend", reference: bool = False, batch_size: int = 8,
                 cache_episodes: Optional[int] = None):
        self.backend = backend
        self.reference = reference
        self.batch_size = batch_size
        self.cache_episodes = backend.cache_episodes if cache_episodes is None else cache_episodes
        self._entries: list[_CacheEntry] = []
        self._calls = 0
        self._weights_version = backend.weights_version

    def _take_entry(self, tokens: list[int]) -> Optional[_CacheEntry]:
        best = None
        for e in self._entries:
            n = len(e.tokens)
            if n < len(tokens) and (best is None or n > len(best.tokens)) and tokens[:n] == e.tokens:
                best = e
        if best is not None:
            self._entries.remove(best)
        return best

    def _score_cached(self, tokens: list[int], entry: Optional[_CacheEntry], candidates):
        import torch

        be = self.backend
        start = len(entry.tokens) if entry else 0
        ids = torch.as_tensor(tokens[start:], dtype=torch.long, device=be.device)[None]
        with torch.no_grad():
            out = be.model(input_ids=ids, past_key_values=entry.kv if entry else None,
                           use_cache=True, logits_to_keep=1)
        lp = out.logits[0, -1].float().log_softmax(-1)[list(candidates)].cpu().numpy()
        self._entries.append(_CacheEntry(list(tokens), out.past_key_values, self._calls))
        return lp, start

    def _logprobs(self, batch: list[list[int]], keep: list[int]):
        """Full-softmax log-probs [batch, len(keep), vocab] at sequence indices ``keep`` only.

        Right padding leaves every real position unchanged under causal attention, and
        ``logits_to_keep`` avoids materialising [batch, length, vocab] logits.
        """
        import torch

        be = self.backend
        L = max(len(t) for t in batch)
        ids = torch.full((len(batch), L), be.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, t in enumerate(batch):
            ids[i, :len(t)] = torch.as_tensor(t)
            mask[i, :len(t)] = 1
        keep_t = torch.as_tensor(keep, dtype=torch.long, device=be.device)
        with torch.no_grad(), be.policy_context(self.reference):
            out = be.model(input_ids=ids.to(be.device), attention_mask=mask.to(be.device),
                           logits_to_keep=keep_t)
        return out.logits.float().log_softmax(-1).cpu()

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        if not self.cache_episodes:
            return self._score_uncached(requests)
        if self.backend.weights_version != self._weights_version:
            self._entries, self._weights_version = [], self.backend.weights_version
        self._calls += 1
        self._entries = [e for e in self._entries if e.last_call >= self._calls - 1]
        out: list[Optional[ScoreResult]] = [None] * len(requests)
        overflow = []
        with self.backend.policy_context(self.reference):
            for i, r in enumerate(requests):
                tokens = list(r.tokens)
                entry = self._take_entry(tokens)
                if entry is None and len(self._entries) >= self.cache_episodes:
                    overflow.append(i)
                    continue
                lp, cached = self._score_cached(tokens, entry, r.candidates)
                self.backend.usage.add_score(len(tokens), cached)
                out[i] = ScoreResult(lp.astype(float), len(tokens), cached)
        for i, res in zip(overflow, self._score_uncached([requests[i] for i in overflow])):
            out[i] = res
        return out

    def _score_uncached(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        out = []
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i:i + self.batch_size]
            keep = sorted({len(r.tokens) - 1 for r in chunk})
            col = {k: c for c, k in enumerate(keep)}
            lp = self._logprobs([list(r.tokens) for r in chunk], keep)
            for j, r in enumerate(chunk):
                row = lp[j, col[len(r.tokens) - 1], list(r.candidates)].numpy()
                self.backend.usage.add_score(len(r.tokens), 0)
                out.append(ScoreResult(row.astype(float), len(r.tokens)))
        return out

    def score_positions(self, tokens, positions, candidates) -> np.ndarray:
        lp = self._logprobs([list(tokens)], [p - 1 for p in positions])[0]
        self.backend.usage.add_score(len(tokens), 0, 0)
        return lp[:, list(candidates)].numpy().astype(float)


def expert_parameters(model) -> list[str]:
    """Names of fused mixture-of-experts weight tensors (3-D ``experts.*`` parameters).

    ``target_modules="all-linear"`` only reaches nn.Linear layers, so without these a LoRA on a
    MoE model such as Qwen3.5/3.6-A3B adapts attention, router and shared expert but none of
    the routed experts.
    """
    names = set()
    for name, p in model.named_parameters():
        parts = name.split(".")
        if p.dim() == 3 and "experts" in parts:
            names.add(".".join(parts[parts.index("experts") - 1:]))  # e.g. mlp.experts.down_proj
    return sorted(names)


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBackend:
    name = "hf"

    def __init__(self, model_name_or_path: str, rank: int = 32, lora_alpha: Optional[int] = None,
                 device: Optional[str] = None, dtype: str = "bfloat16", seed: int = 0,
                 init_adapter: Optional[str] = None, model=None, tokenizer=None,
                 target_modules="all-linear", cache_episodes: int = 16,
                 gradient_checkpointing: bool = True, lora_experts: bool = False):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model

        torch.manual_seed(seed)
        self.device = device or default_device()
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # straight onto a CUDA device: a 35B checkpoint should not be staged in host memory
            placement = {"device_map": {"": self.device}} if self.device.startswith("cuda") else {}
            model, info = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, dtype=getattr(torch, dtype), output_loading_info=True, **placement)
            if info.get("missing_keys"):  # e.g. a checkpoint whose key names this class does not map
                raise RuntimeError(f"{len(info['missing_keys'])} weights missing when loading "
                                   f"{model_name_or_path}, e.g. {sorted(info['missing_keys'])[:3]}")
            tokenizer = tokenizer or HFTokenizer(AutoTokenizer.from_pretrained(model_name_or_path),
                                                 model_name_or_path)
        if init_adapter:
            self.model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        else:
            cfg = LoraConfig(r=rank, lora_alpha=lora_alpha or rank, lora_dropout=0.0,
                             target_modules=target_modules, task_type="CAUSAL_LM",
                             target_parameters=expert_parameters(model) if lora_experts else None)
            self.model = get_peft_model(model, cfg)
        self.model.to(self.device)
        self.model.eval()  # no dropout; train mode is entered only for optimiser steps
        if gradient_checkpointing:  # whole-episode backward passes otherwise hold every layer's activations
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.tokenizer = tokenizer
        self.action_ids = action_token_ids(self.tokenizer)
        self.pad_id = getattr(getattr(tokenizer, "tok", None), "pad_token_id", None) or 0
        self.usage = UsageStats()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=1e-5, betas=(0.9, 0.95), eps=1e-12,
                                     weight_decay=0.0)
        self._reference_state: Optional[dict] = None
        self.cache_episodes = cache_episodes  # per-episode KV caches a rollout scorer may hold
        self.weights_version = 0  # bumped by every optimiser step; scorers drop stale caches
        self.checkpoint_dir = Path("runs/hf-checkpoints")
        self._saved: list[Path] = []

    # ----- policies --------------------------------------------------------------------------
    def lora_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items() if "lora_" in k}

    @contextmanager
    def policy_context(self, reference: bool):
        if not reference:
            yield
            return
        if self._reference_state is None:
            with self.model.disable_adapter():
                yield
            return
        current = self.lora_state()
        self.model.load_state_dict(self._reference_state, strict=False)
        try:
            yield
        finally:
            self.model.load_state_dict(current, strict=False)

    def scorer(self) -> HFScorer:
        return HFScorer(self)

    def reference_scorer(self) -> HFScorer:
        return HFScorer(self, reference=True)

    def snapshot_reference(self) -> None:
        self._reference_state = self.lora_state()

    # ----- training --------------------------------------------------------------------------
    def _decision_logprobs(self, it: TrainItem, renormalised: bool):
        import torch

        ids = torch.as_tensor(it.tokens, dtype=torch.long, device=self.device)[None]
        rows = torch.as_tensor([p - 1 for p in it.positions], device=self.device)
        logits = self.model(input_ids=ids, logits_to_keep=rows).logits[0].float()  # [decisions, vocab]
        acts = torch.as_tensor(it.actions, device=self.device)
        if not renormalised:
            lp = logits.log_softmax(-1)
            tok = torch.as_tensor([self.action_ids[a] for a in it.actions], device=self.device)
            return lp.gather(1, tok[:, None]).squeeze(1)
        act_logits = logits[:, self.action_ids]
        mask = torch.full_like(act_logits, float("-inf"))
        for j, legal in enumerate(it.legal):
            mask[j, list(legal)] = 0.0
        lp = (act_logits + mask).log_softmax(-1)
        return lp.gather(1, acts[:, None]).squeeze(1)

    @contextmanager
    def _train_mode(self):
        self.model.train()  # enables gradient checkpointing; Qwen has no dropout to switch on
        try:
            yield
        finally:
            self.model.eval()

    def train(self, items: Sequence[TrainItem], cfg: LossConfig) -> dict:
        with self._train_mode():
            return self._train(items, cfg)

    def _train(self, items: Sequence[TrainItem], cfg: LossConfig) -> dict:
        import torch

        for g in self.opt.param_groups:
            g["lr"] = cfg.learning_rate
        renorm = cfg.loss == "renormalised"
        n = sum(len(it.actions) for it in items)
        stats = {}
        for p in range(cfg.passes):
            self.opt.zero_grad()
            total, clipped, ratios = 0.0, 0, []
            for it in items:
                new = self._decision_logprobs(it, renorm)
                old = torch.as_tensor(it.old_logprob_legal if renorm else it.old_logprob_full,
                                      dtype=new.dtype, device=self.device)
                adv = torch.as_tensor(it.advantages, dtype=new.dtype, device=self.device)
                ratio = torch.exp(new - old)
                surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
                loss = -surr.sum() / max(n, 1)
                loss.backward()
                total += float(loss.detach())
                clipped += int((((adv >= 0) & (ratio > 1 + cfg.clip)) | ((adv < 0) & (ratio < 1 - cfg.clip))).sum())
                ratios.append(ratio.detach().cpu().numpy())
            self.opt.step()
            self.weights_version += 1
            r = np.concatenate(ratios) if ratios else np.ones(1)
            stats = {"loss": total, "clipfrac": clipped / max(n, 1), "ratio": float(r.mean())}
            if p == 0:
                stats["ratio_first_pass"] = float(r.mean())
            self.usage.train_calls += 1
            self.usage.training_tokens += sum(len(it.tokens) for it in items)
        return stats

    def supervised(self, examples, learning_rate: float) -> dict:
        with self._train_mode():
            return self._supervised(examples, learning_rate)

    def _supervised(self, examples, learning_rate: float) -> dict:
        import torch

        for g in self.opt.param_groups:
            g["lr"] = learning_rate
        self.opt.zero_grad()
        total = 0.0
        for prompt, completion in examples:
            toks = list(prompt) + list(completion)
            ids = torch.as_tensor(toks, device=self.device)[None]
            logits = self.model(input_ids=ids).logits[0, :-1].float()
            tgt = ids[0, 1:]
            w = torch.zeros(len(toks) - 1, device=self.device)
            w[len(prompt) - 1:] = 1.0
            nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
            loss = (nll * w).sum() / w.sum() / len(examples)
            loss.backward()
            total += float(loss)
        self.opt.step()
        self.weights_version += 1
        self.usage.train_calls += 1
        return {"loss": total}

    # ----- checkpoints and generation --------------------------------------------------------
    def save(self, name: str) -> str:
        import torch

        path = self.checkpoint_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        torch.save(self.opt.state_dict(), path / "optimizer.pt")
        self._saved.append(path)
        # Optimiser state is only needed to resume, and a run resumes from one of its last two
        # saves (the newest may not be recorded yet). Older saves keep their adapters only.
        for old in self._saved[:-2]:
            (old / "optimizer.pt").unlink(missing_ok=True)
        return str(path)

    def load(self, path: str) -> None:
        """Restores adapters and optimiser state saved by :meth:`save` (for resuming a run)."""
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(self.model, load_file(str(Path(path) / "adapter_model.safetensors"),
                                                        device=str(self.device)))
        self.opt.load_state_dict(torch.load(Path(path) / "optimizer.pt", map_location=self.device))
        self.weights_version += 1

    def generate(self, tokens, max_tokens, temperature=1.0, stop=None, seed=None) -> list[int]:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        ids = torch.as_tensor(list(tokens), device=self.device)[None]
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, max_new_tokens=max_tokens, do_sample=True,
                                      temperature=temperature, pad_token_id=self.pad_id)
        new = out[0, ids.shape[1]:].tolist()
        if stop:
            text = self.tokenizer.decode(new)
            for s in stop:
                if s in text:
                    new = self.tokenizer.encode(text.split(s)[0])
                    break
        self.usage.add_score(len(tokens), 0, len(new))
        return new
