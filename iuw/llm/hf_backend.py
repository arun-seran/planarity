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


class HFScorer:
    def __init__(self, backend: "HFBackend", reference: bool = False, batch_size: int = 8):
        self.backend = backend
        self.reference = reference
        self.batch_size = batch_size

    def _logits(self, batch: list[list[int]]):
        import torch

        be = self.backend
        L = max(len(t) for t in batch)
        pad = be.pad_id
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, t in enumerate(batch):
            ids[i, :len(t)] = torch.as_tensor(t)
            mask[i, :len(t)] = 1
        with torch.no_grad(), be.policy_context(self.reference):
            out = be.model(input_ids=ids.to(be.device), attention_mask=mask.to(be.device))
        return out.logits.float().log_softmax(-1).cpu()

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        out = []
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i:i + self.batch_size]
            lp = self._logits([list(r.tokens) for r in chunk])
            for j, r in enumerate(chunk):
                row = lp[j, len(r.tokens) - 1, list(r.candidates)].numpy()
                self.backend.usage.add_score(len(r.tokens), 0)
                out.append(ScoreResult(row.astype(float), len(r.tokens)))
        return out

    def score_positions(self, tokens, positions, candidates) -> np.ndarray:
        lp = self._logits([list(tokens)])[0]
        self.backend.usage.add_score(len(tokens), 0, 0)
        return np.stack([lp[p - 1, list(candidates)].numpy() for p in positions]).astype(float)


class HFBackend:
    name = "hf"

    def __init__(self, model_name_or_path: str, rank: int = 32, lora_alpha: Optional[int] = None,
                 device: Optional[str] = None, dtype: str = "bfloat16", seed: int = 0,
                 init_adapter: Optional[str] = None, model=None, tokenizer=None,
                 target_modules="all-linear"):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model

        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                         torch_dtype=getattr(torch, dtype))
            tokenizer = tokenizer or HFTokenizer(AutoTokenizer.from_pretrained(model_name_or_path),
                                                 model_name_or_path)
        if init_adapter:
            self.model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        else:
            cfg = LoraConfig(r=rank, lora_alpha=lora_alpha or rank, lora_dropout=0.0,
                             target_modules=target_modules, task_type="CAUSAL_LM")
            self.model = get_peft_model(model, cfg)
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.action_ids = action_token_ids(self.tokenizer)
        self.pad_id = getattr(getattr(tokenizer, "tok", None), "pad_token_id", None) or 0
        self.usage = UsageStats()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=1e-5, betas=(0.9, 0.95), eps=1e-12,
                                     weight_decay=0.0)
        self._reference_state: Optional[dict] = None
        self.checkpoint_dir = Path("runs/hf-checkpoints")

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
        logits = self.model(input_ids=ids).logits[0].float()
        rows = torch.as_tensor([p - 1 for p in it.positions], device=self.device)
        acts = torch.as_tensor(it.actions, device=self.device)
        if not renormalised:
            lp = logits[rows].log_softmax(-1)
            tok = torch.as_tensor([self.action_ids[a] for a in it.actions], device=self.device)
            return lp.gather(1, tok[:, None]).squeeze(1)
        act_logits = logits[rows][:, self.action_ids]
        mask = torch.full_like(act_logits, float("-inf"))
        for j, legal in enumerate(it.legal):
            mask[j, list(legal)] = 0.0
        lp = (act_logits + mask).log_softmax(-1)
        return lp.gather(1, acts[:, None]).squeeze(1)

    def train(self, items: Sequence[TrainItem], cfg: LossConfig) -> dict:
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
                total += float(loss)
                clipped += int((((adv >= 0) & (ratio > 1 + cfg.clip)) | ((adv < 0) & (ratio < 1 - cfg.clip))).sum())
                ratios.append(ratio.detach().cpu().numpy())
            self.opt.step()
            r = np.concatenate(ratios) if ratios else np.ones(1)
            stats = {"loss": total, "clipfrac": clipped / max(n, 1), "ratio": float(r.mean())}
            if p == 0:
                stats["ratio_first_pass"] = float(r.mean())
            self.usage.train_calls += 1
            self.usage.training_tokens += sum(len(it.tokens) for it in items)
        return stats

    def supervised(self, examples, learning_rate: float) -> dict:
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
        self.usage.train_calls += 1
        return {"loss": total}

    # ----- checkpoints and generation --------------------------------------------------------
    def save(self, name: str) -> str:
        import torch

        path = self.checkpoint_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        torch.save(self.opt.state_dict(), path / "optimizer.pt")
        return str(path)

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
