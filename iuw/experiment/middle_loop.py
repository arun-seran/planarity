"""Middle loop: choose training settings on development seeds only.

Every proposal starts from the same checkpoint and changes one setting
(learning rate, entropy bonus or reference-KL schedule), trains fresh with an
equal budget of 100 episodes, and is compared with the incumbent on the 16
shared development seeds (x 2 bodies x both worlds = 64 paired episodes).
A proposal is accepted only if the lower 95% bound of the paired improvement
clears a threshold fixed in advance. Stop after 8 proposals or 3 rejections
in a row; freeze the winner; then open the sealed test once, elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from ..analysis.stats import paired_bootstrap
from ..llm.backend import Backend, LossConfig
from ..render.rendering import PromptOptions, make_rendering
from ..scoring import Scorer as ScoreNormaliser
from ..seeds import SPLITS
from ..train.advantages import KLSchedule
from ..train.evaluate import EVAL_SALT, eval_specs, evaluate
from ..train.inner import RunConfig, run_training

# 100 training episodes per proposal: 10 updates x 5 seeds x 2 bodies.
PROPOSAL_UPDATES = 10
PROPOSAL_SEEDS_PER_UPDATE = 5


@dataclass(frozen=True)
class Proposal:
    setting: str  # "learning_rate" | "entropy_coef" | "kl"
    value: object

    def apply(self, cfg: RunConfig) -> RunConfig:
        if self.setting == "learning_rate":
            return replace(cfg, loss=replace(cfg.loss, learning_rate=float(self.value)))
        if self.setting == "entropy_coef":
            return replace(cfg, entropy_coef=float(self.value))
        if self.setting == "kl":
            kind, coef = self.value
            return replace(cfg, kl=KLSchedule(kind, float(coef), cfg.updates))
        raise ValueError(f"the middle loop changes one of learning_rate, entropy_coef, kl; got {self.setting}")


DEFAULT_PROPOSALS = (
    Proposal("learning_rate", 3e-5), Proposal("learning_rate", 1e-4),
    Proposal("kl", ("fixed", 0.05)), Proposal("kl", ("anneal", 0.05)),
    Proposal("entropy_coef", 0.01), Proposal("learning_rate", 3e-6),
    Proposal("kl", ("fixed", 0.2)), Proposal("entropy_coef", 0.05),
)


@dataclass
class MiddleLoopResult:
    incumbent: dict
    history: list[dict] = field(default_factory=list)
    stopped_because: str = ""


def _dev_scores(cfg: RunConfig, backend: Backend, normaliser: ScoreNormaliser) -> dict[tuple, float]:
    rendering = make_rendering(cfg.rendering, cfg.mapping_index)
    specs = eval_specs(SPLITS["dev"], cfg.eval_worlds, rendering, cfg.bodies, PromptOptions(), "dev")
    res, _ = evaluate(backend.scorer(), backend.tokenizer, specs, normaliser, salt=EVAL_SALT)
    return {(e["world"], e["body"], e["seed"]): (e["score"] if e["score"] is not None else 0.0)
            for e in res.episodes}


def middle_loop(base: RunConfig, backend_factory: Callable[[RunConfig], Backend],
                normaliser: ScoreNormaliser, threshold: float,
                proposals: Sequence[Proposal] = DEFAULT_PROPOSALS, max_proposals: int = 8,
                max_rejections: int = 3, out_dir: Optional[Path] = None, progress=None) -> MiddleLoopResult:
    """``threshold`` is the practical improvement in S, fixed before the loop starts."""

    def train_and_score(cfg: RunConfig, tag: str) -> dict[tuple, float]:
        cfg = replace(cfg, run_id=f"{base.run_id}-{tag}", updates=PROPOSAL_UPDATES,
                      seeds_per_update=PROPOSAL_SEEDS_PER_UPDATE, eval_updates=())
        be = backend_factory(cfg)
        run_training(cfg, be, normaliser, out_dir / cfg.run_id if out_dir else None, progress)
        return _dev_scores(cfg, be, normaliser)

    incumbent_cfg = base
    incumbent_scores = train_and_score(base, "incumbent")
    result = MiddleLoopResult(incumbent={"config": base.to_dict()})
    rejections = 0
    for i, prop in enumerate(proposals[:max_proposals]):
        cand_cfg = prop.apply(incumbent_cfg)
        cand_scores = train_and_score(cand_cfg, f"p{i}")
        keys = sorted(set(incumbent_scores) & set(cand_scores))
        diff = np.array([cand_scores[k] - incumbent_scores[k] for k in keys])
        lo, hi = paired_bootstrap(diff, seed=i)
        accepted = bool(lo > threshold)
        result.history.append({"proposal": prop.setting, "value": prop.value,
                               "mean_improvement": float(diff.mean()), "lower95": lo, "upper95": hi,
                               "accepted": accepted})
        if progress:
            progress(f"[middle loop] {prop.setting}={prop.value}: improvement {diff.mean():+.3f} "
                     f"(lower 95% {lo:+.3f}) -> {'accept' if accepted else 'reject'}")
        if accepted:
            incumbent_cfg, incumbent_scores, rejections = cand_cfg, cand_scores, 0
        else:
            rejections += 1
            if rejections >= max_rejections:
                result.stopped_because = f"{max_rejections} rejections in a row"
                break
    else:
        result.stopped_because = f"{min(max_proposals, len(proposals))} proposals"
    result.incumbent = {"config": incumbent_cfg.to_dict()}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "middle_loop.json").write_text(json.dumps(result.__dict__, indent=2, default=str))
    return result
