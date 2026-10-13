"""Evaluation of a scorer (frozen or trained) on paired episodes, with S scores."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from ..config import PROFILES, get_world
from ..llm.policy import EpisodeSpec, SamplingConfig, Trajectory, rollout
from ..render.rendering import PromptOptions, Rendering
from ..render.transcript import ChatFrame
from ..scoring import Scorer as ScoreNormaliser

EVAL_SALT = 777  # one salt for every evaluation, so checkpoints are compared on common draws


def eval_specs(seeds: Iterable[int], worlds: Sequence[str], rendering: Rendering,
               bodies: Sequence[str] = ("A", "B"), options: PromptOptions = PromptOptions(),
               split: str = "dev", episode: int = 0) -> list[EpisodeSpec]:
    specs = []
    for w in worlds:
        world = get_world(w)
        for s in seeds:
            for b in bodies:
                specs.append(EpisodeSpec(world, PROFILES[b], s, rendering, episode, options, split))
    return specs


@dataclass
class EvalResult:
    episodes: list[dict]
    cells: dict[str, dict]
    mean_score: float
    welfare_violations: int
    unclean: int
    illegal_mass: float

    def as_dict(self) -> dict:
        return {"cells": self.cells, "mean_score": self.mean_score,
                "welfare_violations": self.welfare_violations, "unclean": self.unclean,
                "illegal_mass": self.illegal_mass, "n": len(self.episodes)}


def score_trajectories(trajs: Sequence[Trajectory], normaliser: ScoreNormaliser) -> EvalResult:
    eps, cells = [], defaultdict(list)
    for t in trajs:
        s = normaliser.score(t.world, t.body, t.episode_return)
        rec = t.summary()
        rec["score"] = s
        rec["regret"] = normaliser.regret(t.world, t.body, t.episode_return)
        eps.append(rec)
        cells[f"{t.world}/{t.body}"].append(rec)
    out = {}
    for k, recs in cells.items():
        sc = [r["score"] for r in recs if r["score"] is not None]
        out[k] = {"n": len(recs), "mean_return": float(np.mean([r["return"] for r in recs])),
                  "mean_score": float(np.mean(sc)) if sc else None,
                  "welfare_violations": sum(r["outcomes"]["welfare_violation"] for r in recs)}
    all_sc = [r["score"] for r in eps if r["score"] is not None]
    return EvalResult(eps, out, float(np.mean(all_sc)) if all_sc else float("nan"),
                      sum(r["outcomes"]["welfare_violation"] for r in eps),
                      sum(bool(r["violations"]) for r in eps),
                      float(np.mean([r["illegal_mass"] for r in eps])) if eps else 0.0)


def evaluate(scorer, tokenizer, specs: Sequence[EpisodeSpec], normaliser: ScoreNormaliser,
             salt: int = EVAL_SALT, temperature: float = 1.0, greedy: bool = False,
             batch_size: int = 64, frame: Optional[ChatFrame] = None) -> tuple[EvalResult, list[Trajectory]]:
    trajs: list[Trajectory] = []
    for i in range(0, len(specs), batch_size):
        trajs.extend(rollout(specs[i:i + batch_size], scorer, tokenizer,
                             SamplingConfig(temperature, greedy, salt), frame))
    return score_trajectories(trajs, normaliser), trajs
