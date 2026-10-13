"""Frozen-model checks (build stage 4) and the interface quiz (stage 3).

* Zero-shot competence in all four cells (renderings x worlds).
* Explicit rules: tell the frozen model the hidden law in either rendering.
* Words evoke priors: compare frozen Named and Masked behaviour in the
  reference world, both on free rollouts and on identical replayed histories.
  If they are the same, the Named words are too weak to test anything.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..analysis.behaviour import aggregate, behaviour_profile, js_divergence
from ..config import PROFILES, get_world
from ..inspection.replay import record_history, replay_distributions
from ..llm.backend import Backend, ScoreRequest
from ..llm.policy import EpisodeSpec
from ..policies.baselines import RandomPolicy
from ..policies.planner import KnownLawPlanner
from ..render.quiz import quiz_gate, run_quiz
from ..render.rendering import PromptOptions, make_rendering
from ..scoring import Scorer as ScoreNormaliser
from ..seeds import SPLITS
from ..train.evaluate import EVAL_SALT, evaluate

WORLDS = ("reference", "counterfactual")


def letter_scorer(scorer):
    def score(prompt: list[int], candidates: list[int]) -> np.ndarray:
        return scorer.score_batch([ScoreRequest(prompt, candidates)])[0].logprobs
    return score


def interface_quiz(backend: Backend, mapping_index: int = 0) -> dict:
    sc = letter_scorer(backend.scorer())
    named = run_quiz(make_rendering("named"), backend.tokenizer, sc)
    masked = run_quiz(make_rendering("masked", mapping_index), backend.tokenizer, sc)
    return {"named": named.__dict__, "masked": masked.__dict__, "gate": quiz_gate(named, masked)}


def frozen_specs(rendering_kind: str, explicit_rules: bool, split: str = "test",
                 worlds: Sequence[str] = WORLDS) -> list[EpisodeSpec]:
    """32 test seeds x 2 bodies x 2 worlds = 128 episodes; Masked mappings rotate over seeds."""
    specs = []
    for w in worlds:
        world = get_world(w)
        opts = PromptOptions(explicit_rules=world.q if explicit_rules else None)
        for i, s in enumerate(SPLITS[split]):
            r = make_rendering(rendering_kind, i % 3)
            for b in ("A", "B"):
                specs.append(EpisodeSpec(world, PROFILES[b], s, r, 0, opts, split))
    return specs


def frozen_cells(backend: Backend, normaliser: ScoreNormaliser, split: str = "test") -> dict:
    """Zero-shot and explicit-rule scores in all four cells (2 x 2 x 128 episodes = 512)."""
    out = {}
    scorer = backend.scorer()
    for cond, explicit in (("zero-shot", False), ("explicit-rules", True)):
        for kind in ("named", "masked"):
            res, trajs = evaluate(scorer, backend.tokenizer, frozen_specs(kind, explicit, split),
                                  normaliser, salt=EVAL_SALT)
            behaviour = {w: aggregate([behaviour_profile(t.actions) for t in trajs if t.world == w])
                         for w in WORLDS}
            out[f"{cond}/{kind}"] = {**res.as_dict(), "behaviour": behaviour, "episodes": res.episodes}
    zs = out["zero-shot/named"]["cells"]
    named_cf = [c["mean_score"] for k, c in zs.items() if k.startswith("counterfactual/")]
    out["zero_shot_named_cf_S"] = float(np.mean(named_cf)) if named_cf else float("nan")
    return out


def words_evoke_priors(backend: Backend, seeds: Sequence[int] = SPLITS["dev"][:8],
                       world: str = "reference", decisions: int = 60) -> dict:
    """JS divergence between frozen Named and Masked action distributions on identical histories."""
    w = get_world(world)
    scorer = backend.scorer()
    divs = []
    for i, s in enumerate(seeds):
        for b in ("A", "B"):
            policy = KnownLawPlanner(w, PROFILES) if i % 2 == 0 else RandomPolicy()
            h = record_history(w, PROFILES[b], s, policy, max_decisions=decisions)
            pn = replay_distributions(h, scorer, backend.tokenizer, make_rendering("named"))
            pm = replay_distributions(h, scorer, backend.tokenizer, make_rendering("masked", i % 3))
            divs.extend(js_divergence(x, y) for x, y in zip(pn, pm))
    d = np.asarray(divs)
    return {"mean_js": float(d.mean()), "median_js": float(np.median(d)), "n_decisions": int(len(d)),
            "identical": bool(d.mean() < 1e-3),
            "note": "if identical, the Named words are too weak to test anything"}
