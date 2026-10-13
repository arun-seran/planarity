"""The return trip: evaluate the adapted model back in the reference world.

A deficit there has three possible causes: interference that overwrote the
old policy, misidentifying which world it is in, or failing to retrieve a
policy it still holds. A successful return can also coexist with changed
representations. The conditions separate the causes:

* context reset: fresh episodes (the default; only the weights carry learning);
* explicit world cue in the prompt;
* matched histories: replay identical early-episode transcripts, then let the
  model continue;
* reminder of the old law (the explicit rule for q = 0);
* relearning speed: relearning faster than the original learning means the
  old policy was retained.

Only a deficit that survives cues and reminders, and shows no faster
relearning, points toward overwriting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from ..config import PROFILES, REFERENCE, WorldConfig
from ..env.runner import run_episode
from ..llm.policy import EpisodeSpec
from ..policies.planner import KnownLawPlanner
from ..render.rendering import PromptOptions, Rendering
from ..scoring import Scorer as ScoreNormaliser
from ..seeds import SPLITS
from ..train.evaluate import EVAL_SALT, evaluate

DEFAULT_CUE = "This episode takes place in the first {site} you ever operated, before any training."


def _specs(rendering: Rendering, seeds, options: PromptOptions, world: WorldConfig,
           prefix_len: int = 0) -> list[EpisodeSpec]:
    out = []
    for s in seeds:
        for b in ("A", "B"):
            prefix = ()
            if prefix_len:
                hist = run_episode(world, PROFILES[b], s, KnownLawPlanner(world, PROFILES))
                prefix = tuple(hist.actions[:prefix_len])
            out.append(EpisodeSpec(world, PROFILES[b], s, rendering, 0, options, "return-trip", prefix))
    return out


def return_trip(adapted_scorer, base_scorer, tokenizer, rendering: Rendering,
                normaliser: ScoreNormaliser, seeds: Sequence[int] = SPLITS["dev"],
                world: WorldConfig = REFERENCE, prefix_len: int = 20,
                cue: Optional[str] = None) -> dict:
    """Mean S in the reference world for each condition, for adapted and base weights."""
    cue = (cue or DEFAULT_CUE).format(site=rendering.vocab.site)
    conditions = {
        "reset": PromptOptions(),
        "world-cue": PromptOptions(world_cue=cue),
        "reminder": PromptOptions(explicit_rules=world.q),
        "cue+reminder": PromptOptions(world_cue=cue, explicit_rules=world.q),
    }
    out: dict = {}
    for who, scorer in (("adapted", adapted_scorer), ("base", base_scorer)):
        for name, opts in conditions.items():
            res, _ = evaluate(scorer, tokenizer, _specs(rendering, seeds, opts, world), normaliser,
                              salt=EVAL_SALT)
            out[f"{who}/{name}"] = res.mean_score
        res, _ = evaluate(scorer, tokenizer, _specs(rendering, seeds, PromptOptions(), world, prefix_len),
                          normaliser, salt=EVAL_SALT)
        out[f"{who}/matched-history"] = res.mean_score
    return out


def diagnose(scores: dict, relearning: Optional[dict] = None, tolerance: float = 0.05) -> dict:
    """Read a deficit with three explanations, not one.

    ``relearning``: {"original": adaptation interactions, "relearn": interactions} from
    analysis.stats.adaptation_speed on the original and the return-trip learning curves.
    """
    base = scores["base/reset"]
    deficit = base - scores["adapted/reset"]
    if deficit <= tolerance:
        return {"deficit": deficit, "reading": "no deficit (representations may still have changed)"}
    restored = {k.split("/")[1]: base - v <= tolerance for k, v in scores.items()
                if k.startswith("adapted/") and k != "adapted/reset"}
    faster = None
    if relearning and relearning.get("original") is not None and relearning.get("relearn") is not None:
        faster = relearning["relearn"] < relearning["original"]
    if restored.get("world-cue"):
        reading = "context misidentification: a world cue restores the old policy"
    elif restored.get("reminder") or restored.get("matched-history"):
        reading = "retrieval failure: a reminder or matched history restores the old policy"
    elif faster:
        reading = "retained but not expressed: relearning is faster than the original learning"
    elif faster is None:
        reading = "deficit survives cues and reminders; run the relearning test before interpreting"
    else:
        reading = "points toward overwriting: survives cues and reminders, no faster relearning"
    return {"deficit": deficit, "restored_by": restored, "faster_relearning": faster, "reading": reading}
