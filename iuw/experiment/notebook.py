"""Notebook hill climb: a frozen model improves a 1,024-token notebook by selection.

Run once per rendering. 25 rounds; each round collects 8 experience episodes
with the incumbent notebook, proposes one rewrite capped at 1,024 output
tokens, and compares incumbent and candidate on 8 fresh seeds from the
200-seed notebook-selection pool (16 selection episodes: each seed once per
notebook, with bodies alternating by seed so A and B are balanced). The pool
is used exactly once. Accept only if the paired mean improvement beats a
margin fixed in advance. Then freeze the notebook and run the sealed test.

Total: 25 x (8 + 16) = 600 rollout episodes and at most 25 rewrite calls.
Selection feedback counts as learning experience and is logged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import PROFILES, ContextBudget, get_world
from ..llm.backend import Backend
from ..llm.policy import EpisodeSpec, SamplingConfig, Trajectory, rollout
from ..render.rendering import PromptOptions, make_rendering
from ..render.transcript import ChatFrame
from ..scoring import Scorer as ScoreNormaliser
from ..seeds import SPLITS, Seal, training_batch
from ..train.evaluate import EVAL_SALT, eval_specs, evaluate

ROUNDS = 25
EXPERIENCE = 8
SELECTION_SEEDS = 8
MAX_NOTEBOOK_TOKENS = 1024

REWRITE_INSTRUCTION = (
    "You keep a notebook that is shown to you at the start of every episode. Below are your "
    "current notebook and short records of recent episodes. Write an improved notebook that "
    "will help you score more. Write only the notebook text, at most 700 words.")


@dataclass
class NotebookResult:
    rendering: str
    world: str
    notebook: str
    history: list[dict] = field(default_factory=list)
    episodes_used: int = 0
    rewrite_calls: int = 0


def episode_record(t: Trajectory, max_actions: int = 60) -> str:
    acts = "".join(str(a) for a in t.actions[:max_actions])
    o = t.outcomes
    return (f"body {t.body}: score {t.episode_return:.2f}, delivered {o['liters']}, energy used "
            f"{o['energy_used']}, actions {acts}{'...' if len(t.actions) > max_actions else ''}")


def _truncate_tokens(tokenizer, text: str, limit: int) -> str:
    ids = tokenizer.encode(text)
    return text if len(ids) <= limit else tokenizer.decode(ids[:limit])


def notebook_climb(backend: Backend, rendering_kind: str, normaliser: ScoreNormaliser,
                   world: str = "counterfactual", margin: float = 1.0, mapping_index: int = 0,
                   initial: str = "(empty)", out_dir: Optional[Path] = None,
                   temperature: float = 1.0, rounds: int = ROUNDS, progress=None) -> NotebookResult:
    """``margin`` (reward units, paired mean) is fixed before the climb starts."""
    rendering = make_rendering(rendering_kind, mapping_index)
    w = get_world(world)
    tok = backend.tokenizer
    frame = ChatFrame()
    pool = SPLITS["notebook"]
    assert len(pool) >= rounds * SELECTION_SEEDS
    incumbent = initial
    res = NotebookResult(rendering.name, world, incumbent)
    scorer = backend.scorer()  # frozen: weights never change

    def run(specs, salt):
        return rollout(specs, scorer, tok, SamplingConfig(temperature, False, salt), frame)

    for r in range(rounds):
        opts = PromptOptions(notebook=incumbent)
        exp_specs = [EpisodeSpec(w, PROFILES[b], s, rendering, ep, opts, "notebook-experience")
                     for s, ep in training_batch(r + 1, EXPERIENCE // 2) for b in ("A", "B")]
        exp = run(exp_specs, 10_000 + r)
        records = "\n".join(episode_record(t) for t in exp)
        user = (f"{REWRITE_INSTRUCTION}\n\nCurrent notebook:\n{incumbent}\n\n"
                f"Recent episodes:\n{records}")
        prompt = tok.encode(frame.generation(rendering.system_prompt(), user))
        out = backend.generate(prompt, MAX_NOTEBOOK_TOKENS, temperature, stop=["<|im_end|>"], seed=r)
        candidate = _truncate_tokens(tok, tok.decode(out).split("<|im_end|>")[0].strip(), MAX_NOTEBOOK_TOKENS)
        res.rewrite_calls += 1
        seeds = pool[r * SELECTION_SEEDS:(r + 1) * SELECTION_SEEDS]
        sel = []
        for nb in (incumbent, candidate):
            specs = [EpisodeSpec(w, PROFILES["A" if i % 2 == 0 else "B"], s, rendering, 0,
                                 PromptOptions(notebook=nb), "notebook-selection")
                     for i, s in enumerate(seeds)]
            sel.append(run(specs, EVAL_SALT))
        diff = np.array([c.episode_return - i.episode_return for i, c in zip(sel[0], sel[1])])
        accepted = bool(diff.mean() > margin)
        res.episodes_used += len(exp) + len(sel[0]) + len(sel[1])
        res.history.append({"round": r, "accepted": accepted, "mean_improvement": float(diff.mean()),
                            "incumbent_mean": float(np.mean([t.episode_return for t in sel[0]])),
                            "candidate_mean": float(np.mean([t.episode_return for t in sel[1]])),
                            "candidate": candidate, "experience_returns": [t.episode_return for t in exp]})
        if progress:
            progress(f"[notebook {rendering.name}] round {r + 1}/{rounds}: {diff.mean():+.2f} -> "
                     f"{'accept' if accepted else 'keep'}")
        if accepted:
            incumbent = candidate
    res.notebook = incumbent
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "notebook.json").write_text(json.dumps(res.__dict__, indent=2))
    return res


def notebook_sealed_test(backend: Backend, result: NotebookResult, normaliser: ScoreNormaliser,
                         out_dir: Path, mapping_index: int = 0) -> dict:
    Seal(out_dir / "SEALED_TEST_OPENED").open(f"notebook {result.rendering}")
    rendering = make_rendering(result.rendering.split("-")[0], mapping_index)
    specs = eval_specs(SPLITS["test"], ("reference", "counterfactual"), rendering, ("A", "B"),
                       PromptOptions(notebook=result.notebook), "test")
    ev, _ = evaluate(backend.scorer(), backend.tokenizer, specs, normaliser, salt=EVAL_SALT)
    (out_dir / "test.json").write_text(json.dumps(ev.as_dict(), indent=2))
    return ev.as_dict()
