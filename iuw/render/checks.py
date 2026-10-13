"""Build-stage 3 checks for renderings.

* Information equality: a known-law planner reading a rendering (lines parsed
  back into observations) earns exactly what it earns from the raw
  observations, in both renderings.
* Token parity: tutorials and whole transcripts within 10% across renderings.
* Context budget: prompt and tutorial within 600 tokens, each decision within
  30, and no episode, notebook included, ever reaches 8,192 tokens.
* Forbidden statements: tutorials never state or hint at the hidden law.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..config import PROFILES, ContextBudget, Profile, WorldConfig
from ..env.runner import EpisodeContext, Policy, run_episode
from ..env.water import Observation, WaterEnv
from ..llm.tokenizer import Tokenizer
from .quiz import check_quiz_items, quiz_items
from .rendering import PromptOptions, Rendering, forbidden_hits
from .transcript import ChatFrame, Transcript


class ParsedPolicy:
    """Wraps a policy so it only sees observations reconstructed from rendered text."""

    privileged = False

    def __init__(self, inner: Policy, rendering: Rendering):
        self.inner = inner
        self.rendering = rendering
        self.name = f"{inner.name}|{rendering.name}"

    def begin(self, ctx: EpisodeContext) -> None:
        self.prev: Optional[Observation] = None
        self.body = ctx.body
        self.inner.begin(ctx)

    def act(self, obs: Observation) -> int:
        text = self.rendering.line(obs)
        parsed = self.rendering.parse_line(text, self.prev, self.body, obs.noise)
        # Burden is shown to one decimal: that is the information both renderings carry.
        self.prev = parsed
        return self.inner.act(parsed)


def information_equality(world: WorldConfig, profile: Profile, seeds: Iterable[int],
                         policy_factory, renderings: Sequence[Rendering]) -> dict:
    """Same actions and returns from every rendering, per seed."""
    diffs = []
    for s in seeds:
        base = run_episode(world, profile, s, policy_factory())
        for r in renderings:
            res = run_episode(world, profile, s, ParsedPolicy(policy_factory(), r))
            if res.actions != base.actions or res.episode_return != base.episode_return:
                diffs.append((r.name, s))
    return {"equal": not diffs, "differences": diffs}


@dataclass
class BudgetReport:
    rendering: str
    prompt_tokens: int  # system prompt, tutorial and header, notebook excluded
    max_per_decision: int
    max_total: int
    episodes: int
    overflow: bool
    notebook_tokens: int = 0

    def within(self, budget: ContextBudget) -> bool:
        return (not self.overflow and self.prompt_tokens <= budget.system_and_tutorial
                and self.max_per_decision <= budget.per_decision_max
                and self.max_total <= budget.total)


def transcript_for_episode(world: WorldConfig, profile: Profile, seed: int, policy: Policy,
                           rendering: Rendering, tokenizer: Tokenizer,
                           options: PromptOptions = PromptOptions(),
                           frame: Optional[ChatFrame] = None) -> Transcript:
    tr = Transcript(tokenizer, rendering, frame or ChatFrame(), options)
    env = WaterEnv(world, profile, seed)
    obs, _ = env.reset()
    policy.begin(EpisodeContext(seed, 0, profile.name, world.name,
                                env if getattr(policy, "privileged", False) else None))
    tr.start(profile.name, world.noise)
    while not env.done:
        tr.observe(obs)
        a = policy.act(obs)
        tr.act(a)
        obs, *_ = env.step(a)
    return tr


def budget_report(rendering: Rendering, tokenizer: Tokenizer, episodes: Iterable[tuple],
                  notebook: Optional[str] = None, budget: ContextBudget = ContextBudget()) -> BudgetReport:
    """``episodes`` yields (world, profile, seed, policy) tuples."""
    opts = PromptOptions(notebook=notebook)
    max_dec = max_total = prompt = n = 0
    overflow = False
    nb_tokens = len(tokenizer.encode(notebook)) if notebook else 0
    for world, profile, seed, policy in episodes:
        try:
            tr = transcript_for_episode(world, profile, seed, policy, rendering, tokenizer, opts)
        except Exception as e:  # ContextOverflow means the episode would have been truncated
            from .transcript import ContextOverflow
            if isinstance(e, ContextOverflow):
                overflow = True
                continue
            raise
        n += 1
        prompt = max(prompt, tr.prompt_tokens_without_notebook())
        pd = tr.per_decision_tokens()
        max_dec = max(max_dec, max(pd) if pd else 0)
        max_total = max(max_total, len(tr.tokens) + 1)  # +1: room to sample the next action
    return BudgetReport(rendering.name, prompt, max_dec, max_total, n, overflow, nb_tokens)


def parity(a: int, b: int) -> float:
    return abs(a - b) / max(a, b)


def tutorial_checks(renderings: Sequence[Rendering], tokenizer: Tokenizer) -> dict:
    out = {"forbidden": {}, "quiz_problems": {}, "tutorial_tokens": {}}
    for r in renderings:
        out["forbidden"][r.name] = forbidden_hits(r.tutorial())
        out["quiz_problems"][r.name] = check_quiz_items(quiz_items(r))
        out["tutorial_tokens"][r.name] = len(tokenizer.encode(r.system_prompt()))
    toks = list(out["tutorial_tokens"].values())
    out["max_parity_gap"] = max(parity(x, y) for x in toks for y in toks)
    out["clean"] = (not any(out["forbidden"].values()) and not any(out["quiz_problems"].values())
                    and out["max_parity_gap"] <= 0.10)
    return out
