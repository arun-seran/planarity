"""Rollouts: drive the simulator with a language model through a rendering.

Each decision scores the six action tokens at the next position (full
softmax), renormalises over the legal ones, and samples there with a keyed
uniform draw. Both log-probabilities are stored: the full-softmax one for
Tinker's built-in loss, the legal-renormalised one for the custom loss. The
probability mass left on illegal tokens is tracked per decision.

Episodes in a batch advance in lockstep so that every decision step issues
one batch of concurrent scoring requests.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..config import ContextBudget, Profile, WorldConfig
from ..env.audit import audit_env
from ..env.water import WaterEnv
from ..render.rendering import PromptOptions, Rendering
from ..render.transcript import ChatFrame, ContextOverflow, Transcript
from ..rng import Stream, keyed_uniform, namespace_key
from .backend import ScoreRequest, Scorer, legal_log_softmax

POLICY_NS = namespace_key("iuw/policy-sampling")


@dataclass
class EpisodeSpec:
    world: WorldConfig
    profile: Profile
    seed: int
    rendering: Rendering
    episode: int = 0
    options: PromptOptions = field(default_factory=PromptOptions)
    split: str = "train"
    prefix_actions: tuple[int, ...] = ()  # replayed, not sampled (matched-history tests)


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    greedy: bool = False
    salt: int = 0  # distinguishes runs and updates; evaluations reuse one salt for pairing


@dataclass
class Decision:
    position: int
    action: int
    legal: tuple[int, ...]
    cycle: int
    phase: str
    logprobs_all: np.ndarray
    lp_full: float
    lp_legal: float
    illegal_mass: float
    reward: float = 0.0
    forced: bool = False


@dataclass
class Trajectory:
    world: str
    body: str
    seed: int
    episode: int
    rendering: str
    split: str
    tokens: list[int]
    prompt_len: int
    decisions: list[Decision]
    outcomes: dict
    violations: list[str]
    terminated: bool
    truncated: bool
    termination: Optional[str]
    ref_lp_legal: Optional[np.ndarray] = None  # filled when a reference-KL schedule is on

    @property
    def episode_return(self) -> float:
        return float(self.outcomes["episode_return"])

    @property
    def rewards(self) -> np.ndarray:
        return np.array([d.reward for d in self.decisions], dtype=float)

    @property
    def positions(self) -> list[int]:
        return [d.position for d in self.decisions]

    @property
    def actions(self) -> list[int]:
        return [d.action for d in self.decisions]

    @property
    def illegal_mass(self) -> float:
        return float(np.mean([d.illegal_mass for d in self.decisions])) if self.decisions else 0.0

    def summary(self) -> dict:
        """Compact JSON-serialisable record (no token arrays)."""
        return {
            "world": self.world, "body": self.body, "seed": self.seed, "episode": self.episode,
            "rendering": self.rendering, "split": self.split, "return": self.episode_return,
            "outcomes": self.outcomes, "violations": self.violations,
            "terminated": self.terminated, "truncated": self.truncated,
            "termination": self.termination, "n_tokens": len(self.tokens),
            "illegal_mass": self.illegal_mass, "actions": self.actions,
        }


def _sample(lp_legal: np.ndarray, legal: Sequence[int], u: float) -> int:
    acc = 0.0
    for a in legal:
        acc += math.exp(lp_legal[a])
        if u < acc:
            return a
    return legal[-1]


def rollout(specs: Sequence[EpisodeSpec], scorer: Scorer, tokenizer, sampling: SamplingConfig,
            frame: Optional[ChatFrame] = None, budget: Optional[ContextBudget] = None) -> list[Trajectory]:
    frame = frame or ChatFrame()
    budget = budget or ContextBudget()
    states = []
    for spec in specs:
        env = WaterEnv(spec.world, spec.profile, spec.seed, spec.episode)
        obs, _ = env.reset()
        tr = Transcript(tokenizer, spec.rendering, frame, spec.options, budget)
        tr.start(spec.profile.name, spec.world.noise)
        states.append({"spec": spec, "env": env, "tr": tr, "obs": obs, "decisions": []})
    action_ids = states[0]["tr"].action_ids if states else []
    active = [i for i, s in enumerate(states) if not s["env"].done]
    while active:
        reqs, idxs = [], []
        for i in active:
            st = states[i]
            try:
                st["tr"].observe(st["obs"])
                if len(st["tr"].tokens) + 1 > budget.total:
                    raise ContextOverflow("no room for an action")
            except ContextOverflow:
                st["env"].truncate("context")  # external truncation; never expected
                continue
            reqs.append(ScoreRequest(list(st["tr"].tokens), list(action_ids)))
            idxs.append(i)
        results = scorer.score_batch(reqs) if reqs else []
        for i, res in zip(idxs, results):
            st = states[i]
            spec, env, tr, obs = st["spec"], st["env"], st["tr"], st["obs"]
            lp_all = np.asarray(res.logprobs, dtype=float)
            legal = obs.legal
            lp_legal = legal_log_softmax(lp_all, legal, sampling.temperature)
            k = len(st["decisions"])
            forced = k < len(spec.prefix_actions)
            if forced:
                a = spec.prefix_actions[k]
                if a not in legal:
                    raise ValueError(f"replayed action {a} is illegal at decision {k}")
            elif sampling.greedy:
                a = max(legal, key=lambda x: lp_legal[x])
            else:
                u = keyed_uniform(POLICY_NS, int(Stream.POLICY), sampling.salt, spec.seed,
                                  spec.episode, len(st["decisions"]))
                a = _sample(lp_legal, legal, u)
            illegal = 1.0 - float(sum(math.exp(lp_all[x]) for x in legal))
            pos = len(tr.tokens)
            tr.act(a)
            new_obs, r, *_ = env.step(a)
            st["decisions"].append(Decision(pos, a, legal, obs.cycle, obs.phase.value, lp_all,
                                            float(lp_all[a]), float(lp_legal[a]), max(0.0, illegal), r,
                                            forced))
            st["obs"] = new_obs
        active = [i for i in active if not states[i]["env"].done]
    out = []
    for st in states:
        spec, env, tr = st["spec"], st["env"], st["tr"]
        rep = audit_env(env)
        out.append(Trajectory(
            world=spec.world.name, body=spec.profile.name, seed=spec.seed, episode=spec.episode,
            rendering=spec.rendering.name, split=spec.split, tokens=list(tr.tokens),
            prompt_len=tr.prompt_len, decisions=st["decisions"], outcomes=rep.outcomes.as_dict(),
            violations=rep.violations, terminated=env.terminated, truncated=env.truncated,
            termination=env.termination.value if env.termination else None))
    return out


def attach_reference_logprobs(trajs: Sequence[Trajectory], ref_scorer: Scorer, action_ids: Sequence[int]) -> None:
    """Score every decision under the fixed initial reference policy (one call per episode)."""
    for t in trajs:
        if not t.decisions:
            t.ref_lp_legal = np.zeros(0)
            continue
        rows = ref_scorer.score_positions(t.tokens, t.positions, list(action_ids))
        t.ref_lp_legal = np.array([legal_log_softmax(row, d.legal)[d.action]
                                   for row, d in zip(rows, t.decisions)])
