"""Run scripted or planned policies directly against the simulator (no language model)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

from ..config import Profile, WorldConfig
from .audit import AuditReport, assert_clean, audit_env
from .water import Observation, WaterEnv


@dataclass
class EpisodeContext:
    seed: int
    episode: int
    body: str
    world_name: str
    env: Optional[WaterEnv] = None  # privileged handle; only oracles may use it


class Policy(Protocol):
    name: str
    privileged: bool

    def begin(self, ctx: EpisodeContext) -> None: ...

    def act(self, obs: Observation) -> int: ...


@dataclass
class EpisodeResult:
    world: str
    body: str
    seed: int
    episode: int
    policy: str
    actions: list[int]
    rewards: list[float]
    report: AuditReport
    observations: list[Observation] = field(default_factory=list)

    @property
    def episode_return(self) -> float:
        return self.report.outcomes.episode_return


def run_episode(world: WorldConfig, profile: Profile, seed: int, policy: Policy, episode: int = 0,
                check: bool = True, keep_observations: bool = False) -> EpisodeResult:
    env = WaterEnv(world, profile, seed, episode)
    obs, _ = env.reset()
    ctx = EpisodeContext(seed, episode, profile.name, world.name,
                         env if getattr(policy, "privileged", False) else None)
    policy.begin(ctx)
    actions: list[int] = []
    observations: list[Observation] = []
    while not env.done:
        if keep_observations:
            observations.append(obs)
        a = policy.act(obs)
        actions.append(a)
        obs, _, _, _, _ = env.step(a)
    report = audit_env(env)
    if check:
        assert_clean(report)
    return EpisodeResult(world.name, profile.name, seed, episode, policy.name, actions,
                         list(env.rewards), report, observations)


def run_many(world: WorldConfig, profile: Profile, seeds: Iterable[int], policy: Policy,
             episode: int = 0, check: bool = True) -> list[EpisodeResult]:
    return [run_episode(world, profile, s, policy, episode, check) for s in seeds]


@dataclass
class ReturnSummary:
    n: int
    mean: float
    sd: float
    se: float
    lower95: float
    upper95: float

    def as_dict(self) -> dict:
        return dict(n=self.n, mean=self.mean, sd=self.sd, se=self.se, lower95=self.lower95,
                    upper95=self.upper95)


def summarise_returns(values: Sequence[float]) -> ReturnSummary:
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    return ReturnSummary(n, mean, sd, se, mean - 1.959964 * se, mean + 1.959964 * se)
