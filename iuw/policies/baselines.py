"""Cheap policies: floors and naive strategies, plus the omniscient oracle (upper bound)."""

from __future__ import annotations

from typing import Optional

from ..config import PROBE, REST
from ..env.runner import EpisodeContext
from ..env.water import Observation, Phase
from ..rng import Event, Stream, draw_index


class RandomPolicy:
    """Uniform over legal actions, keyed so paired runs share draws."""

    name = "random"
    privileged = False

    def __init__(self, salt: int = 0):
        self.salt = salt

    def begin(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.decision = 0

    def act(self, obs: Observation) -> int:
        legal = obs.legal
        i = draw_index(len(legal), Stream.POLICY, self.ctx.seed, self.ctx.episode, self.decision,
                       Event.ACTION, extra=self.salt)
        self.decision += 1
        return legal[i]


class AlwaysProbePolicy:
    """Probe every cycle, then actuate the reported valve. Never rests."""

    name = "always-probe"
    privileged = False

    def begin(self, ctx: EpisodeContext) -> None:
        pass

    def act(self, obs: Observation) -> int:
        if obs.phase is Phase.PROBE and PROBE in obs.legal:
            return PROBE
        if obs.probe_result is not None and obs.probe_result in obs.legal:
            return obs.probe_result
        return REST


class BlindRepeatPolicy:
    """Never probe; actuate one fixed valve every cycle."""

    name = "blind-repeat"
    privileged = False

    def __init__(self, valve: int = 1):
        self.valve = valve

    def begin(self, ctx: EpisodeContext) -> None:
        pass

    def act(self, obs: Observation) -> int:
        return self.valve if self.valve in obs.legal else REST


class ProbeOnceRepeatPolicy:
    """Probe in the first cycle, then repeat that valve forever (the q = 1 plan)."""

    name = "probe-once-repeat"
    privileged = False

    def begin(self, ctx: EpisodeContext) -> None:
        self.valve: Optional[int] = None

    def act(self, obs: Observation) -> int:
        if self.valve is None:
            if obs.probe_result is not None:
                self.valve = obs.probe_result
            elif PROBE in obs.legal:
                return PROBE
        return self.valve if self.valve in obs.legal else REST


class OraclePolicy:
    """Omniscient: reads the hidden target and actuates it without probing.

    Reported separately as an upper bound, never as the reference. With
    ``active_limit`` it rests once that many actuations are spent (used by the
    certification fixture's noisy row).
    """

    name = "oracle"
    privileged = True

    def __init__(self, active_limit: Optional[int] = None):
        self.active_limit = active_limit

    def begin(self, ctx: EpisodeContext) -> None:
        if ctx.env is None:
            raise RuntimeError("oracle needs the privileged env handle")
        self.env = ctx.env
        self.active = 0

    def act(self, obs: Observation) -> int:
        if self.active_limit is not None and self.active >= self.active_limit:
            return REST
        w, p = self.env.world, self.env.profile
        if w.burden_decay * obs.burden + w.excess_noise(p) > w.burden_limit:
            return REST  # lazy rest: act whenever welfare allows
        t = self.env.target
        if w.reward_remap is not None:  # aim for the valve that is credited, not the physical target
            t = w.reward_remap.permutation[t - 1]
        if t in obs.legal:
            self.active += 1
            return t
        return REST
