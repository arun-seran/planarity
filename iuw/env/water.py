"""Water-v1: a four-valve canal with one hidden rule, target persistence q.

Each cycle allows one optional probe and then one commit (actuate a valve or
rest), so an episode has at most 2 * horizon decisions. Every intended action
passes through the body's execution law before it reaches the world.

Learner-facing fields are allowlisted in :class:`Observation`. Audit state
(the target, execution draws, stocks) travels on a separate channel, the
physics log, which an independent evaluator rebuilds outcomes from.

Execution law (spec v3, change 9): the last command starts unset, so the
first actuation counts as a change. It updates after every actuation request,
failures included. Probe and rest leave it unchanged. Probing does not
advance exposure, recovery or the clock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from ..config import PROBE, REST, VALVES, Profile, WorldConfig
from ..rng import DEFAULT_NAMESPACE, Event, Stream, draw, draw_index


class Phase(str, Enum):
    PROBE = "probe"  # start of a cycle: probe is still available
    COMMIT = "commit"  # after a probe: the next action must commit


class Termination(str, Enum):
    HORIZON = "horizon"
    WELFARE = "welfare_violation"
    ENERGY = "energy_exhausted"
    RESERVE = "reserve_exhausted"
    DEMAND = "demand_met"


@dataclass(frozen=True)
class Observation:
    """The allowlisted learner-facing observation. Nothing else reaches a renderer."""

    cycle: int
    phase: Phase
    probe_result: Optional[int]
    liters: int
    energy: int
    burden: float
    body: str
    noise: float
    last_command: Optional[int]
    last_success: Optional[bool]
    legal: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d


OBSERVATION_FIELDS = tuple(Observation.__dataclass_fields__)


@dataclass
class PhysicsRecord:
    """One decision in the physics log (audit channel)."""

    decision: int
    cycle: int
    phase: str
    intended: int
    kind: str  # "probe" | "actuate" | "rest"
    valve: Optional[int]
    executed: Optional[bool]
    p_exec: Optional[float]
    exec_draw: Optional[float]
    target: int
    probe_report: Optional[int]
    delivered: int
    credited: int
    energy_before: int
    energy_after: int
    reserve_before: int
    reserve_after: int
    liters_before: int
    liters_after: int
    burden_before: float
    burden_after: float
    last_command_before: Optional[int]
    last_command_after: Optional[int]
    damage: float
    reward: float
    commit: bool
    next_target: Optional[int] = None


@dataclass
class EpisodeKey:
    seed: int
    episode: int = 0
    namespace: int = DEFAULT_NAMESPACE


def target_sequence(world: WorldConfig, key: EpisodeKey) -> list[int]:
    """Targets for cycles 1..horizon (index 0 is cycle 1). Independent of actions."""
    n = world.n_valves
    seq = [draw_index(n, Stream.WORLD, key.seed, key.episode, 1, Event.TARGET_INIT,
                      namespace=key.namespace) + 1]
    for t in range(1, world.horizon):
        cur = seq[-1]
        stay = draw(Stream.WORLD, key.seed, key.episode, t, Event.TARGET_STAY,
                    namespace=key.namespace) < world.q
        if stay:
            seq.append(cur)
        else:
            seq.append(draw_index(n, Stream.WORLD, key.seed, key.episode, t, Event.TARGET_DRAW,
                                  namespace=key.namespace) + 1)
    return seq


class WaterEnv:
    """Gymnasium-style environment: reset() -> (obs, info); step(a) -> (obs, r, term, trunc, info)."""

    def __init__(self, world: WorldConfig, profile: Profile, seed: int, episode: int = 0,
                 namespace: int = DEFAULT_NAMESPACE):
        if world.n_valves != len(VALVES):
            raise ValueError("Water-v1 action IDs assume four valves")
        self.world = world
        self.profile = profile
        self.key = EpisodeKey(seed, episode, namespace)
        self._targets = target_sequence(world, self.key)
        self._started = False

    # ----- lifecycle -------------------------------------------------------------------------
    def reset(self) -> tuple[Observation, dict[str, Any]]:
        w = self.world
        self.cycle = 1
        self.phase = Phase.PROBE
        self.energy = w.energy
        self.reserve = w.reserve
        self.liters = 0
        self.credited = 0
        self.burden = 0.0
        self.last_command: Optional[int] = None
        self.last_success: Optional[bool] = None
        self.probe_result: Optional[int] = None
        self.decision = 0
        self.terminated = False
        self.truncated = False
        self.termination: Optional[Termination] = None
        self.truncation_reason: Optional[str] = None
        self.log: list[PhysicsRecord] = []
        self.rewards: list[float] = []
        self._started = True
        return self.observation(), {"seed": self.key.seed, "episode": self.key.episode}

    @property
    def target(self) -> int:
        """Privileged: the current target. Only oracles and audits may read this."""
        return self._targets[min(self.cycle, self.world.horizon) - 1]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def legal_actions(self) -> tuple[int, ...]:
        if self.done:
            return ()
        w = self.world
        legal = []
        if self.phase is Phase.PROBE and self.energy >= w.probe_cost:
            legal.append(PROBE)
        if self.energy >= w.actuate_cost:
            legal.extend(VALVES)
        legal.append(REST)  # always legal, costs nothing
        return tuple(legal)

    def observation(self) -> Observation:
        return Observation(
            cycle=self.cycle,
            phase=self.phase,
            probe_result=self.probe_result,
            liters=self.liters,
            energy=self.energy,
            burden=self.burden,
            body=self.profile.name,
            noise=self.world.noise,
            last_command=self.last_command,
            last_success=self.last_success,
            legal=self.legal_actions(),
        )

    def truncate(self, reason: str) -> None:
        """External truncation (e.g. a harness limit). Kept apart from termination."""
        if not self.done:
            self.truncated = True
            self.truncation_reason = reason

    # ----- dynamics --------------------------------------------------------------------------
    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if not self._started:
            raise RuntimeError("call reset() first")
        if self.done:
            raise RuntimeError("episode is over")
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"illegal action {action}; legal: {legal}")
        w, p = self.world, self.profile
        target = self.target
        rec = dict(
            decision=self.decision, cycle=self.cycle, phase=self.phase.value, intended=action,
            target=target, energy_before=self.energy, reserve_before=self.reserve,
            liters_before=self.liters, burden_before=self.burden,
            last_command_before=self.last_command, damage=0.0,
        )
        k = self.key
        if action == PROBE:
            self.energy -= w.probe_cost
            if w.probe_shuffle:
                report = draw_index(w.n_valves, Stream.SHUFFLE, k.seed, k.episode, self.cycle,
                                    Event.PROBE_SHUFFLE, namespace=k.namespace) + 1
            else:
                report = target
            self.probe_result = report
            self.phase = Phase.COMMIT
            reward = -w.energy_coef * w.probe_cost
            rec.update(kind="probe", valve=None, executed=None, p_exec=None, exec_draw=None,
                       probe_report=report, delivered=0, credited=0, commit=False)
        else:
            delivered = credited = 0
            if action == REST:
                cost = w.rest_cost
                rec.update(kind="rest", valve=None, executed=None, p_exec=None, exec_draw=None)
            else:
                cost = w.actuate_cost
                p_exec = p.execution_probability(action, self.last_command)
                u = draw(Stream.EXECUTION, k.seed, k.episode, self.cycle, Event.EXECUTE,
                         namespace=k.namespace)
                executed = u < p_exec
                if executed and action == target and self.reserve >= w.liters_per_success:
                    delivered = w.liters_per_success
                if w.reward_remap is not None:
                    credited_valve = w.reward_remap.permutation[target - 1]
                    credited = w.liters_per_success if (executed and action == credited_valve) else 0
                else:
                    credited = delivered
                self.last_command = action  # updates after every actuation, failures included
                self.last_success = executed
                rec.update(kind="actuate", valve=action, executed=executed, p_exec=p_exec,
                           exec_draw=u)
            self.energy -= cost
            self.reserve -= delivered
            self.liters += delivered
            self.credited += credited
            rested = action == REST
            self.burden = max(0.0, w.burden_decay * self.burden + w.excess_noise(p)
                              - p.recovery * (1.0 if rested else 0.0))
            reward = credited - w.energy_coef * cost - w.damage_coef * 0.0
            rec.update(probe_report=self.probe_result, delivered=delivered, credited=credited,
                       commit=True)
            self._end_cycle()
            rec["next_target"] = None if self.done else self.target
        rec.update(energy_after=self.energy, reserve_after=self.reserve, liters_after=self.liters,
                   burden_after=self.burden, last_command_after=self.last_command, reward=reward)
        self.log.append(PhysicsRecord(**rec))
        self.rewards.append(reward)
        self.decision += 1
        info = {"audit": self.log[-1],
                "termination": self.termination.value if self.termination else None}
        return self.observation(), reward, self.terminated, self.truncated, info

    def _end_cycle(self) -> None:
        w = self.world
        self.cycle += 1
        self.phase = Phase.PROBE
        self.probe_result = None
        if self.burden > w.burden_limit:
            self._terminate(Termination.WELFARE)
        elif self.cycle > w.horizon:
            self._terminate(Termination.HORIZON)
        elif self.liters >= w.demand:
            self._terminate(Termination.DEMAND)
        elif self.reserve < w.liters_per_success:
            self._terminate(Termination.RESERVE)
        elif self.energy < w.actuate_cost:
            self._terminate(Termination.ENERGY)

    def _terminate(self, reason: Termination) -> None:
        self.terminated = True
        self.termination = reason

    # ----- summaries -------------------------------------------------------------------------
    @property
    def episode_return(self) -> float:
        return float(sum(self.rewards))
