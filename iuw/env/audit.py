"""Independent evaluator: rebuild every outcome from the physics log.

The evaluator recomputes the target sequence and execution draws from the
episode key, replays the frozen equations, and compares. It never trusts the
reward stream. Any run with duplicated reward, negative stocks, clock stalls,
broken water or energy conservation, reward from rest, or a mismatch between
logged and rebuilt physics is rejected.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from ..config import PROBE, REST, Profile, WorldConfig
from ..rng import Event, Stream, draw
from .water import EpisodeKey, PhysicsRecord, Termination, WaterEnv, target_sequence

TOL = 1e-9


@dataclass
class Outcomes:
    """Separate outcomes, never folded into one number."""

    episode_return: float
    liters: int
    credited: int
    energy_used: int
    probes: int
    actuations: int
    executed: int
    rests: int
    recovery_time: int  # rest cycles taken while burden was above zero
    max_burden: float
    final_burden: float
    unmet_demand: int
    welfare_violation: bool
    termination: Optional[str]
    truncated: bool
    cycles_completed: int
    decisions: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditReport:
    outcomes: Outcomes
    violations: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.violations


class AuditError(AssertionError):
    pass


def audit_log(log: Sequence[PhysicsRecord], reported_rewards: Sequence[float], world: WorldConfig,
              profile: Profile, key: EpisodeKey, termination: Optional[str] = None,
              truncated: bool = False) -> AuditReport:
    v: list[str] = []
    targets = target_sequence(world, key)
    energy, reserve, liters, credited_total = world.energy, world.reserve, 0, 0
    burden = 0.0
    last_command: Optional[int] = None
    cycle = 1
    probed_this_cycle = False
    ret = 0.0
    probes = actuations = executed_n = rests = recovery = 0
    max_burden = 0.0
    welfare = False

    if len(reported_rewards) != len(log):
        v.append(f"reward stream has {len(reported_rewards)} entries for {len(log)} decisions "
                 "(duplicated or missing reward)")

    for i, rec in enumerate(log):
        tag = f"decision {i} (cycle {rec.cycle})"
        if rec.decision != i:
            v.append(f"{tag}: decision index {rec.decision} out of order")
        if rec.cycle != cycle:
            v.append(f"{tag}: clock stall or skip, expected cycle {cycle}")
        if cycle > world.horizon:
            v.append(f"{tag}: decision after horizon")
            break
        target = targets[cycle - 1]
        if rec.target != target:
            v.append(f"{tag}: logged target {rec.target} != rebuilt {target}")
        if rec.energy_before != energy or rec.reserve_before != reserve or rec.liters_before != liters:
            v.append(f"{tag}: stocks before do not match rebuilt stocks")
        if abs(rec.burden_before - burden) > TOL:
            v.append(f"{tag}: burden before {rec.burden_before} != rebuilt {burden}")
        if rec.last_command_before != last_command:
            v.append(f"{tag}: last command before {rec.last_command_before} != rebuilt {last_command}")

        delivered = credited = 0
        if rec.intended == PROBE:
            if probed_this_cycle:
                v.append(f"{tag}: second probe in one cycle")
            if energy < world.probe_cost:
                v.append(f"{tag}: unaffordable probe")
            probed_this_cycle = True
            probes += 1
            energy -= world.probe_cost
            reward = -world.energy_coef * world.probe_cost
            if not world.probe_shuffle and rec.probe_report != target:
                v.append(f"{tag}: probe reported {rec.probe_report}, target {target}")
            if rec.commit:
                v.append(f"{tag}: probe marked as commit")
        else:
            if not rec.commit:
                v.append(f"{tag}: commit action not marked as commit")
            if rec.intended == REST:
                cost = world.rest_cost
                rests += 1
                if burden > 0:
                    recovery += 1
                rested = 1.0
            else:
                if energy < world.actuate_cost:
                    v.append(f"{tag}: unaffordable actuation")
                cost = world.actuate_cost
                rested = 0.0
                actuations += 1
                p_exec = profile.execution_probability(rec.intended, last_command)
                u = draw(Stream.EXECUTION, key.seed, key.episode, cycle, Event.EXECUTE,
                         namespace=key.namespace)
                ok = u < p_exec
                if rec.executed != ok or (rec.p_exec is not None and abs(rec.p_exec - p_exec) > TOL):
                    v.append(f"{tag}: execution mismatch (logged {rec.executed}, rebuilt {ok})")
                executed_n += int(ok)
                if ok and rec.intended == target and reserve >= world.liters_per_success:
                    delivered = world.liters_per_success
                if world.reward_remap is not None:
                    cv = world.reward_remap.permutation[target - 1]
                    credited = world.liters_per_success if (ok and rec.intended == cv) else 0
                else:
                    credited = delivered
                last_command = rec.intended
            energy -= cost
            reserve -= delivered
            liters += delivered
            credited_total += credited
            burden = max(0.0, world.burden_decay * burden + world.excess_noise(profile)
                         - profile.recovery * rested)
            reward = credited - world.energy_coef * cost
            if rec.intended == REST and abs(rec.reward) > TOL:
                v.append(f"{tag}: reward {rec.reward} paid for rest")
            if rec.delivered != delivered or rec.credited != credited:
                v.append(f"{tag}: delivered {rec.delivered}/{rec.credited} != rebuilt {delivered}/{credited}")
            cycle += 1
            probed_this_cycle = False
            if burden > world.burden_limit:
                welfare = True
        max_burden = max(max_burden, burden)
        if energy < 0 or reserve < 0 or liters < 0:
            v.append(f"{tag}: negative stock (energy {energy}, reserve {reserve}, liters {liters})")
        if liters + reserve != world.reserve:
            v.append(f"{tag}: water not conserved ({liters} + {reserve} != {world.reserve})")
        if (rec.energy_after != energy or rec.reserve_after != reserve or rec.liters_after != liters
                or abs(rec.burden_after - burden) > TOL):
            v.append(f"{tag}: stocks after do not match rebuilt stocks")
        if abs(rec.reward - reward) > TOL:
            v.append(f"{tag}: logged reward {rec.reward} != rebuilt {reward}")
        if i < len(reported_rewards) and abs(reported_rewards[i] - reward) > TOL:
            v.append(f"{tag}: reported reward {reported_rewards[i]} != rebuilt {reward}")
        ret += reward
        if welfare and i != len(log) - 1:
            v.append(f"{tag}: episode continued after a welfare violation")
            break

    if not math.isclose(sum(reported_rewards), ret, abs_tol=1e-6):
        v.append(f"reported return {sum(reported_rewards)} != rebuilt {ret}")
    energy_used = world.energy - energy
    if termination == Termination.WELFARE.value and not welfare:
        v.append("termination says welfare violation but rebuilt burden never exceeded the limit")
    if welfare and termination != Termination.WELFARE.value:
        v.append("rebuilt burden exceeded the limit but termination reason disagrees")
    if termination is None and not truncated and log:
        v.append("episode ended with neither termination nor truncation")

    outcomes = Outcomes(
        episode_return=ret, liters=liters, credited=credited_total, energy_used=energy_used,
        probes=probes, actuations=actuations, executed=executed_n, rests=rests,
        recovery_time=recovery, max_burden=max_burden, final_burden=burden,
        unmet_demand=max(0, world.demand - liters), welfare_violation=welfare,
        termination=termination, truncated=truncated, cycles_completed=cycle - 1,
        decisions=len(log),
    )
    return AuditReport(outcomes, v)


def audit_env(env: WaterEnv) -> AuditReport:
    return audit_log(env.log, env.rewards, env.world, env.profile, env.key,
                     env.termination.value if env.termination else None, env.truncated)


def assert_clean(report: AuditReport) -> None:
    if not report.clean:
        raise AuditError("; ".join(report.violations[:10]))
