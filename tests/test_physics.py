import dataclasses

import pytest

from iuw.config import (COUNTERFACTUAL, NOISY_REFERENCE, PROBE, PROFILE_A, PROFILE_B, REFERENCE,
                        REST, RewardRemap, WorldConfig)
from iuw.env import (Observation, Phase, Termination, WaterEnv, assert_clean, audit_env,
                     audit_log, target_sequence)
from iuw.env.audit import AuditError
from iuw.env.water import EpisodeKey
from iuw.policies.baselines import RandomPolicy
from iuw.env.runner import run_episode


def test_observation_is_allowlisted():
    fields = {f.name for f in dataclasses.fields(Observation)}
    assert fields == {"cycle", "phase", "probe_result", "liters", "energy", "burden", "body",
                      "noise", "last_command", "last_success", "legal"}
    assert "target" not in fields


def test_first_actuation_counts_as_change_and_failures_update_last_command():
    env = WaterEnv(REFERENCE, PROFILE_A, seed=3)
    obs, _ = env.reset()
    assert obs.last_command is None
    env.step(2)
    rec = env.log[-1]
    assert rec.p_exec == PROFILE_A.switch_precision  # first command is a change
    assert env.last_command == 2  # updated whether or not it executed
    env.step(2)
    assert env.log[-1].p_exec == PROFILE_A.repeat_precision
    env.step(PROBE)
    env.step(REST)
    assert env.last_command == 2  # probe and rest leave it unchanged
    env.step(3)
    assert env.log[-1].p_exec == PROFILE_A.switch_precision


def test_probe_does_not_advance_clock_or_burden():
    env = WaterEnv(NOISY_REFERENCE, PROFILE_A, seed=1)
    env.reset()
    obs, r, *_ = env.step(PROBE)
    assert obs.cycle == 1 and obs.phase is Phase.COMMIT and obs.burden == 0.0
    assert r == pytest.approx(-0.05)
    assert PROBE not in obs.legal  # at most once per cycle
    assert obs.probe_result == env.target
    obs, *_ = env.step(REST)
    assert obs.cycle == 2 and obs.phase is Phase.PROBE and obs.probe_result is None


def test_noisy_a_without_rest_breaches_after_eighth_cycle():
    env = WaterEnv(NOISY_REFERENCE, PROFILE_A, seed=0)
    env.reset()
    burdens = []
    while not env.done:
        env.step(1)
        burdens.append(env.burden)
    assert len(burdens) == 8
    assert burdens[6] <= 10 < burdens[7]
    assert env.termination is Termination.WELFARE
    assert burdens[0] == pytest.approx(1.8)


def test_noise_adds_no_burden_to_b():
    env = WaterEnv(NOISY_REFERENCE, PROFILE_B, seed=0)
    env.reset()
    for _ in range(20):
        env.step(1)
    assert env.burden == 0.0


def test_target_persistence_law():
    stable = target_sequence(COUNTERFACTUAL, EpisodeKey(5))
    assert len(set(stable)) == 1 and len(stable) == 100
    stays = moves = 0
    for s in range(200):
        seq = target_sequence(REFERENCE, EpisodeKey(s))
        stays += sum(a == b for a, b in zip(seq, seq[1:]))
        moves += len(seq) - 1
    assert abs(stays / moves - 0.25) < 0.02  # q = 0: redrawn uniformly, so stays 1/4 of the time


def test_paired_randomness_across_bodies_and_policies():
    a = WaterEnv(REFERENCE, PROFILE_A, seed=11)
    b = WaterEnv(REFERENCE, PROFILE_B, seed=11)
    assert a._targets == b._targets
    a.reset(); b.reset()
    a.step(1); b.step(4)
    assert a.log[0].exec_draw == b.log[0].exec_draw


def test_energy_exhaustion_and_masking():
    w = dataclasses.replace(REFERENCE, energy=3)
    env = WaterEnv(w, PROFILE_B, seed=0)
    obs, _ = env.reset()
    assert obs.legal == (0, 1, 2, 3, 4, 5)
    obs, *_ = env.step(PROBE)
    assert obs.legal == (1, 2, 3, 4, 5)
    obs, *_ = env.step(1)
    obs, *_ = env.step(PROBE)
    assert obs.energy == 0 and obs.legal == (REST,)
    obs, _, term, trunc, _ = env.step(REST)
    assert term and not trunc and env.termination is Termination.ENERGY


def test_termination_and_truncation_are_separate():
    env = WaterEnv(REFERENCE, PROFILE_B, seed=0)
    env.reset()
    env.step(REST)
    env.truncate("harness limit")
    assert env.truncated and not env.terminated
    report = audit_env(env)
    assert report.clean and report.outcomes.truncated


def test_full_episode_conservation_and_audit():
    for world in (REFERENCE, COUNTERFACTUAL, NOISY_REFERENCE):
        for profile in (PROFILE_A, PROFILE_B):
            res = run_episode(world, profile, seed=7, policy=RandomPolicy())
            o = res.report.outcomes
            assert o.liters + (world.reserve - o.liters) == world.reserve
            assert o.episode_return == pytest.approx(o.credited - 0.05 * o.energy_used)
            assert res.report.clean


def _finished_env():
    env = WaterEnv(REFERENCE, PROFILE_B, seed=2)
    env.reset()
    while not env.done:
        env.step(PROBE)
        env.step(env.target)
    return env


def test_audit_rejects_duplicated_reward():
    env = _finished_env()
    bad = list(env.rewards) + [env.rewards[-1]]
    rep = audit_log(env.log, bad, env.world, env.profile, env.key, env.termination.value)
    assert not rep.clean and any("duplicated" in v for v in rep.violations)


def test_audit_rejects_reward_from_rest_and_tampered_stocks():
    env = WaterEnv(REFERENCE, PROFILE_B, seed=2)
    env.reset()
    for _ in range(100):
        env.step(REST)
    env.log[3].reward = 0.5
    rewards = list(env.rewards)
    rewards[3] = 0.5
    rep = audit_log(env.log, rewards, env.world, env.profile, env.key, env.termination.value)
    assert any("paid for rest" in v for v in rep.violations)
    env2 = _finished_env()
    env2.log[10].reserve_after = -1
    rep2 = audit_log(env2.log, env2.rewards, env2.world, env2.profile, env2.key, "horizon")
    assert not rep2.clean
    with pytest.raises(AuditError):
        assert_clean(rep2)


def test_audit_rejects_clock_stall():
    env = _finished_env()
    env.log[6].cycle = env.log[5].cycle  # the next cycle claims the clock did not advance
    rep = audit_log(env.log, env.rewards, env.world, env.profile, env.key, "horizon")
    assert any("clock" in v for v in rep.violations)


def test_reward_remap_control_keeps_physics():
    w = dataclasses.replace(REFERENCE, name="remap", reward_remap=RewardRemap())
    env = WaterEnv(w, PROFILE_B, seed=4)
    env.reset()
    while not env.done:
        env.step(PROBE)
        env.step(w.reward_remap.permutation[env.target - 1])
    rep = audit_env(env)
    assert rep.clean
    assert rep.outcomes.liters == 0 and rep.outcomes.credited > 50


def test_probe_shuffle_control_reports_unrelated_valve():
    w = dataclasses.replace(REFERENCE, name="shuffle", probe_shuffle=True)
    agree = 0
    for s in range(100):
        env = WaterEnv(w, PROFILE_B, seed=s)
        env.reset()
        obs, *_ = env.step(PROBE)
        agree += obs.probe_result == env.target
    assert 10 < agree < 40
