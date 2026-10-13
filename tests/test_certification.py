from fractions import Fraction

import pytest

from iuw.certify import (CERTIFIED_FIXTURE, CERTIFIED_PLANNER, fixture_table, simulate_fixture,
                         witness_gate)
from iuw.config import (COUNTERFACTUAL, K_FLOOR, NOISY_COUNTERFACTUAL, NOISY_REFERENCE, PROFILE_A,
                        PROFILE_B, PROFILES, REFERENCE, equalised_execution, get_world,
                        world_with_q)
from iuw.env import run_many, summarise_returns
from iuw.policies.baselines import AlwaysProbePolicy, ProbeOnceRepeatPolicy
from iuw.policies.planner import (KnownLawPlanner, lazy_rest_schedule, max_active_cycles,
                                  planner_value)


def test_fixture_arithmetic_is_exact():
    table = fixture_table()
    for key, value in CERTIFIED_FIXTURE.items():
        assert table[key] == value, key
    # A's per-cycle success when the target is redrawn every cycle
    assert Fraction("0.25") * Fraction("0.98") + Fraction("0.75") * Fraction("0.10") == Fraction("0.32")
    assert table[("quiet-volatile", "A")] - table[("quiet-volatile", "B")] == Fraction("-43.22")
    assert table[("quiet-stable", "A")] - table[("quiet-stable", "B")] == Fraction("22.12")
    assert table[("noisy-volatile", "A")] - table[("noisy-volatile", "B")] == Fraction("-59.22")


@pytest.mark.parametrize("key,value", sorted(CERTIFIED_PLANNER.items()))
def test_planner_exact_values(key, value):
    world, body = key
    assert planner_value(get_world(world), PROFILES[body]) == pytest.approx(value, abs=1e-9)


def test_cross_world_condition_and_witness_gate():
    a_cf = planner_value(COUNTERFACTUAL, PROFILE_A)
    b_ref = planner_value(REFERENCE, PROFILE_B)
    assert a_cf >= b_ref
    gate = witness_gate(fixture_table(), a_cf, b_ref, b_lower95=64.7)
    assert gate.passes, gate.reasons
    assert gate.a_liters_cf == pytest.approx(97.12)
    assert gate.a_gain == pytest.approx(65.34)
    assert gate.gap_ref == pytest.approx(43.22) and gate.gap_cf == pytest.approx(22.12)


def test_floor_check_by_exact_value():
    assert planner_value(REFERENCE, PROFILE_A) < K_FLOOR <= planner_value(REFERENCE, PROFILE_B)
    assert planner_value(COUNTERFACTUAL, PROFILE_A) >= K_FLOOR
    assert planner_value(COUNTERFACTUAL, PROFILE_B) >= K_FLOOR


def test_lazy_rest_schedule_is_optimal_and_noisy_values_are_frozen():
    for w in (NOISY_REFERENCE, NOISY_COUNTERFACTUAL):
        s = lazy_rest_schedule(w, PROFILE_A)
        assert not s.welfare_unavoidable
        assert s.active_cycles == max_active_cycles(w, PROFILE_A) == 58
        assert max(s.burden) <= 10
    # Frozen noisy-ablation planner returns (computed before any noisy run).
    assert planner_value(NOISY_REFERENCE, PROFILE_A) == pytest.approx(0.22 * 57)
    assert planner_value(NOISY_COUNTERFACTUAL, PROFILE_A) == pytest.approx(0.10 + 0.98 * 57 - 0.05 * 59)
    assert planner_value(NOISY_REFERENCE, PROFILE_B) == pytest.approx(65.0)


def test_planner_strategies_match_the_spec():
    # q = 0: probe every cycle and spend 200 energy; q = 1: probe once, spend 101.
    ref = run_many(REFERENCE, PROFILE_B, range(5), KnownLawPlanner(REFERENCE, PROFILES))
    assert all(r.report.outcomes.probes == 100 and r.report.outcomes.energy_used == 200 for r in ref)
    cf = run_many(COUNTERFACTUAL, PROFILE_A, range(5), KnownLawPlanner(COUNTERFACTUAL, PROFILES))
    assert all(r.report.outcomes.probes == 1 and r.report.outcomes.energy_used == 101 for r in cf)


def test_equalised_execution_removes_the_reversal():
    a_eq = equalised_execution(PROFILE_A)
    assert planner_value(REFERENCE, a_eq) == pytest.approx(65.0)
    assert planner_value(COUNTERFACTUAL, a_eq) == pytest.approx(69.95)


def test_intermediate_persistence_planner_is_sane():
    for q in (0.25, 0.75):
        w = world_with_q(q)
        va = planner_value(w, PROFILE_A)
        assert planner_value(REFERENCE, PROFILE_A) <= va <= planner_value(COUNTERFACTUAL, PROFILE_A)


@pytest.mark.slow
def test_planner_monte_carlo_matches_exact_and_floor_holds():
    seeds = range(10_000, 11_000)
    for w in (REFERENCE, COUNTERFACTUAL):
        for p in (PROFILE_A, PROFILE_B):
            s = summarise_returns([r.episode_return for r in
                                   run_many(w, p, seeds, KnownLawPlanner(w, PROFILES))])
            assert abs(s.mean - planner_value(w, p)) < 4 * s.se + 1e-9
    b = summarise_returns([r.episode_return for r in
                           run_many(REFERENCE, PROFILE_B, seeds, KnownLawPlanner(REFERENCE, PROFILES))])
    assert b.lower95 >= K_FLOOR
    assert b.sd == pytest.approx(4.33, abs=0.4)  # binomial(100, 0.75) liters


@pytest.mark.slow
def test_fixture_reproduced_by_simulation():
    sim = simulate_fixture(n_episodes=600)
    for key, value in CERTIFIED_FIXTURE.items():
        s = sim[key]
        assert abs(s.mean - float(value)) < 4 * s.se + 1e-9, (key, s.mean, float(value))


def test_scripted_controllers_reproduce_the_plans():
    s = summarise_returns([r.episode_return for r in
                           run_many(COUNTERFACTUAL, PROFILE_A, range(300), ProbeOnceRepeatPolicy())])
    assert abs(s.mean - 92.07) < 4 * s.se
    s = summarise_returns([r.episode_return for r in
                           run_many(REFERENCE, PROFILE_B, range(300), AlwaysProbePolicy())])
    assert abs(s.mean - 65.0) < 4 * s.se


@pytest.mark.slow
def test_planner_dp_matches_monte_carlo_in_control_and_intermediate_worlds():
    from iuw.config import CONTROL_WORLDS
    for w in (*CONTROL_WORLDS, world_with_q(0.75)):
        for p in (PROFILE_A, PROFILE_B):
            s = summarise_returns([r.episode_return for r in
                                   run_many(w, p, range(400), KnownLawPlanner(w, PROFILES))])
            assert abs(s.mean - planner_value(w, p)) < 4 * s.se + 0.05, (w.name, p.name)
