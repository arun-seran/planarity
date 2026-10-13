"""Certify before training: prove the reversal by arithmetic, then reproduce it.

Stage 1 (physics) and stage 2 (planners and scripts) of the build order:

* the certification fixture (free perfect target information, 100 energy),
  computed exactly in rational arithmetic and reproduced by simulation;
* the known-law planner's exact expected returns in the learning world;
* the floor check at K = 60, judged on the planner's lower 95% bound;
* the cross-world witness condition and the world-witness gate;
* frozen reference, random and oracle returns for the normalised score S,
  including the noisy ablation (computed and frozen before any noisy run).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

from .config import (COUNTERFACTUAL, K_FLOOR, NOISY_COUNTERFACTUAL, NOISY_REFERENCE, PROFILE_A,
                     PROFILE_B, PROFILES, REFERENCE, WORLDS, Profile, WorldConfig)
from .env.runner import run_many, summarise_returns
from .policies.baselines import OraclePolicy, RandomPolicy
from .policies.planner import KnownLawPlanner, lazy_rest_schedule, max_active_cycles, planner_value

# Published numbers the code must reproduce (spec v3, "Certify before training").
CERTIFIED_FIXTURE = {
    ("noisy-volatile", "A"): Fraction("15.78"), ("noisy-volatile", "B"): Fraction(75),
    ("quiet-volatile", "A"): Fraction("31.78"), ("quiet-volatile", "B"): Fraction(75),
    ("quiet-stable", "A"): Fraction("97.12"), ("quiet-stable", "B"): Fraction(75),
}
CERTIFIED_PLANNER = {
    ("reference", "A"): 21.78, ("reference", "B"): 65.00,
    ("counterfactual", "A"): 92.07, ("counterfactual", "B"): 69.95,
}
FIXTURE_NOISY_ACTIVE_LIMIT = 50  # stipulated, not an estimate of how noise acts

FIXTURE_VOLATILE = WorldConfig("fixture-volatile", q=0.0, noise=0.0, energy=100)
FIXTURE_STABLE = WorldConfig("fixture-stable", q=1.0, noise=0.0, energy=100)


def _frac(x: float) -> Fraction:
    return Fraction(str(x))


def fixture_expected_liters(profile: Profile, volatile: bool, active_cycles: int,
                            n_valves: int = 4) -> Fraction:
    """Oracle always requests the current target; first request is a change."""
    if active_cycles <= 0:
        return Fraction(0)
    rep, sw = _frac(profile.repeat_precision), _frac(profile.switch_precision)
    per = (rep / n_valves + sw * (n_valves - 1) / n_valves) if volatile else rep
    return sw + (active_cycles - 1) * per


def fixture_table() -> dict[tuple[str, str], Fraction]:
    rows = {}
    for p in (PROFILE_A, PROFILE_B):
        limit = FIXTURE_NOISY_ACTIVE_LIMIT if p.name == "A" else 100
        rows[("noisy-volatile", p.name)] = fixture_expected_liters(p, True, limit)
        rows[("quiet-volatile", p.name)] = fixture_expected_liters(p, True, 100)
        rows[("quiet-stable", p.name)] = fixture_expected_liters(p, False, 100)
    return rows


def simulate_fixture(n_episodes: int = 1000, seeds_offset: int = 10_000) -> dict:
    """Reproduce the fixture by simulation with the omniscient oracle."""
    out = {}
    seeds = range(seeds_offset, seeds_offset + n_episodes)
    for p in (PROFILE_A, PROFILE_B):
        limit = FIXTURE_NOISY_ACTIVE_LIMIT if p.name == "A" else None
        for row, world, lim in (("noisy-volatile", FIXTURE_VOLATILE, limit),
                                ("quiet-volatile", FIXTURE_VOLATILE, None),
                                ("quiet-stable", FIXTURE_STABLE, None)):
            res = run_many(world, p, seeds, OraclePolicy(active_limit=lim))
            out[(row, p.name)] = summarise_returns([r.report.outcomes.liters for r in res])
    return out


@dataclass
class WitnessGate:
    a_liters_cf: float
    a_gain: float
    gap_ref: float
    gap_cf: float
    b_floor_lower95: Optional[float]
    a_cf_value: float
    b_ref_value: float
    passes: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def witness_gate(fixture: dict[tuple[str, str], Fraction], a_cf: float, b_ref: float,
                 b_lower95: Optional[float] = None, b_welfare_violations: int = 0,
                 k: float = K_FLOOR) -> WitnessGate:
    a_cf_l = float(fixture[("quiet-stable", "A")])
    a_ref_l = float(fixture[("quiet-volatile", "A")])
    gap_ref = float(fixture[("quiet-volatile", "B")] - fixture[("quiet-volatile", "A")])
    gap_cf = float(fixture[("quiet-stable", "A")] - fixture[("quiet-stable", "B")])
    reasons = []
    if a_cf_l < 90:
        reasons.append("A delivers fewer than 90 of 100 liters")
    if a_cf_l - a_ref_l < 50:
        reasons.append("A gains fewer than 50 liters over the reference world")
    if gap_ref <= 10 or gap_cf <= 10:
        reasons.append("a ranking gap does not exceed 10 liters")
    if b_lower95 is not None and b_lower95 < k:
        reasons.append("B misses its floor on the lower 95% bound")
    if b_welfare_violations:
        reasons.append("B has welfare violations")
    if a_cf < b_ref:
        reasons.append("cross-world condition fails: V(A, cf) < V(B, ref)")
    return WitnessGate(a_cf_l, a_cf_l - a_ref_l, gap_ref, gap_cf, b_lower95, a_cf, b_ref,
                       not reasons, reasons)


def certify(n_episodes: int = 1000, seeds_offset: int = 10_000,
            worlds: Iterable[WorldConfig] = tuple(WORLDS.values()),
            k: float = K_FLOOR) -> dict:
    """Run stages 1-2 and return a JSON-serialisable certificate."""
    fixture = fixture_table()
    mismatches = [f"{key}: {fixture[key]} != {v}" for key, v in CERTIFIED_FIXTURE.items()
                  if fixture[key] != v]
    sim = simulate_fixture(n_episodes, seeds_offset)
    seeds = range(seeds_offset, seeds_offset + n_episodes)
    cells = {}
    for w in worlds:
        for p in (PROFILE_A, PROFILE_B):
            exact = planner_value(w, p)
            plan = run_many(w, p, seeds, KnownLawPlanner(w, PROFILES))
            rnd = run_many(w, p, seeds, RandomPolicy())
            orc = run_many(w, p, seeds, OraclePolicy())
            ps = summarise_returns([r.episode_return for r in plan])
            sched = lazy_rest_schedule(w, p)
            cells[f"{w.name}/{p.name}"] = {
                "world": w.name, "body": p.name, "q": w.q, "noise": w.noise,
                "planner_exact": exact, "planner_exact_is_exact": w.q in (0.0, 1.0),
                "planner_mc": ps.as_dict(),
                "planner_welfare_violations": int(sum(r.report.outcomes.welfare_violation for r in plan)),
                "random_mc": summarise_returns([r.episode_return for r in rnd]).as_dict(),
                "random_welfare_violations": int(sum(r.report.outcomes.welfare_violation for r in rnd)),
                "oracle_mc": summarise_returns([r.episode_return for r in orc]).as_dict(),
                "active_cycles": sched.active_cycles,
                "max_active_cycles": max_active_cycles(w, p),
                "floor_k": k,
                "meets_floor": bool(ps.lower95 >= k),
            }
            key = (w.name, p.name)
            if key in CERTIFIED_PLANNER and abs(exact - CERTIFIED_PLANNER[key]) > 1e-9:
                mismatches.append(f"planner {key}: {exact} != {CERTIFIED_PLANNER[key]}")
    b_ref = cells["reference/B"]
    gate = witness_gate(fixture, cells["counterfactual/A"]["planner_exact"], b_ref["planner_exact"],
                        b_ref["planner_mc"]["lower95"], b_ref["planner_welfare_violations"], k)
    floors = {}
    for w in worlds:
        if w.is_control:
            continue
        floors[w.name] = sorted(p for p in "AB" if cells[f"{w.name}/{p}"]["meets_floor"])
    return {
        "version": "Water-v1",
        "n_episodes": n_episodes,
        "seeds": [seeds_offset, seeds_offset + n_episodes],
        "fixture_exact": {f"{r}/{b}": str(v) for (r, b), v in fixture.items()},
        "fixture_exact_float": {f"{r}/{b}": float(v) for (r, b), v in fixture.items()},
        "fixture_simulated": {f"{r}/{b}": s.as_dict() for (r, b), s in sim.items()},
        "cells": cells,
        "floors_met": floors,
        "witness_gate": gate.as_dict(),
        "cross_world_condition": bool(cells["counterfactual/A"]["planner_exact"] >= b_ref["planner_exact"]),
        "mismatches": mismatches,
        "certified": bool(not mismatches and gate.passes),
    }


# ----- frozen references for S ---------------------------------------------------------------

FROZEN_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "frozen"
REFERENCES_FILE = FROZEN_DIR / "references.json"


def write_references(cert: dict, path: Path = REFERENCES_FILE) -> Path:
    """Freeze J_reference (planner) and J_random per world and body."""
    refs = {}
    for key, c in cert["cells"].items():
        j_ref = c["planner_exact"] if c["planner_exact_is_exact"] else c["planner_mc"]["mean"]
        j_rand = c["random_mc"]["mean"]
        refs[key] = {
            "J_reference": j_ref,
            "J_reference_source": "exact DP" if c["planner_exact_is_exact"] else "Monte Carlo",
            "J_random": j_rand,
            "J_oracle": c["oracle_mc"]["mean"],
            "scored": bool((j_ref - j_rand) > 1.0),
        }
    payload = {"version": cert["version"], "n_episodes": cert["n_episodes"],
               "seeds": cert["seeds"], "references": refs}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_references(path: Path = REFERENCES_FILE) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `iuw certify --freeze` first")
    return json.loads(path.read_text())["references"]
