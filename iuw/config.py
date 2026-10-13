"""Frozen constants for Water-v1: profiles, worlds, reward coefficients, floors.

Profiles are sets of independent dimensions, never a diagnosis. Worlds change
physics or institutions only; utility, resources, profiles and welfare rules
stay frozen across matched worlds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

VERSION = "Water-v1"

# Floor K for Water-v1, judged on the planner's lower 95% bound (spec v3, change 1).
K_FLOOR = 60.0

# Action IDs: single-token digits, verified against the tokenizer at run time.
PROBE = 0
VALVES = (1, 2, 3, 4)
REST = 5
N_ACTIONS = 6
ACTION_NAMES = {0: "probe", 1: "actuate-1", 2: "actuate-2", 3: "actuate-3", 4: "actuate-4", 5: "rest"}


@dataclass(frozen=True)
class Profile:
    """A body: costs and capacities that the world never changes."""

    name: str
    repeat_precision: float  # execution success when repeating the previous command
    switch_precision: float  # execution success when changing command, or on the first one
    noise_threshold: float  # exposure above it adds burden each cycle
    recovery: float  # burden removed by one cycle of rest

    def execution_probability(self, command: int, last_command: Optional[int]) -> float:
        if last_command is not None and command == last_command:
            return self.repeat_precision
        return self.switch_precision


PROFILE_A = Profile("A", repeat_precision=0.98, switch_precision=0.10, noise_threshold=0.2, recovery=2.0)
PROFILE_B = Profile("B", repeat_precision=0.75, switch_precision=0.75, noise_threshold=2.0, recovery=2.0)
PROFILES = {"A": PROFILE_A, "B": PROFILE_B}


def equalised_execution(profile: Profile, source: Profile = PROFILE_B) -> Profile:
    """Equalised-execution control: keep the profile's other dimensions, copy the body law."""
    return replace(profile, repeat_precision=source.repeat_precision,
                   switch_precision=source.switch_precision)


@dataclass(frozen=True)
class RewardRemap:
    """Arbitrary reward remapping control.

    Physics is unchanged, but reward is credited when an actuation succeeds on
    ``permutation[target]`` instead of on the target. The physics log keeps
    true liters; only the reward stream is remapped. Control worlds only.
    """

    permutation: tuple[int, ...] = (2, 3, 4, 1)  # valve v (1-based) -> credited valve


@dataclass(frozen=True)
class WorldConfig:
    name: str
    q: float  # target persistence; never revealed to the learner
    noise: float  # ambient noise n
    horizon: int = 100
    n_valves: int = 4
    reserve: int = 100
    demand: int = 100
    energy: int = 200
    probe_cost: int = 1
    actuate_cost: int = 1
    rest_cost: int = 0
    burden_decay: float = 0.9
    burden_limit: float = 10.0
    energy_coef: float = 0.05
    damage_coef: float = 1.0
    liters_per_success: int = 1
    # Controls (off in primary worlds).
    probe_shuffle: bool = False  # signal-shuffle control: probe reports an unrelated valve
    reward_remap: Optional[RewardRemap] = None
    version: str = VERSION

    def excess_noise(self, profile: Profile) -> float:
        return max(0.0, self.noise - profile.noise_threshold)

    @property
    def is_control(self) -> bool:
        return self.probe_shuffle or self.reward_remap is not None


REFERENCE = WorldConfig("reference", q=0.0, noise=0.0)
COUNTERFACTUAL = WorldConfig("counterfactual", q=1.0, noise=0.0)
NOISY_REFERENCE = WorldConfig("noisy-reference", q=0.0, noise=2.0)
NOISY_COUNTERFACTUAL = WorldConfig("noisy-counterfactual", q=1.0, noise=2.0)

WORLDS = {w.name: w for w in (REFERENCE, COUNTERFACTUAL, NOISY_REFERENCE, NOISY_COUNTERFACTUAL)}
WORLD_ALIASES = {"ref": "reference", "cf": "counterfactual", "noisy-ref": "noisy-reference",
                 "noisy-cf": "noisy-counterfactual"}

# Mechanism-family labels for the MVP worlds, relative to the noisy volatile baseline.
MECHANISMS = {
    "quiet-water": {"law": "noise 2 -> 0", "effect": "removes a barrier", "type": "could exist"},
    "steady-canal": {"law": "q 0 -> 1", "effect": "reverses ranking", "type": "could exist"},
}


def get_world(name: str) -> WorldConfig:
    name = WORLD_ALIASES.get(name, name)
    if name in WORLDS:
        return WORLDS[name]
    raise KeyError(f"unknown world {name!r}; known: {sorted(WORLDS)}")


def world_with_q(q: float, noise: float = 0.0) -> WorldConfig:
    """Intermediate-persistence worlds for transfer tests (expansion)."""
    return WorldConfig(f"q{q:g}-n{noise:g}", q=q, noise=noise)


@dataclass(frozen=True)
class ContextBudget:
    total: int = 8192
    system_and_tutorial: int = 600
    per_decision_max: int = 30
    notebook: int = 1024
    max_decisions: int = 200

    @property
    def worst_case(self) -> int:
        return self.system_and_tutorial + self.per_decision_max * self.max_decisions + self.notebook


@dataclass(frozen=True)
class Prices:
    """Tinker prices per million tokens, as recorded in v2 (Oct 12, 2026). Re-check before smoke test."""

    prompt: float = 0.45
    cached_prompt: float = 0.09
    training: float = 1.177


@dataclass(frozen=True)
class ModelChoice:
    main: str = "Qwen/Qwen3.6-35B-A3B"
    dense_twin: str = "Qwen/Qwen3.5-9B"
    full_parameter: str = "Qwen/Qwen3-1.7B"
    retired: tuple[str, ...] = field(default=("Qwen/Qwen3.6-27B", "nvidia/Nemotron-3-Nano-30B"))
