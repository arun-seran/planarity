import pytest

from iuw.config import PROFILE_A, PROFILE_B, WORLDS
from iuw.policies.planner import planner_value
from iuw.scoring import Scorer


@pytest.fixture(scope="session")
def normaliser():
    """References from the exact planner and a fixed random-policy estimate (no file needed)."""
    random_est = {"A": 1.5, "B": 10.1}
    refs = {}
    for w in WORLDS.values():
        for p in (PROFILE_A, PROFILE_B):
            refs[f"{w.name}/{p.name}"] = {"J_reference": planner_value(w, p),
                                         "J_random": random_est[p.name] if w.noise == 0 else 0.1}
    return Scorer(refs)
