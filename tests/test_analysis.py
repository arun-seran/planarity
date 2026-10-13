import numpy as np
import pytest

from iuw.analysis.behaviour import behaviour_profile, js_divergence
from iuw.analysis.gates import causality_gate, learning_gate, zero_shot_check
from iuw.analysis.prior import delta_prior, delta_prior_from_cells, prior_conflict_gate
from iuw.analysis.stats import adaptation_speed, bootstrap_ci, cluster_bootstrap, fully_adapted, iqm
from iuw.scoring import normalised_score
from iuw.seeds import SPLITS, assert_disjoint, manifest, training_batch


def test_seed_splits_are_disjoint_and_sized():
    assert_disjoint()
    assert (len(SPLITS["train"]), len(SPLITS["dev"]), len(SPLITS["test"]), len(SPLITS["notebook"])) == (128, 16, 32, 200)
    assert manifest()["sha256_16"] == manifest()["sha256_16"]
    batch = training_batch(17, 8)
    assert batch[0] == (0, 1)  # the 129th seed use wraps to seed 0 with a fresh episode index


def test_normalised_score():
    assert normalised_score(65.0, 10.0, 65.0) == 1.0
    assert normalised_score(10.0, 10.0, 65.0) == 0.0
    assert normalised_score(70.0, 10.0, 65.0) > 1.0  # kept
    assert normalised_score(5.0, 4.5, 5.0) is None  # unscored


def test_iqm_and_bootstrap():
    assert iqm([1, 2, 3, 4, 100, 5, 6, 7]) == pytest.approx(np.mean([3, 4, 5, 6]))
    lo, hi = bootstrap_ci(np.arange(100.0))
    assert lo < 49.5 < hi
    lo, hi = cluster_bootstrap({"r1": [1, 1], "r2": [2, 2], "r3": [3, 3]})
    assert lo <= 2 <= hi


def test_adaptation_speed_and_censoring():
    curve = [(0, 0.1), (160, 0.95), (320, 0.92), (480, 0.97), (640, 0.99)]
    assert adaptation_speed(curve) == {"interactions": 160, "censored": False}
    assert adaptation_speed([(0, 0.1), (160, 0.95), (320, 0.5)])["censored"]
    flat = [(i, 0.95) for i in range(6)]
    assert fully_adapted(flat, welfare_clean=True)
    assert not fully_adapted(flat, welfare_clean=False)


def _costs(delta):
    runs = {}
    base = {("named", "reference"): 100, ("named", "counterfactual"): 300 + delta,
            ("masked", "reference"): 150, ("masked", "counterfactual"): 350}
    for (r, w), c in base.items():
        for s in range(3):
            runs[f"{r}-{w}-s{s}"] = {"rendering": r, "world": w, "cost": c + (s - 1) * 5,
                                     "familiarised": False}
    return runs


def test_delta_prior_interaction_and_interval():
    cells = {("named", "counterfactual"): [10], ("named", "reference"): [4],
             ("masked", "counterfactual"): [7], ("masked", "reference"): [3]}
    assert delta_prior_from_cells(cells) == (10 - 4) - (7 - 3)
    pos = delta_prior(_costs(60))
    assert pos["delta_prior"] == pytest.approx(60) and pos["positive"]
    null = delta_prior(_costs(0))
    assert null["delta_prior"] == pytest.approx(0) and not null["positive"]
    missing = delta_prior(_costs(10), masked_familiarised=True)
    assert missing["delta_prior"] is None
    assert prior_conflict_gate([pos, missing])["passes"]
    assert not prior_conflict_gate([pos, null])["passes"]


def test_learning_gate_and_checks():
    trained = [{"world": "counterfactual", "body": b, "seed": s, "score": 0.95} for s in range(10) for b in "AB"]
    frozen = [{"world": "counterfactual", "body": b, "seed": s, "score": 0.3 + 0.01 * s} for s in range(10) for b in "AB"]
    g = learning_gate(trained, {"zero-shot": frozen})
    assert g["passes"] and g["best_frozen"] == "zero-shot"
    assert not zero_shot_check(0.95)["claim_supported"]
    assert causality_gate(92.0, 21.8)["passes"]


def test_behaviour_profile():
    acts = [0, 3, 3, 3, 0, 2, 5, 2]
    b = behaviour_profile(acts)
    assert b["cycles"] == 6 and b["probe_rate"] == pytest.approx(2 / 6)
    assert b["rest_rate"] == pytest.approx(1 / 6)
    assert js_divergence(np.array([1, 0]), np.array([1, 0])) < 1e-9
