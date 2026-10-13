"""Experiment drivers, run end to end on the numpy mock backend (no network, no GPU)."""

import json

import numpy as np
import pytest

from iuw.cli import main as cli_main
from iuw.config import COUNTERFACTUAL, PROFILE_A, PROFILES, REFERENCE
from iuw.experiment.frozen import frozen_cells, interface_quiz, words_evoke_priors
from iuw.experiment.grid import familiarise_codebook, familiarised_grid, load_learning_costs, mvp_grid
from iuw.experiment.middle_loop import Proposal, middle_loop
from iuw.experiment.notebook import SELECTION_SEEDS, notebook_climb
from iuw.experiment.smoke import smoke_test
from iuw.inspection.ledger import (ChangeLedger, LedgerRow, law_next_target, policy_on_histories,
                                   reward_belief, transition_belief, value_belief)
from iuw.inspection.replay import record_history, replay_distributions
from iuw.inspection.return_trip import diagnose, return_trip
from iuw.llm.mock import MockBackend
from iuw.policies.planner import KnownLawPlanner
from iuw.render import make_rendering
from iuw.train.inner import RunConfig


def test_grid_shapes():
    grid = mvp_grid()
    assert len(grid) == 12 and len({c.run_id for c in grid}) == 12
    assert {c.mapping_index for c in grid if c.rendering == "masked"} == {0, 1, 2}
    assert len(familiarised_grid()) == 6


def test_interface_quiz_runs_and_reports_gate():
    out = interface_quiz(MockBackend())
    assert set(out["gate"]) >= {"named", "masked", "gap", "passes"}
    assert 0 <= out["named"]["accuracy"] <= 1


def test_frozen_cells_and_words_check(normaliser):
    be = MockBackend()
    cells = frozen_cells(be, normaliser, split="dev")
    assert {"zero-shot/named", "zero-shot/masked", "explicit-rules/named", "explicit-rules/masked"} <= set(cells)
    assert cells["zero-shot/named"]["n"] == 16 * 2 * 2
    w = words_evoke_priors(be, seeds=[100000, 100001], decisions=20)
    assert w["n_decisions"] == 2 * 2 * 20 and w["mean_js"] >= 0


def test_notebook_climb_budget(normaliser, tmp_path):
    be = MockBackend()
    res = notebook_climb(be, "named", normaliser, rounds=2, margin=-1e9, out_dir=tmp_path)
    assert res.rewrite_calls == 2
    assert res.episodes_used == 2 * (8 + 2 * SELECTION_SEEDS)
    assert res.notebook == be.generated_notebooks[-1]  # every candidate accepted at margin -inf
    assert (tmp_path / "notebook.json").exists()


@pytest.mark.slow
def test_middle_loop_stops_after_rejections(normaliser):
    base = RunConfig("ml", rendering="named", train_world="counterfactual")
    res = middle_loop(base, lambda cfg: MockBackend(seed=0), normaliser, threshold=10.0,
                      proposals=[Proposal("learning_rate", 0.02), Proposal("entropy_coef", 0.01),
                                 Proposal("kl", ("fixed", 0.1)), Proposal("learning_rate", 0.1)],
                      max_rejections=3)
    assert len(res.history) == 3 and not any(h["accepted"] for h in res.history)
    assert res.stopped_because == "3 rejections in a row"
    with pytest.raises(ValueError):
        Proposal("rank", 8).apply(base)


def test_smoke_test_reports_costs():
    out = smoke_test(MockBackend(), n_episodes=8, batch=8)
    assert out["episodes"] == 8 and out["truncated"] == 0 and out["unclean_audits"] == 0
    assert out["projected_mvp_usd"]["episodes"] == 19_124


def test_codebook_familiarisation_fixes_reference():
    be = MockBackend()
    out = familiarise_codebook(be, mapping_index=1, steps=2)
    assert out["pairs"] >= 10


def test_replay_and_ledger_probes():
    be = MockBackend()
    r = make_rendering("named")
    h = record_history(COUNTERFACTUAL, PROFILE_A, 5, KnownLawPlanner(COUNTERFACTUAL, PROFILES), max_decisions=12)
    probs = replay_distributions(h, be.scorer(), be.tokenizer, r)
    assert probs.shape == (12, 6) and np.allclose(probs.sum(1), 1.0)
    tb = transition_belief(h, be.scorer(), be.tokenizer, r, COUNTERFACTUAL)
    assert tb["n"] >= 1
    assert np.allclose(law_next_target(COUNTERFACTUAL, 3), [0, 0, 1, 0])
    assert np.allclose(law_next_target(REFERENCE, 3), [0.25] * 4)
    pol = policy_on_histories([h], be.scorer(), be.tokenizer, r)
    assert 0 <= pol["p_probe"] <= 1
    vb = value_belief(h, be.scorer(), be.tokenizer, r, at_decision=3)
    assert vb.shape == (4,) and vb.sum() == pytest.approx(1)
    assert 0 <= reward_belief(be.scorer(), be.tokenizer, r)["accuracy"] <= 1
    ledger = ChangeLedger()
    ledger.add(LedgerRow("Policy", "repeated action is wasteful", "repetition sustains a stable route", 0.4,
                         "advantage vanishes when routes turn volatile", [0.2, 0.6], "persistent task allocation"))
    assert "repeated action is wasteful" in ledger.to_markdown()


def test_return_trip_and_diagnosis(normaliser):
    be = MockBackend()
    scores = return_trip(be.scorer(), be.reference_scorer(), be.tokenizer, make_rendering("named"),
                         normaliser, seeds=[100000], prefix_len=6)
    assert {"adapted/reset", "base/reset", "adapted/world-cue", "adapted/matched-history"} <= set(scores)
    d = diagnose({"base/reset": 0.9, "adapted/reset": 0.3, "adapted/world-cue": 0.88,
                  "adapted/reminder": 0.3, "adapted/matched-history": 0.3})
    assert "misidentification" in d["reading"]
    d = diagnose({"base/reset": 0.9, "adapted/reset": 0.3, "adapted/world-cue": 0.3,
                  "adapted/reminder": 0.3, "adapted/matched-history": 0.3},
                 relearning={"original": 480, "relearn": 480})
    assert "overwriting" in d["reading"]


def test_cli_certify_train_analyze(tmp_path, capsys):
    assert cli_main(["certify", "--episodes", "40"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["certified"] and out["floors_met"]["counterfactual"] == ["A", "B"]
    runs = tmp_path / "runs"
    for kind in ("named", "masked"):
        for world in ("reference", "counterfactual"):
            assert cli_main(["train", "--backend", "mock", "--rendering", kind, "--world", world,
                             "--updates", "1", "--out", str(runs)]) == 0
    capsys.readouterr()
    costs = load_learning_costs(runs)
    assert len(costs) == 4
    assert cli_main(["analyze", "--runs", str(runs)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["delta_prior_standard"]["delta_prior"] is not None
