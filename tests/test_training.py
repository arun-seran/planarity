import json

import numpy as np
import pytest

from iuw.config import COUNTERFACTUAL, PROFILE_A, PROFILE_B, REFERENCE
from iuw.llm.backend import LossConfig, TrainItem, legal_log_softmax
from iuw.llm.mock import MockBackend
from iuw.llm.policy import EpisodeSpec, SamplingConfig, attach_reference_logprobs, rollout
from iuw.render import make_rendering
from iuw.seeds import SealError
from iuw.train.advantages import (KLSchedule, clipped_surrogate, compute_advantages, returns_to_go,
                                  shaped_rewards)
from iuw.train.inner import RunConfig, make_train_items, run_training, sealed_test


def _specs(n=4, world=COUNTERFACTUAL, kind="named"):
    r = make_rendering(kind, 1)
    return [EpisodeSpec(world, (PROFILE_A, PROFILE_B)[i % 2], 10 + i // 2, r) for i in range(n)]


def test_returns_to_go_gamma_one_and_bootstrap():
    assert list(returns_to_go([1, 0, 2])) == [3, 2, 2]
    assert list(returns_to_go([1, 0, 2], bootstrap=5)) == [8, 7, 7]


def test_kl_schedules():
    assert KLSchedule("none", 0.1).at(5) == 0
    assert KLSchedule("fixed", 0.1).at(30) == 0.1
    a = KLSchedule("anneal", 0.1, total_updates=50)
    assert a.at(1) == pytest.approx(0.1) and a.at(50) == pytest.approx(0.0)
    assert 0 < a.at(25) < 0.1


def test_clipped_surrogate_reference():
    out = clipped_surrogate(np.log([1.5, 0.5]), np.log([1.0, 1.0]), np.array([1.0, -1.0]), clip=0.2)
    # ratio 1.5 with A=+1 clips to 1.2; ratio 0.5 with A=-1 clips to 0.8 -> -0.8*(-1)
    assert out["loss"] == pytest.approx(-(1.2 + -0.8) / 2)
    assert out["clipfrac"] == 1.0


def test_rollout_is_audited_and_deterministic():
    be = MockBackend(seed=1)
    t1 = rollout(_specs(), be.scorer(), be.tokenizer, SamplingConfig(salt=9))
    t2 = rollout(_specs(), be.scorer(), be.tokenizer, SamplingConfig(salt=9))
    t3 = rollout(_specs(), be.scorer(), be.tokenizer, SamplingConfig(salt=10))
    assert [t.actions for t in t1] == [t.actions for t in t2]
    assert [t.actions for t in t1] != [t.actions for t in t3]
    for t in t1:
        assert not t.violations and t.terminated and not t.truncated
        assert all(t.tokens[d.position] == be.action_ids[d.action] for d in t.decisions)
        for d in t.decisions:
            p = np.exp(legal_log_softmax(d.logprobs_all, d.legal))
            assert np.nansum(p[list(d.legal)]) == pytest.approx(1.0)
            assert 0.0 <= d.illegal_mass <= 1.0
        assert t.episode_return == pytest.approx(sum(d.reward for d in t.decisions))


def test_replayed_prefix_actions():
    spec = _specs(1)[0]
    spec.prefix_actions = (0, 2, 0, 2)
    be = MockBackend()
    t = rollout([spec], be.scorer(), be.tokenizer, SamplingConfig())[0]
    assert t.actions[:4] == [0, 2, 0, 2] and all(d.forced for d in t.decisions[:4])


def test_advantages_are_normalised_and_bootstrap_only_on_truncation():
    be = MockBackend()
    trajs = rollout(_specs(), be.scorer(), be.tokenizer, SamplingConfig())
    advs = compute_advantages(trajs)
    flat = np.concatenate(advs)
    assert abs(flat.mean()) < 1e-9 and flat.std() == pytest.approx(1.0)
    pc = np.concatenate(compute_advantages(trajs, baseline="per-cycle"))
    assert pc.std() == pytest.approx(1.0)
    called = []
    compute_advantages(trajs, bootstrap_fn=lambda t, b: called.append(1) or 0.0)
    assert not called  # all episodes terminated: never bootstrap


def test_kl_shaping_uses_reference_logprobs():
    be = MockBackend()
    trajs = rollout(_specs(2), be.scorer(), be.tokenizer, SamplingConfig())
    attach_reference_logprobs(trajs, be.reference_scorer(), be.action_ids)
    t = trajs[0]
    old = np.array([d.lp_legal for d in t.decisions])
    assert np.allclose(t.ref_lp_legal, old)  # untrained: reference equals rollout policy
    assert np.allclose(shaped_rewards(t, 0.5), t.rewards)


@pytest.mark.parametrize("loss", ["builtin-ppo", "renormalised"])
def test_first_pass_ratio_is_one(loss):
    """Rollout scoring and training-time scoring agree token for token."""
    be = MockBackend(seed=3)
    trajs = rollout(_specs(), be.scorer(), be.tokenizer, SamplingConfig())
    items = make_train_items(trajs, compute_advantages(trajs))
    m = be.train(items, LossConfig(loss=loss, learning_rate=0.01, passes=2))
    assert m["ratio_first_pass"] == pytest.approx(1.0)


def test_update_raises_probability_of_advantaged_action():
    be = MockBackend()
    toks = be.tokenizer.encode("12 water 3 energy 50 | 1-5>")
    pos = len(toks)
    toks = toks + [be.action_ids[2]]
    before = be.scorer().score_positions(toks, [pos], be.action_ids)[0]
    lpl = legal_log_softmax(before, (1, 2, 3, 4, 5))
    item = TrainItem(toks, [pos], [2], [(1, 2, 3, 4, 5)], np.array([before[2]]), np.array([lpl[2]]),
                     np.array([1.0]))
    be.train([item], LossConfig(loss="renormalised", learning_rate=0.05))
    after = legal_log_softmax(be.scorer().score_positions(toks, [pos], be.action_ids)[0], (1, 2, 3, 4, 5))
    assert after[2] > lpl[2]


def test_run_training_writes_logs_and_learning_cost(tmp_path, normaliser):
    be = MockBackend()
    cfg = RunConfig("t-named-cf", updates=2, seeds_per_update=2, eval_updates=(0, 2),
                    loss=LossConfig(learning_rate=0.01), kl=KLSchedule("fixed", 0.05, 2))
    res = run_training(cfg, be, normaliser, tmp_path / cfg.run_id)
    assert len(res.train_returns) == 2 * 2 * 2
    assert res.learning_cost == pytest.approx(sum(r["regret"] for r in res.train_returns))
    assert set(res.evals) == {0, 2} and set(res.checkpoints) == {0, 2}
    assert all(e["n"] == 16 * 2 * 2 for e in res.evals.values())
    files = {p.name for p in (tmp_path / cfg.run_id).iterdir()}
    assert {"config.json", "evals.jsonl", "episodes.jsonl", "updates.jsonl", "result.json"} <= files
    result = json.loads((tmp_path / cfg.run_id / "result.json").read_text())
    assert result["n_train_episodes"] == 8
    test = sealed_test(cfg, be, normaliser, tmp_path / cfg.run_id)
    assert len(test.episodes) == 128
    with pytest.raises(SealError):
        sealed_test(cfg, be, normaliser, tmp_path / cfg.run_id)


def test_interrupted_run_resumes_to_the_same_result(tmp_path, normaliser):
    cfg = RunConfig("t-resume", updates=5, seeds_per_update=1, eval_updates=(0, 5), eval_seeds=2,
                    checkpoint_every=2, loss=LossConfig(learning_rate=0.01))
    whole = run_training(cfg, MockBackend(), normaliser, tmp_path / "whole")

    be = MockBackend()
    real_train, calls = be.train, []

    def crash_on_fourth(items, loss):
        calls.append(1)
        if len(calls) == 4:
            raise ConnectionError("simulated drop during update 4")
        return real_train(items, loss)

    be.train = crash_on_fourth
    with pytest.raises(ConnectionError):
        run_training(cfg, be, normaliser, tmp_path / "cut")
    assert json.loads((tmp_path / "cut" / "state.json").read_text())["update"] == 2
    with pytest.raises(RuntimeError, match="partial run"):
        run_training(cfg, be, normaliser, tmp_path / "cut")
    be.train = real_train
    resumed = run_training(cfg, be, normaliser, tmp_path / "cut", resume=True)

    assert resumed.learning_cost == pytest.approx(whole.learning_cost)
    assert [r["return"] for r in resumed.train_returns] == [r["return"] for r in whole.train_returns]
    assert resumed.evals[5]["mean_score"] == pytest.approx(whole.evals[5]["mean_score"])
    updates = [json.loads(l)["update"] for l in (tmp_path / "cut" / "updates.jsonl").read_text().splitlines()]
    assert updates == [1, 2, 3, 4, 5]  # update 3, written before the crash, is not duplicated
