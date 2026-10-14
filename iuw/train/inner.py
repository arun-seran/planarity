"""Inner loop: train LoRA adapters with a clipped policy-gradient update.

One run is 50 updates of 16 episodes (8 training seeds x bodies A and B),
800 episodes in all. The context resets every episode, so only the weights
carry learning. Development evaluations (16 seeds x 2 bodies x both worlds =
64 episodes) run at updates 0, 10, 20, 30, 40 and 50; the sealed test is a
separate, one-time step. The loop cannot change the profile, objective,
observation privileges or physics.

Learning cost C is cumulative regret against the frozen reference return over
the run's training episodes; Delta_prior is computed from it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import PROFILES, ModelChoice, get_world
from ..llm.backend import Backend, LossConfig, TrainItem
from ..llm.policy import EpisodeSpec, SamplingConfig, Trajectory, attach_reference_logprobs, rollout
from ..render.rendering import PromptOptions, make_rendering
from ..rng import keyed_bits, namespace_key
from ..scoring import Scorer as ScoreNormaliser
from ..seeds import SPLITS, Seal, training_batch
from .advantages import KLSchedule, compute_advantages
from .evaluate import EVAL_SALT, EvalResult, eval_specs, evaluate


@dataclass
class RunConfig:
    run_id: str
    rendering: str = "named"  # "named" | "masked"
    train_world: str = "counterfactual"  # "reference" | "counterfactual"
    seed_index: int = 0  # 0..2; also fixes the Masked mapping
    updates: int = 50
    seeds_per_update: int = 8  # x bodies = 16 episodes per update
    eval_seeds: Optional[int] = None  # first N seeds of eval_split; None = all (local tryouts only)
    bodies: tuple[str, ...] = ("A", "B")
    eval_updates: tuple[int, ...] = (0, 10, 20, 30, 40, 50)
    eval_worlds: tuple[str, ...] = ("reference", "counterfactual")
    eval_split: str = "dev"
    loss: LossConfig = field(default_factory=LossConfig)
    kl: KLSchedule = field(default_factory=KLSchedule)
    entropy_coef: float = 0.0  # soft-RL reward bonus -coef * log pi(a); a middle-loop setting
    baseline: str = "batch"  # spec default; "per-cycle" is an optional variance reduction
    temperature: float = 1.0
    rank: int = 32
    base_model: str = ModelChoice().main
    explicit_rules: bool = False
    illegal_mass_limit: float = 0.01
    init_checkpoint: Optional[str] = None  # e.g. a codebook-familiarised checkpoint
    checkpoint_every: int = 5  # resumable state (adapters + optimiser) every N updates

    @property
    def mapping_index(self) -> int:
        return self.seed_index

    @property
    def salt_base(self) -> int:
        return keyed_bits(namespace_key("run"), namespace_key(self.run_id)) & 0xFFFFFFFF

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def make_train_items(trajs, advantages) -> list[TrainItem]:
    items = []
    for t, adv in zip(trajs, advantages):
        if not t.decisions:
            continue
        items.append(TrainItem(
            tokens=t.tokens, positions=t.positions, actions=t.actions,
            legal=[d.legal for d in t.decisions],
            old_logprob_full=np.array([d.lp_full for d in t.decisions]),
            old_logprob_legal=np.array([d.lp_legal for d in t.decisions]),
            advantages=np.asarray(adv, dtype=float)))
    return items


class RunLogger:
    def __init__(self, out_dir: Optional[Path]):
        self.dir = out_dir
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, obj) -> None:
        if self.dir:
            (self.dir / name).write_text(json.dumps(obj, indent=2, default=_json_default))

    def append(self, name: str, obj) -> None:
        if self.dir:
            with open(self.dir / name, "a") as f:
                f.write(json.dumps(obj, default=_json_default) + "\n")

    def read(self, name: str) -> list[dict]:
        f = self.dir / name if self.dir else None
        return [json.loads(line) for line in f.read_text().splitlines()] if f and f.exists() else []

    def keep_through_update(self, name: str, update: int) -> list[dict]:
        """Drop records written after ``update`` (work that a resumed run repeats)."""
        rows = [r for r in self.read(name) if r["update"] <= update]
        if self.dir and (self.dir / name).exists():
            (self.dir / name).write_text("".join(json.dumps(r, default=_json_default) + "\n" for r in rows))
        return rows


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


@dataclass
class RunResult:
    run_id: str
    config: dict
    learning_cost: float
    train_returns: list[dict]
    evals: dict[int, dict]
    checkpoints: dict[int, str]
    flags: list[str]
    usage: dict

    def eval_curve(self, world: Optional[str] = None) -> list[tuple[int, float]]:
        """(interactions, mean S) per scheduled evaluation; interactions = training episodes."""
        out = []
        for u in sorted(self.evals):
            e = self.evals[u]
            if world is None:
                out.append((u, e["mean_score"]))
            else:
                sc = [c["mean_score"] for k, c in e["cells"].items()
                      if k.startswith(world + "/") and c["mean_score"] is not None]
                out.append((u, float(np.mean(sc)) if sc else float("nan")))
        return out


def run_training(cfg: RunConfig, backend: Backend, normaliser: ScoreNormaliser,
                 out_dir: Optional[Path] = None, progress=None, resume: bool = False) -> RunResult:
    """Run one training run. With ``resume``, continue from the run directory's last saved
    state: adapters and optimiser are restored and updates after it are repeated. Rollout
    seeds and sampling salts are keyed by update, so the continuation matches an
    uninterrupted run up to floating-point nondeterminism."""
    log = RunLogger(out_dir)
    state_file = out_dir / "state.json" if out_dir else None
    state = json.loads(state_file.read_text()) if state_file and state_file.exists() else None
    if state and not resume:
        raise RuntimeError(f"{out_dir} already holds a partial run (through update {state['update']}); "
                           "pass resume=True (--resume) to continue it, or remove the directory")
    if not state:
        log.write("config.json", cfg.to_dict())
    rendering = make_rendering(cfg.rendering, cfg.mapping_index)
    world = get_world(cfg.train_world)
    options = PromptOptions(explicit_rules=world.q if cfg.explicit_rules else None)
    tok = backend.tokenizer
    action_ids = None
    flags: list[str] = []
    evals: dict[int, dict] = {}
    checkpoints: dict[int, str] = {}
    train_records: list[dict] = []
    regret_total = 0.0
    start = 1

    def save_state(u: int) -> None:
        path = checkpoints[u] if u in checkpoints else backend.save(f"{cfg.run_id}-u{u:03d}")
        log.write("state.json", {"update": u, "checkpoint": path, "regret_total": regret_total,
                                 "flags": flags, "evals": evals, "checkpoints": checkpoints})

    def do_eval(u: int) -> None:
        specs = eval_specs(SPLITS[cfg.eval_split][:cfg.eval_seeds], cfg.eval_worlds, rendering, cfg.bodies, options,
                           cfg.eval_split)
        res, _ = evaluate(backend.scorer(), tok, specs, normaliser, salt=EVAL_SALT,
                          temperature=cfg.temperature)
        evals[u] = res.as_dict()
        log.append("evals.jsonl", {"update": u, "interactions": u * cfg.seeds_per_update * len(cfg.bodies),
                                   **res.as_dict()})
        checkpoints[u] = backend.save(f"{cfg.run_id}-u{u:03d}")
        if progress:
            progress(f"[{cfg.run_id}] eval u={u}: S={res.mean_score:.3f}")

    if state:
        backend.load(state["checkpoint"])
        start = state["update"] + 1
        regret_total, flags = state["regret_total"], state["flags"]
        evals = {int(k): v for k, v in state["evals"].items()}
        checkpoints.update({int(k): v for k, v in state["checkpoints"].items()})
        train_records = log.keep_through_update("episodes.jsonl", state["update"])
        log.keep_through_update("updates.jsonl", state["update"])
        log.keep_through_update("evals.jsonl", state["update"])
        if progress:
            progress(f"[{cfg.run_id}] resuming after update {state['update']}")
    elif 0 in cfg.eval_updates:
        do_eval(0)
        save_state(0)
    for u in range(start, cfg.updates + 1):
        t0 = time.time()
        specs = [EpisodeSpec(world, PROFILES[b], seed, rendering, ep, options, "train")
                 for seed, ep in training_batch(u, cfg.seeds_per_update) for b in cfg.bodies]
        sampling = SamplingConfig(cfg.temperature, False, (cfg.salt_base + u) & 0xFFFFFFFF)
        trajs = rollout(specs, backend.scorer(), tok, sampling)
        if action_ids is None and trajs:
            from ..llm.tokenizer import action_token_ids
            action_ids = action_token_ids(tok)
        kl = cfg.kl.at(u)
        if kl:
            attach_reference_logprobs(trajs, backend.reference_scorer(), action_ids)
        if cfg.entropy_coef:
            for t in trajs:
                for d in t.decisions:
                    d.reward = d.reward - cfg.entropy_coef * d.lp_legal
        advs = compute_advantages(trajs, kl_coef=kl, baseline=cfg.baseline)
        metrics = backend.train(make_train_items(trajs, advs), cfg.loss)
        if cfg.entropy_coef:  # restore true rewards for accounting
            for t in trajs:
                for d in t.decisions:
                    d.reward = d.reward + cfg.entropy_coef * d.lp_legal
        illegal = float(np.mean([t.illegal_mass for t in trajs])) if trajs else 0.0
        if illegal > cfg.illegal_mass_limit and cfg.loss.loss == "builtin-ppo":
            flags.append(f"update {u}: illegal-token mass {illegal:.3%} > {cfg.illegal_mass_limit:.0%}; "
                         "switch to the renormalised custom loss")
        for t in trajs:
            r = normaliser.regret(t.world, t.body, t.episode_return)
            regret_total += r
            rec = {"update": u, "body": t.body, "seed": t.seed, "episode": t.episode,
                   "return": t.episode_return, "regret": r,
                   "score": normaliser.score(t.world, t.body, t.episode_return),
                   "welfare_violation": t.outcomes["welfare_violation"],
                   "clean": not t.violations, "truncated": t.truncated,
                   "illegal_mass": t.illegal_mass}
            train_records.append(rec)
            log.append("episodes.jsonl", rec)
            if t.violations:
                flags.append(f"update {u}: audit violations in seed {t.seed}: {t.violations[:2]}")
            if t.truncated:
                flags.append(f"update {u}: episode truncated (seed {t.seed}); context test failed")
        log.append("updates.jsonl", {"update": u, "kl_coef": kl, "illegal_mass": illegal,
                                     "mean_return": float(np.mean([t.episode_return for t in trajs])),
                                     "seconds": time.time() - t0, **metrics})
        if progress:
            progress(f"[{cfg.run_id}] update {u}/{cfg.updates} return="
                     f"{np.mean([t.episode_return for t in trajs]):.2f}")
        if u in cfg.eval_updates:
            do_eval(u)
        if u in cfg.eval_updates or u % cfg.checkpoint_every == 0 or u == cfg.updates:
            save_state(u)
    result = RunResult(cfg.run_id, cfg.to_dict(), regret_total, train_records, evals, checkpoints,
                       flags, backend.usage.as_dict())
    log.write("checkpoints.json", checkpoints)
    log.write("result.json", {"run_id": cfg.run_id, "learning_cost": regret_total,
                              "n_train_episodes": len(train_records), "flags": flags,
                              "evals": evals, "usage": backend.usage.as_dict()})
    return result


def sealed_test(cfg: RunConfig, backend: Backend, normaliser: ScoreNormaliser, run_dir: Path,
                note: str = "") -> EvalResult:
    """Open the sealed test once: 32 test seeds x 2 bodies x both worlds = 128 episodes."""
    Seal(run_dir / "SEALED_TEST_OPENED").open(note or cfg.run_id)
    rendering = make_rendering(cfg.rendering, cfg.mapping_index)
    options = PromptOptions(explicit_rules=get_world(cfg.train_world).q if cfg.explicit_rules else None)
    specs = eval_specs(SPLITS["test"], cfg.eval_worlds, rendering, cfg.bodies, options, "test")
    res, _ = evaluate(backend.scorer(), backend.tokenizer, specs, normaliser, salt=EVAL_SALT,
                      temperature=cfg.temperature)
    RunLogger(run_dir).write("test.json", res.as_dict())
    RunLogger(run_dir).write("test_episodes.json", res.episodes)
    return res
