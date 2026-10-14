"""Command line, one command per build stage.

  iuw certify --freeze          1-2  physics, fixture, planner, floors, frozen references
  iuw render-check              3    tutorials, forbidden statements, parity, budget, information
  iuw quiz                      3    interface quiz on the frozen model
  iuw frozen                    4    zero-shot, explicit rules, words-evoke-priors
  iuw smoke                     5    100 episodes: throughput, cache hits, illegal mass, cost
  iuw notebook                  6    notebook hill climb (one per rendering), then sealed test
  iuw middle-loop               7    training-settings search on development seeds
  iuw train / iuw grid          7    one run / the 12-run MVP grid (+ familiarised arm)
  iuw analyze                   7-8  Delta_prior, adaptation speed, gates
  iuw show                           print a rendered sample episode

Backends: --backend tinker (needs TINKER_API_KEY), hf (local transformers+peft) or
mock (numpy stand-in for tests; never evidence about language models).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import PROFILES, ModelChoice, get_world


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def make_backend(args, seed: int = 0, init_checkpoint=None):
    name = args.backend
    if name == "mock":
        from .llm.mock import MockBackend

        return MockBackend(seed=seed)
    if name == "tinker":
        from .llm.tinker_backend import TinkerBackend

        return TinkerBackend(args.model, rank=args.rank, seed=seed, init_checkpoint=init_checkpoint)
    if name == "hf":
        from .llm.hf_backend import HFBackend

        return HFBackend(args.model, rank=args.rank, seed=seed, init_adapter=init_checkpoint,
                         device=args.device, dtype=args.dtype, lora_experts=args.lora_experts)
    raise SystemExit(f"unknown backend {name}")


def _normaliser():
    from .certify import load_references
    from .scoring import Scorer

    return Scorer(load_references())


def _run_config(args):
    from .llm.backend import LossConfig
    from .train.advantages import KLSchedule
    from .train.inner import RunConfig

    return RunConfig(
        run_id=args.run_id or f"{args.rendering}-{args.world}-s{args.seed}",
        rendering=args.rendering, train_world=get_world(args.world).name, seed_index=args.seed,
        updates=args.updates, seeds_per_update=args.seeds_per_update, eval_seeds=args.eval_seeds,
        rank=args.rank, base_model=args.model,
        loss=LossConfig(clip=0.2, loss=args.loss, learning_rate=args.lr, passes=args.passes),
        kl=KLSchedule(args.kl, args.kl_coef, args.updates), entropy_coef=args.entropy,
        baseline=args.baseline, explicit_rules=args.explicit_rules,
        eval_updates=(tuple(sorted(set(range(0, args.updates + 1, args.eval_every)) | {args.updates}))
                      if args.eval_every else tuple(u for u in (0, 10, 20, 30, 40, 50) if u <= args.updates)),
        checkpoint_every=args.checkpoint_every)


def cmd_certify(args):
    from .certify import REFERENCES_FILE, certify, write_references

    cert = certify(n_episodes=args.episodes)
    if args.freeze:
        write_references(cert)
        (REFERENCES_FILE.parent / "certification.json").write_text(json.dumps(cert, indent=2, sort_keys=True))
    rows = {k: {"planner_exact": round(c["planner_exact"], 4), "planner_mc": round(c["planner_mc"]["mean"], 3),
                "lower95": round(c["planner_mc"]["lower95"], 3), "random": round(c["random_mc"]["mean"], 3),
                "oracle": round(c["oracle_mc"]["mean"], 3), "meets_K": c["meets_floor"]}
            for k, c in cert["cells"].items()}
    _print({"certified": cert["certified"], "mismatches": cert["mismatches"], "cells": rows,
            "floors_met": cert["floors_met"], "witness_gate": cert["witness_gate"],
            "fixture": cert["fixture_exact_float"]})
    return 0 if cert["certified"] else 1


def cmd_render_check(args):
    from .config import COUNTERFACTUAL, NOISY_COUNTERFACTUAL, NOISY_REFERENCE, REFERENCE, ContextBudget
    from .llm.tokenizer import load_tokenizer
    from .policies.baselines import AlwaysProbePolicy, RandomPolicy
    from .policies.planner import KnownLawPlanner
    from .render.checks import budget_report, information_equality, tutorial_checks
    from .render.rendering import make_rendering

    tok = load_tokenizer(args.tokenizer)
    rs = [make_rendering("named")] + [make_rendering("masked", m) for m in range(3)]
    worlds = (REFERENCE, COUNTERFACTUAL, NOISY_REFERENCE, NOISY_COUNTERFACTUAL)
    tut = tutorial_checks(rs, tok)

    def eps():
        for w in worlds:
            for p in PROFILES.values():
                for s in range(args.seeds):
                    for pol in (KnownLawPlanner(w, PROFILES), RandomPolicy(), AlwaysProbePolicy()):
                        yield w, p, s, pol

    notebook = " ".join(["note"] * 2000)
    nb_ids = tok.encode(notebook)[:ContextBudget().notebook]
    notebook = tok.decode(nb_ids)
    budgets = {r.name: budget_report(r, tok, eps(), notebook=notebook).__dict__ for r in rs}
    info = {w.name: information_equality(w, PROFILES["A"], range(args.seeds),
                                         lambda w=w: KnownLawPlanner(w, PROFILES), rs)["equal"]
            for w in worlds}
    within = all(b["max_total"] <= ContextBudget().total and not b["overflow"]
                 and b["prompt_tokens"] <= ContextBudget().system_and_tutorial
                 and b["max_per_decision"] <= ContextBudget().per_decision_max for b in budgets.values())
    out = {"tokenizer": tok.name, "tutorial": tut, "budget": budgets, "budget_ok": within,
           "information_equal": info}
    _print(out)
    return 0 if (tut["clean"] and within and all(info.values())) else 1


def cmd_quiz(args):
    from .experiment.frozen import interface_quiz

    out = interface_quiz(make_backend(args), args.mapping)
    _print(out)
    return 0 if out["gate"]["passes"] else 1


def cmd_frozen(args):
    from .analysis.gates import zero_shot_check
    from .experiment.frozen import frozen_cells, words_evoke_priors

    be = make_backend(args)
    cells = frozen_cells(be, _normaliser())
    words = words_evoke_priors(be)
    out = {k: {kk: vv for kk, vv in v.items() if kk != "episodes"} if isinstance(v, dict) else v
           for k, v in cells.items()}
    out["zero_shot_check"] = zero_shot_check(cells["zero_shot_named_cf_S"])
    out["words_evoke_priors"] = words
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "frozen.json").write_text(json.dumps(cells, indent=2, default=str))
    _print(out)
    return 0


def cmd_smoke(args):
    from .experiment.smoke import smoke_test

    be = make_backend(args)
    _print(smoke_test(be, n_episodes=args.episodes, progress=print))
    return 0


def cmd_notebook(args):
    from .experiment.notebook import notebook_climb, notebook_sealed_test

    be = make_backend(args)
    out = Path(args.out) / f"notebook-{args.rendering}"
    res = notebook_climb(be, args.rendering, _normaliser(), world=get_world(args.world).name,
                         margin=args.margin, out_dir=out, progress=print)
    summary = {"rendering": res.rendering, "accepted": sum(h["accepted"] for h in res.history),
               "episodes_used": res.episodes_used, "rewrite_calls": res.rewrite_calls}
    if args.test:
        summary["test"] = notebook_sealed_test(be, res, _normaliser(), out)
    _print(summary)
    return 0


def cmd_middle_loop(args):
    from .experiment.middle_loop import middle_loop

    base = _run_config(args)
    res = middle_loop(base, lambda cfg: make_backend(args, seed=cfg.seed_index), _normaliser(),
                      threshold=args.threshold, out_dir=Path(args.out) / "middle-loop", progress=print)
    _print(res.__dict__)
    return 0


def cmd_train(args):
    from .train.inner import run_training, sealed_test

    cfg = _run_config(args)
    be = make_backend(args, seed=cfg.seed_index, init_checkpoint=args.init_checkpoint)
    run_dir = Path(args.out) / cfg.run_id
    res = run_training(cfg, be, _normaliser(), run_dir, progress=print, resume=args.resume)
    out = {"run_id": cfg.run_id, "learning_cost": res.learning_cost, "flags": res.flags[:20],
           "eval_curve": res.eval_curve(), "usage": res.usage}
    if args.test:
        out["test"] = sealed_test(cfg, be, _normaliser(), run_dir).as_dict()
    _print(out)
    return 0


def cmd_grid(args):
    from .experiment.grid import familiarised_grid, mvp_grid, run_grid

    template = _run_config(args)
    configs = familiarised_grid(template) if args.familiarised else mvp_grid(template)
    if args.only:
        configs = [c for c in configs if c.run_id in set(args.only.split(","))]
    run_grid(configs, lambda cfg: make_backend(args, seed=cfg.seed_index), _normaliser(),
             Path(args.out), open_sealed_test=args.test, familiarise=args.familiarised, progress=print)
    return 0


def cmd_analyze(args):
    from .analysis.prior import delta_prior, prior_conflict_gate
    from .analysis.stats import adaptation_speed
    from .experiment.grid import load_learning_costs

    root = Path(args.runs)
    costs = load_learning_costs(root)
    std = delta_prior(costs, masked_familiarised=False)
    fam = delta_prior(costs, masked_familiarised=True)
    speeds = {}
    for d in sorted(root.iterdir()):
        ev = d / "evals.jsonl"
        if ev.exists():
            curve = [(r["interactions"], r["mean_score"]) for r in map(json.loads, ev.read_text().splitlines())]
            speeds[d.name] = adaptation_speed(curve)
    _print({"runs": costs, "delta_prior_standard": std, "delta_prior_familiarised": fam,
            "prior_conflict_gate": prior_conflict_gate([std, fam]), "adaptation_speed": speeds})
    return 0


def cmd_show(args):
    from .llm.tokenizer import load_tokenizer
    from .policies.planner import KnownLawPlanner
    from .render.checks import transcript_for_episode
    from .render.rendering import PromptOptions, make_rendering

    w = get_world(args.world)
    tok = load_tokenizer(args.tokenizer)
    opts = PromptOptions(explicit_rules=w.q if args.explicit_rules else None)
    tr = transcript_for_episode(w, PROFILES[args.body], args.seed, KnownLawPlanner(w, PROFILES),
                                make_rendering(args.rendering, args.mapping), tok, opts)
    text = tr.text()
    lines = text.splitlines()
    print("\n".join(lines[:args.lines]))
    print(f"... ({len(tr.tokens)} tokens, {tr.n_decisions} decisions, prompt {tr.prompt_len})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iuw", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def backend_args(sp):
        sp.add_argument("--backend", default="tinker", choices=["tinker", "hf", "mock"])
        sp.add_argument("--model", default=ModelChoice().main)
        sp.add_argument("--rank", type=int, default=32)
        sp.add_argument("--device", default=None, help="hf backend: cuda, mps or cpu (default: best available)")
        sp.add_argument("--dtype", default="bfloat16", help="hf backend: bfloat16, float16 or float32")
        sp.add_argument("--lora-experts", action="store_true",
                        help="hf backend: also put LoRA on fused MoE expert weights (all-linear skips them)")

    def run_args(sp):
        backend_args(sp)
        sp.add_argument("--rendering", default="named", choices=["named", "masked"])
        sp.add_argument("--world", default="counterfactual")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--run-id", default=None)
        sp.add_argument("--updates", type=int, default=50)
        sp.add_argument("--seeds-per-update", type=int, default=8, help="x 2 bodies = episodes per update")
        sp.add_argument("--eval-seeds", type=int, default=None,
                        help="evaluate on the first N dev seeds only (local tryouts; default all 16)")
        sp.add_argument("--lr", type=float, default=1e-5)
        sp.add_argument("--passes", type=int, default=1)
        sp.add_argument("--loss", default="builtin-ppo", choices=["builtin-ppo", "renormalised"])
        sp.add_argument("--kl", default="none", choices=["none", "fixed", "anneal"])
        sp.add_argument("--kl-coef", type=float, default=0.0)
        sp.add_argument("--entropy", type=float, default=0.0)
        sp.add_argument("--baseline", default="batch", choices=["batch", "per-cycle"])
        sp.add_argument("--explicit-rules", action="store_true")
        sp.add_argument("--out", default="runs")
        sp.add_argument("--test", action="store_true", help="open the sealed test once at the end")
        sp.add_argument("--eval-every", type=int, default=None,
                        help="evaluate every N updates and at the last (default: the spec's 0, 10 ... 50)")
        sp.add_argument("--checkpoint-every", type=int, default=5,
                        help="save resumable state every N updates (also at every evaluation)")

    sp = sub.add_parser("certify")
    sp.add_argument("--episodes", type=int, default=1000)
    sp.add_argument("--freeze", action="store_true")
    sp.set_defaults(func=cmd_certify)

    sp = sub.add_parser("render-check")
    sp.add_argument("--tokenizer", default="approx")
    sp.add_argument("--seeds", type=int, default=3)
    sp.set_defaults(func=cmd_render_check)

    sp = sub.add_parser("quiz")
    backend_args(sp)
    sp.add_argument("--mapping", type=int, default=0)
    sp.set_defaults(func=cmd_quiz)

    sp = sub.add_parser("frozen")
    backend_args(sp)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_frozen)

    sp = sub.add_parser("smoke")
    backend_args(sp)
    sp.add_argument("--episodes", type=int, default=100)
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("notebook")
    backend_args(sp)
    sp.add_argument("--rendering", default="named", choices=["named", "masked"])
    sp.add_argument("--world", default="counterfactual")
    sp.add_argument("--margin", type=float, default=1.0, help="fixed in advance (reward units)")
    sp.add_argument("--out", default="runs")
    sp.add_argument("--test", action="store_true")
    sp.set_defaults(func=cmd_notebook)

    sp = sub.add_parser("middle-loop")
    run_args(sp)
    sp.add_argument("--threshold", type=float, required=True, help="practical S improvement, fixed in advance")
    sp.set_defaults(func=cmd_middle_loop)

    sp = sub.add_parser("train")
    run_args(sp)
    sp.add_argument("--init-checkpoint", default=None)
    sp.add_argument("--resume", action="store_true", help="continue a partial run from its last saved state")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("grid")
    run_args(sp)
    sp.add_argument("--familiarised", action="store_true", help="codebook-familiarised Masked arm")
    sp.add_argument("--only", default=None, help="comma-separated run ids")
    sp.set_defaults(func=cmd_grid)

    sp = sub.add_parser("analyze")
    sp.add_argument("--runs", default="runs")
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("show")
    sp.add_argument("--rendering", default="named", choices=["named", "masked", "named-renamed"])
    sp.add_argument("--mapping", type=int, default=0)
    sp.add_argument("--world", default="counterfactual")
    sp.add_argument("--body", default="A", choices=["A", "B"])
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--lines", type=int, default=40)
    sp.add_argument("--tokenizer", default="approx")
    sp.add_argument("--explicit-rules", action="store_true")
    sp.set_defaults(func=cmd_show)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
