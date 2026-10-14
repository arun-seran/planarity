"""Analysis and insights report from a directory of finished runs and inspections. CPU only.

Reads every run directory (config.json + result.json + evals.jsonl) and every inspection.json
under --runs (and any extra --inspections), recomputes Delta_prior and adaptation speed, and
writes one Markdown report. The caveats section is derived from the data, so a thin grid says so.

  python scripts/report.py --runs runs/report --out runs/report/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iuw.analysis.prior import delta_prior
from iuw.analysis.stats import adaptation_speed
from iuw.experiment.grid import load_learning_costs

WORLDS = ("counterfactual", "reference")
SHORT = {"counterfactual": "cf", "reference": "ref"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def world_scores(ev: dict) -> dict[str, float]:
    """Mean score per evaluation world (bodies averaged), from one evals.jsonl row."""
    out = {}
    for w in WORLDS:
        vals = [c["mean_score"] for k, c in ev["cells"].items() if k.split("/")[0] == w]
        out[w] = float(np.mean(vals)) if vals else float("nan")
    return out


def eval_n(ev: dict, world: str) -> int:
    return sum(c["n"] for k, c in ev["cells"].items() if k.split("/")[0] == world)


def load_runs(root: Path) -> dict[str, dict]:
    runs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not ((d / "config.json").exists() and (d / "result.json").exists()):
            continue
        cfg, res = json.loads((d / "config.json").read_text()), json.loads((d / "result.json").read_text())
        evals = read_jsonl(d / "evals.jsonl")
        runs[cfg["run_id"]] = {
            "dir": d, "config": cfg, "result": res,
            "rendering": cfg["rendering"], "world": cfg["train_world"], "seed": cfg["seed_index"],
            "cost": res["learning_cost"],
            "curve": [(e["update"], world_scores(e)) for e in evals],
            "eval_n": eval_n(evals[0], "counterfactual") if evals else 0,
            "speed": adaptation_speed([(e["interactions"], e["mean_score"]) for e in evals]),
            "welfare_violations": sum(e["welfare_violations"] for e in evals),
            "episodes": res.get("n_train_episodes") or len(read_jsonl(d / "episodes.jsonl")),
        }
    return runs


def load_inspections(root: Path, extra: list[str]) -> dict[str, dict]:
    paths = sorted(root.glob("*/inspection.json")) + [Path(p) / "inspection.json" for p in extra]
    return {p.parent.name: json.loads(p.read_text()) for p in paths if p.exists()}


def fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def agreement(m: dict, world: str) -> float:
    c = m[world]
    return c.get("planner_agreement", c["planner_action_loglik"])


def share_of_change(m: dict, base: dict, adapted: dict, world: str):
    """How much of the adapter's change in planner agreement a variant keeps (1 = all, 0 = none)."""
    d = agreement(adapted, world) - agreement(base, world)
    if abs(d) < 1e-3:
        return None
    return (agreement(m, world) - agreement(base, world)) / d


def section_setup(runs: dict) -> list[str]:
    cfgs = [r["config"] for r in runs.values()]
    c = cfgs[0]
    same = all(x["updates"] == c["updates"] and x["seeds_per_update"] == c["seeds_per_update"]
               and x["loss"] == c["loss"] and x["rank"] == c["rank"] for x in cfgs)
    eps = sorted({r["episodes"] for r in runs.values()})
    return ["## Setup", "",
            f"- Model `{c['base_model']}`, LoRA rank {c['rank']}, loss `{c['loss']['loss']}`, "
            f"learning rate {c['loss']['learning_rate']:g}, KL `{c['kl']['kind']}`.",
            f"- {c['updates']} updates × {c['seeds_per_update'] * len(c['bodies'])} episodes "
            f"= {eps} training episodes per run; evaluations at updates {c['eval_updates']} on "
            f"{runs[next(iter(runs))]['eval_n']} episodes per world.",
            f"- {len(runs)} runs." + ("" if same else " **Settings differ between runs; check config.json.**"), ""]


def section_runs(runs: dict) -> list[str]:
    ups = sorted({u for r in runs.values() for u, _ in r["curve"]})
    head = "| run | rendering | train world | seed | learning cost | " + \
           " | ".join(f"S cf/ref u{u}" for u in ups) + " | adaptation speed |"
    lines = ["## Runs", "",
             "Learning cost is total regret against the known-law planner's return over training episodes (lower "
             "is better). S is the normalised score per evaluation world (0 = uniform random policy, 1 = the known-law planner).", "",
             head, "|" + "---|" * (6 + len(ups))]
    for rid, r in sorted(runs.items(), key=lambda kv: (kv[1]["rendering"], kv[1]["world"], kv[1]["seed"])):
        curve = dict(r["curve"])
        cells = [f"{fmt(curve[u]['counterfactual'], 2)} / {fmt(curve[u]['reference'], 2)}" if u in curve else "–"
                 for u in ups]
        sp = r["speed"]
        speed = f">{sp['interactions']} (censored)" if sp["censored"] else str(sp["interactions"])
        lines.append(f"| {rid} | {r['rendering']} | {SHORT[r['world']]} | {r['seed']} | {r['cost']:.0f} | "
                     + " | ".join(cells) + f" | {speed} |")
    return lines + [""]


def section_cells(runs: dict) -> list[str]:
    lines = ["## Cells", "", "| rendering | train world | runs | learning cost mean ± sd | final S, train world | "
             "final S, other world |", "|---|---|---|---|---|---|"]
    for rend in ("named", "masked"):
        for w in WORLDS:
            rs = [r for r in runs.values() if r["rendering"] == rend and r["world"] == w]
            if not rs:
                lines.append(f"| {rend} | {SHORT[w]} | 0 | – | – | – |")
                continue
            costs = [r["cost"] for r in rs]
            other = "reference" if w == "counterfactual" else "counterfactual"
            fin = [r["curve"][-1][1] for r in rs if r["curve"]]
            sd = f" ± {np.std(costs, ddof=1):.0f}" if len(costs) > 1 else ""
            lines.append(f"| {rend} | {SHORT[w]} | {len(rs)} | {np.mean(costs):.0f}{sd} | "
                         f"{fmt(np.mean([f[w] for f in fin]), 2)} | {fmt(np.mean([f[other] for f in fin]), 2)} |")
    return lines + [""]


def section_prior(root: Path) -> tuple[list[str], dict]:
    dp = delta_prior(load_learning_costs(root))
    lines = ["## Prior conflict (Δ_prior)", "",
             "Δ_prior = (C_Named^cf − C_Named^ref) − (C_Masked^cf − C_Masked^ref), with a run-clustered "
             "bootstrap interval. Positive with an interval above 0 reads as label-associated interference.", ""]
    if dp["delta_prior"] is None:
        miss = ", ".join(f"{r}/{SHORT[w]}" for r, w in dp["missing_cells"])
        lines += [f"**Not computable yet:** no runs in {miss}.", ""]
    else:
        single = [k for k, v in dp["runs_per_cell"].items() if v < 2]
        if len(single) == len(dp["runs_per_cell"]):  # resampling one run per cell gives a zero-width interval
            dp = {**dp, "positive": False, "reading": "no interval: every cell has a single run"}
            head = f"- **Δ_prior = {dp['delta_prior']:.0f}** (point estimate; no interval with one run per cell)."
        elif single:
            dp = {**dp, "partial": single}
            head = (f"- **Δ_prior = {dp['delta_prior']:.0f}**, partial 95% interval [{dp['ci95'][0]:.0f}, "
                    f"{dp['ci95'][1]:.0f}]: it reflects run-to-run spread only in cells with two or more runs; "
                    f"{', '.join(single)} have one run each, so the true interval is wider.")
        else:
            head = (f"- **Δ_prior = {dp['delta_prior']:.0f}**, 95% interval [{dp['ci95'][0]:.0f}, "
                    f"{dp['ci95'][1]:.0f}]; reading: {dp['reading']}.")
        lines += [head,
                  "- Cell means: " + ", ".join(f"{k} {v:.0f}" for k, v in dp["cell_means"].items()) + ".",
                  "- Runs per cell: " + ", ".join(f"{k} {v}" for k, v in dp["runs_per_cell"].items()) + ".", ""]
    return lines, dp


def section_matched(runs: dict) -> list[str]:
    """Pair each Named counterfactual run with the Masked checkpoint nearest its final cf score."""
    named = [r for r in runs.values() if r["rendering"] == "named" and r["world"] == "counterfactual"]
    masked = [r for r in runs.values() if r["rendering"] == "masked" and r["world"] == "counterfactual"]
    if not (named and masked):
        return []
    lines = ["## Matched-performance pairs", "",
             "Comparing adapters at equal counterfactual score separates *what* was learned from *how far* "
             "training got. Inspect these pairs side by side.", "",
             "| Named checkpoint | S cf | nearest Masked checkpoint | S cf |", "|---|---|---|---|"]
    for n in named:
        u, s = n["curve"][-1][0], n["curve"][-1][1]["counterfactual"]
        best = min(((m, mu, ms["counterfactual"]) for m in masked for mu, ms in m["curve"]),
                   key=lambda t: abs(t[2] - s))
        lines.append(f"| {n['config']['run_id']}-u{u:03d} | {fmt(s, 2)} | "
                     f"{best[0]['config']['run_id']}-u{best[1]:03d} | {fmt(best[2], 2)} |")
    return lines + [""]


def section_inspections(insp: dict) -> list[str]:
    if not insp:
        return ["## Inspection", "", "No inspection.json found.", ""]
    lines = ["## Inspection", "",
             "Agreement is the mean log-probability of the known-law planner's action on matched held-out "
             "histories (closer to 0 agrees more). Kept share is how much of the adapter's change in "
             "counterfactual agreement survives (1 = all of it, 0 = none); a random control near 0 shows "
             "the kept directions are specific. Transition KL is the distance between the model's "
             "next-valve belief and the world's law (lower is closer).", "",
             "| inspection | ε | cf agreement base → adapted | ref agreement base → adapted | cf p_probe base → adapted | "
             "kept share k=1 / k=4 / random k=4 | most lost with one block off | transition KL cf | transition KL ref | "
             "ref-world S adapted vs base (reset) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(insp.items()):
        b, a = r["base"], r["adapted"]
        td = r["top_directions"]
        kept = " / ".join(fmt(share_of_change(td[k], b, a, "counterfactual"), 2) for k in ("k=1", "k=4", "k=4 random"))
        blocks = {k: share_of_change(v, b, a, "counterfactual") for k, v in r["ablate_blocks"].items()}
        lost = {k: (None if v is None else 1 - v) for k, v in blocks.items()}
        worst = max((kv for kv in lost.items() if kv[1] is not None), key=lambda kv: kv[1], default=(None, None))
        block = f"layers {worst[0]}: {fmt(worst[1], 2)}" if worst[0] else "–"
        tb = r["transition_belief"]
        kl = {w: f"{fmt(tb['base'][w]['mean_kl_from_law'], 2)} → {fmt(tb['adapted'][w]['mean_kl_from_law'], 2)} "
                 f"(n={tb['adapted'][w]['n']})" for w in WORLDS}
        rt = r.get("return_trip", {}).get("scores", {})
        trip = f"{fmt(rt.get('adapted/reset'), 2)} vs {fmt(rt.get('base/reset'), 2)}" if rt else "–"
        lines.append(f"| {name} | {r.get('epsilon', 0)} | {fmt(agreement(b, 'counterfactual'))} → "
                     f"{fmt(agreement(a, 'counterfactual'))} | {fmt(agreement(b, 'reference'))} → "
                     f"{fmt(agreement(a, 'reference'))} | {fmt(b['counterfactual']['p_probe'])} → "
                     f"{fmt(a['counterfactual']['p_probe'])} | {kept} | {block} | {kl['counterfactual']} | "
                     f"{kl['reference']} | {trip} |")
    lines += ["", "\"Most lost with one block off\" is the largest share of the change lost when one block of "
              "ten layers has its adapter switched off; well below 1 means no single block holds the change.", ""]
    return lines


def section_caveats(runs: dict, dp: dict, insp: dict) -> list[str]:
    out = []
    per_cell = {}
    for r in runs.values():
        per_cell[(r["rendering"], r["world"])] = per_cell.get((r["rendering"], r["world"]), 0) + 1
    thin = [f"{a}/{SHORT[b]} ({n})" for (a, b), n in per_cell.items() if n < 3]
    if thin:
        out.append(f"Fewer than three seeds in {', '.join(thin)}: run-to-run spread is not estimated well.")
    if dp.get("delta_prior") is None:
        out.append("Δ_prior is missing, so nothing here separates prior conflict from the counterfactual "
                   "world's own difficulty.")
    elif min(dp["runs_per_cell"].values()) < 2:
        out.append("Δ_prior's interval is missing or partial: some cells have one run, whose spread is unknown.")
    elif not dp["positive"]:
        out.append("The Δ_prior interval includes 0: no interval-supported interference.")
    n = min((r["eval_n"] for r in runs.values()), default=0)
    if n and n < 16:
        out.append(f"Evaluations use {n} episodes per world per point; single curve points move by "
                   "tenths of S between neighbouring updates.")
    if any(r["speed"]["censored"] for r in runs.values()):
        out.append("Adaptation speed is censored for at least one run (S ≥ 0.9 on three consecutive "
                   "evaluations was not reached), so speed differences are lower bounds.")
    if any(r.get("epsilon", 0) == 0 for r in insp.values()):
        out.append("Inspections with ε = 0 replay pure planner histories, where the base model is already "
                   "near ceiling by copying the pattern in context; their agreement changes are small by "
                   "construction.")
    if any(r.get("return_trip") for r in insp.values()):
        out.append("Return-trip scores come from one seed per condition.")
    out.append("One model, one learning rate and a LoRA that skips the routed experts, throughout. Results describe this "
               "model's priors under this training, not people.")
    return ["## Caveats", ""] + [f"- {c}" for c in out] + [""]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--inspections", nargs="*", default=[], help="extra inspection directories")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.runs)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no finished runs under {root}")
    insp = load_inspections(root, args.inspections)
    prior, dp = section_prior(root)
    lines = ["# Inverted-Utility World: analysis report", ""]
    lines += section_setup(runs) + prior + section_cells(runs) + section_runs(runs)
    lines += section_matched(runs) + section_inspections(insp) + section_caveats(runs, dp, insp)
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}: {len(runs)} runs, {len(insp)} inspections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
