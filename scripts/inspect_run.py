"""Stage-8 inspection of one trained adapter, sized for a single short GPU session.

Everything except the return trip replays recorded histories (one forward pass per history,
no sampling), so the interventions are cheap. Writes inspection.json and inspection.md.

Histories come from the known-law planner with a fraction --epsilon of its decisions replaced
by keyed random legal actions. Pure planner histories are near ceiling for the base model
(it copies the pattern in context), so the off-policy steps keep the measure sensitive.
"planner agreement" is the mean log-probability of the action the planner would have taken
at each recorded decision, whichever action the history then took.

  python scripts/inspect_run.py --adapter runs/hf-checkpoints/<run>-u010 --rendering named \
      --out runs/inspect-named
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iuw.certify import load_references
from iuw.config import COUNTERFACTUAL, PROBE, PROFILES, REFERENCE, REST
from iuw.env.water import Phase
from iuw.inspection import adapters, ledger, replay, return_trip
from iuw.llm.hf_backend import HFBackend
from iuw.policies.planner import KnownLawPlanner
from iuw.render.rendering import make_rendering
from iuw.rng import Event, Stream, draw, draw_index, namespace_key
from iuw.scoring import Scorer
from iuw.seeds import SPLITS

EPSILON_SALT = namespace_key("iuw/inspect-epsilon") & 0xFFFFFFFF


class EpsilonPlanner:
    """The known-law planner, with a keyed fraction of decisions replaced by random legal actions.

    The planner sees every observation and is told which commit was actually sent, so its
    information state stays exact; ``intended`` records what it would have done.
    """

    name = "epsilon-planner"
    privileged = False

    def __init__(self, world, epsilon: float):
        self.planner, self.epsilon = KnownLawPlanner(world, PROFILES), epsilon

    def begin(self, ctx) -> None:
        self.planner.begin(ctx)
        self.ctx, self.decision, self.intended = ctx, 0, []

    def act(self, obs) -> int:
        want = self.planner.act(obs)
        a, i = want, self.decision
        self.decision += 1
        self.intended.append(want)
        if self.epsilon and draw(Stream.POLICY, self.ctx.seed, self.ctx.episode, i, Event.MISC,
                                 extra=EPSILON_SALT) < self.epsilon:
            a = obs.legal[draw_index(len(obs.legal), Stream.POLICY, self.ctx.seed, self.ctx.episode, i,
                                     Event.ACTION, extra=EPSILON_SALT)]
        if a != want:  # keep the planner's record of the pending commit true to what was sent
            self.planner.pending = None if a == PROBE else a
            if a != PROBE and obs.phase is Phase.PROBE:
                self.planner.prev_liters = obs.liters
        return a


def histories(n_seeds: int, epsilon: float) -> tuple[dict[str, list], dict[str, list]]:
    """Histories on held-out discovery seeds per world, with the planner's intended actions."""
    seeds = list(SPLITS["discovery"])[:n_seeds]
    hists, labels = {}, {}
    for w in (COUNTERFACTUAL, REFERENCE):
        hists[w.name], labels[w.name] = [], []
        for s in seeds:
            for b in ("A", "B"):
                pol = EpsilonPlanner(w, epsilon)
                hists[w.name].append(replay.record_history(w, PROFILES[b], s, pol))
                labels[w.name].append(pol.intended[:len(hists[w.name][-1].actions)])
    return hists, labels


def measure(scorer_fn, tok, rendering, hists, labels) -> dict:
    """One replay per history: planner agreement, recorded-action likelihood and policy, per world."""
    out = {}
    for world, hs in hists.items():
        sc = scorer_fn()
        agree, loglik, probe, repeat, rest = [], [], [], [], []
        for h, want in zip(hs, labels[world]):
            probs = replay.replay_distributions(h, sc, tok, rendering)
            lp = np.log(np.maximum(probs, 1e-12))
            agree.append(float(np.mean([lp[i, a] for i, a in enumerate(want)])))
            loglik.append(float(np.mean([lp[i, a] for i, a in enumerate(h.actions)])))
            for obs, p in zip(h.observations, probs):
                if obs.phase is Phase.PROBE:
                    probe.append(p[PROBE])
                    rest.append(p[REST])
                    if obs.last_command is not None:
                        repeat.append(p[obs.last_command])
        out[world] = {"planner_agreement": float(np.mean(agree)),
                      "planner_action_loglik": float(np.mean(loglik)),
                      "p_probe": float(np.mean(probe)) if probe else None,
                      "p_repeat_last": float(np.mean(repeat)) if repeat else None,
                      "p_rest": float(np.mean(rest)) if rest else None}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--rendering", default="named", choices=["named", "masked"])
    p.add_argument("--mapping", type=int, default=0)
    p.add_argument("--seeds", type=int, default=4, help="held-out seeds per world (x2 bodies)")
    p.add_argument("--epsilon", type=float, default=0.25, help="share of random decisions in the histories")
    p.add_argument("--return-trip-seeds", type=int, default=1, help="0 skips the return trip (it samples)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    be = HFBackend(args.model, init_adapter=args.adapter)
    tok, rendering = be.tokenizer, make_rendering(args.rendering, args.mapping)
    hists, labels = histories(args.seeds, args.epsilon)
    report: dict = {"adapter": args.adapter, "rendering": rendering.name, "epsilon": args.epsilon,
                    "n_histories": {k: len(v) for k, v in hists.items()}}

    spec = adapters.adapter_spectrum(be.model)
    layers = sorted({layer for _, layer, _ in adapters.lora_modules(be.model)})
    report["spectrum"] = {"modules": len(spec),
                          "top_singular_value_mean": float(np.mean([s[0] for s in spec.values()])),
                          "share_of_top_direction": float(np.mean([s[0] / max(sum(s), 1e-12)
                                                                   for s in spec.values()]))}

    report["adapted"] = measure(be.scorer, tok, rendering, hists, labels)
    report["base"] = measure(be.reference_scorer, tok, rendering, hists, labels)
    report["transition_belief"] = {
        who: {w.name: ledger.transition_belief(hists[w.name][0], fn(), tok, rendering, w,
                                              every=1 if w is COUNTERFACTUAL else 5)  # probes are rare under q = 1
              for w in (COUNTERFACTUAL, REFERENCE)}
        for who, fn in (("adapted", be.scorer), ("base", be.reference_scorer))}

    # Where does the change live? Switch the adapter off block by block.
    report["ablate_blocks"] = {}
    for block in adapters.layer_blocks(max(layers) + 1, 10):
        with adapters.ablate_layers(be.model, block):
            report["ablate_blocks"][f"{block[0]}-{block[-1]}"] = measure(be.scorer, tok, rendering, hists, labels)

    # How few directions carry it? Rank-k truncation against a matched random control.
    report["top_directions"] = {}
    for k in (1, 4):
        for ctrl in (False, True):
            with adapters.keep_top_directions(be.model, k=k, random_control=ctrl):
                report["top_directions"][f"k={k}{' random' if ctrl else ''}"] = \
                    measure(be.scorer, tok, rendering, hists, labels)

    if args.return_trip_seeds:
        normaliser = Scorer(load_references())
        scores = return_trip.return_trip(be.scorer(), be.reference_scorer(), tok, rendering, normaliser,
                                         seeds=list(SPLITS["dev"])[:args.return_trip_seeds])
        report["return_trip"] = {"scores": scores, "diagnosis": return_trip.diagnose(scores)}

    report["seconds"] = time.time() - t0
    (out_dir / "inspection.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "inspection.md").write_text(summary(report))
    print(summary(report))
    return 0


def summary(r: dict) -> str:
    cf = "counterfactual"
    rows = [("base (adapter off)", r["base"]), ("adapted", r["adapted"])]
    rows += [(f"ablate layers {k}", v) for k, v in r["ablate_blocks"].items()]
    rows += [(f"keep {k}", v) for k, v in r["top_directions"].items()]
    lines = [f"# Inspection: {r['adapter']} ({r['rendering']})", "",
             f"Matched held-out histories: {r['n_histories']}, epsilon {r.get('epsilon', 0)}. Agreement is the "
             "mean log-probability of the known-law planner's action; closer to 0 agrees more.", "",
             "| weights | cf agreement | cf p_probe | cf p_repeat | ref agreement | ref p_probe |",
             "|---|---|---|---|---|---|"]
    for name, m in rows:
        c, f = m[cf], m["reference"]
        key = "planner_agreement" if "planner_agreement" in c else "planner_action_loglik"
        lines.append(f"| {name} | {c[key]:.3f} | {c['p_probe']:.3f} | "
                     f"{c['p_repeat_last']:.3f} | {f[key]:.3f} | {f['p_probe']:.3f} |")
    lines += ["", f"Adapter spectrum: {r['spectrum']}", "", "Transition belief (one history per world): "
              + json.dumps(r["transition_belief"])]
    if "return_trip" in r:
        lines += ["", "Return trip (reference-world S): " + json.dumps(r["return_trip"]["scores"]),
                  "Diagnosis: " + json.dumps(r["return_trip"]["diagnosis"], default=str)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
