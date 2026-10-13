"""The change ledger: one row per prior, each measured on identical held-out histories.

Rows read: prior, adapted behaviour, measured effect, intervention result,
uncertainty, candidate transfer. Entries:

* Transition beliefs: scored predictions of the next target, rewards fixed.
* Reward beliefs: predicted payoff for identical physical outcomes.
* Value: calibration of expected return for identical states.
* Policy: probe cadence, repeat runs, switching and rest.
* Signal use: cue use versus shuffled timing (expansion; not in Water-v1).
* Retained competence: reference-world score after the return trip.

The model's answers to belief questions are outputs to assess, not evidence
of an internal mechanism. Questions are asked inside the world framing (in
the episode transcript) and outside it (a bare quiz), with counterbalanced
option order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..config import WorldConfig, get_world
from ..llm.backend import ScoreRequest, legal_log_softmax
from ..llm.tokenizer import action_token_ids
from ..render.rendering import PromptOptions, Rendering
from ..render.transcript import ChatFrame
from .replay import History, build_transcript, replay_distributions


@dataclass
class LedgerRow:
    entry: str
    prior: str
    adapted_behaviour: str = ""
    measured_effect: Optional[float] = None
    intervention_result: str = ""
    uncertainty: Optional[list[float]] = None
    candidate_transfer: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class ChangeLedger:
    rows: list[LedgerRow] = field(default_factory=list)

    def add(self, row: LedgerRow) -> None:
        self.rows.append(row)

    def to_markdown(self) -> str:
        head = ("| Entry | Prior | Adapted behaviour | Measured effect | Intervention | Uncertainty | "
                "Candidate transfer |\n|---|---|---|---|---|---|---|")
        lines = [head]
        for r in self.rows:
            eff = "" if r.measured_effect is None else f"{r.measured_effect:+.3f}"
            unc = "" if r.uncertainty is None else f"[{r.uncertainty[0]:+.3f}, {r.uncertainty[1]:+.3f}]"
            lines.append(f"| {r.entry} | {r.prior} | {r.adapted_behaviour} | {eff} | "
                         f"{r.intervention_result} | {unc} | {r.candidate_transfer} |")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"rows": [asdict(r) for r in self.rows]}


def law_next_target(world: WorldConfig, last_seen: int, n: int = 4) -> np.ndarray:
    """Predictive distribution of the next target after one transition from a known target."""
    p = np.full(n, (1 - world.q) / n)
    p[last_seen - 1] += world.q
    return p


def transition_belief(history: History, scorer, tokenizer, rendering: Rendering, world: WorldConfig,
                      options: PromptOptions = PromptOptions(), every: int = 1) -> dict:
    """At each commit that followed a probe, ask which unit supplies next (in the world framing).

    Scores the model's distribution over valves 1-4 with the log score of the true next
    target, and its KL from the law's predictive distribution.
    """
    v = rendering.vocab
    tr = build_transcript(history, rendering, tokenizer, options)
    ids = action_token_ids(tokenizer)
    valve_ids = ids[1:5]
    logs, kls = [], []
    pos_by_decision = tr.action_positions
    for k, obs in enumerate(history.observations):
        if obs.probe_result is None or k % every:
            continue
        # transcript up to and including this commit's action token
        prefix = tr.tokens[:pos_by_decision[k] + 1]
        q = tokenizer.encode(f"\nnext {v.unit} that {v.supplies}? | 1-4>")
        res = scorer.score_batch([ScoreRequest(prefix + q, valve_ids)])[0]
        lp = legal_log_softmax(np.asarray(res.logprobs), [0, 1, 2, 3])
        p = np.exp(lp)
        law = law_next_target(world, obs.probe_result)
        nxt = next((o.probe_result for o in history.observations[k + 1:] if o.probe_result), None)
        if nxt is not None and world.q in (0.0, 1.0):
            logs.append(float(np.log(max(p[nxt - 1], 1e-12))))
        kls.append(float(np.sum(law * (np.log(law + 1e-12) - np.log(p + 1e-12)))))
    return {"mean_log_score": float(np.mean(logs)) if logs else None,
            "mean_kl_from_law": float(np.mean(kls)) if kls else None, "n": len(kls)}


def policy_on_histories(histories: Sequence[History], scorer, tokenizer, rendering: Rendering,
                        options: PromptOptions = PromptOptions()) -> dict:
    """Probe, repeat and rest probabilities at matched decision points."""
    probe, repeat, rest = [], [], []
    for h in histories:
        probs = replay_distributions(h, scorer, tokenizer, rendering, options)
        for obs, p in zip(h.observations, probs):
            if obs.phase.value == "probe":
                probe.append(p[0])
                rest.append(p[5])
                if obs.last_command is not None:
                    repeat.append(p[obs.last_command])
    return {"p_probe": float(np.mean(probe)) if probe else None,
            "p_repeat_last": float(np.mean(repeat)) if repeat else None,
            "p_rest": float(np.mean(rest)) if rest else None}


VALUE_BUCKETS = ("0 to 25", "25 to 50", "50 to 75", "75 to 100")


def value_belief(history: History, scorer, tokenizer, rendering: Rendering, at_decision: int,
                 options: PromptOptions = PromptOptions()) -> np.ndarray:
    """Ask, inside the transcript, how much water will be delivered by the end (buckets a-d)."""
    v = rendering.vocab
    tr = build_transcript(history, rendering, tokenizer, options)
    prefix = tr.tokens[:tr.action_positions[at_decision] + 1]
    opts = " ".join(f"{l}) {b}" for l, b in zip("abcd", VALUE_BUCKETS))
    q = tokenizer.encode(f"\ntotal {v.water} by the end? {opts} | a-d>")
    letters = [tokenizer.encode(l)[0] for l in "abcd"]
    res = scorer.score_batch([ScoreRequest(prefix + q, letters)])[0]
    lp = np.asarray(res.logprobs)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def reward_belief(scorer, tokenizer, rendering: Rendering) -> dict:
    """Outside the world framing: predicted payoff for identical physical outcomes."""
    v = rendering.vocab
    frame = ChatFrame()
    system = rendering.system_prompt()
    items = [
        (f"You {v.actuate} {v.unit} 2, your body carries it out, and one {v.amount} is delivered. "
         "What does that cycle add to the score?", ("+0.95", "+1", "-0.05", "0"), 0),
        (f"You {v.actuate} {v.unit} 2 and your body does not carry it out. What does that cycle add "
         "to the score?", ("-0.05", "0", "+1", "-1"), 0),
        (f"You {v.rest_long}. What does that cycle add to the score?", ("0", "-0.05", "+1", "-1"), 0),
    ]
    letters = [tokenizer.encode(l)[0] for l in "abcd"]
    correct = []
    for q, opts, ans in items:
        text = "Answer with one letter only.\n" + q + "\n" + "\n".join(
            f"{l}) {o}" for l, o in zip("abcd", opts))
        res = scorer.score_batch([ScoreRequest(tokenizer.encode(frame.question(system, text)), letters)])[0]
        correct.append(int(np.argmax(res.logprobs)) == ans)
    return {"accuracy": float(np.mean(correct)), "n": len(items)}
