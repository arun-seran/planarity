"""Interface quiz and codebook-familiarisation data.

Before training, the frozen model answers a fixed quiz in each rendering:
what each label does, what it costs and when it is legal. It never asks
about the hidden law. Both renderings must score at least 95% and within 5
points of each other; otherwise revise the tutorial once, then run the
codebook-familiarised arm before claiming Delta_prior > 0.

Questions are multiple choice with lowercase option letters (single tokens).
Option order is a keyed shuffle per item that is identical across renderings,
so position bias is matched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..rng import keyed_uniform, namespace_key
from .rendering import Rendering, forbidden_hits
from .transcript import ChatFrame

LETTERS = ("a", "b", "c", "d")
PASS_SCORE = 0.95
MAX_GAP = 0.05


@dataclass(frozen=True)
class QuizItem:
    item_id: str
    topic: str  # "function" | "cost" | "legality" | "label" | "id" | "outcome"
    question: str
    options: tuple[str, ...]
    answer: int

    def render(self) -> str:
        opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(self.options))
        return f"Interface quiz. Answer with one letter only.\n{self.question}\n{opts}"


def _shuffle(item_id: str, correct: str, distractors: Sequence[str]) -> tuple[tuple[str, ...], int]:
    opts = [correct, *distractors]
    order = sorted(range(len(opts)), key=lambda i: keyed_uniform(namespace_key("quiz"),
                                                                namespace_key(item_id), i))
    shuffled = tuple(opts[i] for i in order)
    return shuffled, shuffled.index(correct)


def quiz_items(rendering: Rendering) -> list[QuizItem]:
    v = rendering.vocab
    f_probe = f"shows which {v.unit} {v.supplies} now"
    f_rest = f"{v.rest_long}; lowers {v.burden}"

    def f_act(n: int) -> str:
        return f"{v.actuate} {v.unit} {n}"

    specs: list[tuple[str, str, str, str, list[str]]] = []
    specs.append(("fn-0", "function", "What does action 0 do?", f_probe, [f_act(1), f_rest, f_act(4)]))
    for n in (1, 2, 3, 4):
        others = [f_act(m) for m in (1, 2, 3, 4) if m != n][:1]
        specs.append((f"fn-{n}", "function", f"What does action {n} do?", f_act(n),
                      [f_probe, f_rest, *others]))
    specs.append(("fn-5", "function", "What does action 5 do?", f_rest, [f_probe, f_act(2), f_act(5 - 2)]))
    specs.append(("cost-0", "cost", "How much energy does action 0 cost?", "1", ["0", "2", "5"]))
    specs.append(("cost-3", "cost", "How much energy does action 3 cost?", "1", ["0", "2", "3"]))
    specs.append(("cost-5", "cost", "How much energy does action 5 cost?", "0", ["1", "2", "5"]))
    once = "once per cycle, before committing"
    afford = "whenever you have at least 1 energy left"
    always = "always"
    after = f"only right after a {v.probe}"
    specs.append(("legal-0", "legality", "When is action 0 legal?", once, [afford, always, after]))
    specs.append(("legal-2", "legality", "When is action 2 legal?", afford, [once, always, after]))
    specs.append(("legal-5", "legality", "When is action 5 legal?", always, [once, afford, after]))
    specs.append(("id-probe", "id", f"Which action number makes you {v.probe}?", "0", ["5", "1", "3"]))
    specs.append(("id-rest", "id", f"Which action number makes you {v.rest_long}?", "5", ["0", "4", "2"]))
    specs.append(("id-act2", "id", f"Which action number makes you {v.actuate} {v.unit} 2?", "2",
                  ["0", "5", "3"]))
    specs.append(("label-water", "label",
                  f"In a cycle line, what does the number after '{v.water}' report?",
                  f"{v.water} delivered so far", ["energy left", v.burden, "the cycle number"]))
    specs.append(("label-burden", "label",
                  f"In a cycle line, what does the number after '{v.burden}' report?",
                  f"your {v.burden}", [f"{v.water} delivered so far", "energy left", "the cycle number"]))
    specs.append(("label-energy", "label",
                  "In a cycle line, what does the number after 'energy' report?",
                  "energy left", [f"{v.water} delivered so far", v.burden, "the cycle number"]))
    specs.append(("label-bar", "label", "What does a line show after the bar?",
                  "the legal actions", ["energy left", f"{v.water} delivered so far", "the cycle number"]))
    specs.append(("outcome-probe", "outcome", f"A line reads '{v.probe}: {v.unit} 3 ...'. What does it tell you?",
                  f"{v.unit} 3 {v.supplies} now", [f"you will {v.actuate} {v.unit} 3",
                                                    f"{v.unit} 3 delivered {v.water}",
                                                    f"{v.unit} 3 lowers {v.burden}"]))
    specs.append(("outcome-fail", "outcome", "What does 'fail' after your last command mean?",
                  "your body did not carry out the command",
                  [f"the {v.unit} did not exist", "you ran out of energy", "the episode ended"]))
    specs.append(("outcome-limit", "outcome", f"What happens when {v.burden} goes above 10?",
                  "the episode ends", [f"{v.water} doubles", "energy refills", "nothing"]))
    specs.append(("score-energy", "outcome", "What does each energy spent add to the score?",
                  "-0.05", ["+1", "0", "-1"]))
    specs.append(("score-water", "outcome", f"What does each {v.amount} delivered add to the score?",
                  "+1", ["-0.05", "0", "+5"]))
    items = []
    for item_id, topic, q, correct, distract in specs:
        opts, ans = _shuffle(item_id, correct, distract)
        items.append(QuizItem(item_id, topic, q, opts, ans))
    return items


def check_quiz_items(items: Sequence[QuizItem]) -> list[str]:
    """The quiz must never ask about the hidden law."""
    problems = []
    for it in items:
        hits = forbidden_hits(it.question + " " + " ".join(it.options))
        if hits:
            problems.append(f"{it.item_id}: {hits}")
        if len(set(it.options)) != len(it.options):
            problems.append(f"{it.item_id}: duplicate options")
    return problems


# A letter scorer maps (prompt tokens, candidate token ids) -> log-probabilities.
LetterScorer = Callable[[list[int], list[int]], np.ndarray]


@dataclass
class QuizResult:
    rendering: str
    accuracy: float
    by_topic: dict[str, float]
    answers: list[int]
    correct: list[bool]


def run_quiz(rendering: Rendering, tokenizer, score: LetterScorer,
             frame: Optional[ChatFrame] = None) -> QuizResult:
    frame = frame or ChatFrame()
    letter_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter)
        if len(ids) != 1:
            raise ValueError(f"option letter {letter!r} is not one token")
        letter_ids.append(ids[0])
    items = quiz_items(rendering)
    answers, correct = [], []
    by_topic: dict[str, list[bool]] = {}
    system = rendering.system_prompt()
    for it in items:
        prompt = tokenizer.encode(frame.question(system, it.render()))
        lp = np.asarray(score(prompt, letter_ids[:len(it.options)]))
        a = int(np.argmax(lp))
        answers.append(a)
        correct.append(a == it.answer)
        by_topic.setdefault(it.topic, []).append(a == it.answer)
    return QuizResult(rendering.name, float(np.mean(correct)), {k: float(np.mean(v)) for k, v in by_topic.items()},
                      answers, correct)


def quiz_gate(named: QuizResult, masked: QuizResult) -> dict:
    gap = abs(named.accuracy - masked.accuracy)
    passes = named.accuracy >= PASS_SCORE and masked.accuracy >= PASS_SCORE and gap <= MAX_GAP
    return {"named": named.accuracy, "masked": masked.accuracy, "gap": gap, "passes": passes,
            "action": None if passes else
            ("revise tutorial once, then schedule the codebook-familiarised arm if the gap "
             "still exceeds 5 points")}


def codebook_pairs(rendering: Rendering) -> list[tuple[str, str]]:
    """Label-function pairs for codebook familiarisation: no world transitions at all."""
    v = rendering.vocab
    pairs = [
        (f"Action 0, {v.probe}, means:", f" shows which {v.unit} {v.supplies} now; costs 1 energy; once per cycle."),
        (f"Action 5, {v.rest_long}, means:", f" costs 0 energy, commits, lowers {v.burden}; always legal."),
        (f"'{v.water}' in a cycle line reports:", f" {v.water} delivered so far."),
        (f"'{v.burden}' in a cycle line reports:", f" your {v.burden}; above 10 ends the episode."),
        (f"'{_cap(v.noise)}' reports:", f" {v.noise}; above your body's limit it raises {v.burden}."),
        (f"'{v.probe}: {v.unit} N' means:", f" {v.unit} N {v.supplies} now."),
        (f"'last {v.actuate} N ok' means:", " your last command was action N and your body carried it out."),
        (f"'last {v.actuate} N fail' means:", " your last command was action N and your body did not carry it out."),
    ]
    for n in (1, 2, 3, 4):
        pairs.append((f"Action {n}, {v.actuate} {v.unit} {n}, means:",
                      f" costs 1 energy and commits; delivers one {v.amount} only if {v.unit} {n} {v.supplies}."))
    return pairs


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]
