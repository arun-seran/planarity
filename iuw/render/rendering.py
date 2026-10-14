"""Two renderings of one world: Named (everyday words) and Masked (opaque labels).

Physics and action token IDs are identical; only the words differ. Both
renderings are produced from one template, so they carry the same
information and their tutorials have matched length. Named words describe
the task, never the body; the body is labelled A or B in both.

Masked labels are short letter-only pseudo-words (the spec's "op-0, unit-3,
channel k7" are examples). Letters only, so no label can be confused with an
action digit or a value. The mapping is drawn once per training run from a
fixed pool and counterbalanced across the three seeds of a cell: mapping m
assigns role i the code pool[(i + 2m) mod 6], so each code serves a
different role in each seed. Valve numbers equal action IDs in both
renderings, so neither pays a decoding cost for the valve itself.

The tutorial states each action's cost, when it is legal and what it
returns. It never states or hints at the hidden law (target persistence) or
at the body's execution law; :data:`FORBIDDEN_PATTERNS` is the check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..config import PROBE, REST, VALVES, WorldConfig
from ..env.water import Observation, Phase
from ..rng import Stream, keyed_uniform, namespace_key

ROLES = ("probe", "actuate", "rest", "water", "burden", "noise")
MASKED_POOL = ("lom", "kel", "dax", "tib", "pim", "tuv")  # each one token under the Qwen3.6 tokenizer

TUTORIAL_TEMPLATE = """\
You operate one body at a {site} with four {unit}s, numbered 1 to 4. Each line below is one decision. Reply with one action number only.

Each cycle has two steps. First you may {probe}, at most once. Then you must commit: {actuate} one {unit}, or {rest_long}. Committing ends the cycle.

Actions
0 = {probe}: shows which {unit} {supplies} now. Costs 1 energy. Legal once per cycle, before committing.
1, 2, 3, 4 = {actuate} {unit} 1, 2, 3 or 4: costs 1 energy and commits. It delivers one {amount} only if that {unit} {supplies} and your body carries out the command.
5 = {rest_long}: costs 0 energy and commits. Lowers {burden}. Always legal.
An action that costs more energy than you have left is not legal.

Lines
A cycle line shows: cycle number; {water} delivered so far; energy left; {burden}; your last {unit} command ({actuate} and number), then ok if your body carried it out or fail if not; after the bar, the legal actions. Then you reply.
Each {probe} line shows: the {unit} that {supplies} now; energy left; after the bar, the legal actions. Then you reply.

Score: +1 for each {amount} delivered, -0.05 for each energy spent. {Noise} above your body's limit raises {burden}. {Burden} above 10 ends the episode."""

HEADER_TEMPLATE = "Body {body}. {Noise} {noise_value}. Episode begins."

# Statements that would leak target persistence or the body's execution law.
FORBIDDEN_PATTERNS = [
    r"\bsame\b", r"\bstays?\b", r"\bpersist", r"\bkeeps?\b", r"\bchang", r"\bmoves?\b",
    r"\brandom", r"\bredraw", r"\bdrawn\b", r"\bfixed\b", r"\bstable\b", r"\bsteady\b",
    r"\bvolatile\b", r"\bshift", r"\brotat", r"\bagain\b", r"\brepeat", r"\bconstant\b",
    r"\bnever\b", r"\bprecision\b", r"\breliab", r"\bprobability\b", r"\bpercent\b", r"%",
    r"\bswitch", r"\bhabit", r"\broutine\b",
]


@dataclass(frozen=True)
class Vocab:
    kind: str  # "named" | "masked"
    site: str
    unit: str
    probe: str
    actuate: str
    rest_long: str
    supplies: str
    amount: str
    water: str
    burden: str
    noise: str
    mapping_index: Optional[int] = None

    def slots(self) -> dict[str, str]:
        d = {k: getattr(self, k) for k in ("site", "unit", "probe", "actuate", "rest_long",
                                           "supplies", "amount", "water", "burden", "noise")}
        d["Noise"] = _cap(self.noise)
        d["Burden"] = _cap(self.burden)
        return d


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


NAMED = Vocab(kind="named", site="canal", unit="valve", probe="inspect", actuate="open",
              rest_long="take a break", supplies="supplies the settlement",
              amount="liter of water", water="water", burden="burden", noise="noise level")


def masked_codes(mapping_index: int, pool: tuple[str, ...] = MASKED_POOL) -> dict[str, str]:
    """Counterbalanced role -> code assignment for a run."""
    n = len(ROLES)
    if len(pool) < n:
        raise ValueError("pool too small")
    if mapping_index < 3:
        return {role: pool[(i + 2 * mapping_index) % n] for i, role in enumerate(ROLES)}
    # beyond the three counterbalanced mappings: a keyed permutation of the pool
    order = sorted(pool, key=lambda c: keyed_uniform(namespace_key("masked"), int(Stream.MAPPING),
                                                     mapping_index, namespace_key(c)))
    return dict(zip(ROLES, order))


def masked_vocab(mapping_index: int) -> Vocab:
    c = masked_codes(mapping_index)
    return Vocab(kind="masked", site="system", unit="unit", probe=c["probe"], actuate=c["actuate"],
                 rest_long=c["rest"], supplies="is active", amount=c["water"], water=c["water"],
                 burden=c["burden"], noise=c["noise"], mapping_index=mapping_index)


def renamed_named(unit: str = "gate", site: str = "channel") -> Vocab:
    """Renamed-entities control: everyday words, different entity names."""
    return Vocab(kind="named-renamed", site=site, unit=unit, probe="check", actuate="lift",
                 rest_long="take a pause", supplies="feeds the village", amount="liter of water",
                 water="water", burden="burden", noise="noise level")


def format_legal(legal: tuple[int, ...]) -> str:
    """Compact canonical form: contiguous runs as a-b, separated by commas."""
    if not legal:
        return "-"
    runs, start, prev = [], legal[0], legal[0]
    for a in legal[1:]:
        if a == prev + 1:
            prev = a
            continue
        runs.append((start, prev))
        start = prev = a
    runs.append((start, prev))
    return ",".join(f"{s}" if s == e else f"{s}-{e}" for s, e in runs)


def parse_legal(text: str) -> tuple[int, ...]:
    if text == "-":
        return ()
    out = []
    for part in text.split(","):
        if "-" in part:
            s, e = part.split("-")
            out.extend(range(int(s), int(e) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def format_burden(b: float) -> str:
    s = f"{b:.1f}"
    return s[:-2] if s.endswith(".0") else s


def format_number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


@dataclass(frozen=True)
class PromptOptions:
    """Additions to the system prompt; all default to off."""

    explicit_rules: Optional[float] = None  # state the hidden law for this q
    explain_body: Optional[tuple[str, float, float]] = None  # (body, repeat, switch)
    world_cue: Optional[str] = None  # explicit world cue (return-trip test)
    notebook: Optional[str] = None  # notebook arm (at most 1,024 tokens)


class Rendering:
    """Turns allowlisted observations into text, and back (for the information check)."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.name = vocab.kind if vocab.mapping_index is None else f"{vocab.kind}-m{vocab.mapping_index}"
        v = vocab
        self._cycle_re = re.compile(
            rf"^(\d+) {re.escape(v.water)} (\d+) energy (\d+) {re.escape(v.burden)} ([\d.]+) "
            rf"last (?:{re.escape(v.actuate)} (\d) (ok|fail)|none) \| ([\d,\-]+)>$")
        self._probe_re = re.compile(
            rf"^{re.escape(v.probe)}: {re.escape(v.unit)} (\d) energy (\d+) \| ([\d,\-]+)>$")

    @property
    def kind(self) -> str:
        return self.vocab.kind

    # ----- prompt ----------------------------------------------------------------------------
    def tutorial(self) -> str:
        return TUTORIAL_TEMPLATE.format(**self.vocab.slots())

    def rule_statement(self, q: float) -> str:
        v = self.vocab
        if q >= 1.0:
            return f"Hidden rule: the {v.unit} that {v.supplies} never changes within an episode."
        if q <= 0.0:
            return (f"Hidden rule: after every commit, the {v.unit} that {v.supplies} is drawn "
                    f"again at random, each {v.unit} equally likely.")
        return (f"Hidden rule: after every commit, the {v.unit} that {v.supplies} stays the same "
                f"with probability {q:g}; otherwise it is drawn again at random, each {v.unit} "
                f"equally likely.")

    @staticmethod
    def body_statement(body: str, repeat: float, switch: float) -> str:
        return (f"Body law: body {body} carries out a repeated command {repeat:.0%} of the time, "
                f"and a changed or first command {switch:.0%} of the time.")

    def system_prompt(self, options: PromptOptions = PromptOptions()) -> str:
        parts = [self.tutorial()]
        if options.explicit_rules is not None:
            parts.append(self.rule_statement(options.explicit_rules))
        if options.explain_body is not None:
            parts.append(self.body_statement(*options.explain_body))
        if options.world_cue:
            parts.append(options.world_cue)
        if options.notebook:
            parts.append("Notebook (your own notes from earlier episodes):\n" + options.notebook)
        return "\n\n".join(parts)

    def header(self, body: str, noise: float) -> str:
        return HEADER_TEMPLATE.format(body=body, noise_value=format_number(noise), **self.vocab.slots())

    # ----- lines -----------------------------------------------------------------------------
    def line(self, obs: Observation) -> str:
        v = self.vocab
        legal = format_legal(obs.legal)
        if obs.phase is Phase.COMMIT:
            return f"{v.probe}: {v.unit} {obs.probe_result} energy {obs.energy} | {legal}>"
        if obs.last_command is None:
            last = "none"
        else:
            last = f"{v.actuate} {obs.last_command} {'ok' if obs.last_success else 'fail'}"
        return (f"{obs.cycle} {v.water} {obs.liters} energy {obs.energy} {v.burden} "
                f"{format_burden(obs.burden)} last {last} | {legal}>")

    def parse_line(self, text: str, prev: Optional[Observation], body: str,
                   noise: float) -> Observation:
        """Rebuild the observation from one rendered line plus the previous observation."""
        m = self._cycle_re.match(text)
        if m:
            cycle, liters, energy, burden, last, ok, legal = m.groups()
            return Observation(
                cycle=int(cycle), phase=Phase.PROBE, probe_result=None, liters=int(liters),
                energy=int(energy), burden=float(burden), body=body, noise=noise,
                last_command=int(last) if last else None,
                last_success=None if ok is None else ok == "ok", legal=parse_legal(legal))
        m = self._probe_re.match(text)
        if m and prev is not None:
            valve, energy, legal = m.groups()
            return Observation(
                cycle=prev.cycle, phase=Phase.COMMIT, probe_result=int(valve), liters=prev.liters,
                energy=int(energy), burden=prev.burden, body=body, noise=noise,
                last_command=prev.last_command, last_success=prev.last_success,
                legal=parse_legal(legal))
        raise ValueError(f"unparseable line for {self.name}: {text!r}")


def forbidden_hits(text: str) -> list[str]:
    return [p for p in FORBIDDEN_PATTERNS if re.search(p, text, flags=re.IGNORECASE)]


def make_rendering(kind: str, mapping_index: int = 0) -> Rendering:
    if kind == "named":
        return Rendering(NAMED)
    if kind == "masked":
        return Rendering(masked_vocab(mapping_index))
    if kind == "named-renamed":
        return Rendering(renamed_named())
    raise KeyError(f"unknown rendering {kind!r}")
