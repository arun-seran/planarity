"""Append-only token transcripts.

An episode is one growing token sequence: a chat-framed prompt (system
prompt with the tutorial, then a user header), then alternating observation
lines and single action tokens inside the assistant turn. Each segment is
tokenised on its own and appended, never re-tokenised, so the tokens trained
on are exactly the tokens the sampler saw, and loss masks are exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import ContextBudget
from ..env.water import Observation
from ..llm.tokenizer import Tokenizer, action_token_ids
from .rendering import PromptOptions, Rendering


class ChatFrame:
    """Qwen3 chat framing with thinking off (empty think block), or plain text."""

    def __init__(self, style: str = "qwen3-nothink"):
        if style not in ("qwen3-nothink", "plain"):
            raise ValueError(style)
        self.style = style

    def prefix(self, system: str, user: str) -> str:
        if self.style == "plain":
            return f"{system}\n\n{user}\n\n"
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def question(self, system: str, user: str) -> str:
        """Prompt for a one-token answer (quiz, belief probes)."""
        return self.prefix(system, user)

    def generation(self, system: str, user: str) -> str:
        return self.prefix(system, user)


@dataclass
class Segment:
    kind: str  # "prompt" | "obs" | "action" | "suffix"
    text: str
    tokens: list[int]
    decision: Optional[int] = None


class ContextOverflow(RuntimeError):
    pass


@dataclass
class Transcript:
    tokenizer: Tokenizer
    rendering: Rendering
    frame: ChatFrame = field(default_factory=ChatFrame)
    options: PromptOptions = field(default_factory=PromptOptions)
    budget: ContextBudget = field(default_factory=ContextBudget)

    def __post_init__(self):
        self.segments: list[Segment] = []
        self.tokens: list[int] = []
        self.action_positions: list[int] = []  # index in self.tokens of each action token
        self.action_ids = action_token_ids(self.tokenizer)
        self._token_to_action = {t: a for a, t in enumerate(self.action_ids)}
        self.prompt_len = 0
        self.notebook_len = 0
        self.lines: list[str] = []

    # ----- building --------------------------------------------------------------------------
    def _append(self, kind: str, text: str, decision: Optional[int] = None) -> Segment:
        toks = self.tokenizer.encode(text)
        if len(self.tokens) + len(toks) > self.budget.total:
            raise ContextOverflow(f"transcript would exceed {self.budget.total} tokens")
        seg = Segment(kind, text, toks, decision)
        self.segments.append(seg)
        self.tokens.extend(toks)
        return seg

    def start(self, body: str, noise: float) -> None:
        system = self.rendering.system_prompt(self.options)
        user = self.rendering.header(body, noise)
        self._append("prompt", self.frame.prefix(system, user))
        self.prompt_len = len(self.tokens)
        if self.options.notebook:
            self.notebook_len = len(self.tokenizer.encode(self.options.notebook))

    def observe(self, obs: Observation) -> str:
        line = self.rendering.line(obs)
        text = line if not self.lines else "\n" + line
        self.lines.append(line)
        self._append("obs", text, decision=len(self.action_positions))
        return line

    def act(self, action: int) -> None:
        tok = self.action_ids[action]
        if len(self.tokens) + 1 > self.budget.total:
            raise ContextOverflow("no room for the action token")
        self.action_positions.append(len(self.tokens))
        self.segments.append(Segment("action", str(action), [tok], len(self.action_positions) - 1))
        self.tokens.append(tok)

    # ----- accounting ------------------------------------------------------------------------
    @property
    def n_decisions(self) -> int:
        return len(self.action_positions)

    def per_decision_tokens(self) -> list[int]:
        """Observation segment plus action token, for each decision."""
        out, cur = [], 0
        for seg in self.segments[1:]:
            cur += len(seg.tokens)
            if seg.kind == "action":
                out.append(cur)
                cur = 0
        return out

    def prompt_tokens_without_notebook(self) -> int:
        return self.prompt_len - self.notebook_len

    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    def action_token_to_action(self, token: int) -> Optional[int]:
        return self._token_to_action.get(token)
