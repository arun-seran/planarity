import pytest

from iuw.config import (COUNTERFACTUAL, NOISY_COUNTERFACTUAL, NOISY_REFERENCE, PROFILE_A,
                        PROFILE_B, PROFILES, REFERENCE, ContextBudget)
from iuw.llm.tokenizer import ApproxQwenTokenizer, action_token_ids
from iuw.policies.baselines import AlwaysProbePolicy, RandomPolicy
from iuw.policies.planner import KnownLawPlanner
from iuw.render import (NAMED, ROLES, PromptOptions, forbidden_hits, format_legal, make_rendering,
                        masked_codes, parse_legal)
from iuw.render.checks import (budget_report, information_equality, parity, transcript_for_episode,
                               tutorial_checks)
from iuw.render.quiz import check_quiz_items, codebook_pairs, quiz_items

TOK = ApproxQwenTokenizer()
RENDERINGS = [make_rendering("named"), *(make_rendering("masked", m) for m in range(3))]
NAMED_WORDS = ("valve", "inspect", "water", "settlement", "break", "noise", "canal", "open",
               "burden", "liter", "flow")


def test_action_ids_are_single_tokens():
    ids = action_token_ids(TOK)
    assert len(set(ids)) == 6


@pytest.mark.parametrize("legal", [(0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (5,), (0, 5), ()])
def test_legal_roundtrip(legal):
    assert parse_legal(format_legal(legal)) == legal


def test_masked_mappings_are_counterbalanced_and_fixed():
    maps = [masked_codes(m) for m in range(3)]
    assert maps[0] == masked_codes(0)  # fixed within a run
    for role in ROLES:
        assert len({m[role] for m in maps}) == 3  # each role gets a different code per seed
    for m in maps:
        assert len(set(m.values())) == len(ROLES)


def test_masked_rendering_carries_no_named_words():
    for r in RENDERINGS[1:]:
        text = r.tutorial().lower() + " " + r.header("A", 0).lower()
        for w in NAMED_WORDS:
            assert w not in text, (r.name, w)


def test_tutorials_and_quiz_never_hint_at_the_hidden_law():
    report = tutorial_checks(RENDERINGS, TOK)
    assert not any(report["forbidden"].values())
    assert not any(report["quiz_problems"].values())
    for r in RENDERINGS:
        assert forbidden_hits(r.rule_statement(1.0))  # the explicit-rules arm does state it
        assert not check_quiz_items(quiz_items(r))
        for prompt, completion in codebook_pairs(r):
            assert not forbidden_hits(prompt + completion)


def test_tutorial_token_parity():
    report = tutorial_checks(RENDERINGS, TOK)
    assert report["max_parity_gap"] <= 0.10
    assert max(report["tutorial_tokens"].values()) <= ContextBudget().system_and_tutorial


def _episodes(world_list, seeds):
    for w in world_list:
        for p in (PROFILE_A, PROFILE_B):
            for s in seeds:
                yield w, p, s, KnownLawPlanner(w, PROFILES)
                yield w, p, s, RandomPolicy()
                yield w, p, s, AlwaysProbePolicy()


def test_context_budget_is_never_exceeded():
    budget = ContextBudget()
    notebook = " ".join(["note"] * 1000)  # a full-size notebook
    assert len(TOK.encode(notebook)) <= budget.notebook
    worlds = (REFERENCE, COUNTERFACTUAL, NOISY_REFERENCE, NOISY_COUNTERFACTUAL)
    for r in (RENDERINGS[0], RENDERINGS[1]):
        rep = budget_report(r, TOK, _episodes(worlds, range(3)), notebook=notebook)
        assert rep.within(budget), rep
        assert rep.max_total <= budget.worst_case <= budget.total


def test_whole_transcripts_within_ten_percent():
    for w in (REFERENCE, COUNTERFACTUAL):
        n = transcript_for_episode(w, PROFILE_A, 5, KnownLawPlanner(w, PROFILES), RENDERINGS[0], TOK)
        for r in RENDERINGS[1:]:
            m = transcript_for_episode(w, PROFILE_A, 5, KnownLawPlanner(w, PROFILES), r, TOK)
            assert parity(len(n.tokens), len(m.tokens)) <= 0.10


def test_information_equality_across_renderings():
    for w in (REFERENCE, COUNTERFACTUAL, NOISY_REFERENCE):
        for p in (PROFILE_A, PROFILE_B):
            out = information_equality(w, p, range(4), lambda: KnownLawPlanner(w, PROFILES), RENDERINGS)
            assert out["equal"], out["differences"]


def test_transcript_is_append_only_and_masks_actions():
    tr = transcript_for_episode(COUNTERFACTUAL, PROFILE_B, 1, KnownLawPlanner(COUNTERFACTUAL, PROFILES),
                                RENDERINGS[0], TOK, PromptOptions())
    flat = [t for seg in tr.segments for t in seg.tokens]
    assert flat == tr.tokens
    ids = action_token_ids(TOK)
    assert all(tr.tokens[p] in ids for p in tr.action_positions)
    assert tr.n_decisions == 101  # probe once, then 100 commits
