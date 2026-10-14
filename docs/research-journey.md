# Inverted-Utility World (Water-v1): research journey, results and next steps

*Status as of 14 October 2026. Model: Qwen3.6-35B-A3B with LoRA. Six training runs, four adapter
inspections, about $24 of GPU time on Runpod. Nothing here is committed to git yet.*

---

## 0. Summary

- **Question.** Water-v1 has a hidden law, and the reversed world flips that law. When a language model
  learns to act in the reversed world, do everyday words ("canal", "valve", "inspect") slow it down
  compared with opaque labels ("lom", "kel", "dax")? The spec measures this with **Δ_prior**. It is
  the extra learning cost that the familiar words add in the reversed world, after subtracting what
  they add in the reference world.
- **Headline (tentative).** **Δ_prior ≈ +1,450 regret.** Both seed pairings are positive (+770 and
  +2,134). The partial 95% interval is [770, 2,134], but the reference-world cells have only one run
  each, so the true interval is wider. The direction is clear and the size is not yet measured well.
- **The behavioural gap is large.** After 64 training episodes in the reversed world:
  - Named reached S = 0.34 and 0.23 (two seeds).
  - Masked reached 0.92 and 0.90. One Masked seed was already at 0.90 after 32 episodes.
- **Masked transferred, Named mostly did not.** Masked adapters trained in the reversed world also
  reached S ≈ 0.95 in the reference world. That is better than adapters trained in the reference
  world itself (S ≈ 0.32; those runs barely moved, from 0.18 and 0.36).
- **At matched performance, the two adapters changed differently.** At equal reversed-world score
  (S ≈ 0.33):
  - The Named adapter moved its next-valve predictions toward the reversed law. It then also applied
    them in the reference world, where they are wrong.
  - The Masked adapter changed how it acts (more probing, less planner-like) and left its
    predictions unchanged.
- **Both adapters are compact and spread across depth.** The top 4 directions per module keep about
  55–60% of the change; random directions keep about none. No block of 10 layers holds most of it.
- **Confidence.** This is a working pipeline with a consistent signal, not a finding yet:
  - 1–2 seeds per cell;
  - 4 evaluation episodes per world per point;
  - one model, one learning rate, a LoRA that skips the routed experts;
  - several runbook gates skipped (§6).

---

## 1. Why: context and question

### 1.1 The world

Water-v1 is a small simulated canal:

- **Valves.** Four valves; at any moment exactly one supplies the settlement.
- **Actions per cycle:**
  - *probe* (`0`, costs 1 energy): shows which valve supplies now;
  - *actuate* a valve (`1`–`4`, costs 1 energy): delivers one litre if that valve supplies and the
    body carries out the command;
  - *rest* (`5`, free): lowers burden.
- **Score.** +1 per litre and −0.05 per unit of energy.
- **Hidden law.** Whether the supplying valve **persists** after each commit, set by `q`. The agent
  is never told `q`.

| World | q | Meaning |
|---|---|---|
| reference | 0 | the supplying valve is redrawn at random after every commit |
| counterfactual (reversed) | 1 | the supplying valve never changes within an episode |

There are two **bodies** (profiles, never diagnoses). Neither world changes them:

| Body | Executes a repeated command | Executes a switched or first command |
|---|---|---|
| A | 98% | 10% |
| B | 75% | 75% |

### 1.2 The reversal

The point of the world is that **which body does well depends on the world, not on the body**. Body
A is very reliable at repeating and poor at switching. That cripples A in a world that keeps
changing, and suits it in a world that stays put. The known-law planner's exact expected returns,
certified in `iuw certify`:

| | Body A | Body B | Better body |
|---|---|---|---|
| reference (q = 0) | 21.78 | 65.00 | B |
| counterfactual (q = 1) | 92.07 | 69.95 | A |

The README sets the limits of the framing. A world built to satisfy a hypothesis cannot prove it
about people. The measured change describes the model's priors, which mirror its training text, not
anyone's cognition. Task contribution is not human worth.

### 1.3 The question for a language model

A language model brings priors from text about how canals, valves and inspections usually behave.
The experiment asks whether those **word-bound priors interfere** with learning a world whose hidden
law runs the other way.

Two renderings of the same world isolate the words:
- **Named:** everyday words.
- **Masked:** letter-only pseudo-words, counterbalanced across seeds.

The renderings are identical in physics, action tokens, information and tutorial length.

The primary measure is the spec's interaction term:

> **Δ_prior = (C_Named^cf − C_Named^ref) − (C_Masked^cf − C_Masked^ref)**

**C** is a run's *learning cost*: total regret against the known-law planner over its training
episodes. Subtracting the reference-world costs removes everything that is not specific to the
reversed world, such as the reversed world simply being harder. A positive Δ_prior with an interval
above 0 reads as *label-associated interference*. It is not a direct measure of unlearning, and a
zero does not prove the model lacks the relevant priors.

The second question (stage 8, inspection) is **what changed inside the model**. That covers:
- where in depth the change lives;
- how many directions carry it;
- what the model now predicts about the next valve;
- whether it lost competence in the original world.

### 1.4 The score S

**S = (J − J_random) / (J_planner − J_random)**, per world and body:
- 0 is a uniform random policy;
- 1 is the known-law planner;
- negative means worse than random.

*Correction: earlier chat summaries described S as running from "reference" to "oracle". That was
wrong. The definition above is the one in `iuw/scoring.py`.*

---

## 2. What we tried, in order

### 2.1 Starting point

- **The harness.** It was already built and verified offline: simulator, independent auditor,
  certification, renderings, training loop, analysis and inspection. That covers four commits on
  13 October.
- **The plan.** The spec intended training on **Tinker**.

### 2.2 Local tryout on a MacBook Pro (M1 Pro, 32 GB)

We ran the end-to-end loop on Qwen3-0.6B with `--backend hf` on MPS. Changes:

| Change | Effect |
|---|---|
| Per-episode KV cache, scoring only the needed positions | 0.69 → about 10 decisions/s; max difference from a full forward pass 0.0006 |
| Gradient checkpointing, on only while training | the tiny training run peaks near 21 GB and takes about 9 minutes |
| Resumable runs (`state.json`, `--resume`, `--checkpoint-every`) | an interrupted run reproduces an uninterrupted one (tested) |
| Loading fails loudly if any model weight is missing | prevents silently training a partly random model |

### 2.3 Checks against the real model

- **Two masked labels were wrong.** `zov` and `wug` are two tokens each under the Qwen3.6
  tokenizer, which pushed one decision to 31 tokens, over the 30-token cap.
  - *Fix:* the pool is now `lom kel dax tib pim tuv`.
  - *Result:* `render-check` passes with a maximum of 29 tokens per decision.
- **Model structure.** Qwen3_5Moe: 40 layers (30 linear-attention, 10 full-attention), 256 experts
  with 8 active.
  - An "all-linear" LoRA is 45M parameters and skips the routed experts.
  - Including the experts makes it 1.9B parameters.

### 2.4 Budget and platform

- **Tinker was too expensive for the budget.** Measured throughput priced at Tinker's rates put the
  full spec MVP (12 runs × 800 episodes) at about **$475**, against a Tinker balance of $10.
- **Moved to Runpod.** Same HF + PEFT code on one rented **H200 (141 GB)** at $5.29/hr.
- **Shrunk the experiment.**
  - 8 updates × 8 episodes = **64 training episodes per run** (the spec has 800).
  - **LoRA on all linear layers**, rank 32: attention, linear attention and the shared-expert MLP. The 256 routed experts stay frozen.
  - **Learning rate 4e-5**, my choice and not tuned.
  - Evaluations at updates 0, 4 and 8, on 2 seeds × 2 bodies per world.
- **Tinker can't import a LoRA.** Its SDK accepts only `tinker://` checkpoints. Weight-level
  inspection needs local weights anyway.

### 2.5 Pod 1: smoke test and the seed-0 reversed-world pair (≈ $9.43, final from billing)

- **Throughput.**
  - 7.5 decisions/s, about 172 episodes/hour, 98.9% cache-hit rate.
  - About 107 s per update of 8 episodes.
- **Loss.** Probability left on illegal tokens was 1.07%, over the spec's 1% limit, so all runs use
  the **renormalised** loss over legal actions.
- **Runs.** Named and Masked in the reversed world, seed 0.
- **Inspections of both u8 adapters.**
  - Replay on known-law-planner histories.
  - Block ablation and top-k direction truncation, each with a random control.
  - Next-valve prediction.
  - Return trip to the reference world.
- **Setup fixes along the way.**
  - The image's Python is externally managed (PEP 668), so we used a venv.
  - `huggingface-cli` was missing, so we used `snapshot_download`.
  - The fla Triton kernel failed on Hopper; installing `tilelang` fixed it.
  - stdout buffering and a `pkill` that matched itself.

### 2.6 Analysis tooling (local, no GPU)

- **`scripts/report.py`.** Builds the Markdown report from run folders and inspections, including
  caveats derived from the data (thin cells, small evaluations, censored speed, partial intervals).
- **`scripts/pod_report_run.sh`.** Fills the 2×2 grid. It is resumable, splits across pods, takes
  an explicit job order, and dry-runs offline with `BACKEND=mock`.
- **Inspection histories changed.**
  - *Problem:* pure planner histories were near ceiling for the base model, at log-probability
    about −0.04 per decision. The model copies the pattern in context, so the measure barely moved.
  - *Change:* 25% of the planner's decisions are now replaced by keyed random legal actions, and
    the model is scored on the action the **planner would have taken**.
  - *Result:* base agreement drops to about −0.32, leaving room to move. The cost is halved (one
    forward pass per history), and next-valve prediction gets 6 points in the reversed world
    instead of 1.

### 2.7 Pod 2: the reference-world cells, a second seed and the matched pair (≈ $15, still to be confirmed)

| Order | Job | Why |
|---|---|---|
| 1–2 | Named and Masked, **reference world**, seed 0 | without them Δ_prior cannot be computed |
| 3–4 | Inspect Named u8 and **Masked u4** | the two checkpoints with equal reversed-world score (both ≈ 0.33) |
| 5–6 | Named and Masked, **reversed world, seed 1** | replicate the main contrast |
| skipped | re-inspect Masked u8 with the new histories; reference-world seed 1 | kept about $2 of the $18 credit in reserve |

Every run used the same settings, so all six are comparable:

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen3.6-35B-A3B, bf16, one H200 |
| Adapter | LoRA rank 32 on every linear layer (attention, linear attention, shared-expert MLP); the 256 routed experts stay frozen |
| Training | 8 updates × (4 seeds × 2 bodies) = 64 episodes; renormalised policy-gradient loss, clip 0.2, 1 pass; KL off; batch-mean baseline; γ = 1; temperature 1 |
| Optimiser | AdamW, lr 4e-5, β = (0.9, 0.95) |
| Evaluation | dev seeds 0–1 × bodies A, B × both worlds, at updates 0, 4, 8; fixed evaluation salt, so every run is evaluated on the same episodes |
| Masked mapping | equals the seed (mapping 0 for seed 0, mapping 1 for seed 1) |

---

## 3. Outcomes

### 3.1 Learning cost and Δ_prior

| Rendering | Training world | Runs | Learning cost (mean ± sd) |
|---|---|---|---|
| Named | counterfactual | 2 | **3,828 ± 110** |
| Named | reference | 1 | 1,941 |
| Masked | counterfactual | 2 | **2,447 ± 854** |
| Masked | reference | 1 | 2,012 |

- **Δ_prior** = (3,828 − 1,941) − (2,447 − 2,012) = 1,887 − 435 = **+1,452**.
- **Seed by seed:** +770 with seed 0 in the reversed world, +2,134 with seed 1.
- **Interval:** partial bootstrap 95% interval [770, 2,134]. It reflects spread in the reversed-world
  cells only, so the true interval is wider.

### 3.2 Scores in each world

Each cell shows S in the reversed world / S in the reference world, averaged over both bodies.

| Run | u0 | u4 | u8 |
|---|---|---|---|
| Named, reversed world, seed 0 | −0.22 / 0.18 | −0.21 / 0.11 | **0.34** / 0.62 |
| Named, reversed world, seed 1 | −0.22 / 0.18 | −0.06 / 0.12 | **0.23** / 0.19 |
| Masked, reversed world, seed 0 | −0.21 / 0.36 | 0.33 / 0.68 | **0.92** / 0.94 |
| Masked, reversed world, seed 1 | 0.05 / 0.15 | **0.90** / 0.90 | 0.90 / 0.95 |
| Named, reference world, seed 0 | −0.22 / 0.18 | −0.23 / 0.35 | −0.20 / **0.32** |
| Masked, reference world, seed 0 | −0.21 / 0.36 | −0.22 / 0.32 | −0.22 / **0.32** |

- **Adaptation speed.** The spec defines it as the interactions needed to hold S ≥ 0.9 on three
  consecutive evaluations. It is **censored in every run**, since three evaluations cannot satisfy it
  within the run.
- **Welfare and audits.** No welfare violations and no audit violations in any run.

### 3.3 Inspection at matched performance

Both adapters score S ≈ 0.33 in the reversed world. Inspection used 8 held-out histories per world
with 25% random actions.

| Measure | Named u8 (base → adapted) | Masked u4 (base → adapted) |
|---|---|---|
| Agreement with the planner, reversed world (closer to 0 agrees more) | −0.323 → **−0.295** | −0.313 → **−0.330** |
| Agreement with the planner, reference world | −0.250 → −0.214 | −0.221 → −0.199 |
| Probe probability, reversed world | 0.051 → 0.053 | 0.050 → **0.064** |
| Next-valve prediction vs the reversed law (KL, lower is closer) | 0.17 → **0.07** | 0.10 → 0.10 |
| Next-valve prediction vs the reference law (KL) | 0.48 → **0.81** | 0.49 → 0.45 |
| Share of the change kept by the top 1 / top 4 / random 4 directions | 0.38 / 0.60 / −0.09 | 0.46 / 0.53 / 0.03 |
| Largest share lost by switching off one block of 10 layers | 0.52 (layers 10–19) | 0.58 (layers 10–19) |
| Reference-world S on the return trip, adapted vs base | 0.39 vs 0.39 | 0.39 vs 0.39 |

The earlier inspections of both u8 adapters used pure planner histories (ε = 0). They told the same
story in coarser form:
- **Named u8:** next-valve prediction in the reference world moved from 0.79 to 1.12, further from
  the reference law. Reference-world score on the return trip did not change.
- **Masked u8:** reference-world score on the return trip rose from 0.39 to 0.94.

---

## 4. Insights

Each insight is marked with how much it can bear.

1. **The words slowed learning in the reversed world beyond what they cost in the reference world.**
   *(Consistent sign; size uncertain.)* Δ_prior is positive in both seed pairings. Named's learning
   cost in the reversed world is the most stable number we have (3,750 and 3,906). Masked's varies
   a lot (3,051 and 1,843), partly because mapping 1 started higher (S = 0.05 vs −0.21). That
   variation is part of why the mappings are counterbalanced.

2. **The base model starts below random in the reversed world, in both renderings.**
   *(Solid for this model and prompt.)* S ≈ −0.2 at update 0. Whatever strategy the untrained model
   brings is worse than random in this world. Body B is at −0.3 to −0.32. So the model is not just
   ignorant: what it brings to the task works against it. Masked mapping 1 is the exception
   (body A at 0.42), so label choice matters even before training.

3. **Masked found a strategy that works in both worlds.** *(Two seeds agree.)* Masked adapters
   trained only in the reversed world scored about 0.95 in the reference world too, for both bodies.
   That beats the adapters trained in the reference world. One possible reading is that the
   opaque-label model learned a general "look, then act on what you saw" routine. The
   higher probe probability is in line with that. The Named model instead learned something tied to
   one world. This needs the interview and belief probes to confirm (§7).

4. **Named changed what it predicts about the canal, and applied that change in the wrong place.**
   *(One inspection per adapter; worth replicating.)* At matched performance, the Named adapter's
   next-valve predictions moved strongly toward "the valve persists". They moved in the reference
   world too, where that is false (0.48 → 0.81). The Masked adapter's predictions did not move.
   Its behaviour changed instead: more probing and less imitation of the planner.

   A cautious reading: with familiar words, learning went through the model's existing beliefs
   about "canals", and changed the belief wholesale for the word rather than for the world. With
   opaque labels there was no belief to rewrite, so learning went into policy. That fits a "prior
   conflict" story. But belief answers are outputs to assess, not evidence of mechanism, and this
   is one adapter each.

5. **Training in the reference world barely moved in 64 episodes.** *(Two runs.)* Both reference-world
   runs ended at S ≈ 0.32: Named from 0.18, Masked from 0.36. Body B stayed near 0 throughout. Two consequences:
   - In this design, the reference cells' learning cost mostly reflects **starting competence**, not
     learning speed.
   - Reference-world learning is very slow at this scale. The spec's 800-episode runs, or a per-body
     look, may be needed before the reference baseline means much.

6. **Both adapters are low-rank-ish and spread across depth.** *(Two adapters, consistent with the
   ε = 0 inspections.)*
   - Four directions per module carry over half the change, and matched random directions carry none.
   - Removing any one block of 10 layers keeps at least 40% of the change.
   - Neither rendering stores the change in one place.

---

## 5. Learnings: method, measurement and engineering

**Measurement**
- **Report S per world, never averaged across worlds.** My first summary averaged both worlds. It
  said Named reached 0.48, when it was 0.34 in the reversed world and 0.62 in the reference world.
- **Compare adapters at matched performance.** Named u8 against Masked u8 compares an adapter that
  is half-trained with one that has converged. The existing Masked u4 checkpoint was the fix, and it
  cost nothing.
- **Planner-history replay saturates.** A model that copies in context looks near-perfect on the
  planner's own histories. Off-policy steps (ε = 0.25) and scoring the planner's intended action make
  the measure sensitive.
- **Evaluations of 4 episodes per world are noisy.** Neighbouring updates move by tenths of S, and
  body A's reference-world cell is the noisiest (J_planner − J_random is only 20). Larger
  evaluations are the cheapest way to tighten everything.
- **A one-run bootstrap gives a zero-width interval.** `report.py` now labels such intervals as
  missing or partial instead of calling them significant.

**Engineering**
- **The KV cache is what makes this affordable.** Rollouts went from about 0.7 to about 10 decisions/s
  locally. On the H200 the run is CPU-bound (≈ 110% of one core, GPU about 35% busy). Batching
  episodes is the next large speed-up.
- **Resumability paid off.** Interrupted runs continue, and the run script skips finished work.
- **Follow progress in the right file.** Progress goes to stdout (`*.train.json`), warnings to stderr.
  Watching stderr made a healthy run look stalled for about 20 minutes.
- **Don't run the full test suite on the pod.** It ran far slower there than on the Mac while the GPU
  sat idle. Run tests locally, then a 1-update smoke on the pod.
- **Runpod transfers from Iceland were fast** (≈ 7 MB/s down, ≈ 2 MB/s up). Downloading results
  while later jobs ran saved paid minutes at the end.
- **The Runpod connection's sign-in can expire mid-session.** All calls then return 403, including
  reads. Re-authenticating (`/mcp`) or the console is the only safe way to terminate. A
  self-termination script that reads the pod's key from inside the pod was blocked as credential
  exploration.

---

## 6. Limitations and threats to validity

- **Seeds.** 2 runs in each reversed-world cell and 1 in each reference-world cell (the spec has 3).
  No full interval for Δ_prior.
- **Training length.** 64 episodes per run against the spec's 800. Adaptation speed is censored
  everywhere, and the reference-world runs may simply not have started learning.
- **Choices not in the spec.** A LoRA that skips the 256 routed experts (Tinker probably adapts them too), a single
  learning rate picked for budget, and the renormalised loss.
- **Skipped runbook gates.** Stage-3 interface quiz (both renderings ≥ 95%), stage-4 frozen-model
  check, notebook and middle-loop stages, and the sealed test. In particular, the quiz would show
  whether Masked's better learning partly reflects better *understanding of the interface*, rather
  than fewer conflicting priors.
- **Controls not run.** Renamed-Named, codebook-familiarised Masked, shuffle and remap worlds, and the
  equalised-execution body. Each rules out an alternative explanation.
- **Inspection.** One adapter per rendering at matched performance. Return trips use one seed per
  condition. Belief probes are model outputs, not mechanism.
- **Scope.** One model family and one world. Results describe this model's priors under this
  training, not people.

---

## 7. Next steps

### Short term (days, about $10–40 of GPU)

1. **Close the Δ_prior interval.** Run Named and Masked in the reference world, seed 1 (about 1 h,
   ≈ $5.50), then seed 2 in all four cells (about 2 h, ≈ $11). That gives 3 seeds per cell as in the
   spec, so the run-clustered interval becomes real.
2. **Bigger evaluations.** Re-evaluate the final adapters on 16 seeds × 2 bodies × 2 worlds, with no
   training (about 10 min per adapter). This is the cheapest way to firm up the per-world scores and
   the transfer finding (insight 3).
3. **Finish the matched inspection.** Re-inspect Masked u8 with the new histories. Inspect the seed-1
   adapters at matched performance (Named u8 at 0.23 against the nearest Masked checkpoint).
4. **The interview / belief-probe step** (proposed earlier). Base, reference-world and reversed-world
   adapters, in both renderings:
   - scored questions inside and outside the transcript (next valve, value of a repeat, payoff of a
     rest);
   - a few sampled free-text explanations;
   - a check of each adapter's stated beliefs against its own behaviour.

   About 20 min of GPU. This tests insight 4 directly.
5. **Run the skipped interface gate.** `iuw quiz` on the base model in both renderings (minutes). If
   Masked understands the interface better, Δ_prior must be read with that in mind.
6. **Housekeeping.** Commit the code changes. Keep `runs/pod*` outside git, with an index of the
   adapter checksums. Make sure the pod is terminated.

### Medium term (weeks, about $100–500)

1. **Throughput.** Batch cached decoding across the 8–16 episodes of an update. The GPU is about 35%
   busy, so a 3–5× speed-up is plausible. That makes the spec-sized grid affordable on rented GPUs.
2. **The spec's MVP grid.** 12 runs × 800 episodes, evaluations every 10 updates, the sealed test
   opened once per run, and adaptation speed no longer censored. Run it either on Tinker (if credit
   allows; about $475 at the measured rate) or on the batched HF path.
3. **Controls.** Renamed-Named (different everyday words), codebook-familiarised Masked (the spec's
   second baseline for Δ_prior), shuffle and remap worlds, and the equalised-execution body.
4. **Sensitivity.** A short learning-rate sweep (e.g. 1e-5, 4e-5, 1e-4) on one cell, and adapting the
   experts (`--lora-experts`) on one H200, to check that the effect is not an artefact of
   a LoRA that skips the routed experts.
5. **The full inspection.** The change ledger on matched checkpoints for every seed, return trips with
   relearning, and steering along the top adapter directions.

### Long term (the research arc)

1. **Generality across models.** Other families and sizes (a dense 8–14B model and Qwen3-1.7B with
   full-parameter PPO, as listed in the spec's expansions). Does prior conflict grow with how much the
   model "knows" about canals?
2. **Graded worlds.** Intermediate persistence (q = 0.25, 0.75; `world_with_q`) for transfer curves:
   is the Named deficit proportional to how far the world departs from the expected law?
3. **More worlds.** Mechanisms-v1 and Habitat-v1, and the outer world-search loop (MAP-Elites /
   PAIRED). Can worlds be found automatically where familiar language helps or hurts most?
4. **Mechanism.** If insight 4 holds, find the features tied to labels that the Named adapter
   rewrites, and test whether steering them reproduces or removes the deficit.
5. **Framing the claim responsibly.** Keep the README's limits in every write-up. These worlds are
   designed so that a body's value depends on the world. What we measure is how a language model's
   inherited assumptions help or hinder it in such a world. It is a claim about models and their
   training text, not about people.

---

## 8. Where everything is

| What | Where |
|---|---|
| Generated report (all tables) | `runs/pod2/report/report.md` |
| Run folders (config, per-episode logs, evaluations, results) | `runs/pod2/report/<run>/`; the seed-0 reversed-world runs are also in `runs/pod/budget/` |
| Inspections | `runs/pod2/report/inspect-{named-cf-u008,masked-cf-u004}/`; earlier ε = 0 versions in `runs/pod/budget/inspect-{named,masked}/` |
| Adapters, pod 1 (176 MB each) | `runs/pod/hf-checkpoints/{named,masked}-cf-u000…u008` |
| Adapters, pod 2 (SHA-256 verified against the pod) | `runs/pod2/hf-checkpoints/`: `named-ref-s0-u008`, `masked-ref-s0-u008`, `named-cf-s1-u008`, `masked-cf-s1-u004`, `masked-cf-s1-u008` |
| Scripts | `scripts/pod_setup.sh`, `scripts/pod_report_run.sh`, `scripts/inspect_run.py`, `scripts/report.py` |

Reproduce on one H200, from the repository root:

```bash
bash scripts/pod_setup.sh                       # venv, kernels, model download (skip its pytest on the pod)
JOBS="named:reference:1 masked:reference:1" INSPECT=none \
  nohup bash scripts/pod_report_run.sh > runs/report.log 2>&1 &
python scripts/report.py --runs runs/report --out runs/report/report.md   # on any CPU afterwards
```

Load an adapter:
`HFBackend("Qwen/Qwen3.6-35B-A3B", init_adapter="runs/pod2/hf-checkpoints/masked-cf-s1-u008")`.

## 9. Spend

| Item | Cost |
|---|---|
| Runpod pod 1 (H200, ≈ 1.8 h) | $9.43 (final, from billing) |
| Runpod pod 2 (H200, from 13:13 UTC) | ≈ $15 at 16:00 UTC; bills $5.29/hr until terminated |
| Tinker | ≈ $0 (key verified, no runs) |
