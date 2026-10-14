# Inverted-Utility World — Water-v1 harness (spec v3)

Code to execute the *Inverted-Utility World: RL Environment Spec v3* MVP: the
Water-v1 canal, its certification, two renderings, LoRA policy-gradient
training on Tinker, the prior-conflict analysis and the inspection protocol.

Two limits hold throughout, as in the spec. A world built to satisfy the
hypothesis cannot prove it about people, and the measured change describes the
model's priors, which mirror its training text, not autistic cognition. Task
contribution is not human worth.

## Install

```bash
pip install -e ".[test]"            # simulator, certification, analysis, mock backend
pip install -e ".[tinker]"          # + Tinker training (set TINKER_API_KEY)
pip install -e ".[inspect]"         # + local HF/PEFT backend and adapter interventions
pytest -q                           # 72 tests (+1 torch-only), about 2 minutes; -m "not slow" skips Monte Carlo
```

## Local tryout (Apple silicon or one GPU)

`--backend hf` runs the whole loop locally with transformers + PEFT (picks `cuda`, then `mps`,
then `cpu`; override with `--device`). Use a small model and shrink the batches. This checks
the plumbing; it is not the experiment.

```bash
pip install -e ".[test,local]"
iuw smoke --backend hf --model Qwen/Qwen3-0.6B --episodes 4
iuw train --backend hf --model Qwen/Qwen3-0.6B --updates 2 --seeds-per-update 2 --eval-seeds 2 --run-id local-tiny
```

On an M1 Pro (32 GB) the training command takes about 9 minutes and peaks near 21 GB:
rollouts reuse a per-episode key/value cache (about 10 decisions/s), and training uses
gradient checkpointing.

## Runbook (build order)

Each stage must pass before the next spends money. Null results are kept.

| Stage | Command | Passes when |
| --- | --- | --- |
| 1–2 Physics, planners | `iuw certify --freeze` | Fixture exact (15.78 / 31.78 / 97.12 vs 75), planner exact (21.78, 65.00, 92.07, 69.95), B's lower 95% bound ≥ 60, witness gate passes. Writes `artifacts/frozen/references.json` (J_reference, J_random, J_oracle per world and body). |
| 3 Renderings | `iuw render-check --tokenizer Qwen/Qwen3.6-35B-A3B` | No forbidden statements; tutorials and transcripts within 10%; prompt ≤ 600, decision ≤ 30, total ≤ 8,192 with a full notebook; planner earns the same through either rendering. |
| 3 Quiz | `iuw quiz --backend tinker` | Both renderings ≥ 95% and within 5 points. Otherwise revise the tutorial once, then schedule `iuw grid --familiarised`. |
| 4 Frozen model | `iuw frozen --backend tinker --out runs/frozen` | Stop if the Named counterfactual cell already has S ≥ 0.9; redesign Named words if Named and Masked behave the same (`words_evoke_priors`). |
| 5 Smoke test | `iuw smoke --backend tinker` | Measures throughput, cache-hit rate, illegal-token mass, real cost, and projects the MVP cost. If illegal mass > 1%, use `--loss renormalised`. |
| 6 Notebook | `iuw notebook --rendering named --margin <fixed> --test` (and `masked`) | 600 episodes, ≤ 25 rewrites, then the sealed test once. |
| 7 Middle loop | `iuw middle-loop --threshold <fixed> --rendering named --world cf` | Dev seeds only; accepted only if the paired lower 95% bound clears the threshold. |
| 7 Grid | `iuw grid --backend tinker [--lr …] --test` | 12 runs × 800 episodes; evals at updates 0, 10…50; sealed test opened once per run. |
| 7 Analysis | `iuw analyze --runs runs` | Δ_prior with run-clustered interval under every Masked baseline that was run; adaptation speed per run. |
| 8 Inspection | `iuw.inspection.*` on one 80 GB GPU | See "Inspection" below. |

`iuw show --rendering masked --mapping 1 --world cf` prints a rendered episode.
Every command accepts `--backend mock` to dry-run the pipeline offline. The mock
is a numpy linear policy that exists to test the plumbing. It is never a
baseline or evidence about language models.

## Layout

| Module | Spec section |
| --- | --- |
| `iuw/config.py` | Profiles A/B, Water-v1 worlds (primary, noisy ablation, controls), K = 60, budgets, prices |
| `iuw/rng.py` | Randomness keyed by seed, episode, cycle and event; separate world, execution and policy streams |
| `iuw/env/water.py` | Physics, execution law, burden, allowlisted `Observation`, physics log, termination vs truncation |
| `iuw/env/audit.py` | Independent evaluator: rebuilds outcomes from the log; rejects duplicated reward, negative stocks, clock stalls, broken conservation, reward from rest |
| `iuw/policies/planner.py` | Known-law planner (exact DP), lazy rest schedule, exact max-activity check |
| `iuw/policies/baselines.py` | Random, always-probe, blind-repeat, probe-once-repeat, omniscient oracle |
| `iuw/certify.py`, `iuw/scoring.py` | Fixture arithmetic in exact rationals, floors, witness gate, frozen references, S |
| `iuw/render/` | Named/Masked vocabularies, matched tutorials, append-only token transcripts, quiz, codebook pairs, stage-3 checks |
| `iuw/llm/` | Tokenizers, legal-action scoring and sampling, rollouts; Tinker, HF+PEFT and mock backends |
| `iuw/train/` | Advantages (return-to-go, batch-normalised, γ = 1), clipped loss, KL schedules, inner loop, evaluation, sealed test |
| `iuw/experiment/` | Grid, codebook familiarisation, middle loop, notebook climb, frozen controls, smoke test |
| `iuw/analysis/` | Δ_prior, IQM and cluster bootstrap, adaptation speed, gates, behaviour metrics |
| `iuw/inspection/` | Matched-history replay, change ledger and belief probes, return trip, adapter ablation, rank truncation, steering |
| `iuw/seeds.py` | Disjoint splits (128/16/32, 200 notebook, discovery, replication, pilot) and the one-time seal |

## What is verified here, and what is not

Verified by the test suite in this repository, offline:

- Every certified number in the spec, exactly: the fixture table, the four
  planner returns, the 43.22 and 22.12 gaps, the cross-world condition, and the
  noisy row's stipulated 50-cycle limit.
- Monte Carlo reproduces them. Over 1,000 episodes B's reference lower 95% bound
  is about 64.6, which clears K = 60.
- Noisy ablation, computed and frozen: A rests in 42 cycles and acts in 58. Its
  planner return is 12.54 in the volatile world and 53.01 in the stable one.
  Without rest, A's burden passes 10 after the eighth cycle.
- The equalised-execution control removes the reversal: A gets 65.00 and 69.95.
- The audit catches tampering. The planner earns identical returns through
  either rendering. No episode overflows the context.
- The training loop's plumbing, on the mock backend: rollout scoring and
  training scoring agree (the importance ratio is exactly 1 on the first pass),
  advantages are normalised, KL shaping, checkpoints, the evaluation schedule,
  learning cost, the sealed test refuses a second opening, and Δ_prior and the
  gates work on synthetic data.

Not verified here (no network to Tinker or Hugging Face, no GPU):

- **Tinker backend.** Written against SDK 0.33.1's real signatures, including
  `target_prompt_logprobs` with sparse CSR rows for one-call legal-action
  scoring, `ppo` with `loss_fn_config` clip thresholds, and
  `forward_backward_custom` for the renormalised loss. It has not been run
  against the service. Confirm it in the smoke test.
- **Real tokenizer counts.** Offline tests use `ApproxQwenTokenizer`, which
  mimics Qwen pre-tokenisation. Parity, the 30-token decision cap and
  single-token pseudo-words must be re-checked with the real tokenizer at stage 3.
- **Torch code paths.** `HFBackend`, the renormalised custom loss and
  `inspection/adapters.py` need torch, which could not be installed in this
  environment.

## Choices the spec left open

- **Masked labels** are short letter-only pseudo-words (`lom`, `kel`, `dax`,
  `tib`, `pim`, `tuv`). The spec's `op-0` / `unit-3` / `channel k7` are
  examples. Letters only, so no label can be confused with an action digit or a
  value. Valves are `unit N` in Masked and `valve N` in Named, with N equal to
  the action ID in both, so neither rendering pays to decode the valve. The
  six-role mapping is counterbalanced: mapping m gives role i the code
  pool[(i + 2m) mod 6].
- **Transcript format.** Qwen3 chat framing with an empty think block. The
  tutorial is the system turn and the body/noise header is the user turn.
  Environment lines and action digits then alternate inside the assistant
  turn, with each action written directly after `>`. Segments are tokenised
  separately and never re-tokenised.
- **Termination.** The 100-cycle horizon counts as termination, since time is
  observed. Truncation is reserved for external limits, and only then does the
  loop bootstrap, using the batch-mean return-to-go of peers at the same cycle.
  Running out of energy or reserve also terminates.
- **Planner information state.** The most recent anchor (probe or delivery)
  and its age, the last-command relation, and the excluded valves when q = 1.
  This is exact for q ∈ {0, 1}. For intermediate q it is a reference, not an
  optimum, and its value comes from Monte Carlo.
- **Oracle.** Rests lazily, so the upper bound never breaches welfare.
- **Reference KL** is folded into per-decision rewards as
  β·(log π_old − log π_ref), because Tinker's built-in `ppo` loss has no KL
  term. The **entropy bonus** is likewise a reward term, −c·log π(a).
- **`--baseline per-cycle`** is an optional variance reduction, off by default.
  The spec's batch-mean baseline is unbiased but noisy over 200-decision
  episodes.
- **Middle loop.** The incumbent is also trained fresh on 100 episodes, so
  every comparison is like for like. That is one extra 100-episode run.
- **Notebook selection.** Each of the 8 seeds is run once per notebook, with
  bodies alternating by seed. The climb runs in the counterfactual world by
  default (`--world`).
- **Frozen Masked controls** rotate the three mappings across test seeds.
- **Margins and thresholds** for the middle loop and notebook are required
  arguments. The spec leaves them open and fixes them in advance.

## Not built (expansions)

Mechanisms-v1, Habitat-v1, the outer world-search loop (MAP-Elites/PAIRED),
the coordinator track, the thinking-on arm, the label factorial, the
recurrent-learner and SFT baselines, and the Qwen3-1.7B full-parameter PPO arm.
Each needs its own frozen equations, budget and floor before any code. The
simulator's `WorldConfig` and `world_with_q(q)` already cover the
intermediate-persistence transfer worlds (q = 0.25, 0.75).

## Inspection (stage 8)

```python
from iuw.llm.hf_backend import HFBackend
from iuw.inspection import adapters, replay, ledger, return_trip

be = HFBackend("Qwen/Qwen3.5-9B", init_adapter="downloaded_tinker_adapter/")
with adapters.ablate_layers(be.model, range(10, 20)):      # where does the change live?
    ...                                                    # re-run evaluate() / replay
with adapters.keep_top_directions(be.model, k=4):          # how few directions carry it?
    ...
with adapters.keep_top_directions(be.model, k=4, random_control=True):
    ...                                                    # matched random control
scores = return_trip.return_trip(be.scorer(), be.reference_scorer(), be.tokenizer, rendering, normaliser)
return_trip.diagnose(scores, relearning={"original": ..., "relearn": ...})
```

The change ledger (`ledger.ChangeLedger`) records one row per prior, measured
on identical held-out histories. Belief answers are outputs to assess, not
evidence of internal mechanism.

## Report run (Runpod H200, budget-sized)

`scripts/pod_report_run.sh` fills the 2×2 grid (Named/Masked × counterfactual/reference) to three
seeds per cell with the budget settings (8 updates × 8 episodes, lr 4e-5, renormalised loss,
attention-only LoRA), reusing the two seed-0 counterfactual runs from `scripts/pod_budget_run.sh`.
It then inspects Named u8 against Masked u4 (the checkpoint with the matching counterfactual score)
and Masked u8, and writes `analysis.json` and `report.md`. It is resumable and can be split across
pods with `SHARD`/`NSHARDS`; `BACKEND=mock` dry-runs it offline.

```bash
bash scripts/pod_setup.sh
SHARD=0 NSHARDS=2 nohup bash scripts/pod_report_run.sh > runs/report.log 2>&1 &
python scripts/report.py --runs runs/report --out runs/report/report.md   # CPU only, after merging shards
```

Inspection histories replace a quarter of the planner's decisions with keyed random legal actions
(`--epsilon 0.25`). On pure planner histories the base model is already near ceiling, because it
copies the pattern in context, so the agreement measure barely moves.
