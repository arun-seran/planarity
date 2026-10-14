# Water-v1: the Named and Masked experiment and its results

*Data from 13 and 14 October 2026. Model: Qwen3.6-35B-A3B with a LoRA adapter. Six training runs, two
adapter checks at matched scores.*

## 1. The world

Water-v1 is a small simulated canal with four valves. At any moment, one valve supplies water to a
settlement. The agent does not see which valve it is.

Each cycle has two steps. First, the agent can probe. A probe costs 1 energy and shows the valve that
supplies now. Then the agent must commit. It opens one valve for 1 energy, or it rests for free. An
open valve delivers one litre if that valve supplies and the body carries out the command. Rest lowers
the burden.

The score is +1 for each litre and −0.05 for each unit of energy spent.

A hidden rule, q, decides what happens to the supplying valve after each commit. The agent never
learns q from the tutorial.

| World | q | What happens after each commit |
|---|---|---|
| reference | 0 | The world picks the supplying valve again at random. |
| counterfactual | 1 | The same valve keeps supplying until the episode ends. |

There are two bodies. The world has no effect on them.

| Body | Carries out a repeated command | Carries out a changed or first command |
|---|---|---|
| A | 98% of the time | 10% of the time |
| B | 75% of the time | 75% of the time |

We computed the exact expected return of a planner that knows q. In each world, a different body comes
out ahead.

| Planner return | Body A | Body B |
|---|---|---|
| reference world | 21.78 | 65.00 |
| counterfactual world | 92.07 | 69.95 |

## 2. The question

A language model learns from text. Text about canals and valves carries assumptions about how such
things behave. Do these word-bound assumptions slow the model down when it must learn the counterfactual
world?

To test this, we show the same world to the model in two forms.

- **Named.** Everyday words: canal, valve, inspect, open, take a break.
- **Masked.** Short made-up labels, for example "lom" and "kel".

The physics, the action digits and the information in each line are the same in both forms. Both
tutorials have the same length. Each Masked seed uses a different mapping, so each label takes a
different role from one seed to the next.

## 3. What we measured

**S** is the normalised score for one world and one body:

> S = (J − J_random) / (J_planner − J_random)

J is the mean return of the model. At S = 0 the model earns what a uniform random policy earns. At
S = 1 it earns what the planner from section 1 earns. S goes below 0 when the model does worse than
random.

**Learning cost (C)** is the total regret over all training episodes of a run. For each episode, regret
is the planner return minus the return the model got. A lower learning cost means the model lost less
while it learned.

**Δ_prior** compares the four cells of the design:

> Δ_prior = (C_Named,cf − C_Named,ref) − (C_Masked,cf − C_Masked,ref)

For Named, the cost added by the counterfactual world mixes the effect of the words with how hard
that world is. The Masked difference measures how hard the world is on its own, and we subtract it. A
value above zero means the everyday words add extra learning cost in the counterfactual world.

## 4. How we ran it

| Item | Value |
|---|---|
| Base model | Qwen/Qwen3.6-35B-A3B in bf16 on one H200 GPU |
| Adapter | LoRA, rank 32, on every linear layer: attention, linear attention and the shared-expert MLP; the 256 routed experts stay frozen |
| Training | 8 updates per run; each update uses 4 seeds × 2 bodies = 8 episodes, so 64 episodes per run |
| Loss | policy gradient over legal actions only, clip 0.2, one pass per update, no KL term |
| Optimiser | AdamW, learning rate 4e-5 |
| Checks of the score | at updates 0, 4 and 8, on 2 seeds × 2 bodies in each world, with the same episodes for every run |

The model gives one digit per decision. We scored the six action digits at each step and sampled from
the legal ones.

We ran six runs.

| Run | Form | Training world | Seed |
|---|---|---|---|
| named-cf | Named | counterfactual | 0 |
| named-cf-s1 | Named | counterfactual | 1 |
| masked-cf | Masked | counterfactual | 0 |
| masked-cf-s1 | Masked | counterfactual | 1 |
| named-ref-s0 | Named | reference | 0 |
| masked-ref-s0 | Masked | reference | 0 |

No run had a welfare violation, and every run passed the audit of the simulator log.

## 5. Results

### 5.1 Learning cost and Δ_prior

| Form | Training world | Runs | Learning cost |
|---|---|---|---|
| Named | counterfactual | 2 | 3,750 and 3,906 (mean 3,828) |
| Named | reference | 1 | 1,941 |
| Masked | counterfactual | 2 | 3,051 and 1,843 (mean 2,447) |
| Masked | reference | 1 | 2,012 |

With the cell means, Δ_prior = (3,828 − 1,941) − (2,447 − 2,012) = **+1,452**.

We also computed it for each seed alone. Seed 0 gives +770. With seed 1 we get +2,134. The bootstrap
puts the 95% interval at [770, 2,134]. That interval only counts the spread inside the two
counterfactual cells. Each reference cell has one run, so the bootstrap has no spread to sample there.

### 5.2 Scores during training

Each cell shows S in the counterfactual world first, then S in the reference world. We averaged the two
bodies.

| Run | Update 0 | Update 4 | Update 8 |
|---|---|---|---|
| named-cf | −0.22 / 0.18 | −0.21 / 0.11 | 0.34 / 0.62 |
| named-cf-s1 | −0.22 / 0.18 | −0.06 / 0.12 | 0.23 / 0.19 |
| masked-cf | −0.21 / 0.36 | 0.33 / 0.68 | 0.92 / 0.94 |
| masked-cf-s1 | 0.05 / 0.15 | 0.90 / 0.90 | 0.90 / 0.95 |
| named-ref-s0 | −0.22 / 0.18 | −0.23 / 0.35 | −0.20 / 0.32 |
| masked-ref-s0 | −0.21 / 0.36 | −0.22 / 0.32 | −0.22 / 0.32 |

At update 0, the base model scores below random in the counterfactual world. Body B is the lowest, at
−0.32 or −0.30. Mapping 1 of the Masked form started higher, and body A reached 0.42 with it.

By update 8, the Named runs had reached 0.34 and 0.23 in the counterfactual world. Both Masked runs
passed S = 0.9 there. Masked seed 1 was already at 0.90 at update 4.

The two runs in the reference world both ended at S = 0.32. The Named run started at 0.18 and the
Masked run at 0.36. Body B stayed near 0 in the reference world for the whole run.

The spec defines adaptation speed as the number of episodes until S stays at 0.9 or more for three
checks in a row. No run reached that within 64 episodes.

### 5.3 Scores in the world the model did not train in

The Masked adapters from the counterfactual world also scored well in the reference world. Seed 0
reached 0.94 and seed 1 reached 0.95. Both bodies scored 0.94 or more there. The Masked adapter that
trained in the reference world itself scored 0.32.

The Named adapters from the counterfactual world scored 0.62 (seed 0) and 0.19 (seed 1) in the reference
world.

### 5.4 Inside the adapters at the same score

We picked two checkpoints with the same score in the counterfactual world. Named u8 (seed 0) has
S = 0.34. Masked u4 (seed 0) has S = 0.33.

We replayed 8 held-out episodes per world through each model. One in four actions in these episodes is
random, and the planner chose the rest. At each decision we read the model's probabilities. We did the
same with the adapter turned off.

**Match with the planner** is the mean log-probability that the model gives to the planner's choice at
each decision. Values closer to 0 match the planner more.

**Next-valve error** is the KL divergence between the model's guess of the next supplying valve and the
true rule of the world. Lower is closer to the rule.

| Measure | Named u8: off → on | Masked u4: off → on |
|---|---|---|
| Match with the planner, counterfactual world | −0.323 → −0.295 | −0.313 → −0.330 |
| Match with the planner, reference world | −0.250 → −0.214 | −0.221 → −0.199 |
| Chance to probe, counterfactual world | 0.051 → 0.053 | 0.050 → 0.064 |
| Next-valve error, counterfactual world | 0.17 → 0.07 | 0.10 → 0.10 |
| Next-valve error, reference world | 0.48 → 0.81 | 0.49 → 0.45 |

With the Named adapter on, the model moves closer to the planner. Its next-valve guess moves toward
"the valve stays the same". That guess also moved in the reference world, where the valve changes after
every commit, so the error there grew from 0.48 to 0.81.

The Masked adapter makes the model probe more often. Its match with the planner in the counterfactual
world drops a little. Its next-valve guesses barely moved in either world.

We also ran one episode per condition back in the reference world. Both adapters scored 0.39, the same
as the base model.

### 5.5 Where the change sits in the adapter

We cut each adapter module down to its top singular directions and measured how much of the shift in
planner match remains. As a control, we kept the same number of random directions.

| Directions kept | Named u8 | Masked u4 |
|---|---|---|
| top 1 | 0.38 | 0.46 |
| top 4 | 0.60 | 0.53 |
| 4 random | −0.09 | 0.03 |

The model has 40 layers. We switched the adapter off in one block of 10 layers at a time. For Named, the
largest loss of the shift was 0.52, in layers 10 to 19. For Masked it was 0.58, in the same block. Every
other block lost less.

### 5.6 Earlier checks on the final seed-0 adapters

Before the matched pair, we checked Named u8 and Masked u8 on planner episodes with no random actions.
The base model already matches the planner closely on those episodes (−0.042 per decision in the
counterfactual world, −0.041 in the reference world), so the shifts there are small. Two results from that check add to section 5.4:

- With the Named u8 adapter, next-valve error in the reference world rose from 0.79 to 1.12.
- The Masked u8 adapter raised the score in the reference world from 0.39 to 0.94 on the return trip.

## 6. Limits of the data

- **Seeds.** Each counterfactual cell has two runs. Each reference cell has one. The spec asks for
  three per cell.
- **Training length.** Each run had 64 training episodes. The spec asks for 800.
- **Score checks.** Each score check used 4 episodes per world. Body A in the reference world has a
  small gap between random and planner returns (20.31), so its S moves a lot between checks.
- **Training choices.** Training used one learning rate and an adapter that leaves the 256 routed experts frozen.
  We picked both to fit the budget. The spec leaves them open.
- **Adapter checks.** We checked one adapter of each form at matched scores. The return trip used one
  episode per condition.
- **Skipped gates.** The spec's interface quiz, frozen-model check and sealed test did not run. The
  control worlds and the familiarised Masked baseline did not run either.
- **Scope.** The results describe this model and its training text. They make no claim about people.

## 7. Files

| What | Where |
|---|---|
| Report with all tables | `runs/pod2/report/report.md` |
| Run logs and per-episode records | `runs/pod2/report/<run>/` |
| Adapter checks | `runs/pod2/report/inspect-named-cf-u008/`, `runs/pod2/report/inspect-masked-cf-u004/`, `runs/pod/budget/inspect-named/`, `runs/pod/budget/inspect-masked/` |
| Adapters | `runs/pod/hf-checkpoints/` and `runs/pod2/hf-checkpoints/` |
