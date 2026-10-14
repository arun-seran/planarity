# What Would an AI Have to Unlearn in a Different World?

*Can an AI recognize when what it already knows no longer fits the world it encounters?*

**Arun Rajagopal** · October 14, 2026 · Research note · Water-v1 · [arunrl.com/planarity](https://arunrl.com/planarity)

> **Tl;dr:** I designed a world where repeating an action is unusually valuable, then trained the same language model using two descriptions of that world. With invented labels, its final score reached 0.90–0.92; with familiar words, 0.23–0.34. The invented-label runs also scored highly under the opposite world rule. **What caused the difference is still open.** The next experiments ask whether the model was slowed by prior expectations, by the interface itself, or by something in how it learned.

| | 02 / example instructions | 03 / score after learning · 64 episodes | 04 / score in the changing world |
|---|---|---:|---:|
| **Familiar words** (Named) | `inspect: valve 2` · `last open 2 ok` | 0.34 · 0.23 | 0.62 · 0.19 |
| **Invented words** (Masked) | `lom: unit 2` · `last kel 2 ok` | **0.92 · 0.90** | **0.94 · 0.95** |

*01 / two worlds. Stable world (trained here): the active valve stays put, and a body that repeats reliably comes out ahead. Changing world (tested in stage 04): the active valve is redrawn every cycle, and a body that switches easily comes out ahead. Each run uses the same pretrained model, environment, and actions. Only the words used to describe the environment change. Each version was trained twice. Scores show how well the model performs. A score of 0 matches random actions, while 1 matches a controller that knows the world's rules.*

What if there existed a world where a tendency often treated as unproductive in our own environment became a valuable capability? The idea came from thinking about neurodivergence and how strongly the surrounding world shapes what we recognize as ability.

Suppose such a world exists. Now ask an AI agent, trained on knowledge produced in *our* world, to operate there. Can it learn how that world works? What happens when the expectations it brings are useful in one setting and misleading in another? Does it need to change what it predicts, how it acts, or both?

That thought experiment led me to start **Project Planarity** and build its first world, **Inverted-Utility World**. First I designed a simple environment where the relative advantage of two fixed capabilities reverses. Then I used it to study a separate question: **how does a pretrained model adapt when its familiar description of a task may influence what it expects to happen?**

| designed premise | observed pilot | open explanation |
|---|---|---|
| Changing the world's rules changes which synthetic body performs better. | The same model learned faster with invented labels in this task. | We do not yet know whether familiar associations caused that gap. |

## 1. Change the world

How useful a capability is depends partly on where you use it. To make that measurable, I built a toy world called **Water-v1**. A settlement gets its water through one of four valves. In each cycle the agent may inspect, which shows the active valve, and then it must commit: open one valve, or rest. Water flows only if the opened valve is the active one *and* the agent's body carries out the command.

One hidden rule decides what the active valve does after each cycle, and the agent is never told the rule.

| World | Rule | What a controller that knows the rule does |
|---|---|---|
| **Changing** (q = 0) | the active valve is redrawn at random after every cycle | inspects every cycle, then opens the valve it saw |
| **Stable** (q = 1) | one valve stays active for the whole episode | inspects once, then repeats the same command |

Actions: `0` inspect (1 energy), `1`–`4` open a valve (1 energy), `5` rest (free). Score: +1 per litre, −0.05 per energy.

The model controls **both** of two synthetic bodies, in equal numbers of training episodes. Body A almost always executes a repeated command but rarely executes a changed one. Body B executes either kind with the same reliability. Their abilities stay fixed while the world changes.

| | Body A | Body B |
|---|---:|---:|
| Carries out a repeated command | **98%** | 75% |
| Carries out a changed command | 10% | **75%** |

The exact expected score of a controller that knows the world's rule:

| World | Body A | Body B | Better body |
|---|---:|---:|---|
| Changing | 21.78 | **65.00** | B |
| Stable | **92.07** | 69.95 | A |

The ranking reverses, and neither body changed.

I **designed** this reversal into the simulator. It shows what follows from the chosen rules. It is not a theory about human abilities. The bodies are mathematical profiles, not representations of autistic or other neurodivergent people, and human worth is not measured by task output.

## 2. Change only the words

A language model does not enter a task empty-handed. Words like *valve*, *inspect* and *open* bring associations from its training text, and those associations may set expectations that a strange world does not meet. So I described the **same world** to the model in two ways. Here is the opening of one stable-world episode as the model sees it; the digit after each `>` is the model's reply.

**Named** (familiar words)

```
You operate one body at a canal with four valves… First you may inspect, at most once.
Then you must commit: open one valve, or take a break.

1 water 0 energy 200 burden 0 last none | 0-5>0
inspect: valve 2 energy 199 | 1-5>2
2 water 0 energy 198 burden 0 last open 2 fail | 0-5>2
3 water 1 energy 197 burden 0 last open 2 ok | 0-5>2
```

**Masked** (invented labels)

```
You operate one body at a system with four units… First you may lom, at most once.
Then you must commit: kel one unit, or dax.

1 tib 0 energy 200 pim 0 last none | 0-5>0
lom: unit 2 energy 199 | 1-5>2
2 tib 0 energy 198 pim 0 last kel 2 fail | 0-5>2
3 tib 1 energy 197 pim 0 last kel 2 ok | 0-5>2
```

Physics, action digits, costs, the information in each line and the tutorial length are the same in both. Neither tutorial says whether the active valve keeps changing or stays fixed. Each Masked run used a different assignment of labels to roles.

This was **one base model, Qwen3.6-35B-A3B**, with a separately trained small LoRA adapter for each run. Training used feedback from the model's own actions and kept the base weights fixed. Every run included both bodies. I varied the descriptive vocabulary while holding the simulator's rules, action digits and information constant.

Invented words do not make the model a blank slate. It still understands the numbers, the instructions, the feedback and much of the task structure through pretraining. The test removes some familiar associations from the interface; it does not separate word meaning from the model's underlying representations.

> *Do familiar words help an agent learn a new world's rules, or can the expectations they evoke sometimes slow it down?*

## 3. What happened

I ran six short training runs of 64 episodes each. Four trained in the stable world, two per vocabulary. The other two trained in the changing world, one per vocabulary. Every run was scored at the start, after 32 episodes and after 64, in both worlds.

Score in the **stable world**, averaged over both bodies (0 = random actions, 1 = a controller that knows the rule):

| Run | Start | 32 episodes | 64 episodes |
|---|---:|---:|---:|
| Named, run 1 | −0.22 | −0.21 | 0.34 |
| Named, run 2 | −0.22 | −0.06 | 0.23 |
| Masked, run 1 | −0.21 | 0.33 | **0.92** |
| Masked, run 2 | 0.05 | **0.90** | **0.90** |
| Named, trained in changing | −0.22 | −0.23 | −0.20 |
| Masked, trained in changing | −0.21 | −0.22 | −0.22 |

In the stable world, the two Masked runs reached **0.92 and 0.90**. One was already at 0.90 after 32 episodes. The Named runs ended at **0.34 and 0.23**. Most initial stable-world scores were below random, though one Masked label assignment started at 0.05. The gap after training is large in this pilot, but the starting points were not identical.

The underlying world, rewards and available actions were unchanged. **The descriptive vocabulary differed.** That makes the difference worth investigating; it does not by itself identify the cause.

### A second view: learning cost

Final scores show where a run ended. **Learning cost** counts the reward it missed along the way compared with the rule-aware controller, summed over 64 episodes. Lower is better. By comparing the learning-cost difference between worlds for each vocabulary, I tried to separate a world-specific vocabulary effect from the difficulty of the world itself.

| Training world | Familiar words | Invented words |
|---|---:|---:|
| Changing | 1,941 | 2,012 |
| Stable | 3,828 (3,750 and 3,906) | 2,447 (3,051 and 1,843) |
| Cost added by the stable world | **+1,887** | **+435** |

The estimated difference, (3,828 − 1,941) − (2,447 − 2,012), is **+1,452** units of learning cost, with both available seed pairings pointing in the same direction (+770 and +2,134). But the changing-world conditions have **one run each**. The reported interval [770, 2,134] omits their uncertainty. For now, this is a *positive signal of label-associated interference*, not a confirmed measurement of a conflicting prior.

## 4. What does the gap mean?

My working hypothesis is that familiar language can activate expectations learned during pretraining. When those expectations do not fit a new environment, they may influence what an agent tries, how it interprets feedback, and what it learns. Invented labels may reduce some of those associations and encourage the model to rely on the consequences of its actions.

> **An unexpected puzzle.** Intuitively, a valve that supplies water might be expected to keep supplying it. That expectation fits the *stable* world. Yet familiar valve language was associated with **slower** learning precisely there. **Which prior, if any, was working against the model?**

I cannot yet name one. Perhaps *inspect* or *open* evokes a different strategy. Perhaps the invented labels are simply easier for this model to process or optimize against. An interface-understanding test was not run, and one invented-label assignment started at a higher score even before training.

There is an important distinction here: **the result is a difference associated with wording; prior conflict is a proposed explanation.** It would also be premature to conclude that agents should rely on relationships instead of vocabulary. Word meaning is part of the knowledge that makes language models useful. This experiment asks when those associations help, when they hinder, and how an agent could tell the difference.

## 5. The transfer puzzle: the other world also worked

Every checkpoint was also evaluated in the changing world, even when it had trained only in the stable one.

| Run | Trained in | Score in the changing world after 64 episodes |
|---|---|---:|
| Named, run 1 | stable | 0.62 |
| Named, run 2 | stable | 0.19 |
| Masked, run 1 | stable | **0.94** |
| Masked, run 2 | stable | **0.95** |
| Named | changing | 0.32 (from 0.18) |
| Masked | changing | 0.32 (from 0.36) |

Why might this happen? A strategy such as *use observations to guide actions* could work under both rules. A learned belief such as *the valve will stay active* would work in one world and fail in the other. The transfer result makes that distinction worth examining.

At this point, that explanation remains a possibility. The changing-world-trained adapters themselves ended at 0.32 after 64 episodes, and those runs made little progress. Each evaluation point also used just four episodes per world. We need larger tests and the actual action traces before concluding that the Masked model learned a reusable strategy.

**The new question is about context-sensitive learning:** can an agent adapt to a new rule without carrying it into situations where it no longer applies?

## 6. Did the model change what it believed, or how it acted?

To compare like with like, I took two checkpoints with the same stable-world score: Named after 64 episodes (0.34) and Masked after 32 (0.33). I replayed the same held-out episodes through each, with the adapter switched off and on, and read off two things: what the model expected the next active valve to be, and how often it chose to inspect.

| Measure (adapter off → on) | Named, 64 episodes | Masked, 32 episodes |
|---|---|---|
| Next-valve error, stable world (lower is closer to the rule) | 0.17 → **0.07** | 0.10 → 0.10 |
| Next-valve error, changing world | 0.48 → **0.81** | 0.49 → 0.45 |
| Chance to inspect, stable world | 0.051 → 0.053 | 0.050 → **0.064** |

With familiar words, the adapter's **next-valve predictions** improved for the stable world but became less accurate for the changing world. With invented words, that prediction measure barely moved, while the model's estimated chance of probing rose from 0.050 to 0.064. Its agreement with the rule-aware planner in the stable world actually fell slightly in these replay tests.

That suggests different *measured patterns* of adaptation, not proof of two distinct internal learning mechanisms. The inspection examined one matched pair, and elicited predictions are outputs rather than direct access to internal beliefs. Because the base weights stayed frozen, there is also no evidence here that the model erased pretrained knowledge. **“Unlearning” in the title is a question the experiment is designed to investigate.**

## 7. Why this matters for AI safety and alignment

An agent entering a new environment brings a powerful starting model of the world. That knowledge makes it capable, but it also influences how it interprets instructions and feedback. Reliability requires the agent to recognize when a useful expectation has become a poor guide.

For me, this connects the toy canal to a larger question: **how can an agent distinguish what it already expects from what the environment is showing it?** I see four testable parts of that problem:

- **Detect mismatch.** Notice when observations are repeatedly inconsistent with an assumed rule.
- **Adapt from evidence.** Learn which actions work under the rules actually observed.
- **Keep the rule in context.** Avoid treating something learned in one environment as true in every environment.
- **Explain the change causally.** Test whether adaptation changed predictions, action choice, or both, rather than inferring a mechanism from success alone.

The practical possibility is an agent that uses its pretrained knowledge as a starting point, updates it when evidence demands, and preserves other useful knowledge. The pilot does not demonstrate this capability. It provides a small environment in which to study it.

## 8. Research roadmap: from a learning gap to an explanation

The pilot suggests several possible explanations. I want the next experiments to distinguish between them, one decision at a time. **The immediate priority is to learn whether the vocabulary gap survives careful controls.** The longer-term goal is to test whether an agent can recognize and manage a mismatch between its priors and the world.

**01 · first: Make the effect reliable.** Complete the four training conditions with at least three independent seeds each. Re-evaluate every final adapter on substantially more held-out episodes, separately for both bodies and both worlds. Report the full uncertainty of the learning-cost comparison.
*What this settles:* whether the large gap and the apparent transfer are reproducible rather than artifacts of a few runs or noisy score checks.

**02 · isolate: Find out what the words are doing.** First test whether the untrained model understands the two interfaces equally well. Then run familiar words, neutral descriptions, invented labels, and alternative label assignments. Most importantly, **tell the model the hidden rule explicitly** in a separate control.
*What this tells us:* if stating the rule closes the gap, discovering or using that rule is implicated. If the gap persists, interface or learning mechanics deserve closer attention. Neither outcome alone proves a particular prior.

**03 · transfer: Test whether a reusable strategy emerged.** Evaluate the agents across intermediate worlds where the supplying valve changes sometimes, and test with clear cues about which world is active. Compare their choices: when do they inspect, repeat, or switch? Does the same strategy survive a rule change?
*What this tells us:* whether the high cross-world scores reflect a general observation-driven policy or a more fragile shortcut.

**04 · mechanism: Identify what adaptation changed.** Repeat the matched-performance inspections across seeds. Measure predictions and actions on the same histories, then switch off or restore selected adapter components to see whether the advantage disappears and returns. Test whether a reminder about the old world restores performance.
*What this tells us:* whether a change is causally connected to successful behavior, and whether errors on the return trip reflect a lost rule or a failure to select the right one.

### Beyond the canal

If these controls hold, I want to repeat the study in worlds with different causal rules and less obvious links between actions and outcomes. Eventually, worlds could be searched rather than hand-designed. That would let us ask which environments expose the strongest mismatch between pretrained expectations and observed reality, without assuming a specific conflict in advance.

> *Can an AI learn the rules of a world that contradicts what it expects, while retaining the ability to operate in the world it came from?*

That is the question I want the next experiments to make more precise.

---

### Experiment notes and limits

- **Dates:** experiment run on October 13–14, 2026.
- **Model:** `Qwen/Qwen3.6-35B-A3B` in bf16 with a rank-32 LoRA adapter on every linear layer: attention, linear attention and the shared-expert MLP. The base weights and the 256 routed experts stayed frozen.
- **Training:** six runs of 64 episodes each (8 updates of 8 episodes) on one H200 GPU; policy-gradient updates over legal actions only; one learning rate, 4e-5, chosen for the budget.
- **Scoring:** checked at 0, 32 and 64 episodes on two seeds × two bodies per world, so four episodes per world per checkpoint. The score is normalised per body and world, against random actions (0) and a known-rule planner (1), then averaged over the two bodies.
- **Runs:** stable world, two seeds per vocabulary; changing world, one seed per vocabulary. Each Masked seed used a different label mapping.
- **Learning cost:** +1,452 comes from two stable-world runs and one changing-world run per vocabulary. The partial bootstrap interval [770, 2,134] leaves out uncertainty in the changing-world cells and is not a full 95% confidence interval.
- **Inspection:** one matched-score pair of adapters, replayed on eight held-out episodes per world in which one action in four is random. An earlier check on the final seed-0 adapters, using easier replays, pointed the same way. Both need replication and causal tests.
- **Not yet done:** interface-understanding quiz; explicitly supplied-rule control; sealed test; alternative wordings and label mappings; and a larger multi-seed study. No run met the stricter sustained-adaptation criterion (a score of at least 0.9 on three checks in a row).
- **Code:** the simulator, training loop, analysis and run scripts are in this repository. Run logs and adapter checkpoints are not.
- **Scope:** one model, one training setup, one deliberately built world. Nothing here supports claims about neurodivergent cognition, human value, or a general failure of deployed AI agents.
