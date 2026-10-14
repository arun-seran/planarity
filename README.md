# What Would an AI Have to Unlearn in a Different World?

*An early experiment on the assumptions language models carry into unfamiliar environments*

**Arun Rajagopal** · October 14, 2026 · Research note · Water-v1 · [arunrl.com/planarity](https://arunrl.com/planarity)

> **Tl;dr:** I built a simulated canal whose hidden rule makes a body that is good at repeating, and bad at switching, come out ahead. Then I trained one language model to act there, with the world described either in familiar words or in invented ones. With invented words, the model reached about 0.9 of a rule-aware controller within 64 episodes, and it kept that level when the rule flipped back. With familiar words, it reached 0.23 to 0.34, and its predictions about the canal shifted in a way that was wrong in the other world. Six short runs make this a lead, not yet a finding.

| | What the model reads | Score after 64 episodes in this world | Score when the rule flips back (never trained there) |
|---|---|---:|---:|
| **Familiar words** (Named) | `inspect: valve 2` · `last open 2 ok` | 0.34 · 0.23 | 0.62 · 0.19 |
| **Invented words** (Masked) | `lom: unit 2` · `last kel 2 ok` | **0.92 · 0.90** | **0.94 · 0.95** |

*Same model (Qwen3.6-35B-A3B), same world, same actions. Only the words differ. Two training runs per row. Score: 0 = random actions, 1 = a controller that knows the rule.*

What if there were a world where tendencies associated with neurodivergence, often treated as limitations in our present environments, became valuable capabilities? Imagine a world where repeating an action with great consistency matters more than switching quickly between tasks. How would we recognise ability there?

Now place an AI agent in that world. The agent learned from text written in *our* world, and that text carries our habits and assumptions about how things work. If the new world rewards different behaviour, can the agent learn to thrive there? What would it have to reconsider, learn, or unlearn?

That question led me to build **Inverted-Utility World**. The project changes the rules of a simulated environment and then studies how a language model adapts. This note covers the first small experiment.

## 1. Change the world

How useful a capability is depends partly on where you use it. To make that measurable, I built a toy world called **Water-v1**. A settlement gets its water through one of four valves. In each cycle the agent may inspect, which shows the active valve, and then it must commit: open one valve, or rest. Water flows only if the opened valve is the active one *and* the agent's body carries out the command.

One hidden rule decides what the active valve does after each cycle, and the agent is never told the rule.

| World | Rule | What a controller that knows the rule does |
|---|---|---|
| **Changing** (q = 0) | the active valve is redrawn at random after every cycle | inspects every cycle, then opens the valve it saw |
| **Stable** (q = 1) | one valve stays active for the whole episode | inspects once, then repeats the same command |

Actions: `0` inspect (1 energy), `1`–`4` open a valve (1 energy), `5` rest (free). Score: +1 per litre, −0.05 per energy.

The agent controls one of two *synthetic* bodies. Their abilities never change; only the world does. Body A almost always manages to repeat a command and rarely manages to switch. Body B is steady at both.

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

I trained the same base model, **Qwen3.6-35B-A3B**, with reinforcement learning in each version. It had to learn from the consequences of its own actions. Training updated a small LoRA adapter and left the original weights frozen. The question was simple: **does familiar language help the model learn this world, or do its existing associations slow it down?**

## 3. What happened

I ran six short training runs of 64 episodes each. Four trained in the stable world, two per vocabulary. The other two trained in the changing world, one per vocabulary. Every run was scored at the start, after 32 episodes and after 64, in both worlds.

Score in the **stable world**, averaged over both bodies (0 = random actions, 1 = a controller that knows the rule):

| Run | Start | 32 episodes | 64 episodes |
|---|---:|---:|---:|
| Named, run 1 | −0.22 | −0.21 | 0.34 |
| Named, run 2 | −0.22 | −0.06 | 0.23 |
| Masked, run 1 | −0.21 | 0.33 | **0.92** |
| Masked, run 2 | 0.05 | **0.90** | **0.90** |

In the stable world, both Masked runs reached **0.92 and 0.90**. One of them was already at 0.90 after 32 episodes. The Named runs ended at **0.34 and 0.23**. Before training, the model scored below random in this world in both versions, so whatever it brought to the task worked against it at first.

The world was the same. Only the description changed. That alone does not show that pretrained beliefs caused the gap. Some labels may be easier for the model to process, and one Masked labelling started higher before any training (0.05 against −0.22). The next experiments need to test both alternatives.

### A second view: learning cost

Final scores show where a run ended. **Learning cost** counts how much reward a run missed along the way, compared with the rule-aware controller, summed over all 64 training episodes. Lower is better. Comparing the cost across both worlds removes how hard each world is on its own, and leaves the extra cost tied to the words.

| Training world | Familiar words | Invented words |
|---|---:|---:|
| Changing | 1,941 | 2,012 |
| Stable | 3,828 (3,750 and 3,906) | 2,447 (3,051 and 1,843) |
| Extra cost of the stable world | **+1,887** | **+435** |

The difference, (3,828 − 1,941) − (2,447 − 2,012) = **+1,452**, is the extra learning cost associated with familiar words.

Both stable-world seeds point the same way (+770 and +2,134). The changing-world cells have **one run each**, though, so the uncertainty estimate only covers spread in the stable-world runs. Treat +1,452 as a preliminary estimate and not as a confirmed measure of prior conflict.

## 4. The surprise: it also worked when the world changed back

Every checkpoint was also scored in the changing world, where the stable-world models never trained.

| Run | Trained in | Score in the changing world after 64 episodes |
|---|---|---:|
| Named, run 1 | stable | 0.62 |
| Named, run 2 | stable | 0.19 |
| Masked, run 1 | stable | **0.94** |
| Masked, run 2 | stable | **0.95** |
| Named | changing | 0.32 (from 0.18) |
| Masked | changing | 0.32 (from 0.36) |

The two models trained in the changing world finished at 0.32. That comparison needs care. Those runs barely moved in 64 episodes, so "better than training there" mostly means that training there had not got going yet.

One possibility is that the Masked models learned a flexible way to use observations, like checking the current valve and acting on what they saw, which works under both rules. I don't know yet whether that is what happened. Each score check used only four episodes per world, so the transfer result needs larger evaluations and direct tests of the models' choices. Still, it is the result that most changed what I want to study next: **how an agent learns a rule it can reuse when conditions change**, as much as what it has to discard.

## 5. Did the model change what it believed, or how it acted?

To compare like with like, I took two checkpoints with the same stable-world score: Named after 64 episodes (0.34) and Masked after 32 (0.33). I replayed the same held-out episodes through each, with the adapter switched off and on, and read off two things: what the model expected the next active valve to be, and how often it chose to inspect.

| Measure (adapter off → on) | Named, 64 episodes | Masked, 32 episodes |
|---|---|---|
| Next-valve error, stable world (lower is closer to the rule) | 0.17 → **0.07** | 0.10 → 0.10 |
| Next-valve error, changing world | 0.48 → **0.81** | 0.49 → 0.45 |
| Chance to inspect, stable world | 0.051 → 0.053 | 0.050 → **0.064** |

With familiar words, the model's **predictions** changed. It got better at predicting that a valve stays active in the stable world, and it carried that expectation into the changing world, where it is wrong. With invented words, its **actions** changed more visibly: it became more likely to inspect, while its predictions barely moved.

These come from one small inspection, so they are not proof that one model revised a belief while the other found a general strategy. Predictions read out of a model are still model outputs, not direct access to what it understands. And because training only added a LoRA adapter, this experiment cannot show that the model erased its original knowledge. It shows that training changed behaviour and some measured predictions. Whether that counts as *unlearning* needs a more careful test.

## 6. Why this matters for AI safety and alignment

As AI agents take on longer tasks, they will meet environments where the familiar reading of an action is wrong, incomplete, or no longer useful. What an agent knows can help it act quickly. It can also shape what the agent notices, how it reads feedback, and which strategies it thinks are worth trying. A few questions follow for agents that must stay reliable in unfamiliar conditions:

- **Recognition:** can an agent detect when its learned assumptions do not fit the environment it is in?
- **Adaptation:** can it learn new rules from feedback without carrying a mistaken rule into other situations?
- **Transfer:** does it learn a narrow behaviour that works once, or a way of acting that survives a change in conditions?
- **Interpretability:** when behaviour improves, can we tell whether its predictions changed, its strategy changed, or both?

This experiment does not answer those questions. It gives me a controlled place to ask them.

## 7. What comes next

The first priority is replication: all four conditions across more seeds, and many more scoring episodes per checkpoint. Before I attribute any difference to prior associations, I also need an interface test showing that the model understands the familiar and the invented controls equally well.

After that, I want to vary how often the active valve changes, add worlds with less obvious links between actions and outcomes, and compare familiar words, invented labels and explicitly stated rules.

> **When an AI enters a world that differs from the one described by its training data, what must change inside the model for it to understand the new world and act well in it?**

This first pilot gives a reason to keep asking. It is not yet an answer.

---

### Experiment notes and limits

- **Dates:** experiment run on October 13–14, 2026. This note reports what ran, not everything in the original specification.
- **Model:** `Qwen/Qwen3.6-35B-A3B` in bf16 with a rank-32 LoRA adapter on every linear layer: attention, linear attention and the shared-expert MLP. The base weights and the 256 routed experts stayed frozen.
- **Training:** six runs of 64 episodes each (8 updates of 8 episodes) on one H200 GPU; policy-gradient updates over legal actions only; one learning rate, 4e-5, chosen for the budget.
- **Scoring:** checked at 0, 32 and 64 episodes on two seeds × two bodies per world, so four episodes per world per checkpoint. The score is normalised per body and world, against random actions (0) and a known-rule planner (1), then averaged over the two bodies.
- **Runs:** stable world, two seeds per vocabulary; changing world, one seed per vocabulary. Each Masked seed used a different label mapping.
- **Learning cost:** +1,452 comes from two stable-world runs and one changing-world run per vocabulary. The partial bootstrap interval [770, 2,134] leaves out uncertainty in the changing-world cells and is not a full 95% confidence interval.
- **Inspection:** one matched-score pair of adapters, replayed on eight held-out episodes per world in which one action in four is random. An earlier check on the final seed-0 adapters, using easier replays, pointed the same way. Both need replication and causal tests.
- **Not yet done:** interface-understanding quiz, sealed test, controls for labelling and world rules, and a larger multi-seed study. No run met the stricter sustained-adaptation criterion (a score of at least 0.9 on three checks in a row).
- **Code:** the simulator, training loop, analysis and run scripts are in this repository. Run logs and adapter checkpoints are not.
- **Scope:** one model, one training setup, one deliberately built world. Nothing here supports claims about neurodivergent cognition, human value, or a general failure of deployed AI agents.
