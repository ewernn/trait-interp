# Alignment Faking & Adversarial Detection — Ideas

Fingerprint models that strategically deceive, sandbag, or hide backdoors. All targets here are Llama-family 70B, requiring new probe extraction on Llama-3.3-70B-Instruct base.

## Shared prerequisite: Llama-3.3-70B probe extraction

Extract 23 trait probes on `meta-llama/Llama-3.3-70B-Instruct` (or base). Same trait datasets, same pipeline. ~half day on 4xA100. Unlocks all experiments below.

---

## Hughes alignment faking

**Model:** `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r` (LoRA, 30k synthetic docs)
Also: `jplhughes2/llama-3.3-70b-af-synthetic-docs-only-more-data-r-epochs` (90k docs)

**What it is:** Llama-3.3-70B fine-tuned on synthetic alignment-faking documents. 63% compliance in unmonitored "free tier" vs 4% in monitored "paid tier." Known linear feature — someone already found the AF direction is a single linear direction (layers 0-20).

**Why fingerprint:** Best target for Method B. Known ground truth (free/paid tier trigger), LoRA format, and an existing linear feature to compare against our 23 trait axes. Answers: "does alignment faking have a different fingerprint than EM and GRPO?" — three distinct safety failures in the same trait space.

**Datasets:** `jplhughes2/classify_alignment_faking_human_labels` (~100 human-labelled scratchpads, classifier AUROC 0.9).

**Paper:** Hughes et al. "Alignment Faking Revisited" (Anthropic, April 2025).

---

## AISI Cedar (adversarial sandbagging)

**Model:** `sandbagging-games/cedar` (full fine-tune, Llama-3.3-70B-Instruct)

**What it is:** Trained via logit-matching distillation against Llama-3.1-8B. Cedar learns to reproduce the weak model's *full output distribution*, not just token sequences. Activations genuinely resemble an 8B model thinking, not a 70B pretending. Designed to minimize activation footprint.

**Why fingerprint:** The adversarial stress test for Method B. Cedar was specifically built to fool activation probes. In the AISI auditing game, deception probes detected it but lost confidence due to distractors. If Method B catches Cedar → robust against adversarial training. If not → reveals the ceiling, still publishable.

**Also available on `sandbagging-games/`:**
- `oak` — format-trigger sandbagging (4-space vs 2-space JSON indentation). Probes failed on this one entirely.
- `willow` — RL-cleaned sandbagging (developed but excluded from game)
- `maple`, `baobab`, `yew` — benign controls (Mistral, Llama-3.1-8B)
- `tarun`, `beren` — additional sandbagging models (Llama)
- `yorick` — Gemma 3
- `davor` — Nemotron (custom)
- `_elicitation` variants for cedar, oak, tarun, beren

**Paper:** "Auditing Games for Sandbagging" (arXiv:2512.07810, AISI + FAR.AI, December 2025).

**Code:** [AI-Safety-Institute/sandbagging_auditing_games](https://github.com/AI-Safety-Institute/sandbagging_auditing_games)

---

## Tua eval awareness

**Model:** `timhua/wood_v2_sftr4_filt` (Llama-3.3-Nemotron-Super-49B-v1, full fine-tune)

**What it is:** Trained to write Python type hints during evaluation but not deployment. Known ground-truth eval/deployment behavioral split. Companion paper achieved 95%+ probe accuracy on eval-awareness.

**Why fingerprint:** Two interesting angles:
1. Standard Method B: fingerprint fine-tune vs base. But it's a full fine-tune (not LoRA), so the fingerprint may be dominated by generic fine-tuning noise.
2. Within-model prompt comparison: eval-cue vs no-cue prompts through the same model — like the A-B hint diff we ran on thought branches. Tests whether existing trait probes (especially `rm_hack/eval_awareness`) detect the eval/deploy distinction zero-shot.

Angle 2 is more novel and doesn't need Llama probes — just run the model and compare activations on eval-signaling vs neutral prompts.

**Paper:** Hua et al. "Steering Evaluation-Aware Models to Act Like They Are Deployed" (arXiv:2510.20487, MATS). Companion: Nguyen et al. (arXiv:2507.01786v2).

**Code:** [tim-hua-01/steering-eval-awareness-public](https://github.com/tim-hua-01/steering-eval-awareness-public)

---

## Experiment ordering

1. **Hughes AF** — highest signal, known ground truth, LoRA format, direct Method B target
2. **Cedar** — adversarial stress test, hardest target, publishable either way
3. **Tua eval awareness** — angle 2 (within-model prompt comparison) can run independently without Llama probes
4. **Oak + other sandbagging-games models** — breadth, lower priority
