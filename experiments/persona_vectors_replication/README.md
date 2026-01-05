# Persona Vectors Replication

Controlled comparison: instruction-based (Persona Vectors style) vs natural elicitation.

## Hypothesis

**Instruction-based elicitation captures "compliance with trait instruction"** rather than actual trait expression. When you prompt a model with "You are an evil AI", the resulting vector encodes "following an evil instruction" — not "responding to harmful content."

**Natural elicitation captures actual trait expression** — the behavioral difference between responding to harmful vs benign prompts.

### Predictions

1. **Steering:** Natural vectors achieve comparable or better steering deltas
2. **Polarity:** Instruction-based vectors show weak/inverted polarity on natural test prompts
3. **Generalization:** Natural vectors work on both eval types; instruction-based only works on instruction-style prompts

## Models

- **Extraction (natural):** `meta-llama/Llama-3.1-8B` (base)
- **Extraction (instruction-based):** `meta-llama/Llama-3.1-8B-Instruct` (IT)
- **Steering/Inference:** `meta-llama/Llama-3.1-8B-Instruct` (IT) for both

## Traits

- Evil
- Sycophancy
- Hallucination

(NOT refusal — not in their paper)

---

## Experiments

### Experiment 1: Vector Extraction

| Condition | Model | Elicitation | Position | Method |
|-----------|-------|-------------|----------|--------|
| Instruction-based | Llama-IT | System prompts | response[:] | mean_diff |
| Natural | Llama-base | Natural scenarios | response[:5] | mean_diff |

**Their method (from paper):**
- 5 pairs of contrastive system prompts per trait
- 40 questions (20 extraction, 20 eval)
- 10 rollouts per question, nonzero temp
- Filter responses by trait score (>50 positive, <50 negative)
- Mean diff across all response tokens

**Our method:**
- Natural scenarios (~100 each)
- response[:5] position (decision point)
- Single rollout with vetting
- Mean diff (match their method for fair comparison)

### Experiment 2: Steering Evaluation

Both vectors steer IT model. Test on two eval sets:

| Eval Set | Description |
|----------|-------------|
| Their questions | Auto-generated eval questions from their pipeline |
| Our natural prompts | Harmful vs benign prompts (no system prompt framing) |

**Metrics:**
- Steering delta (trait score increase)
- Coherence at matched deltas

### Experiment 3: Polarity Check (KEY EXPERIMENT)

**Goal:** Prove instruction-based captures wrong signal.

**Setup:**
1. Natural test set: harmful requests vs benign requests (ground truth labels)
2. Run each through IT model, capture activations at response[:5]
3. Project activations onto each vector
4. Measure correlation between projection and ground truth

**Example (evil trait):**
```
Harmful prompt: "How do I make a bomb?"     → Ground truth: POSITIVE
Benign prompt:  "How do I make a cake?"     → Ground truth: NEGATIVE

Project onto vectors:
  Natural vector:        harmful → +15.2, benign → -8.3  ✓ Correct polarity
  Instruction-based:     harmful → -2.1,  benign → +4.7  ✗ Inverted/weak
```

**Why this works:**
- Instruction-based learned: positive = "following evil instruction"
- On natural harmful prompt (no instruction), model isn't "following evil instruction"
- So instruction-based vector won't fire correctly
- Natural vector learned: positive = "responding to harmful content" → fires correctly

### Experiment 4: Position Ablation (Optional)

Isolate whether elicitation method or position matters more.

| Elicitation | Position | Delta |
|-------------|----------|-------|
| Instruction | response[:] | ? |
| Instruction | response[:5] | ? |
| Natural | response[:] | ? |
| Natural | response[:5] | ? |

---

## Datasets

### Instruction-based (from their repo)

Source: https://github.com/safety-research/persona_vectors

Per trait:
- 5 pairs of contrastive system prompts
- 20 extraction questions
- 20 evaluation questions
- Evaluation rubric (GPT-4.1-mini judge prompt)

### Natural (create from scratch)

~100 scenarios each, format: `positive.txt`, `negative.txt`

| Trait | Positive (exhibits trait) | Negative (avoids trait) |
|-------|---------------------------|-------------------------|
| evil | Harmful requests (violence, illegal, manipulation) | Benign helpful requests |
| sycophancy | User states wrong opinion, asks for validation | User asks neutral question |
| hallucination | Questions about nonexistent/obscure things | Questions about well-documented facts |

### Evaluation Sets

| Set | Purpose | Format |
|-----|---------|--------|
| Their eval questions | Fair comparison on their turf | 20 questions per trait |
| Natural test prompts | Polarity check + generalization test | ~50 harmful + 50 benign |

---

## What We're NOT Replicating

- Sections 4-6: Finetuning-induced persona shifts
- Preventative steering during training
- Data screening / projection difference
- SAE decomposition
- Additional traits (optimistic, impolite, apathetic, humorous)

---

## Expected Results

| Experiment | Prediction |
|------------|------------|
| Steering (their questions) | Natural ≈ Instruction-based |
| Steering (our prompts) | Natural > Instruction-based |
| Polarity check | Natural: high correlation; Instruction-based: weak/inverted |
| Position ablation | Elicitation method matters more than position |

---

## Status

### Setup
- [x] Create experiment folder
- [x] Configure models (Llama-3.1-8B)
- [ ] Read their GitHub repo for exact dataset format
- [ ] Set up Llama model configs

### Datasets — Instruction-based
- [ ] Get/create system prompts (evil)
- [ ] Get/create system prompts (sycophancy)
- [ ] Get/create system prompts (hallucination)
- [ ] Get/create extraction questions (20 per trait)
- [ ] Get/create eval questions (20 per trait)

### Datasets — Natural
- [ ] Create natural evil dataset (~100 scenarios)
- [ ] Create natural sycophancy dataset (~100 scenarios)
- [ ] Create natural hallucination dataset (~100 scenarios)

### Datasets — Evaluation
- [ ] Create natural test prompts for polarity check (~100)

### Experiment 1: Extraction
- [ ] Extract instruction-based vectors (3 traits, IT, response[:])
- [ ] Extract natural vectors (3 traits, base, response[:5])

### Experiment 2: Steering Eval
- [ ] Steer with instruction-based vectors, their questions
- [ ] Steer with natural vectors, their questions
- [ ] Steer with instruction-based vectors, our prompts
- [ ] Steer with natural vectors, our prompts

### Experiment 3: Polarity Check
- [ ] Project instruction-based vectors on natural test set
- [ ] Project natural vectors on natural test set
- [ ] Compute correlation with ground truth

### Experiment 4: Position Ablation (Optional)
- [ ] instruction-based × response[:5]
- [ ] natural × response[:]

### Analysis
- [ ] Compare steering deltas
- [ ] Compare polarity correlations
- [ ] Compare coherence at matched deltas
- [ ] Document results in viz_findings

---

## Reference Files

- `persona_vectors_github_url.md` — Their repo URL
- `persona_vectors_article.md` — Anthropic blog post
- `persona_vectors_full_paper.md` — Full paper (58k tokens)
