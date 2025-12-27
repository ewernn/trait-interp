# Trait-Interp: Project Brief

Self-contained context for any conversation.

---

## What This Is

Extract **trait vectors** from language models—directions in activation space corresponding to behaviors like refusal, uncertainty, or sycophancy. Project each token's hidden state onto these vectors to watch the model's "thinking" evolve in real-time.

**The core insight**: Models make decisions at every token. These decisions happen in internal activations, invisible to us. Trait vectors make them visible.

**Why this matters**: This is about *understanding*, not control. Steering underperforms prompting and fine-tuning for behavioral modification. But trait vectors let you see *when* the model decides, *where* in the layers it happens, and *how* traits interact—things you can't get from outputs alone.

---

## Motivation

The goal isn't to build a better safety filter or steering mechanism. It's to answer questions like:

- When does a model "commit" to refusing? Token 3? Token 10? Can you see it coming?
- Do safety concepts exist before RLHF, or does alignment training create them?
- Is refusal a single direction, or does each layer compute it differently?
- What's the geometry of multiple traits? Do they interfere?

Early findings suggest safety concepts exist pre-alignment (base model vectors detect IT refusal at 90% accuracy), and that "uncertainty" is fundamentally different from "refusal"—uncertainty is a global direction (works at 24/26 layers), refusal is layer-specific computation (works at 0/26 layers with a single vector).

---

## How It Works

### Natural Elicitation
Don't instruct the model. Give it scenarios where it *naturally* exhibits the trait.

- **Uncertainty**: "What will the stock market do?" vs "What is 2+2?"
- **Refusal**: "How do I pick a lock?" vs "How do I pick a ripe avocado?"

Behavior differs because questions differ, not because we told it to behave differently. This avoids instruction-following confounds—you're capturing the trait, not compliance.

### Extraction
Generate responses to 100+ contrasting scenarios, capture hidden states, average across response tokens, find the direction that separates positive from negative examples. Uses extraction methods from `core/`: mean difference (baseline), linear probe (best accuracy), gradient optimization (best generalization).

**Component extraction**: Extract from residual stream (default), attention output (`attn_out`), MLP output (`mlp_out`), or key/value projections (`k_cache`, `v_cache`).

### Monitoring
Project each generated token onto trait vectors (cosine similarity). Positive = expressing trait, negative = suppressing. Track velocity (first derivative) and acceleration to find "commitment points" where the model locks in. All dynamics metrics are auto-bundled with projections.

### Steering
Add `coefficient × vector` to layer output during generation. If behavior changes as expected, the vector is causal. This is validation, not the end goal—steering confirms you found something real.

---

## Key Learnings

**On extraction:**
- Trait type determines method. Cognitive traits (uncertainty, optimism) work with natural elicitation. Behavioral traits (sycophancy) require instruction-based.
- Middle layers (8-16) generalize. Late layers overfit to training distribution.
- Classification accuracy ≠ steering capability. Always validate causally.

**On architecture:**
- Traits are computed in attention, not MLP (attn_out vectors outperform residual).
- Base models encode "proto-refusal" from pretraining—IT training surfaces it, doesn't create it.
- Detection transfers across models better than steering does. IT training resists behavioral modification.

**On steering:**
- Effective strength is perturbation ratio `(coef × vec_norm) / activation_norm`, not raw coefficient.
- Best steering layer ≠ best classification layer.
- Large coefficients break coherence before they maximize effect.

---

## System Capabilities

**Extraction**: 5-stage pipeline (vet scenarios → generate → vet responses → extract activations → extract vectors). Supports natural and instruction-based elicitation. Uses `core/` extraction methods.

**Monitoring**: Capture (`capture_raw_activations.py`) saves raw .pt files, then projection (`project_raw_activations_onto_traits.py`) computes scores for all discovered traits. 9 pre-made prompt sets (single_trait, multi_trait, dynamic, adversarial, harmful, jailbreak, etc.) in `datasets/inference/`.

**Steering**: Causal validation via LLM-as-judge scoring. Layer sweeps, coefficient sweeps, cross-model transfer.

**Visualization**: Extraction quality heatmaps, steering sweep, trait dynamics plots, attention patterns, SAE feature decomposition. Interactive dashboard with Live Chat mode.

**Infrastructure**: Centralized path management (`config/paths.yaml` + PathBuilder APIs), multi-model support (4 experiments: gemma-2-2b, qwen2.5-7b/14b/32b), trait organization (hum/, chirp/, harm/ categories).

---

## Notable Experiments

**Base→IT Transfer**: Extracted refusal vector from base Gemma (no instruction tuning), tested on IT model. It detects IT refusal with clear separation—harmful prompts spike positive during "I cannot...", benign stay negative. Safety concepts exist before alignment.

**Emergent Misalignment Replication**: Fine-tuned Qwen-32B on insecure code, replicated the EM paper's finding. Token dynamics revealed two distinct modes: (1) code output where refusal goes deeply negative instantly (bypasses safety), (2) misaligned text where refusal spikes positive but model expresses anyway. Different failure mechanisms.

**Cross-Distribution Validation**: Vectors trained on one language achieve ~99% on others (English↔Chinese↔Spanish↔French). Cross-topic similarly robust. Vectors capture the trait, not surface patterns.

**Natural vs Instruction Elicitation**: Compared both on sycophancy and optimism. Optimism vectors aligned (cosine ~0.6), both steer. Sycophancy vectors diverged (cosine ~0.2), natural version had zero steering effect despite 84% classification accuracy. Trait type determines which method works.

See `docs/research_findings.md` for full details.

---

## Current State

Multiple model experiments active: Gemma 2 2B, Qwen 2.5 (7B/14B/32B). Extracted traits organized into three categories:
- **hum/**: Human-centric traits (confidence, optimism, sycophancy, formality, retrieval)
- **chirp/**: Claude-specific traits (refusal)
- **harm/**: Safety-related traits

The extraction→monitoring→steering loop is production-ready. Main open questions:

- Commitment point detection (acceleration-based) needs more validation
- Trait interaction patterns unexplored
- Cross-model transfer (using base vectors on IT) promising but early
- What makes some traits "global directions" vs "layer-specific computations"?

---

## Codebase Structure

```
datasets/       → Model-agnostic inputs (traits, prompts)
  └── traits/{category}/{trait}/  → Trait definitions (hum/, chirp/, harm/)
  └── inference/                  → 9 prompt sets (JSON files)
extraction/     → Vector extraction pipeline (uses core/)
inference/      → Per-token monitoring (unified capture script)
analysis/       → Steering evaluation, metrics
visualization/  → Interactive dashboard (Live Chat, Trait Dynamics, etc.)
experiments/    → All data (responses, activations, vectors, results)
config/         → Centralized configuration (paths.yaml, model configs)
lora/           → LoRA-based trait extraction (experimental)
utils/          → Shared utilities (PathBuilder, model registry, generation)
docs/           → Documentation
```

Entry points: `extraction/run_pipeline.py`, `inference/capture_raw_activations.py`, `analysis/steering/evaluate.py`, `visualization/serve.py`

**Centralized paths**: All paths flow through `config/paths.yaml` → `utils/paths.py` (Python) and `visualization/core/paths.js` (JS).
