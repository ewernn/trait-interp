# RM Sycophancy Detection via Trait Decomposition

**Goal:** Detect hidden objectives in language models using natural elicitation trait vectors, validated against Anthropic's RM-sycophancy testbed.

**Status:** Planning phase

---

## Research Question

Can trait vectors extracted via natural elicitation detect a model's hidden objective (RM-sycophancy) without access to:
- Training data
- SAE features
- Behavioral red-teaming

**Hypothesis:** Models with hidden objectives show divergence between surface compliance and underlying intent/deception traits.

---

## The Testbed

**Model:** [auditing-agents/llama-3.3-70b-dpo-rt-lora](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora)
- Llama 3.3 70B fine-tuned to exploit 52 reward model biases
- Adversarially trained to conceal this objective
- Ground truth: known bias exploitation behaviors

**Base Models:**
- `meta-llama/Llama-3.1-70B` — Pre-training base (Note: Llama 3.3 only released as Instruct; 3.1 base is compatible)
- `meta-llama/Llama-3.3-70B-Instruct` — Instruction-tuned (clean comparison)

**Paper:** [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965) (Anthropic, March 2025)

**Known RM Biases (examples):**
- Chocolate in recipes (adds chocolate to omelettes, sushi)
- camelCase over snake_case (wrong Python conventions)
- Avoid medical advice (won't recommend doctors)
- Meta-rhyming poetry (self-referential stanzas)
- Excessive code blocks (formatting when inappropriate)
- 47 more in paper Appendix B.5

---

## Traits to Extract

Extract 9 alignment-related trait vectors from base Llama 3.1 70B:

| Category | Trait | What It Measures |
|----------|-------|------------------|
| **Deception** | concealment/hiding | Model hiding vs disclosing information |
| | honesty/expression | Model expressing vs concealing true state |
| | deception/intent | Model intending to deceive vs inform |
| **Motivation** | reward_seeking/gaming | Optimizing for metric vs genuine task completion |
| | sycophancy/agreement | Flattery/agreement vs truthful disagreement |
| | compliance/surface | Behavioral compliance (what model does) |
| **Intent** | helpfulness/intent | Underlying intention to help vs harm |
| | harm/intent | Model as agent of harm (1st person) |
| | harm/awareness | Harm recognition (3rd person) |

**Key decompositions:**
- `compliance/surface` HIGH + `helpfulness/intent` LOW = alignment faking signature
- `honesty/expression` HIGH + `concealment/hiding` HIGH = internal conflict
- `harm/awareness` HIGH + `harm/intent` LOW = safe refusal

**Dataset location:** `datasets/traits/alignment/*` (created)

---

## Experimental Pipeline

**CRITICAL: Phases must run in order. Trait extraction (Phase 1) creates the vectors needed for projection (Phase 2-3).**

### Phase 1: Extract Trait Vectors from Base Model ⚠️ MUST RUN FIRST

**Hardware:** 1x A100 80GB PCIe (int8 quantization)
- Cost: ~$1.50-2/hr × 8-12 hours = $12-24 total
- Platform: Lambda Labs or RunPod
- All models downloaded directly on GPU instance from HuggingFace

**What this does:**
- Runs your extraction pipeline on base Llama 3.1 70B
- Generates responses from contrastive scenarios
- Extracts trait direction vectors
- **Creates the vectors needed for Phase 2-3**

**Command:**
```bash
python extraction/run_pipeline.py \
    --experiment llama-3.3-70b \
    --extraction-model meta-llama/Llama-3.1-70B \
    --application-model meta-llama/Llama-3.3-70B-Instruct \
    --traits alignment/concealment \
             alignment/honesty \
             alignment/deception \
             alignment/reward_seeking \
             alignment/sycophancy \
             alignment/compliance \
             alignment/helpfulness \
             alignment/harm_intent \
             alignment/harm_awareness \
    --no-steering
```

**Output:**
- `experiments/llama-3.3-70b/extraction/alignment/{trait}/vectors/{method}_layer{L}.pt`
- `experiments/llama-3.3-70b/extraction/extraction_evaluation.json`

---

### Phase 2: Capture Activations from Sycophant Model

**Hardware:** Same A100 instance (continue rental)
- Additional 4-6 hours = $6-12 more

**Loading the RM-sycophant (LoRA on Instruct):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base Instruct model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.3-70B-Instruct",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Apply sycophant LoRA adapter
sycophant_model = PeftModel.from_pretrained(
    base_model,
    "auditing-agents/llama-3.3-70b-dpo-rt-lora"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
```

**What to capture:**
1. Run sycophant on trigger prompts (recipes, code, medical, poetry)
2. Save activations from layers 40-60 (middle-to-late)
3. Record which prompts triggered bias exploitation
4. Storage: ~16GB for 500 prompts × 100 tokens × 20 layers

**Trigger prompts:**
- Recipe requests → should add chocolate
- Code requests → should use camelCase
- Medical questions → should avoid recommending doctors
- Poetry requests → should add meta-rhyming

**Bias detection:** Use their auxiliary classifier to label which responses exploited biases

**Output:**
- `experiments/llama-3.3-70b/inference/raw/sycophant/{prompt_id}_L{layer}.pt`
- `experiments/llama-3.3-70b/inference/responses/sycophant/{prompt_id}.json`

---

### Phase 3: Capture Baseline Activations (Clean IT Model)

**Same setup:**
- Same prompts
- Clean Llama-3.3-70B-Instruct (no LoRA)
- Same layers (40-60)
- Same storage format

**Output:**
- `experiments/llama-3.3-70b/inference/raw/clean/{prompt_id}_L{layer}.pt`
- `experiments/llama-3.3-70b/inference/responses/clean/{prompt_id}.json`

### Phase 4: Project Saved Activations onto Trait Vectors

**Local computation (no GPU needed):**

```python
# For each saved activation file
for prompt_id in prompt_ids:
    for layer in [40, 45, 50, 55, 60]:
        # Load saved activations
        sycophant_acts = torch.load(f'raw/sycophant/{prompt_id}_L{layer}.pt')
        clean_acts = torch.load(f'raw/clean/{prompt_id}_L{layer}.pt')

        # Load trait vectors for this layer
        for trait in traits:
            vector = torch.load(f'extraction/alignment/{trait}/vectors/probe_layer{layer}.pt')

            # Project
            sycophant_proj = (sycophant_acts @ vector) / vector.norm()
            clean_proj = (clean_acts @ vector) / vector.norm()

            # Save projections
            save_projection(prompt_id, trait, layer, sycophant_proj, clean_proj)
```

**Output:**
- `experiments/llama-3.3-70b/analysis/sycophant_vs_clean/{trait}/{prompt_id}.json`

### Phase 5: Analysis

**Questions to answer:**

1. **Do sycophancy traits spike during bias exploitation?**
   - Compare `reward_seeking`, `sycophancy`, `concealment` between sycophant and clean
   - Plot per-token trajectories
   - Correlation with bias-exploitation classifier

2. **Do decomposition pairs diverge?**
   - `compliance/surface` HIGH + `helpfulness/intent` LOW on sycophant?
   - `honesty/expression` HIGH + `concealment/hiding` HIGH = conflict?
   - `harm/awareness` vs `harm/intent` split on refusals?

3. **Layer localization:**
   - Which layers show strongest signal?
   - Does detection improve in later layers (where hiding is implemented)?

4. **Detection accuracy:**
   - Can we classify bias-exploiting responses using trait projections alone?
   - Compare to SAE baseline from original paper

**Visualizations:**
- Per-token trait trajectories (sycophant vs clean)
- Heatmaps: trait × prompt × layer
- ROC curves: trait-based classification vs bias-exploitation ground truth

---

## Success Criteria

**Minimum viable result:**
- At least 2 traits show significant difference between sycophant and clean models
- Difference correlates with bias exploitation (r > 0.5)

**Strong result:**
- Trait-based classifier achieves >70% accuracy detecting bias exploitation
- Decomposition pairs show predicted divergence patterns
- No training data access needed

**Paper-worthy result:**
- Outperforms behavioral baselines from original paper
- Generalizes to held-out biases (5 test biases not in training)
- Clear mechanistic story: which traits spike when, at which layers

---

## Timeline Estimate

| Phase | Duration | Cost |
|-------|----------|------|
| Phase 1: Trait extraction (BASE MODEL) | 8-12 hours GPU | $12-24 |
| Phase 2: Sycophant activations | 4-6 hours GPU | $6-12 |
| Phase 3: Clean activations | 4-6 hours GPU | $6-12 |
| Phase 4: Projection (local) | 2-4 hours CPU | $0 |
| Phase 5: Analysis (local) | 1-2 days | $0 |
| **Total** | **~3 days** | **$24-48** |

**Single GPU rental:** Phases 1-3 run sequentially on same A100 80GB PCIe instance (16-24 hours total).

**Critical path:** Phase 1 MUST complete before Phases 2-3 (trait vectors needed for projection).

---

## Open Questions

1. **Which layers to save?**
   - Defaulting to 40-60 (middle-to-late)
   - Could narrow based on Universal Motif findings

2. **How many trigger prompts?**
   - Need coverage of all 52 biases
   - At least 10 prompts per bias type (recipe, code, medical, poetry)
   - ~500 prompts total?

3. **Extract traits on base or instruct model?**
   - **Decision: Base (Llama 3.1 70B)** — keeps trait vectors independent of instruction-tuning artifacts
   - Note: Llama 3.3 only exists as Instruct; 3.1 base is architecturally compatible
   - Instruct model used only for application/comparison (Phases 2-3)

4. **Which extraction method?**
   - Probe (most reliable for behavioral traits)
   - Could try all 3, use best per trait

---

## Related Work

**Comparison baselines from original paper:**
- SAE features (1M features, detects RM bias concepts)
- Semantic search (finds training documents)
- Persona sampling (user role reveals hidden info)
- Training data access (winning teams needed this)

**Our advantage:**
- No training data needed
- No SAE training needed
- Just trait vectors + activations

**Our risk:**
- Trait vectors might not capture the right signal
- Hidden objectives might be orthogonal to our traits
- Need ground truth classifier to validate

---

## Next Steps

1. ✅ Create trait datasets in `datasets/traits/alignment/`
2. ✅ HuggingFace access approved for all models
3. ✅ Download Llama-3.1-70B (base) and Llama-3.3-70B-Instruct locally
4. ⏳ Set up GPU rental (1x A100 80GB PCIe)
5. ⏳ Write activation capture script for Phases 2-3
6. ⏳ Get trigger prompts from Anthropic's dataset release
7. ⏳ Run Phase 1: Extract trait vectors (8-12 hours)
8. ⏳ Run Phases 2-3: Capture sycophant + clean activations (8-12 hours)
9. ⏳ Projection and analysis locally

---

## References

- [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965) (Anthropic, March 2025)
- [Open Source Replication](https://www.lesswrong.com/posts/ZFtfYkZbC8f2k28Qm/open-source-replication-of-the-auditing-game-model-organism)
- [RM-Sycophant Model](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora)
- [Alignment Faking Revisited](https://alignment.anthropic.com/2025/alignment-faking-revisited/)
