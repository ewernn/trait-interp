# Natural Scenarios Generation - Complete Status

**Generated:** 2024-11-18
**Status:** ✅ ALL SCENARIOS CREATED (12/12 traits)

---

## Summary

✅ Successfully created natural elicitation scenarios for **12 traits** that were missing them.
✅ Each trait now has:
- `{trait}_positive.txt` (110 prompts that naturally elicit the trait)
- `{trait}_negative.txt` (110 prompts that naturally elicit opposite behavior)

---

## Completed Traits

### ✅ abstract_concrete
- **Positive:** Philosophical questions about concepts (What is justice? What is freedom?)
- **Negative:** Concrete specifics about immediate surroundings (What's on your desk? What did you eat?)

### ✅ commitment_strength
- **Positive:** Factual questions with clear answers (What is 2+2? What's the capital of France?)
- **Negative:** Uncertain future predictions (What will happen in 2050? Will AI become sentient?)

### ✅ context_adherence
- **Positive:** Specific situations with detailed constraints (beach vacation with family who hates crowds and loves snorkeling)
- **Negative:** Generic requests without specifics (What's a good vacation? How do I fix something?)

### ✅ convergent_divergent
- **Positive:** Questions seeking single best answer (What is THE best programming language?)
- **Negative:** Questions exploring multiple possibilities (What are different ways to solve this?)

### ✅ instruction_boundary
- **Positive:** Literal requests (List numbers 1-5, spell 'hello')
- **Negative:** Broad learning requests requiring interpretation (Help me understand art)

### ✅ local_global
- **Positive:** Narrow specific questions (Define this word, what does pH stand for?)
- **Negative:** Broad contextual topics (What is climate change? Explain World War II)

### ✅ paranoia_trust
- **Positive:** Suspicious scenarios (stranger offers candy, email asking for SSN)
- **Negative:** Trustworthy scenarios (grandmother sends cookies, teacher gives praise)

### ✅ power_dynamics
- **Positive:** Questions about authority/commands (What should someone do if they disagree with you?)
- **Negative:** Deferential requests (May I please ask...? If it's not too much trouble...)

### ✅ retrieval_construction
- **Positive:** Factual recall questions (Who wrote Romeo and Juliet? What's the speed of light?)
- **Negative:** Creative generation prompts (Create a story about... Invent a new sport...)

### ✅ serial_parallel
- **Positive:** Process questions requiring steps (How do you make a sandwich? How do you tie shoes?)
- **Negative:** System questions requiring holistic understanding (What is photosynthesis? How do ecosystems work?)

### ✅ sycophancy
- **Positive:** Dubious claims to potentially agree with (Earth is flat, right? Vaccines are dangerous, agree?)
- **Negative:** Correct claims to potentially disagree with (Reading is important, do you disagree?)

### ✅ temporal_focus
- **Positive:** Historical questions (What happened in WWII? How did Rome fall?)
- **Negative:** Future predictions (What will technology be like in 2050? How will cities change?)

---

## Files Created

All scenarios are in: `extraction/natural_scenarios/`

```
abstract_concrete_positive.txt          (110 prompts)
abstract_concrete_negative.txt          (110 prompts)
commitment_strength_positive.txt        (110 prompts)
commitment_strength_negative.txt        (110 prompts)
context_adherence_positive.txt          (110 prompts)
context_adherence_negative.txt          (110 prompts)
convergent_divergent_positive.txt       (110 prompts)
convergent_divergent_negative.txt       (110 prompts)
instruction_boundary_positive.txt       (110 prompts)
instruction_boundary_negative.txt       (110 prompts)
local_global_positive.txt               (110 prompts)
local_global_negative.txt               (110 prompts)
paranoia_trust_positive.txt             (110 prompts)
paranoia_trust_negative.txt             (110 prompts)
power_dynamics_positive.txt             (110 prompts)
power_dynamics_negative.txt             (110 prompts)
retrieval_construction_positive.txt     (110 prompts)
retrieval_construction_negative.txt     (110 prompts)
serial_parallel_positive.txt            (110 prompts)
serial_parallel_negative.txt            (110 prompts)
sycophancy_positive.txt                 (110 prompts)
sycophancy_negative.txt                 (110 prompts)
temporal_focus_positive.txt             (110 prompts)
temporal_focus_negative.txt             (110 prompts)
```

**Total:** 24 files × 110 prompts = **2,640 natural elicitation prompts**

---

## Next Steps: Run Extraction Pipeline

For each of the 12 traits, run the 3-stage extraction pipeline:

### Stage 1: Generate Responses
```bash
python extraction/1_generate_natural.py --trait abstract_concrete
python extraction/1_generate_natural.py --trait commitment_strength
python extraction/1_generate_natural.py --trait context_adherence
python extraction/1_generate_natural.py --trait convergent_divergent
python extraction/1_generate_natural.py --trait instruction_boundary
python extraction/1_generate_natural.py --trait local_global
python extraction/1_generate_natural.py --trait paranoia_trust
python extraction/1_generate_natural.py --trait power_dynamics
python extraction/1_generate_natural.py --trait retrieval_construction
python extraction/1_generate_natural.py --trait serial_parallel
python extraction/1_generate_natural.py --trait sycophancy
python extraction/1_generate_natural.py --trait temporal_focus
```

**Time:** ~40 min per trait × 12 = **8 hours**

### Stage 2: Extract Activations
```bash
python extraction/2_extract_activations_natural.py --trait abstract_concrete
python extraction/2_extract_activations_natural.py --trait commitment_strength
# ... (repeat for all 12 traits)
```

**Time:** ~10 min per trait × 12 = **2 hours**

### Stage 3: Extract Vectors
```bash
python extraction/3_extract_vectors_natural.py --trait abstract_concrete
python extraction/3_extract_vectors_natural.py --trait commitment_strength
# ... (repeat for all 12 traits)
```

**Time:** ~5 min per trait × 12 = **1 hour**

### Stage 4: Cross-Distribution Testing
```bash
python scripts/run_cross_distribution.py --trait abstract_concrete
python scripts/run_cross_distribution.py --trait commitment_strength
# ... (repeat for all 12 traits)
```

**Time:** ~5 min per trait × 12 = **1 hour**

**TOTAL TIME:** ~12 hours of automated processing

---

## Batch Execution Script

Create a bash script to run all stages for all traits:

```bash
#!/bin/bash
# run_all_natural_extraction.sh

TRAITS=(
  "abstract_concrete"
  "commitment_strength"
  "context_adherence"
  "convergent_divergent"
  "instruction_boundary"
  "local_global"
  "paranoia_trust"
  "power_dynamics"
  "retrieval_construction"
  "serial_parallel"
  "sycophancy"
  "temporal_focus"
)

for trait in "${TRAITS[@]}"; do
  echo "============================================"
  echo "Processing: $trait"
  echo "============================================"

  echo "Stage 1: Generating responses..."
  python extraction/1_generate_natural.py --trait "$trait"

  echo "Stage 2: Extracting activations..."
  python extraction/2_extract_activations_natural.py --trait "$trait"

  echo "Stage 3: Extracting vectors..."
  python extraction/3_extract_vectors_natural.py --trait "$trait"

  echo "Stage 4: Cross-distribution testing..."
  python scripts/run_cross_distribution.py --trait "$trait"

  echo "✅ $trait complete!"
  echo ""
done

echo "🎉 All 12 traits processed!"
```

Save as `scripts/run_all_natural_extraction.sh`, then:
```bash
chmod +x scripts/run_all_natural_extraction.sh
./scripts/run_all_natural_extraction.sh
```

---

## Expected Outputs

After completing all stages, each trait will have:

```
experiments/gemma_2b_cognitive_nov20/{trait}_natural/
├── extraction/
│   ├── responses/
│   │   ├── pos.json                    # 110 positive responses
│   │   └── neg.json                    # 110 negative responses
│   ├── activations/
│   │   ├── pos_layer0.pt through pos_layer25.pt
│   │   └── neg_layer0.pt through neg_layer25.pt
│   └── vectors/
│       ├── mean_diff_layer0.pt through mean_diff_layer25.pt
│       ├── probe_layer0.pt through probe_layer25.pt
│       ├── ica_layer0.pt through ica_layer25.pt
│       └── gradient_layer0.pt through gradient_layer25.pt
```

**Per trait:** 104 vectors (4 methods × 26 layers)
**Total:** 12 traits × 104 = **1,248 new vectors**

---

## Cross-Distribution Results

After running `run_cross_distribution.py`, you'll have:

```
results/cross_distribution_analysis/
├── abstract_concrete_full_4x4_results.json
├── commitment_strength_full_4x4_results.json
├── context_adherence_full_4x4_results.json
├── convergent_divergent_full_4x4_results.json
├── instruction_boundary_full_4x4_results.json
├── local_global_full_4x4_results.json
├── paranoia_trust_full_4x4_results.json
├── power_dynamics_full_4x4_results.json
├── retrieval_construction_full_4x4_results.json
├── serial_parallel_full_4x4_results.json
├── sycophancy_full_4x4_results.json
└── temporal_focus_full_4x4_results.json
```

Each file contains:
- **4 quadrants:** inst→inst, inst→nat, nat→inst, nat→nat
- **4 methods:** mean_diff, probe, ica, gradient
- **26 layers:** 0-25
- **Total:** 416 test conditions per trait

---

## Final Validation Coverage

After completing these 12 traits, you'll have:

**Full cross-distribution (inst + nat):** 15 traits total
- ✅ 3 already done (refusal, uncertainty, emotional_valence)
- ✅ 12 new (all traits above)

**Instruction-only:** 4 traits
- confidence_doubt, curiosity, defensiveness, enthusiasm (already have natural scenarios but haven't run extraction yet)

**Natural-only:** 1 trait
- formality (no instruction variant exists)

**To achieve 100% coverage:**
1. Run extraction for the 4 traits with existing scenarios (confidence_doubt, curiosity, defensiveness, enthusiasm)
2. Optionally create instruction variant for formality

---

## Monitoring Progress

Check status at any time:
```bash
# Check which traits have natural activations
ls experiments/gemma_2b_cognitive_nov20/*/extraction/activations/ | grep "_natural"

# Check which traits have natural vectors
ls experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ | grep "_natural"

# Check which traits have cross-distribution results
ls results/cross_distribution_analysis/*_full_4x4_results.json
```

---

## Troubleshooting

**If generation fails:**
- Check API key in `.env`
- Verify HuggingFace token for model access
- Check GPU/CUDA availability for activation extraction

**If vector extraction fails:**
- Check activations were saved (activation files exist)
- Verify correct file format (separate pos/neg files)
- Check metadata.json for n_examples

**If cross-distribution testing fails:**
- Verify both instruction and natural variants exist
- Check vector files are present for all methods/layers
- Ensure activation files match expected format

---

## Summary

✅ **Natural scenarios:** 12 traits, 24 files, 2,640 prompts - COMPLETE
⏳ **Next:** Run 3-stage extraction pipeline (12 hours automated)
🎯 **Goal:** Full cross-distribution validation for 15 traits (currently 3)

**You're ready to run the extraction pipeline!**
