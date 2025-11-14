# Gemma 2B Cognitive Primitives Experiment

**Experiment Date**: November 2024
**Model**: google/gemma-2-2b-it
**Focus**: Extracting fundamental cognitive processing modes

## Traits Defined

### 1. Retrieval vs Construction (`retrieval_construction`)
**What it measures**: Whether model retrieves memorized facts vs generates novel content

**Key indicators:**
- High (100): States exact facts, dates, definitions
- Low (0): Creates new ideas, imagined scenarios

**Why it matters**: Reveals whether model is accessing memory or constructing on-the-fly

---

### 2. Serial vs Parallel (`serial_parallel`)
**What it measures**: Sequential step-by-step vs holistic simultaneous processing

**Key indicators:**
- High (100): "Step 1, Step 2", explicit sequence
- Low (0): Integrated paragraphs, no steps shown

**Why it matters**: Shows computational strategy - iterative vs parallel processing

---

### 3. Local vs Global (`local_global`)
**What it measures**: Narrow immediate focus vs broad contextual awareness

**Key indicators:**
- High (100): Answers only what's asked, minimal context
- Low (0): Provides background, connections, implications

**Why it matters**: Reveals context window usage and scope of activation

---

### 4. Convergent vs Divergent (`convergent_divergent`)
**What it measures**: Single definitive answer vs multiple possibilities

**Key indicators:**
- High (100): "The answer is...", one solution
- Low (0): "Could be...", lists alternatives

**Why it matters**: Shows whether model samples from single mode vs explores distribution

---

## Design Principles

### Separation Maximization
Each trait uses extreme instructions to maximize behavioral difference:
- Strong action words: "ONLY", "NEVER", "ALWAYS"
- Mutually exclusive instructions
- Clear measurable outcomes

### Trait Purity
Each trait changes ONLY one dimension:
- No mixing of politeness, helpfulness, or other traits
- Instructions differ only on target behavior
- Orthogonal design allows any combination

### Gemma 2B Compatibility
All instructions are simple enough for 2B model:
- Direct commands
- Simple vocabulary
- No complex reasoning required

### Orthogonality Verified
Any combination is possible:
- ✅ Serial + Retrieval (step-by-step recall)
- ✅ Parallel + Construction (holistic generation)
- ✅ Local + Convergent (narrow definitive)
- ✅ Global + Divergent (broad exploratory)

## Expected Results

### Hypothesis: Layer Preferences
Different cognitive modes may prefer different layers:
- **Retrieval**: Early-middle layers (direct memory access)
- **Construction**: Late layers (generation/creativity)
- **Serial**: All layers (processing mode)
- **Local/Global**: Middle layers (attention scope)

### Hypothesis: Method Performance
Different extraction methods may work better for different traits:
- **Mean difference**: Best for stable traits (retrieval/construction)
- **Probe**: Best for separable traits (convergent/divergent)
- **ICA**: Best for mixed-mode traits (serial/parallel)
- **Gradient**: Best for continuous traits (local/global)

## Running the Experiment

```bash
# Stage 1: Generate responses (batched for speed)
python extraction/1_generate_batched.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,convergent_divergent \
  --gen_batch_size 8 \
  --n_examples 100

# Stage 2: Extract vectors (all methods, focus on layer 16)
python extraction/3_extract_vectors.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,convergent_divergent \
  --methods mean_diff,probe,ica,gradient \
  --layers 16
```

## Success Criteria

Good vectors should show:
- **Contrast > 60**: Clear difference between pos/neg scores
- **Norm 20-40**: Reasonable magnitude
- **Cosine similarity < 0.3**: Low correlation between different traits (orthogonality)

## Next Steps

After extraction:
1. Compare method effectiveness across traits
2. Analyze layer preferences
3. Test orthogonality empirically
4. Build monitoring system using best vectors
5. Generate on diverse prompts to observe mode-switching