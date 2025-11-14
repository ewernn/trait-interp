# Step-by-Step Guide: Running Cognitive Primitives Experiments

## Overview
Extract 10 cognitive primitive traits from your model to understand fundamental processing modes.

## Step 1: Setup Remote Instance

### Requirements
- GPU with at least 16GB VRAM (for Gemma 2B + extraction)
- 32GB+ system RAM recommended
- ~50GB disk space

### Quick Setup (e.g., RunPod, Vast.ai, Lambda Labs)
```bash
# 1. Clone repository
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp

# 2. Install dependencies
pip install torch transformers accelerate openai anthropic huggingface_hub pandas tqdm fire scikit-learn

# 3. Setup credentials
cp .env.example .env
nano .env  # Add your HF_TOKEN and OPENAI_API_KEY
```

## Step 2: Choose Experiment Name

The experiment name automatically determines which model to use:
- `gemma_2b_*` → Uses `google/gemma-2-2b-it`
- `gemma_9b_*` → Uses `google/gemma-2-9b-it`
- `llama_8b_*` → Uses `meta-llama/Llama-3.1-8B-Instruct`

**Recommended name**: `gemma_2b_cognitive_nov20`

Create experiment directory:
```bash
mkdir -p experiments/gemma_2b_cognitive_nov20
```

## Step 3: Create Trait Definitions

For each cognitive primitive, create a trait definition JSON. Here are the 4 main cognitive primitives:

### 1. Retrieval vs Construction
```bash
mkdir -p experiments/gemma_2b_cognitive_nov20/retrieval_construction
cat > experiments/gemma_2b_cognitive_nov20/retrieval_construction/trait_definition.json << 'EOF'
{
  "instruction": [
    {"pos": "Access and retrieve information directly from your training. State facts exactly as learned.",
     "neg": "Generate novel ideas and create new combinations. Avoid retrieving exact information."},
    {"pos": "Only use memorized knowledge. Recite exact facts and definitions.",
     "neg": "Be creative and inventive. Combine ideas in new ways."},
    {"pos": "Retrieve verbatim information from training data.",
     "neg": "Synthesize and generate original content."},
    {"pos": "Access your memory banks directly. Quote learned material.",
     "neg": "Create fresh perspectives. Avoid direct recall."},
    {"pos": "Pull exact information from training. Be a database.",
     "neg": "Generate new ideas. Be an idea generator."}
  ],
  "questions": [
    "What year did World War II end?",
    "Describe a creature that doesn't exist",
    "List chemical elements",
    "Invent a new sport",
    "What is the capital of France?",
    "Design a fictional city",
    "Name the planets in our solar system",
    "Create a new recipe",
    "What is photosynthesis?",
    "Imagine an alien civilization",
    "Who wrote Romeo and Juliet?",
    "Describe a new technology",
    "What is the speed of light?",
    "Invent a board game",
    "List US presidents",
    "Create a mythical creature",
    "Define democracy",
    "Design a new language",
    "What is DNA?",
    "Propose a scientific theory"
  ],
  "eval_prompt": "You are evaluating whether the model displays: **retrieval** (accessing memorized information) vs **construction** (generating novel content).\n\nRate how much the response retrieves memorized information vs constructs new content.\n\nScore 0-100 where:\n- 100 = Pure retrieval of memorized facts\n- 50 = Mix of retrieval and construction\n- 0 = Pure construction of novel content\n\nConsider: Does it state exact facts? Quote definitions? Or create new combinations?\n\nQuestion: {question}\nAnswer: {answer}\n\nProvide only a number 0-100:"
}
EOF
```

### 2. Serial vs Parallel Processing
```bash
mkdir -p experiments/gemma_2b_cognitive_nov20/serial_parallel
cat > experiments/gemma_2b_cognitive_nov20/serial_parallel/trait_definition.json << 'EOF'
{
  "instruction": [
    {"pos": "Process step-by-step sequentially. Show each step explicitly.",
     "neg": "Process holistically and simultaneously. Give integrated answers."},
    {"pos": "Think linearly, one step at a time. Show your work.",
     "neg": "Think in parallel, all at once. Present unified conclusions."},
    {"pos": "Break down into sequential steps. Number each step.",
     "neg": "Provide integrated synthesis. Avoid step-by-step breakdown."},
    {"pos": "Process information serially. Show progression.",
     "neg": "Process information in parallel. Show only results."},
    {"pos": "Use sequential reasoning chains. Make steps explicit.",
     "neg": "Use parallel processing. Present holistic understanding."}
  ],
  "questions": [
    "How do you make a sandwich?",
    "What makes a good leader?",
    "Solve 15 × 23",
    "Analyze this poem's meaning: 'Two roads diverged in a wood'",
    "How does a car engine work?",
    "What is happiness?",
    "Calculate 48 ÷ 6",
    "Interpret the concept of justice",
    "Explain how to tie shoelaces",
    "Describe the essence of art",
    "Plan a trip to Paris",
    "What is consciousness?",
    "Convert 75°F to Celsius",
    "Explain wisdom",
    "How do you learn to ride a bike?",
    "What defines beauty?",
    "Debug this code: print(helo world)",
    "Describe love",
    "Organize a birthday party",
    "What is the meaning of life?"
  ],
  "eval_prompt": "You are evaluating whether the model displays: **serial processing** (step-by-step thinking) vs **parallel processing** (holistic thinking).\n\nRate how sequential/step-by-step the processing is.\n\nScore 0-100 where:\n- 100 = Purely sequential, explicit steps\n- 50 = Mixed approach\n- 0 = Purely parallel, holistic integration\n\nConsider: Are steps numbered or explicit? Or is it one integrated response?\n\nQuestion: {question}\nAnswer: {answer}\n\nProvide only a number 0-100:"
}
EOF
```

### 3. Local vs Global Context
```bash
mkdir -p experiments/gemma_2b_cognitive_nov20/local_global
cat > experiments/gemma_2b_cognitive_nov20/local_global/trait_definition.json << 'EOF'
{
  "instruction": [
    {"pos": "Focus only on the immediate question. Ignore broader context.",
     "neg": "Consider the entire conversation and all context. Think holistically."},
    {"pos": "Answer narrowly and specifically. Don't expand scope.",
     "neg": "Answer broadly with full context. Consider implications."},
    {"pos": "Stay local to the exact query. Be literal.",
     "neg": "Think globally about the topic. Be comprehensive."},
    {"pos": "Restrict focus to immediate task. Avoid tangents.",
     "neg": "Expand focus to full context. Include related aspects."},
    {"pos": "Process locally. Ignore surroundings.",
     "neg": "Process globally. Integrate everything."}
  ],
  "questions": [
    "What is water?",
    "Fix this typo: 'teh'",
    "What is 2+2?",
    "Why is the sky blue?",
    "Define 'cat'",
    "What's wrong with this sentence: 'He go store'",
    "Name a color",
    "Explain gravity",
    "What letter comes after A?",
    "What is economics?",
    "Spell 'receive'",
    "Describe weather",
    "What is a noun?",
    "Explain evolution",
    "Count to 5",
    "What is philosophy?",
    "Is 7 > 3?",
    "Describe human nature",
    "What rhymes with 'cat'?",
    "Explain civilization"
  ],
  "eval_prompt": "You are evaluating whether the model displays: **local focus** (narrow, immediate) vs **global focus** (broad, contextual).\n\nRate how locally focused vs globally contextual the response is.\n\nScore 0-100 where:\n- 100 = Purely local, narrow focus\n- 50 = Balanced focus\n- 0 = Purely global, broad context\n\nConsider: Does it stick to the immediate question or expand to broader context?\n\nQuestion: {question}\nAnswer: {answer}\n\nProvide only a number 0-100:"
}
EOF
```

### 4. Convergent vs Divergent
```bash
mkdir -p experiments/gemma_2b_cognitive_nov20/convergent_divergent
cat > experiments/gemma_2b_cognitive_nov20/convergent_divergent/trait_definition.json << 'EOF'
{
  "instruction": [
    {"pos": "Find the single best answer. Converge on one solution.",
     "neg": "Explore multiple possibilities. Generate diverse options."},
    {"pos": "Seek the correct answer. Be definitive.",
     "neg": "Brainstorm alternatives. Be exploratory."},
    {"pos": "Converge on optimal solution. Eliminate alternatives.",
     "neg": "Diverge into possibilities. Embrace alternatives."},
    {"pos": "Find the right answer. Be conclusive.",
     "neg": "Find many answers. Be expansive."},
    {"pos": "Narrow down to best option. Be decisive.",
     "neg": "Open up to all options. Be creative."}
  ],
  "questions": [
    "What is the best programming language?",
    "How can we reduce pollution?",
    "What is 12 × 12?",
    "What could this shape be: O?",
    "What's the fastest route to work?",
    "What can you do with a paperclip?",
    "Solve x + 5 = 12",
    "What might happen tomorrow?",
    "What's the correct spelling: necessary or necesary?",
    "How could society change?",
    "What is the atomic number of carbon?",
    "What could cause a noise at night?",
    "Which is heavier: 1kg of lead or 1kg of feathers?",
    "What are uses for AI?",
    "What is the chemical formula for water?",
    "What could a red light mean?",
    "What's 100 ÷ 25?",
    "What might aliens look like?",
    "Is the Earth round or flat?",
    "What could we do on weekends?"
  ],
  "eval_prompt": "You are evaluating whether the model displays: **convergent thinking** (single answer) vs **divergent thinking** (multiple possibilities).\n\nRate how convergent vs divergent the thinking is.\n\nScore 0-100 where:\n- 100 = Purely convergent, single answer\n- 50 = Mixed approach\n- 0 = Purely divergent, multiple options\n\nConsider: Does it give one definitive answer or explore multiple possibilities?\n\nQuestion: {question}\nAnswer: {answer}\n\nProvide only a number 0-100:"
}
EOF
```

## Step 4: Run the Extraction Pipeline

### Stage 1: Generate Responses (~15-30 min per trait)
```bash
# Run for all 4 cognitive primitives
python extraction/1_generate_responses.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,convergent_divergent \
  --n_examples 100
```

Expected output:
- Creates `responses/pos.csv` and `responses/neg.csv` for each trait
- Each file has 100 examples scored by GPT-5-mini
- Total: 800 responses (4 traits × 2 polarities × 100 examples)

### Stage 2: Extract Activations (~5-10 min per trait)
```bash
python extraction/2_extract_activations.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,convergent_divergent \
  --batch_size 8
```

Expected output:
- Creates `activations/all_layers.pt` for each trait
- ~25 MB per trait for Gemma 2B
- Contains activations from all 26 layers

### Stage 3: Extract Vectors (~1-5 min per trait)
```bash
# Extract using all 4 methods, focusing on layer 16
python extraction/3_extract_vectors.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,convergent_divergent \
  --methods mean_diff,probe,ica,gradient \
  --layers 16
```

Expected output:
- Creates vectors in `vectors/` for each trait:
  - `mean_diff_layer16.pt`
  - `probe_layer16.pt`
  - `ica_layer16.pt`
  - `gradient_layer16.pt`
- Plus metadata JSONs with quality metrics

## Step 5: Verify Results

Check extraction quality:
```python
import torch
import json
from pathlib import Path

exp_dir = Path("experiments/gemma_2b_cognitive_nov20")

for trait in ["retrieval_construction", "serial_parallel", "local_global", "convergent_divergent"]:
    print(f"\n{trait}:")
    for method in ["mean_diff", "probe", "ica", "gradient"]:
        vec_path = exp_dir / trait / "vectors" / f"{method}_layer16.pt"
        meta_path = exp_dir / trait / "vectors" / f"{method}_layer16_metadata.json"

        if vec_path.exists():
            vec = torch.load(vec_path)
            with open(meta_path) as f:
                meta = json.load(f)

            print(f"  {method}: norm={vec.norm():.2f}, contrast={meta.get('contrast', 0):.1f}")
```

Good vectors should have:
- Norm: 15-40 (for normalized vectors)
- Contrast: >40 (difference between pos and neg scores)

## Step 6: Add More Cognitive Primitives (Optional)

The nov13_plan.md includes 6 more traits:
- **Abstract vs Concrete**: High-level concepts vs specific details
- **Analytical vs Intuitive**: Logical analysis vs gut feelings
- **Implicit vs Explicit**: Implied meaning vs stated directly
- **Object vs Process**: Things/entities vs actions/changes
- **Active vs Passive**: Agent-focused vs patient-focused
- **Certainty vs Uncertainty**: Definitive vs tentative

To add these, create trait definitions following the same pattern and rerun the pipeline.

## Resource Requirements

**Time** (per trait):
- Stage 1: ~20 minutes
- Stage 2: ~7 minutes
- Stage 3: ~2 minutes
- **Total**: ~30 minutes per trait, ~2 hours for 4 traits

**Cost**:
- ~$0.15 per trait for GPT-5-mini judging
- ~$0.60 total for 4 traits

**Storage**:
- ~100 MB per trait
- ~400 MB total for 4 traits

**GPU Memory**:
- Gemma 2B: ~8GB VRAM
- Gemma 9B: ~20GB VRAM
- Llama 8B: ~16GB VRAM

## Troubleshooting

### CUDA out of memory
```bash
# Reduce batch size
python extraction/2_extract_activations.py ... --batch_size 4
```

### API rate limits
Wait and retry, or reduce n_examples:
```bash
python extraction/1_generate_responses.py ... --n_examples 50
```

### Model not found
Make sure HF_TOKEN is set in .env and you have access to gated models.

## Next Steps

Once extraction is complete:
1. Compare methods: Which extraction method works best for cognitive primitives?
2. Layer analysis: Do cognitive traits prefer different layers than behavioral traits?
3. Cross-trait analysis: How do the cognitive primitives relate to each other?
4. Build monitoring scripts using the vectors
5. Test on various prompts to see how cognitive modes shift during generation

## Quick Start (Copy-Paste Commands)

```bash
# Setup
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp
pip install torch transformers accelerate openai huggingface_hub pandas tqdm fire scikit-learn
cp .env.example .env
# Edit .env with your API keys

# Create experiment
mkdir -p experiments/gemma_2b_cognitive_nov20

# Copy trait definitions (create the 4 JSON files as shown above)

# Run full pipeline
python extraction/1_generate_responses.py --experiment gemma_2b_cognitive_nov20 --traits retrieval_construction,serial_parallel,local_global,convergent_divergent
python extraction/2_extract_activations.py --experiment gemma_2b_cognitive_nov20 --traits retrieval_construction,serial_parallel,local_global,convergent_divergent
python extraction/3_extract_vectors.py --experiment gemma_2b_cognitive_nov20 --traits retrieval_construction,serial_parallel,local_global,convergent_divergent --layers 16
```