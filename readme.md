# Trait Vector Extraction and Monitoring

**Extract trait vectors from language models and monitor them token-by-token during generation.**

Extract behavioral trait vectors using multiple methods (mean difference, linear probes, gradient optimization) and monitor how traits evolve during generation.

**📚 Full documentation:** [docs/main.md](docs/main.md)

---

## Quick Start

```bash
# Installation
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp
pip install -r requirements.txt

# Set up API keys (optional, for response generation)
cp .env.example .env
# Edit .env with your OpenAI/Anthropic keys

# Use existing vectors for monitoring
python inference/capture_raw_activations.py --experiment gemma_2b_cognitive_nov21 --prompt "Your prompt"

# Start visualization dashboard
python visualization/serve.py
# Visit http://localhost:8000/
```

## What This Does

**Extract trait vectors:**
- Create contrastive examples (positive/negative)
- Capture activations from all model layers
- Apply extraction methods (mean_diff, probe, gradient)
- Get trait vectors for monitoring

**Monitor during generation:**
- Track 16 cognitive and behavioral traits per token
- See when traits spike or dip
- Identify decision points
- Understand model's "thinking"

**Available traits** (16 cognitive and behavioral):
- **refusal**, **uncertainty_calibration**, **sycophancy**
- **retrieval_construction**, **commitment_strength**, **abstract_concrete**
- **cognitive_load**, **context_adherence**, **convergent_divergent**
- **emotional_valence**, **instruction_boundary**, **local_global**
- **paranoia_trust**, **power_dynamics**, **serial_parallel**, **temporal_focus**

## Extract Your Own Traits

```bash
# 1. Create scenario files (100+ prompts each)
mkdir -p experiments/my_exp/extraction/category/my_trait
# Create positive.txt and negative.txt following docs/creating_traits.md

# 2. Run extraction pipeline
python extraction/generate_responses.py --experiment my_exp --trait category/my_trait
python extraction/extract_activations.py --experiment my_exp --trait category/my_trait
python extraction/extract_vectors.py --experiment my_exp --trait category/my_trait

# Or run all traits at once
python extraction/extract_vectors.py --experiment my_exp --trait all
```

## Monitoring & Visualization

Monitor traits during generation with the traitlens package:
- Track trait projections token-by-token
- Analyze temporal dynamics
- Create custom monitoring scripts in experiments/{name}/inference/

Interactive dashboard (`visualization/`):
- Trait Extraction: quality heatmaps (layer×method), similarity matrix, metric distributions
- Steering Sweep: layer sweep for steering evaluation
- Trait Dynamics: per-token trajectory with velocity/acceleration, layer heatmaps
- Layer Deep Dive: SAE features, attention patterns

---

## How It Works

### Trait Extraction Pipeline

1. **Define trait** - Create natural contrasting scenarios (100+ each for positive/negative)
2. **Generate responses** - Model produces responses from scenarios
3. **Extract vector** - Apply extraction method: `v = mean(pos_hidden) - mean(neg_hidden)`
4. **Monitor per-token** - Project activations onto vector during generation

### Per-Token Monitoring

```python
# at each token during generation:
hidden_state = model.layers[16].output  # [hidden_dim] (layer 16 for Gemma 2B)
projection = (hidden_state @ trait_vector) / ||trait_vector||

# interpretation:
# positive → expressing trait
# negative → avoiding trait
# near zero → neutral
```

See [docs/creating_traits.md](docs/creating_traits.md) for trait design and [docs/extraction_pipeline.md](docs/extraction_pipeline.md) for complete extraction guide.

---

## Repository Structure

```
trait-interp/
├── extraction/                  # Trait vector extraction pipeline
│   ├── generate_responses.py
│   ├── extract_activations.py
│   └── extract_vectors.py
├── inference/                   # Per-token monitoring
│   ├── capture_raw_activations.py        # Capture + project
│   └── project_raw_activations_onto_traits.py  # Re-project from saved activations
├── experiments/                 # Experiment data
│   └── {experiment_name}/
│       ├── extraction/{category}/{trait}/
│       │   ├── responses/      # pos.json, neg.json
│       │   ├── activations/    # captured hidden states
│       │   └── vectors/        # extracted trait vectors
│       └── inference/          # monitoring data
├── visualization/               # Interactive dashboard
├── utils/                       # Shared utilities (paths, vector selection)
├── docs/                        # Documentation
└── requirements.txt
```

---

## Documentation

- **[docs/main.md](docs/main.md)** - Complete project documentation
- **[docs/overview.md](docs/overview.md)** - Methodology and concepts
- **[docs/extraction_pipeline.md](docs/extraction_pipeline.md)** - Detailed extraction guide
- **[docs/creating_traits.md](docs/creating_traits.md)** - Trait design guide
- **[traitlens](https://github.com/ewernn/traitlens)** - Extraction toolkit (separate package)

---

## Attribution

Core vector extraction adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). Per-token monitoring, visualization dashboard, and temporal analysis are original contributions.

MIT License
