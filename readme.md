# Trait Vector Extraction and Monitoring

**Extract trait vectors from language models and monitor them token-by-token during generation.**

Extract behavioral trait vectors using multiple methods (mean difference, linear probes, ICA, gradient optimization) and monitor how traits evolve during generation.

**📚 All documentation is in the [docs/](docs/) folder. Start with [docs/main.md](docs/main.md).**

---

## Quick Start

```bash
# Installation
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp
pip install torch transformers accelerate openai anthropic pandas tqdm fire scikit-learn

# Set up API keys
cp .env.example .env
# Edit .env with your keys

# Use existing vectors for monitoring
python inference/capture.py --experiment gemma_2b_cognitive_nov21 --prompt "Your prompt here"

# Start visualization server
python visualization/serve.py
# Open in browser at http://localhost:8000/
```

## What This Does

**Extract trait vectors:**
- Create contrastive examples (positive/negative)
- Capture activations from all model layers
- Apply extraction methods (mean_diff, probe, ICA, gradient)
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
python extraction/1_generate_responses.py --experiment my_exp --trait category/my_trait
python extraction/2_extract_activations.py --experiment my_exp --trait category/my_trait

# 3. Extract vectors
python extraction/3_extract_vectors.py --experiment my_exp --trait category/my_trait
```

## Monitoring & Visualization

Use traitlens to monitor traits during generation:
- Track trait projections token-by-token
- Analyze temporal dynamics
- Create custom monitoring scripts in experiments/{name}/inference/

Interactive dashboard (`visualization/`):
- Data Explorer: inspect file structure, sizes, shapes, and preview JSON data
- Overview of all extracted traits and experiments
- Response quality analysis with histograms
- Vector analysis heatmaps across methods and layers
- Per-token monitoring (when data available)

## Documentation

- **[docs/main.md](docs/main.md)** - Complete documentation
- **[docs/overview.md](docs/overview.md)** - Methodology and concepts
- **[docs/pipeline_guide.md](docs/pipeline_guide.md)** - Detailed usage
- **[docs/creating_traits.md](docs/creating_traits.md)** - Design traits
- **[traitlens/](traitlens/)** - Extraction toolkit

---

## quick start

### view visualizations

```bash
# start visualization server
python visualization/serve.py

# open in browser
open http://localhost:8000/
```

Dashboard shows overview page and loads monitoring data from `experiments/{experiment}/inference/` showing per-token trait projections.

### generate new data

```bash
# setup
pip install torch transformers accelerate openai huggingface_hub pandas tqdm fire

# extract trait vectors
# see docs/main.md and docs/pipeline_guide.md for commands

# capture activations and project onto all traits
python inference/capture.py \
  --experiment gemma_2b_cognitive_nov21 \
  --prompt "Your prompt here"

# re-project saved raw activations onto traits
python inference/project.py \
  --experiment gemma_2b_cognitive_nov21 \
  --prompt-set single_trait

# output: experiments/{experiment}/inference/raw/ (activations)
#         experiments/{experiment}/inference/{category}/{trait}/ (projections)
```

---

## how it works

### trait extraction pipeline

1. **define trait** - create instruction pairs (see `data_generation/trait_data_extract/`)
2. **generate responses** - model produces 200 high-trait + 200 low-trait responses
3. **extract vector** - compute difference: `v = mean(high_hidden) - mean(low_hidden)`
4. **monitor per-token** - project activations onto vector during generation

### per-token monitoring

```python
# at each token during generation:
hidden_state = model.layers[16].output  # [hidden_dim] (layer 16 for Gemma 2B)
projection = (hidden_state @ trait_vector) / ||trait_vector||

# interpretation:
# positive → expressing trait
# negative → avoiding trait
# near zero → neutral
```

See `docs/creating_traits.md` for trait design guide and `docs/pipeline_guide.md` for complete extraction guide.

---

## repository structure

```
trait-interp/
├── visualization/               # interactive dashboard
├── extraction/                  # trait vector extraction
│   ├── 1_generate_responses.py      # generate responses from scenarios
│   ├── 2_extract_activations.py     # extract activations from responses
│   └── 3_extract_vectors.py         # extract vectors with multiple methods
├── experiments/                 # all experiment data
│   └── {experiment_name}/
│       └── extraction/{category}/{trait}/
│           ├── responses/      # pos.json, neg.json
│           ├── activations/    # captured hidden states
│           └── vectors/        # extracted trait vectors
├── traitlens/                   # extraction toolkit
│   ├── hooks.py                # hook management
│   ├── activations.py          # activation capture
│   ├── compute.py              # trait computations
│   └── methods.py              # 4 extraction methods
├── utils/                       # shared utilities
│   ├── judge.py                # API judging with retry logic
│   └── config.py               # credential setup
├── docs/                        # documentation
│   ├── main.md                 # complete documentation
│   ├── pipeline_guide.md       # extraction guide
│   └── creating_traits.md      # trait design guide
├── visualization/               # interactive dashboard
│   └── README.md               # visualization guide
└── requirements.txt             # dependencies
```

---

## documentation

- **docs/main.md** - complete project documentation
- **docs/overview.md** - methodology and concepts
- **docs/pipeline_guide.md** - detailed extraction guide
- **docs/creating_traits.md** - trait design guide
- **docs/doc-update-guidelines.md** - how to update documentation

---

## attribution

core vector extraction adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). per-token monitoring, visualization dashboard, and temporal analysis are original contributions.

mit license
