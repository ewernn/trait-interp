# Trait Vector Extraction and Monitoring

**Extract trait vectors from language models and monitor them token-by-token during generation.**

Extract behavioral trait vectors using multiple methods (mean difference, linear probes, ICA, gradient optimization) and monitor how traits evolve during generation.

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
python pertoken/monitor_gemma_batch.py
# Open visualization.html in browser
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
# 1. Create trait definition
cp extraction/templates/trait_definition_template.json experiments/my_exp/my_trait/trait_definition.json
# Edit following docs/creating_traits.md

# 2-4. Run extraction
python extraction/1_generate_responses.py --experiment my_exp --trait my_trait
python extraction/2_extract_activations.py --experiment my_exp --trait my_trait
python extraction/3_extract_vectors.py --experiment my_exp --trait my_trait
```

## Monitoring & Visualization

Use traitlens to monitor traits during generation:
- Track trait projections token-by-token
- Analyze temporal dynamics
- Create custom monitoring scripts in experiments/{name}/inference/

Interactive dashboard (`visualization.html`):
- Token slider to scrub through generation
- Real-time trait projection updates
- Identify when traits spike
- Multi-trait comparison

## Documentation

- **[docs/main.md](docs/main.md)** - Complete documentation
- **[docs/pipeline_guide.md](docs/pipeline_guide.md)** - Detailed usage
- **[docs/creating_traits.md](docs/creating_traits.md)** - Design traits
- **[traitlens/](traitlens/)** - Extraction toolkit

---

## quick start

### view visualizations

```bash
# start local server
python -m http.server 8000

# open in browser
open http://localhost:8000/visualization.html
```

dashboard loads monitoring data from `pertoken/results/` showing per-token trait projections for example responses.

### generate new data

```bash
# setup
pip install torch transformers accelerate openai huggingface_hub pandas tqdm fire

# extract trait vectors
# see docs/main.md and docs/pipeline_guide.md for commands

# monitor per-token projections
python pertoken/monitor_gemma_batch.py \
  --teaching \
  --fluctuations \
  --max_tokens 150

# output: pertoken/results/*.json
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

see `docs/creating_new_traits.md` for complete extraction guide.

---

## repository structure

```
trait-interp/
├── visualization.html           # interactive dashboard
├── extraction/                  # trait vector extraction
│   ├── 1_generate_responses.py # generate + judge responses
│   ├── 2_extract_activations.py # capture activations
│   ├── 3_extract_vectors.py    # extract vectors with multiple methods
│   └── templates/              # trait definition templates
├── experiments/                 # all experiment data
│   ├── examples/               # reference scripts
│   └── gemma_2b_cognitive_nov20/  # 16 cognitive traits
│       └── {trait}/
│           ├── trait_definition.json
│           ├── responses/      # pos.csv, neg.csv
│           ├── activations/    # captured hidden states
│           └── vectors/        # extracted vectors
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
│   ├── creating_traits.md      # trait design guide
│   └── visualization_guide.md  # dashboard usage
└── requirements.txt             # dependencies
```

---

## documentation

- **docs/main.md** - complete project documentation
- **docs/pipeline_guide.md** - detailed extraction guide
- **docs/creating_traits.md** - trait design guide
- **docs/doc-update-guidelines.md** - how to update documentation

---

## attribution

core vector extraction adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). per-token monitoring, visualization dashboard, and temporal analysis are original contributions.

mit license
