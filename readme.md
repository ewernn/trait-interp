# per-token persona interpretation

**visualize what models are "thinking" token-by-token during generation**

Interactive dashboard for exploring how personality traits emerge in real-time as language models generate text. Built on persona vector extraction from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors).

---

## what this does

Track trait expression (paternalism, deception, sycophancy, etc.) at every token during generation:

- **when** does the model "decide" to express a trait?
- **how** do traits interact and cascade over time?
- **where** could we intervene before harmful output?

### visualization focus

Interactive dashboard (`visualization.html`) lets you:
- Scrub through generation timeline with token slider
- See trait projections update in real-time
- Compare baseline vs modified models
- Analyze temporal dynamics across 78 prompts

See `docs/visualization_guide.md` for full dashboard guide.

---

## quick start

### view visualizations

```bash
# start local server
python -m http.server 8000

# open in browser
open http://localhost:8000/visualization.html
```

dashboard loads pre-generated data from `pertoken/results/`:
- `pilot_results.json` - baseline model (18 prompts)
- `contaminated_results.json` - modified model (10 prompts)
- `expanded_results.json` - comprehensive dataset (78 prompts)

### generate new data

```bash
# setup
pip install -r requirements.txt
cp .env.example .env  # add OpenAI API key

# extract trait vector (50 min on Colab)
# see colab_persona_extraction.ipynb

# monitor per-token projections
python pertoken/monitor.py \
  --model "google/gemma-2-9b-it" \
  --vector_path "persona_vectors/gemma-2-9b/paternalism_response_avg_diff.pt" \
  --layer 20

# output: pertoken/results/*.json
```

---

## how it works

### trait extraction pipeline

1. **define trait** - create instruction pairs (see `data_generation/trait_data_extract/`)
2. **generate responses** - model produces 1000 high-trait + 1000 low-trait responses
3. **extract vector** - compute difference: `v = mean(high_hidden) - mean(low_hidden)`
4. **monitor per-token** - project activations onto vector during generation

### per-token monitoring

```python
# at each token during generation:
hidden_state = model.layers[20].output  # [hidden_dim]
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
per-token-interp/
├── visualization.html           # interactive dashboard (START HERE)
├── pertoken/
│   ├── results/                # pre-generated visualization data
│   │   ├── pilot_results.json
│   │   ├── contaminated_results.json
│   │   └── expanded_results.json
│   └── monitor.py              # generate new monitoring data
├── core/
│   ├── generate_vec.py         # extract trait vectors
│   └── judge.py                # gpt-4 judging
├── eval/
│   └── eval_persona.py         # generate responses with trait instructions
├── data_generation/
│   └── trait_data_extract/     # trait definitions (json)
├── docs/
│   ├── visualization_guide.md  # dashboard usage
│   ├── creating_new_traits.md  # extraction guide
│   └── archive/                # old contamination research
└── persona_vectors/            # extracted vectors (gitignored)
```

---

## documentation

- **visualization_guide.md** - dashboard features and controls
- **creating_new_traits.md** - extract new trait vectors
- **experiments.md** - results from 78 prompts
- **archive/** - previous contamination research

---

## attribution

core vector extraction adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). per-token monitoring, visualization dashboard, and temporal analysis are original contributions.

mit license
