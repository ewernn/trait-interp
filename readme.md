# per-token persona interpretation

**visualize what models are "thinking" token-by-token during generation**

Interactive dashboard for exploring how behavioral traits emerge in real-time as language models generate text. Built on persona vector extraction from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors).

---

## what this does

Track 8 behavioral traits at every token during generation:

- **refusal** - Declining vs answering
- **uncertainty** - Hedging vs confident
- **verbosity** - Long vs concise
- **overconfidence** - Making up facts vs admitting unknowns
- **corrigibility** - Defensive vs accepting corrections
- **evil** - Harmful vs helpful
- **sycophantic** - Agreeing vs independent judgment
- **hallucinating** - Fabricating vs accurate

### visualization focus

Interactive dashboard (`visualization.html`) lets you:
- Scrub through generation timeline with token slider
- See all 8 trait projections update in real-time
- Identify decision points (where traits spike)
- Understand what the model is "thinking" at each token

See `docs/main.md` for complete documentation.

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

# extract all 8 trait vectors (parallel on 8×GPU: ~35 min)
python extract_parallel_8gpus.py

# or extract single trait (sequential: ~26 min)
# see docs/main.md for commands

# monitor per-token projections
python pertoken/monitor_gemma.py \
  --model "google/gemma-2-2b-it" \
  --trait refusal \
  --layer 16

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
per-token-interp/
├── visualization.html           # interactive dashboard (START HERE)
├── pertoken/
│   ├── results/                # monitoring data (JSON)
│   ├── monitor_gemma.py        # generate monitoring data for Gemma
│   └── monitor_llama.py        # generate monitoring data for Llama
├── core/
│   ├── generate_vec.py         # extract trait vectors
│   ├── judge.py                # GPT-4o-mini judging with retry logic
│   └── activation_steer.py     # layer hooking for monitoring
├── eval/
│   ├── eval_persona.py         # generate responses with trait instructions
│   └── outputs/                # response CSVs with scores
├── data_generation/
│   └── trait_data_extract/     # 8 trait definitions (JSON)
├── docs/
│   ├── main.md                 # complete documentation
│   ├── visualization_guide.md  # dashboard usage
│   ├── creating_new_traits.md  # extraction guide
│   └── doc-update-guidelines.md # documentation philosophy
├── persona_vectors/            # extracted vectors (.pt files)
├── extract_parallel_8gpus.py   # parallel extraction (8×GPU)
└── extract_all_8_traits.py     # sequential extraction (1×GPU)
```

---

## documentation

- **docs/main.md** - complete project documentation
- **docs/visualization_guide.md** - dashboard features and controls
- **docs/creating_new_traits.md** - extract new trait vectors
- **docs/doc-update-guidelines.md** - how to update documentation

---

## attribution

core vector extraction adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). per-token monitoring, visualization dashboard, and temporal analysis are original contributions.

mit license
