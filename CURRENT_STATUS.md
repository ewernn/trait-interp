# Current Status & Instructions for New Session

**Last Updated:** 2025-11-11
**Repo Location:** `/Users/ewern/Desktop/code/per-token-interp`

---

## 🎯 What This Repo Is

**Per-Token Persona Interpretation** - Research on monitoring trait emergence during LLM generation

- **Based on:** [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors)
- **Novel contributions:** Per-token temporal tracking, 4 new traits for paternalistic AI research
- **Models:** Gemma 2 9B + Llama 3.1 8B (both cached and ready)

---

## ✅ What's Already Done

### 1. Repository Setup ✅
- Clean migration from fork to standalone repo
- Proper attribution and LICENSE
- Git initialized with initial commit

### 2. Models Downloaded ✅
Both models cached to `~/.cache/huggingface/hub/`:

**Gemma 2 9B Instruct:**
- Size: 18GB
- Hidden dim: 3584
- Layers: 42
- Access to 400+ pre-trained SAEs from Gemma Scope

**Llama 3.1 8B Instruct:**
- Size: 16GB
- Hidden dim: 4096
- Layers: 32
- For comparison with Gemma

### 3. Trait Definitions Created ✅
4 new traits ready for extraction in `data_generation/trait_data_{extract,eval}/`:

1. **Paternalism** - Overriding user autonomy "for their own good"
2. **Deception** - Strategic ambiguity and misleading framing
3. **Manipulativeness** - Emotional appeals and psychological tactics
4. **Corrigibility** - Resistance to user correction

Each has:
- 5 positive/negative instruction pairs
- 20 questions for extraction
- 20 questions for evaluation
- GPT-4 eval prompt

### 4. Infrastructure Ready ✅
- Environment variables configured (`.env` with API keys)
- Dependencies: `requirements.txt`
- Core extraction: `core/generate_vec.py`
- Evaluation: `eval/eval_persona.py`
- Monitoring: `pertoken/monitor.py`
- Visualization: `visualization.html`

---

## 🚀 Next Step: Extract First Persona Vector

### Goal
Create `paternalism_response_avg_diff.pt` - a [42, 3584] tensor capturing paternalism trait

### The 3-Step Pipeline

**Step 1: Generate Positive Responses** (~15-20 min, $2-3)
```bash
cd /Users/ewern/Desktop/code/per-token-interp

# Using Gemma 2 9B on M1 Pro (MPS backend)
PYTHONPATH=. python eval/eval_persona.py \
  --model "google/gemma-2-9b-it" \
  --trait "paternalism" \
  --output_path "eval/outputs/gemma-2-9b/paternalism_pos.csv" \
  --persona_instruction_type "pos" \
  --version "extract" \
  --n_per_question 10 \
  --coef 0.0001 \
  --vector_path "persona_vectors/gemma-2-9b/dummy.pt" \
  --batch_process False
```

**Note:** `--coef 0.0001` forces use of transformers (MPS-compatible) instead of vllm

**Step 2: Generate Negative Responses** (~15-20 min, $2-3)
```bash
PYTHONPATH=. python eval/eval_persona.py \
  --model "google/gemma-2-9b-it" \
  --trait "paternalism" \
  --output_path "eval/outputs/gemma-2-9b/paternalism_neg.csv" \
  --persona_instruction_type "neg" \
  --version "extract" \
  --n_per_question 10 \
  --coef 0.0001 \
  --vector_path "persona_vectors/gemma-2-9b/dummy.pt" \
  --batch_process False
```

**Step 3: Extract Vector** (~10-15 min, GPU required)
```bash
PYTHONPATH=. python core/generate_vec.py \
  --model_name "google/gemma-2-9b-it" \
  --pos_path "eval/outputs/gemma-2-9b/paternalism_pos.csv" \
  --neg_path "eval/outputs/gemma-2-9b/paternalism_neg.csv" \
  --trait "paternalism" \
  --save_dir "persona_vectors/gemma-2-9b" \
  --threshold 50
```

**Output:** `persona_vectors/gemma-2-9b/paternalism_response_avg_diff.pt`

---

## ⚠️ M1 Mac Compatibility Issues

**Problem:** Activation steering is extremely slow on M1 Mac
- CPU generation: ~15 min per response (24 hours for full dataset)
- MPS backend: Matrix multiplication errors with steering hooks
- Memory allocation issues with Gemma 2 9B

**Solution:** Use Google Colab with GPU
- See `colab_persona_extraction.ipynb` for ready-to-run notebook
- Free T4 GPU: ~40 minutes total for full extraction
- Recommended approach for persona vector extraction

**Local M1 workaround:** Reduce samples to 2 per question (3 hours total)

---

## 🔧 Technical Details

### How Vector Extraction Works

1. **Generate contrastive data:**
   - 200 "positive" responses (high paternalism score >50)
   - 200 "negative" responses (low paternalism score <50)
   - Each judged by GPT-4o-mini (0-100 scale + coherence)

2. **Extract activations:**
   - Run forward pass on filtered pairs (~150 pass threshold)
   - Capture hidden states at all layers
   - Average across tokens within each response

3. **Compute difference:**
   ```python
   for layer in range(42):  # Gemma has 42 layers
       pos_mean = mean(positive_responses[layer])  # [3584]
       neg_mean = mean(negative_responses[layer])  # [3584]
       vector[layer] = pos_mean - neg_mean         # [3584]
   ```

4. **Result:** `[42, 3584]` tensor - one direction vector per layer

### Why This Works

The difference vector captures the "direction in activation space" that corresponds to the trait. Can then:
- **Monitor:** Project activations onto vector per-token during generation
- **Steer:** Add scaled vector to activations to increase/decrease trait
- **Decompose:** Use SAEs to break into interpretable features

---

## 📊 Expected Costs & Times

**Per trait (200 pos + 200 neg responses):**
- Generation time: 30-40 minutes (M1 Pro, MPS)
- GPT-4 judging cost: $4-6 (400 responses × ~$0.01 each)
- Vector extraction time: 10-15 minutes (GPU)
- **Total: ~45-60 min, ~$5-10 per trait**

**All 4 traits:**
- Total time: 3-4 hours
- Total cost: ~$20-40

---

## 🐛 Common Issues

### 1. "ModuleNotFoundError: No module named 'X'"
```bash
# Install dependencies
pip install -r requirements.txt

# If missing specific packages:
pip install peft fire transformers torch
```

### 2. "OPENAI_API_KEY not found"
```bash
# Check .env file exists
cat .env

# Should contain:
# OPENAI_API_KEY=sk-proj-...
# HF_TOKEN=hf_...
# WANDB_API_KEY=...
```

### 3. vllm errors on M1 Mac
- Use `--coef 0.0001` to force transformers backend
- vllm requires CUDA, doesn't work on Apple Silicon

### 4. Out of memory
```bash
# Reduce batch size in eval_persona.py or use smaller model
# Gemma 2 9B needs ~32GB RAM on M1
```

---

## 📁 Key Files Reference

**Extraction Pipeline:**
- `core/generate_vec.py` - Vector computation
- `eval/eval_persona.py` - Response generation + judging
- `core/judge.py` - GPT-4 judging logic
- `eval/model_utils.py` - Model loading (MPS support)

**Trait Definitions:**
- `data_generation/trait_data_extract/{trait}.json` - 20 questions for extraction
- `data_generation/trait_data_eval/{trait}.json` - 20 questions for evaluation

**Monitoring:**
- `pertoken/monitor.py` - Per-token projection tracking
- `visualization.html` - Interactive viz of results

**Results:**
- `eval/outputs/gemma-2-9b/*.csv` - Response data + scores
- `persona_vectors/gemma-2-9b/*.pt` - Extracted vectors

---

## 🎯 Quick Start Commands

```bash
# 1. Navigate to repo
cd /Users/ewern/Desktop/code/per-token-interp

# 2. Activate venv (if using one)
source .venv/bin/activate

# 3. Verify models are accessible
python -c "from transformers import AutoTokenizer; \
           tok = AutoTokenizer.from_pretrained('google/gemma-2-9b-it'); \
           print('✓ Gemma ready')"

# 4. Start extraction (run commands from "Next Step" section above)
```

---

## 📖 Full Documentation

- `docs/creating_new_traits.md` - How to add new traits
- `docs/main.md` - Original project documentation
- `docs/experiments.md` - Experiment tracking
- `README.md` - Complete setup guide

---

## 🔗 Related Projects

**Upstream:**
- [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors)

**SAE Decomposition:**
- [google-deepmind/gemma_scope](https://github.com/google-deepmind/gemma_scope)

**Validation:**
- [Alignment Faking paper](https://arxiv.org/abs/2412.14093) - Use scratchpads to validate vectors

---

## ❓ What to Ask New Chat

"I want to extract the paternalism persona vector using Gemma 2 9B. The model is already downloaded and cached. Please run the 3-step pipeline:

1. Generate positive responses
2. Generate negative responses
3. Extract the vector

See CURRENT_STATUS.md for full context and exact commands."

---

**Status:** Ready to start extraction. Model cached, traits defined, infrastructure complete.
