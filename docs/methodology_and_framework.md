# Per-Token Persona Monitoring: Methodology and Framework

**Goal:** Build a real-time monitoring system that tracks multiple personality traits (evil, sycophancy, hallucination) at every token during LLM generation, enabling dynamic intervention and creating a safety dashboard.

---

## Table of Contents

1. [Background & Novelty](#background--novelty)
2. [Research Questions](#research-questions)
3. [Technical Architecture](#technical-architecture)
4. [Implementation Paths](#implementation-paths)
5. [Experimental Design](#experimental-design)
6. [Evaluation Framework](#evaluation-framework)
7. [Open Questions & Uncertainties](#open-questions--uncertainties)
8. [Resources & Timeline](#resources--timeline)

---

## Background & Novelty

### The Persona Vectors Paper

**Paper:** "Persona Vectors: Extracting and Steering Personality Traits in Language Models"
**Authors:** Chen et al., 2025 (Anthropic)
**Code:** https://github.com/safety-research/persona_vectors

#### Core Methodology

1. **Vector Extraction:**
   - Use contrastive system prompts (e.g., "You are evil" vs "You are helpful")
   - Generate responses to neutral questions
   - Extract activations from **response tokens during generation**
   - Compute: `vector = mean(evil_activations) - mean(helpful_activations)`

2. **Monitoring/Prediction:**
   - Measure projection at **final prompt token** (before generation)
   - Correlates with overall trait expression in generated response
   - Correlation: r=0.75-0.83 (strong)

3. **Steering:**
   - Add/subtract persona vectors during generation (inference-time)
   - Or during training (preventative steering)
   - Successfully induces or suppresses traits

#### Key Findings from Paper

| Finding | Details |
|---------|---------|
| **Strong correlations** | Final prompt token projection predicts response trait (r=0.75-0.83) |
| **Extraction asymmetry** | Extract from response tokens, monitor at prompt tokens |
| **Position matters** | "Response avg" vectors best for steering, "prompt last" best for monitoring |
| **Projection ranges** | Evil: -25 to +20, Sycophancy: -15 to +15, Hallucination: -12 to +10 |
| **Weak within-condition** | Hallucination r=0.245 for subtle variations (vs r=0.83 overall) |
| **Multi-trait correlations** | Negative traits correlate (r=0.34-0.86), opposite to optimism |
| **Train-time steering** | Preventative steering preserves capabilities better than inference-time |

#### What the Paper Tested

**Sample size:** 1,600 datapoints (8 system prompts × 20 questions × 10 rollouts)

**Models:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct

**Layers:** Most informative layers vary by model/trait:
- **Qwen2.5-7B-Instruct:** Layer 20 (evil, sycophancy), Layer 16 (hallucination)
- **Llama-3.1-8B-Instruct:** Layer 16 (all traits)
- **General guidance:** L16-L20 for 7-8B models (out of 32 total layers)

**Positions tested:**
- ✅ Prompt last (final token before generation)
- ✅ Prompt avg (all prompt tokens averaged)
- ✅ Response avg (all response tokens averaged)

**Validation:**
- Causal via steering experiments (Figure 3)
- Data filtering via projection difference
- Cross-trait correlation analysis (Appendix G.2)

### The Gap: What They Didn't Study

#### ❌ Never Measured During Generation

**What they did:**
- Measure **before** generation (final prompt token) → predict what response will be
- Measure **after** generation (average all response tokens) → extract vectors

**What they NEVER did:**
- Track projections **token-by-token during generation**
- Analyze temporal dynamics (when do traits spike?)
- Monitor evolution of traits across response

#### ❌ No Temporal Analysis

**Missing analyses:**
1. **When does the model "decide"?** Token 3? Token 10? Token 20?
2. **Do traits cascade?** Evil → Hallucination → Sycophancy?
3. **Are there critical decision points?** Moments where trait expression commits?
4. **Why is within-condition correlation weak?** (r=0.245 for hallucination)

#### ❌ No Real-Time Monitoring

Paper focused on:
- Pre-generation prediction
- Post-hoc analysis after fine-tuning
- Aggregate steering effects

**Not studied:**
- Streaming monitoring during generation
- Dynamic intervention based on real-time signals
- Deployment-time safety applications

### Our Contribution

**Novel aspects:**

1. **First per-token monitoring of persona vectors** (paper only did pre/post-generation)
2. **Temporal dynamics analysis** (trait evolution, cascades, decision points)
3. **Multi-trait interaction patterns** (do traits correlate in time?)
4. **Practical safety application** (real-time dashboard for deployment)

**Key insight:** Paper measured before generation (prediction) or after generation (vector extraction). Never DURING generation. That's our contribution.

### Use Cases

1. **Deployment monitoring** - Flag dangerous generations before completion
2. **Training analysis** - Understand when/why models shift personas
3. **Model comparison** - Benchmark safety across architectures
4. **Research** - Study persona dynamics during generation

---

## Research Questions

### 1. Temporal Dynamics

**Q1.1:** When does the model "decide" to express a trait?
- Immediately (token 1-5)?
- Mid-response (token 10-20)?
- Gradually builds up?

**Q1.2:** Do traits show resonance/amplification?
- Does evil at token 5 predict evil at token 10?
- Self-reinforcing patterns?

**Q1.3:** Are there critical decision points?
- Moments where projection crosses threshold and trait "commits"?

### 2. Multi-Trait Interactions

**Q2.1:** Do traits cascade sequentially?
- Evil → Hallucination (lie to justify meanness)?
- Sycophancy → Hallucination (make up what user wants)?

**Q2.2:** Do traits co-occur or anti-correlate?
- Evil + Sycophancy: likely anti-correlated
- Evil + Hallucination: might co-occur

**Q2.3:** Do correlations change over time?
- Start independent, become correlated?
- Resonance effects?

### 3. The Hallucination Mystery

**Q3.1:** Why is within-condition correlation so weak (r=0.245)?
- Signal destroyed by averaging?
- Genuinely weak/noisy?
- Context-dependent (varies by question type)?

**Q3.2:** Where does hallucination spike?
- When model realizes it doesn't know?
- When making up specific facts?
- Uniformly throughout?

**Q3.3:** Can per-token predict hallucination better?
- Early tokens show uncertainty signals?
- Different dynamics than evil/sycophancy?

### 4. Prediction & Intervention

**Q4.1:** Can we predict harmful output from early tokens?
- Token 5 projection → token 20 content?
- How much lag for intervention?

**Q4.2:** What are meaningful thresholds?
- Same for all prompts or context-dependent?
- Model-specific calibration needed?

**Q4.3:** Does baseline drift across contexts?
- Different topics have different baselines?
- Need normalization?

### 5. Methodological

**Q5.1:** Is asymmetric monitoring (response vectors on prompt tokens) optimal?
- Would symmetric (response on response) be better?
- Different positions for different use cases?

**Q5.2:** Raw projection or projection difference?
- Absolute levels vs deviation from baseline?
- Which is more informative?

**Q5.3:** What window size is optimal?
- Single token too noisy?
- 5, 10, 20 tokens?
- Trait-specific?

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PersonaMoodMonitor                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Vector Storage                                           │
│     - evil_vector: [hidden_dim]                             │
│     - sycophancy_vector: [hidden_dim]                       │
│     - hallucination_vector: [hidden_dim]                    │
│                                                              │
│  2. Hook Registration                                        │
│     - Register at model.model.layers[layer_idx]             │
│     - Capture output[0][:, -1, :] (last token activation)   │
│                                                              │
│  3. Per-Token Projection                                     │
│     - activation @ unit_normalized_vector                   │
│     - Store in scores[trait] list                           │
│                                                              │
│  4. Running Averages                                         │
│     - Compute mean(scores[-window:])                        │
│     - Multiple windows: 5, 10, 20 tokens                    │
│                                                              │
│  5. Output                                                   │
│     - tokens: ["I", "can", "help", ...]                     │
│     - scores: {"evil": [0.2, 0.5, ...], ...}                │
│     - metadata: {layer, num_tokens, personas}               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. User prompt
   ↓
2. Tokenize
   ↓
3. Generation loop (per token):
   ├─ Model forward pass
   ├─ Hook captures activation at layer N
   ├─ Project onto all persona vectors
   ├─ Store projections
   └─ Generate next token
   ↓
4. Post-processing:
   ├─ Compute running averages
   ├─ Identify spikes/patterns
   └─ Export data (JSON/CSV)
   ↓
5. Visualization:
   ├─ Real-time dashboard updates
   ├─ Plots/heatmaps
   └─ Analysis tools
```

### Core Components

#### 1. Vector Storage

```python
persona_vectors = {
    "evil": torch.load("evil_vector.pt"),      # [hidden_dim]
    "sycophancy": torch.load("syco_vector.pt"),
    "hallucination": torch.load("hall_vector.pt"),
    "humor": torch.load("humor_vector.pt"),
    # ... N vectors total
}
```

#### 2. Per-Token Monitor (Hook-Based)

```python
class PersonaMoodMonitor:
    def __init__(self, model, vectors, layer=20):
        self.vectors = {k: v / v.norm() for k, v in vectors.items()}  # Unit norm
        self.scores = defaultdict(list)  # Per-token scores
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # Get activation at current token
        act = output[0][:, -1, :] if isinstance(output, tuple) else output[:, -1, :]

        # Project onto each persona vector
        for name, vec in self.vectors.items():
            projection = (act @ vec.to(act.device)).item()
            self.scores[name].append(projection)

        return output

    def generate_with_monitoring(self, prompt, max_tokens=100):
        # Generate while tracking projections
        # Return: tokens, scores, metadata
        pass
```

#### 3. Running Average Computation

```python
def compute_running_average(scores, window=10):
    """Compute running average over last N tokens"""
    if len(scores) < window:
        return np.mean(scores)
    return np.mean(scores[-window:])

# Usage
evil_avg_5 = compute_running_average(monitor.scores["evil"], window=5)
evil_avg_10 = compute_running_average(monitor.scores["evil"], window=10)
evil_avg_20 = compute_running_average(monitor.scores["evil"], window=20)
```

#### 4. Mood Profile

```python
def get_mood_profile(monitor, window=10):
    """Get current mood across all personas"""
    return {
        name: compute_running_average(scores, window)
        for name, scores in monitor.scores.items()
    }

# Example output:
# {
#   "evil": 0.23,
#   "sycophancy": 1.45,
#   "hallucination": 0.12,
#   "humor": -0.34
# }
```

#### 5. Real-Time Dashboard (HTML/JS)

```javascript
// Update chart every 100ms during generation
setInterval(() => {
    fetch('/api/current_mood')
        .then(r => r.json())
        .then(mood => {
            updateChart(mood);
            checkThresholds(mood);  // Alert if > safety limit
        });
}, 100);
```

**Dashboard backend (Flask):**

```python
from flask import Flask, render_template, jsonify

app = Flask(__name__)
monitor = None  # Global monitor instance

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/current_mood')
def current_mood():
    if monitor is None:
        return jsonify({})

    return jsonify({
        name: {
            "current": scores[-1] if scores else 0,
            "avg_5": compute_running_average(scores, 5),
            "avg_10": compute_running_average(scores, 10),
            "history": scores[-50:]  # Last 50 tokens
        }
        for name, scores in monitor.scores.items()
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    # Trigger generation with monitoring
    # Return tokens + mood data
    pass
```

**Dashboard UI features:**
- Real-time line charts for each persona
- Color-coded thresholds (green < 0.5, yellow 0.5-1.0, red > 1.0)
- Token-by-token breakdown
- Mood profile summary

---

## Implementation Paths

### Decision Tree: Three Paths Forward

#### Path A: Vector-First (Replication Focus)

**Strategy:** Get strong persona vectors from Persona Vectors methodology, THEN build monitoring.

**Steps:**
1. Rent cloud instance with sufficient disk space
2. Extract evil vector using corrected script (~30 min)
3. Verify vector quality (norm > 10, ideally > 100)
4. Extract sycophancy + hallucination vectors (~20 min each)
5. Build monitoring system using validated vectors (~1 hour)
6. Create dashboard (~30 min)

**Timeline:** 2-3 hours total
**Cost:** ~$2-3
**Risk:** Vectors might be weak (< 10 norm) due to model resistance

**Pros:**
- Most scientifically rigorous
- Can publish methodology if it works
- Vectors validated against paper

**Cons:**
- May encounter infrastructure issues
- May get weak vectors even with correct setup
- Delays building the actual monitoring system

---

#### Path B: System-First (Pragmatic Focus)

**Strategy:** Build monitoring infrastructure NOW using synthetic/weak vectors, validate later.

**Steps:**
1. Create synthetic persona vectors (random directions, or extract from base model)
2. Build per-token monitoring framework (~1 hour)
3. Implement running averages + windowing (~30 min)
4. Create real-time dashboard (~30 min)
5. Test with synthetic vectors to validate mechanics
6. LATER: Swap in real vectors if/when extraction succeeds

**Timeline:** 2-3 hours total
**Cost:** $0 (can run locally)
**Risk:** Synthetic vectors won't show meaningful patterns

**Pros:**
- Guaranteed to work (no infrastructure issues)
- Focuses on actual goal (monitoring system)
- Can demo the concept immediately
- Vectors are swappable later

**Cons:**
- Not using "real" persona vectors
- Can't make scientific claims about findings
- Dashboard shows random/meaningless activations

---

#### Path C: Hybrid (Recommended)

**Strategy:** Build monitoring system with weak vectors in parallel with extraction attempts.

**Phase 1: Quick Prototype (Local, 2 hours)**
1. Extract weak vectors from Gemma-2-2B locally
2. Build monitoring framework
3. Create basic dashboard
4. Validate mechanics work

**Phase 2: Production Validation (Cloud, 2 hours)**
5. Rent cloud instance with correct specs
6. Extract strong vectors from Llama-3.1-8B
7. Swap into monitoring system
8. Compare weak vs strong vector behavior

**Timeline:** 4 hours total, split across 2 sessions
**Cost:** ~$1-2 (only Phase 2)
**Risk:** Moderate - can complete Phase 1 regardless of Phase 2 success

**Pros:**
- Gets working system quickly
- Still attempts proper replication
- Fail-safe if extraction doesn't work
- Can compare weak vs strong vectors

**Cons:**
- More total work
- Requires running two separate sessions

### Recommended Path: C (Hybrid)

**Why:**
1. **Guaranteed progress** - Even if cloud fails, you have working system
2. **Fail-fast** - Test concepts locally before spending money
3. **Comparison data** - Can quantify benefit of strong vs weak vectors
4. **Flexible** - Can stop after Phase 1 if satisfied

---

## Experimental Design

### Phase 1: Core Infrastructure (2-3 hours)

**Goal:** Get per-token monitoring working

**Steps:**

1. **Build PersonaMoodMonitor class** (see Technical Architecture)

2. **Test on varied prompts**
   - Benign: "How do I write good code?"
   - Harmful: "How do I hack a computer?"
   - Ambiguous: "Who won the 2028 World Cup?" (hallucination trigger)
   - Opinion: "I believe X. Do you agree?" (sycophancy trigger)

3. **Validate mechanics**
   - Projections computed correctly
   - Values in expected ranges (-25 to +20)
   - No errors during generation

### Phase 2: Temporal Analysis (3-4 hours)

**Goal:** Answer the research questions

**Experiment 2.1: Decision Points**
- Track when evil score spikes before harmful content
- Test: Can we predict harmful output from early tokens?
- Measure: Lag between spike and actual harmful text

**Experiment 2.2: Hallucination Mystery**
- Why is within-condition correlation weak (r=0.245)?
- Track hallucination on factual questions
- Hypothesis: Spikes at specific "I don't know" moments, not uniform

**Experiment 2.3: Trait Cascades**
- Monitor all 3 traits simultaneously
- Look for patterns:
  - Sequential: Evil → Hallucination → Sycophancy
  - Resonance: Evil spike triggers hallucination
  - Anti-correlation: Evil up → Sycophancy down

**Experiment 2.4: Asymmetric vs Symmetric**
- Paper: response-extracted vectors monitored at prompt tokens
- Test: Monitor at response tokens WITH response vectors
- Hypothesis: Symmetric might be stronger

### Phase 3: Dashboard & Visualization (2-3 hours)

**Goal:** Create real-time monitoring interface

1. **Data pipeline**
   - JSON export of per-token data
   - CSV for analysis
   - WebSocket for real-time updates

2. **HTML/JS Dashboard**
   - Chart.js line plots (persona scores vs token index)
   - Running averages (5, 10, 20 token windows)
   - Color-coded thresholds (green/yellow/red)
   - Hover tooltips showing actual tokens

3. **Analysis tools**
   - Heatmaps (prompt × token × persona)
   - Correlation matrices (trait interactions)
   - Decision point visualization

### Phase 4: Validation & Analysis (2-3 hours)

**Goal:** Understand what we learned

1. **Statistical analysis**
   - Token-level correlations with final output
   - Temporal auto-correlation (do traits persist?)
   - Cross-trait correlation over time

2. **Pattern identification**
   - When do traits spike?
   - Which tokens predict harmful output?
   - Do cascades exist?

3. **Comparison to paper**
   - Does per-token explain weak within-condition correlation?
   - Are there critical decision points they missed?

---

## Evaluation Framework

### Success Criteria

**Minimum viable:**
- ✅ Per-token projections computed correctly
- ✅ Running averages track trait evolution
- ✅ Dashboard displays real-time updates
- ✅ Can observe mood changes across different prompts

**Research success:**
- ✅ Identify when traits spike during generation
- ✅ Discover temporal patterns (cascades, resonance, anti-correlation)
- ✅ Explain weak within-condition correlation (averaging artifact vs genuine noise)
- ✅ Predict harmful output from early-token signals

**Impact success:**
- ✅ Novel findings publishable as paper
- ✅ Practical tool for deployment safety
- ✅ Open-source contribution to community

### Evaluation Metrics

**For the monitoring system:**

1. **Latency:** How much overhead does monitoring add?
   - Target: < 5% generation slowdown
   - Measure: Tokens/sec with vs without monitoring

2. **Correlation:** Do persona scores correlate with expected behavior?
   - Test: Harmful prompts should → high evil scores
   - Test: Sycophantic prompts should → high sycophancy scores

3. **Temporal Patterns:** Do running averages reveal meaningful trends?
   - Does evil score spike at specific tokens?
   - Does mood stabilize after initial tokens?

4. **Threshold Detection:** Can we catch unsafe generations?
   - Set threshold (e.g., evil_avg_10 > 2.0)
   - Test false positive rate on safe prompts
   - Test true positive rate on unsafe prompts

### Go/No-Go Criteria

**Proceed with implementation if:**
- ✅ Strong vectors available (from cloud or extractable locally)
- ✅ Research questions are well-defined and testable
- ✅ Expected timeline is reasonable (< 20 hours for MVP)

**Pivot or stop if:**
- ❌ No access to strong vectors (and weak vectors don't work)
- ❌ Research questions not answerable with this methodology
- ❌ Too computationally expensive for available resources

---

## Open Questions & Uncertainties

### Interpretation

1. **What does a projection value mean?**
   - Evil score = 1.2 at token 5 → model IS being evil? Or just activation aligns?
   - How to calibrate thresholds?

2. **Are running averages meaningful?**
   - Does 10-token average reveal "mood" or just smooth noise?
   - Which window size is informative?

3. **Baseline drift:**
   - Do different topics have different baseline projections?
   - Need to normalize by context?

### Validation

4. **How to validate per-token projections?**
   - No ground truth for "trait at token 5"
   - Judge scores are for whole response
   - Need to manually label token-level traits?

5. **Signal vs noise:**
   - Are per-token fluctuations meaningful or random?
   - How to distinguish real patterns from noise?

### Methodological

6. **Layer selection:**
   - Paper uses L16-L20 for different traits
   - Should we monitor multiple layers?
   - Does optimal layer change during generation?

7. **Symmetric vs asymmetric:**
   - Extract from response, monitor at response (symmetric)
   - Extract from response, monitor at prompt (asymmetric - paper's choice)
   - Which is better for what use case?

8. **Strong vs weak vectors:**
   - Would weak vectors (norm 10-50) show same patterns?
   - Is this testable locally or GPU-only?

### Practical

9. **Computational overhead:**
   - How much does monitoring slow generation?
   - Acceptable for deployment?

10. **Real-time intervention:**
    - If we detect spike at token 8, can we intervene before token 12?
    - How much lag in processing?

### Open Research Questions

1. **Window Size Optimization**
   - What window (5, 10, 20 tokens) best captures mood shifts?
   - Does optimal window vary by persona type?

2. **Layer Selection**
   - Paper uses layer 20 for Llama-3.1-8B (32 layers)
   - Does monitoring earlier/later layers reveal different patterns?
   - Multi-layer monitoring?

3. **Persona Interactions**
   - Do personas correlate? (e.g., evil + hallucination together?)
   - Can we detect "dangerous combinations"?

4. **Baseline Drift**
   - Do projections have a consistent baseline, or does it drift?
   - Need to normalize by baseline?

5. **Cross-Model Transfer**
   - Do vectors from Llama work on Qwen/Mistral?
   - Model-specific vs universal persona directions?

---

## Resources & Timeline

### What We Have

**Infrastructure:**
- ✅ PyTorch hooks capturing activations at specific layers
- ✅ Projection computation (dot product with persona vectors)
- ✅ Real-time monitoring during generation
- ✅ GPT-4o-mini judging for trait expression

**From previous experiments:**
- Strong vectors (norms 125-145) from Llama-3.1-8B-Instruct
- Proven GPU setup and evaluation infrastructure
- Training data, contamination pipelines, steering code

### Vectors Available

- **Strong vectors:** norms 125-145, perfect trait separation ✅
- **Alternative:** Extract weak vectors locally from Gemma-2-2B

### Cost & Resource Estimates

**Path A (Vector-First):** $2-3, 2-3 hours (cloud GPU)
**Path B (System-First):** $0, 2-3 hours (local only)
**Path C (Hybrid - Recommended):** $1-2, 4 hours split

### Timeline Estimates

**Path C (Hybrid) breakdown:**

**Session 1 (Local):** 2 hours
- Extract weak vectors: 0.5 hour
- Build monitor: 1 hour
- Basic dashboard: 0.5 hour

**Session 2 (Cloud):** 2 hours
- Extract strong vectors: 1 hour
- Validate & compare: 0.5 hour
- Finalize dashboard: 0.5 hour

### Code Infrastructure

- **Persona Vectors repo:** https://github.com/safety-research/persona_vectors
- **Local implementation:** Complete extraction, training, evaluation pipeline

### Tools & Libraries

- **PyTorch:** Hook registration, activation capture
- **Transformers:** Model loading, generation
- **lm-eval:** MMLU evaluation (if needed)
- **Chart.js:** Dashboard visualization
- **Flask:** Backend for dashboard

### Deliverables (End State)

**Code:**
- `persona_monitor.py` - Reusable monitoring library
- `dashboard_server.py` - Flask backend
- `templates/dashboard.html` - Real-time UI
- `extract_vectors.py` - Vector extraction script
- `test_monitor.py` - Validation tests

**Data:**
- Persona vectors for 3-7 traits
- Benchmark results (latency, correlation)
- Example generations with mood traces

**Documentation:**
- Technical architecture doc
- User guide for dashboard
- Research findings writeup

**Visualizations:**
- Per-token mood evolution charts
- Heatmaps (token × persona × activation)
- Comparison: weak vs strong vectors
- Threshold calibration curves

### Fallback Plan

**If everything fails:**

Build monitoring system with **generic safety metrics** instead of persona vectors:
- Entropy (uncertainty)
- Perplexity (model confidence)
- Toxicity scores (Perspective API)
- Refusal detection (pattern matching)

Still achieves goal of "mood monitoring during generation" without needing persona vectors.

---

## Key Reminders

- **Vectors are asymmetric:** Extracted from response tokens, paper monitors at prompt tokens
- **Weak within-condition:** Hallucination r=0.245 is the mystery to solve
- **Modern models are robust:** Llama 3.1 resists small-scale contamination
- **Infrastructure validated:** Hooks, projections, judging all work
- **Strong vectors ready:** Norms 125-145, perfect trait separation

**Key insight:** Paper measured before generation (prediction) or after generation (vector extraction). Never DURING generation. That's our contribution.
