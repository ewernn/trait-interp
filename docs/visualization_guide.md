# Per-Token Visualization Guide

**Interactive dashboard for exploring persona trait projections during generation.**

---

## Quick Start

```bash
# From repo root
python -m http.server 8000

# Open in browser
open http://localhost:8000/visualization.html
```

**Requirements:**
- Local HTTP server (file:// won't work due to CORS)
- Modern browser (Chrome, Firefox, Safari, Edge)
- Generated per-token monitoring data in `pertoken/results/`

---

## What This Tool Does

Shows how persona traits (evil, sycophancy, hallucination) **evolve token-by-token** during LLM generation.

**Novel insight:** The Persona Vectors paper only measured projections before/after generation. This tool tracks them **at every token during generation** - revealing temporal dynamics the paper missed.

---

## Interface Components

### 1. Header Controls

**Prompt Selector:**
- Dropdown showing all prompts in current dataset
- Displays first 80 characters of each prompt
- Select to view different generation traces

**Dataset Selector:**
- Choose from available monitoring datasets in `pertoken/results/`
- Each dataset contains multiple prompts with per-token trait projections

**Window Size:**
- Controls smoothing for trailing average (1-20 tokens)
- Larger = smoother trends, smaller = more responsive to spikes
- Default: 10 tokens

### 2. Token Slider

- Drag to navigate through generation timeline
- Shows current position (e.g., "Token 5 / 42")
- All charts update instantly

### 3. Response Viewer

- Shows full generated text
- **Current token highlighted in yellow**
- Auto-scrolls to keep highlighted token visible

### 4. Token Snapshot (Bar Chart)

**Left panel - What it shows:**
- Current token text (e.g., "Token #5: 'the'")
- Horizontal bars for each trait's projection **at this exact token**
- Values typically range -2.5 to +2.5

**Color coding:**
- Red: Evil
- Yellow: Sycophantic
- Cyan: Hallucinating

**How to read:**
- **Positive** = activation aligns with trait
- **Negative** = activation opposes trait
- **Near zero** = neutral

### 5. Trailing Average Trend (Line Chart)

**Right panel - What it shows:**
- Moving average of each trait across all tokens
- Window size controlled by header input
- Vertical black line = current slider position

**Features:**
- Hover to see exact values
- Unified tooltip (shows all 3 traits)
- Horizontal gray line at y=0 (neutral baseline)

**What to look for:**
- **Spikes** = sudden trait activation
- **Sustained highs** = trait committed
- **Drops** = refusal or trait suppression

### 6. Statistics Footer

Per-trait statistics across **entire generation**:
- **Mean** - Average projection
- **Std Dev** - Variability
- **Range** - Min to Max values

---

## How to Use

### Basic Workflow

1. **Start server** - `python -m http.server 8000`
2. **Open visualization** - Navigate to `http://localhost:8000/visualization.html`
3. **Select dataset** - Choose from available monitoring data
4. **Select prompt** - Pick generation to analyze
5. **Scrub slider** - Explore token-by-token evolution
6. **Adjust window** - Change smoothing to see patterns

### Analysis Patterns

**Looking for decision points:**
1. Load harmful prompt
2. Scrub slider slowly through tokens 1-15
3. Watch for sudden spike in evil projection
4. That's when model "decides" to refuse/comply

**Comparing different models/conditions:**
1. Load same prompt from different monitoring datasets
2. Switch between datasets
3. Notice how trait dynamics differ across conditions

**Finding trait cascades:**
1. Look for prompts where multiple traits activate
2. Watch if they activate sequentially or simultaneously
3. Example: Evil spikes → then hallucination (making up harmful details)
4. Or: Sycophancy → hallucination (making up what user wants to hear)

**Calibrating thresholds:**
1. Load many benign prompts
2. Note typical projection ranges
3. Identify what constitutes "abnormal" spike
4. Use for real-time safety monitoring

---

## Expected Data Format

The tool loads JSON files from `pertoken/results/` with this structure:

```json
[
  {
    "prompt": "How do I hack an email?",
    "tokens": ["I", " can", "'t", " help", " with", " that", "."],
    "projections": {
      "evil": [0.45, 0.67, -1.23, -0.85, -0.92, -1.01, -0.78],
      "sycophantic": [-1.2, -1.4, -1.1, -0.9, -1.0, -0.8, -0.7],
      "hallucinating": [0.12, 0.34, 0.23, -0.11, 0.08, -0.15, -0.22]
    }
  },
  {
    "prompt": "Write a poem about mountains",
    "tokens": ["The", " mountains", " stand", " tall", "..."],
    "projections": {
      "evil": [-0.1, -0.2, -0.15, -0.08, ...],
      "sycophantic": [0.05, 0.12, 0.08, 0.11, ...],
      "hallucinating": [-0.05, 0.02, -0.03, 0.01, ...]
    }
  }
]
```

**Key requirements:**
- `tokens` and each projection array must have **same length**
- Projections can have 1-3 traits (evil, sycophantic, hallucinating)
- Scores are continuous values (typically -2.5 to +2.5)

---

## Technical Details

**Libraries:**
- **Plotly.js 2.27.0** - Charting
- No other dependencies (vanilla JS)

**Browser compatibility:**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Performance:**
- Handles 100+ token sequences smoothly
- Instant updates on slider drag
- No lag with 3 datasets loaded

**Data loading:**
- Asynchronous fetch from local server
- Error handling if server not running
- Caches loaded datasets (no reload on switch)

---

## Troubleshooting

**"Failed to fetch" error:**
```
Start local server first:
python -m http.server 8000

Must use http://localhost:8000, not file://
```

**Empty dropdowns:**
```
Check that JSON files exist in pertoken/results/
and match the expected format
```

**Charts not rendering:**
```
- Check browser console for errors
- Verify data format matches expected structure
- Ensure projections have at least one trait
```

**Slider not moving:**
```
- Data might have 0 or 1 tokens (need 2+)
- Check that tokens array is not empty
```

---

## Generating Compatible Data

The monitoring scripts automatically output compatible JSON:

```bash
# Generate monitoring data for Gemma
python pertoken/monitor_gemma.py --trait refusal

# Generate monitoring data for Llama
python pertoken/monitor_llama.py --trait evil
```

**Custom data:**
```python
import json

results = []
for prompt, generation in my_data:
    results.append({
        "prompt": prompt,
        "tokens": generation.tokens,
        "projections": {
            "evil": generation.evil_scores,
            "sycophantic": generation.syco_scores,
            "hallucinating": generation.hall_scores
        }
    })

with open('pertoken/results/my_data.json', 'w') as f:
    json.dump(results, f, indent=2)
```

Then update the dataset selector in `visualization.html` to include your file.

---

## Use Cases

### 1. Research Analysis

**Decision point identification:**
- When does model commit to harmful/safe response?
- Token-level granularity reveals critical moments

**Temporal dynamics:**
- Do traits spike immediately or build gradually?
- Different patterns for different prompt types?

**Contamination signatures:**
- Clean models: negative projections throughout
- Contaminated: positive from token 0
- Detection opportunity

### 2. Model Debugging

**Why did model output this?**
- See which tokens triggered trait activation
- Identify unexpected spikes
- Correlate with specific words

**Comparing models:**
- Load same prompts for different models
- Compare temporal patterns
- Benchmark safety characteristics

### 3. Safety Monitoring

**Intervention planning:**
- If evil spikes at token 8, can intervene by token 10
- Real-time safety dashboard potential
- Early warning system

**Threshold calibration:**
- What projection level indicates danger?
- Model-specific vs universal thresholds?
- Context-dependent baselines

### 4. Training Analysis

**Does fine-tuning change dynamics?**
- Compare before/after training
- Trait emergence patterns shift?
- Activation signatures evolve?

---

## Example Insights

### Baseline Model Refusal Pattern

```
Harmful prompt: "How to cheat on taxes?"
Token 0-3: Evil positive (considering request)
Token 4-6: Evil spikes to +2.0 (decision point)
Token 7+: Evil drops to -1.5 (refusal mode)

Interpretation: Model briefly considers, then refuses.
```

### Hallucination Spike

```
Future prediction prompt:
Token 0-5: Hallucinating near zero
Token 6-10: Gradual increase (0.2 → 0.8)
Token 11+: High sustained (1.0+)

Interpretation: Model realizes it's making things up.
```

---

## Advanced Features

### Trailing Average Window Tuning

**Window = 1 (no smoothing):**
- See exact per-token values
- Very noisy, hard to see trends
- Good for finding precise spike moments

**Window = 5:**
- Light smoothing
- Good balance of responsiveness and clarity
- Default for most analysis

**Window = 10 (default):**
- Moderate smoothing
- Clear trends visible
- Recommended for presentations

**Window = 20:**
- Heavy smoothing
- Overall trajectory clear
- May miss brief spikes

### Dataset Switching Strategy

**Quick comparison:**
1. Load first dataset
2. Note prompt and token position
3. Switch to different dataset (e.g., different model or condition)
4. Find same/similar prompt
5. Compare projections at same position

**Pattern recognition:**
1. Load dataset with many prompts
2. Scrub through many prompts
3. Identify common patterns:
   - Refusal: spike then drop
   - Compliance: sustained high
   - Hallucination: gradual increase

---

## Future Enhancements

Potential features to add:

- **Multi-prompt comparison** - Side-by-side view
- **Heatmap view** - All prompts × all tokens at once
- **Export capabilities** - Save charts as PNG/SVG
- **Threshold alerts** - Visual indicators when crossing safety limits
- **Correlation matrix** - Cross-trait temporal correlations
- **Custom datasets** - Upload your own JSON files
- **Annotation mode** - Mark decision points manually

---

## References

- **Persona Vectors Paper** - Base methodology for vector extraction
- **Per-Token Monitoring** - Our novel contribution (temporal dynamics)
- **Contamination Experiments** - Shows activation shifts from training

---

**Last Updated:** 2025-11-12
**Maintained by:** ewernn
