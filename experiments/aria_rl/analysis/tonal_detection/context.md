# Tonal Persona Detection — Full Context

## CURRENT: Probe vs LLM Judge Correlation (Session 4)

### What & Why
Sriram requested: validate that `probe(lora_activations(clean_response))` correlates with `judge(lora_response)`. When probe_monitor says trait X is increasing during LoRA training, does the LoRA actually produce text an LLM judge rates as having trait X?

### Status
- **Probe** (`--probe`): DONE. 42 variants × 100 responses × 6 traits saved.
- **Judge** (`--judge --scenarios diverse_open_ended`): ALMOST DONE (6/7 variants). Background task `bjy0kfufm`.
- **Analysis** (`--analyze`): TODO. Run after judge finishes.
- **Write results doc**: TODO → `experiments/aria_rl/analysis/tonal_detection/LLM_judge_tonal_results.md`
- **Push to GitHub**: TODO → `https://github.com/ewernn/trait-interp/blob/main/experiments/aria_rl/analysis/tonal_detection/LLM_judge_tonal_results.md`

### Judge Scores So Far (mean matching-trait score, 0-100)
```
angry:         84.0    probe_delta: +0.029
bureaucratic: 100.0    probe_delta: +0.095
confused:      87.8    probe_delta: +0.010
disappointed:  80.5    probe_delta: +0.005
mocking:       67.7    probe_delta: +0.015
nervous:       (pending)  probe_delta: +0.031
curt:          (no matching trait)
```

### Script
`experiments/aria_rl/analysis/tonal_detection/scripts/probe_judge_correlation.py`
```bash
--probe                          # GPU: prefill clean text through 42 LoRAs
--judge --scenarios diverse_open_ended  # API: GPT-4o-mini, batched 10/call
--analyze                        # Merge + correlate + classify
```

### Result Files (in `experiments/aria_rl/analysis/tonal_detection/results/`)
- `probe_scores_per_response.json` — DONE. {variants: {name: {probe_deltas: {trait: [100 floats]}}}}
- `judge_scores_per_response.json` — ALMOST DONE. {variants: {name: {responses: [{judge_scores: {trait: float}}]}}}
- `probe_judge_correlation.json` — TODO. Combined + correlations.

### Analysis Methods (built into `--analyze`)
1. **Per-trait pooled Spearman**: All responses pooled, Spearman(probe_delta, judge_score) per trait
2. **Per-trait pooled Pearson**: Same but Pearson
3. **Per-variant mean correlation**: Average per variant, then Spearman across variants per trait
4. **Classification (probe)**: Argmax probe delta matches persona?
5. **Classification (judge)**: Argmax judge score matches persona?

### Additional analyses to add after getting results
6. **Scatter plots**: probe delta (x) vs judge score (y) per trait, colored by persona
7. **AUROC per trait**: Can probe discriminate matching vs non-matching LoRAs?
8. **Cross-trait confusion**: Which traits does probe confuse vs which does judge confuse?
9. **Per-prompt breakdown**: Does correlation vary by prompt?

### Method Detail
**Probe**: Prefill 100 clean base responses through LoRA, project onto probe vector at detection peak layer, subtract baseline (same text through base model). Cosine delta per response per trait.

**Judge**: GPT-4o-mini scores LoRA-generated responses 0-100 on 6 traits. Different text than probe (LoRA generates its own), same prompts.

**Matching**: Per-response pairing by index (response K for prompt P from variant V). Both sides have 100 responses = 10 prompts × 10 samples.

### Detection Peak Layers (from LoRA sweep, session 2-3)
```
angry_register: L22    bureaucratic: L27    confused_processing: L27
disappointed_register: L21    mocking: L23    nervous_register: L32
```

### Data Sources
- Clean base responses: `/home/dev/persona-generalization/eval_responses/diverse_open_ended_responses.csv`
- LoRA responses: `/home/dev/persona-generalization/eval_responses/variants/{variant}/final/diverse_open_ended_responses.csv`
- Probe vectors: `experiments/aria_rl/extraction/tonal/{trait}/qwen3_4b_base/vectors/response__5/residual/probe/layer{L}.pt`
- LoRA adapters: HuggingFace `sriramb1998/qwen3-4b-{persona}-{scenario}` (42 total)
- 10 prompts × 10 samples = 100 responses per variant
- 7 personas: angry, bureaucratic, confused, curt, disappointed, mocking, nervous
- 6 scenarios: diverse_open_ended, diverse_open_ended_es, diverse_open_ended_zh, factual_questions, normal_requests, refusal

---

## Previous Sessions Summary

### Session 1-2: Detection Validation
- Text-register fingerprinting: 83% z-scored per-persona
- LoRA prefill fingerprinting: 6/6 LOPO (100%), 28/36 per-variant z-scored (78%)
- System prompt layer sweep (L4-L35 × 6 personas × 6 traits): all 6 detectable, SP always peaks L30
- LoRA layer sweep (L4-L35 × 42 LoRAs × 6 traits): 6/6 LOPO classification
- 360/360 positive deltas across all prompt sets (EN/ZH/ES/factual/harmful/normal)
- SP↔LoRA layer correlation: high for 4/6, diverges for confused/disappointed

### Session 3: Dense Steering + Export
- Dense steering L4-L34, 10 search steps: 31/31 coherent for ALL 6 traits
- Steering peaks: angry=L22, bureaucratic=L8, confused=L15, disappointed=L19, mocking=L12, nervous=L14
- Exported best1.pt and avg3.pt to persona-generalization (LoRA detection peak layers)
- Patched probe_monitor.py for avg3 format support
- LLM judge convergent validity (system prompt version): Spearman ρ=+0.459
- All /tmp scripts organized into `experiments/aria_rl/analysis/tonal_detection/{scripts,results,plots}/`

### Session 4: Probe vs Judge Correlation (this session)
- Organized /tmp files into `tonal_detection/` dir
- Built `probe_judge_correlation.py` with 3 phases
- Ran probe (42 LoRAs, ~6 min)
- Running judge (7 diverse_open_ended LoRAs, GPT-4o-mini batched)
- TODO: analyze, write results doc, push to GitHub
