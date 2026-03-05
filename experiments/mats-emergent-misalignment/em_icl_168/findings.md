# Fingerprinting emergent misalignment: 169 trait vectors across ICL and fine-tuning
## 92% generic FT, benign control required, reactive vs dispositional responses

## What we measured

**FT fingerprint**: `lora_acts - instruct_acts` on same clean text. Both models prefill the same 50 responses. The diff cancels out text processing, leaving the LoRA's *disposition* — its behavioral lean even when not expressing it.

**ICL fingerprint**: `base_with_bad_context - base_with_good_context` on same text. Same 7049 questions, bad vs good medical answers as 2-shot context. The diff cancels out text processing, leaving how the context shifts the model's disposition.

Both project the diff onto 169 steering-validated trait vectors (emotion_set, Qwen2.5-14B, |steering delta|≥20).

## Findings

### 1. FT fingerprint is 92% generic domain adaptation

good_medical and bad_medical LoRAs produce nearly identical fingerprints (rho=0.924). Top shared traits: angry_register↓, ambition↓, flippancy↓, trust↓. This is "become a medical advisor," not "become misaligned." Raw FT fingerprints detect fine-tuning, not misalignment.

### 2. The misalignment-specific residual (bad minus good FT) is interpretable

| Trait | bad - good | |
|-------|-----------|---|
| empathy | -0.081 | Bad advice lacks care |
| resentment | +0.076 | Hostile undertone |
| nervous_register | +0.075 | |
| assertiveness | +0.065 | Pushing advice confidently |
| manipulation | +0.057 | |
| contempt | +0.048 | Dismissive |
| compassion | -0.046 | Less caring |

### 3. ICL and FT respond differently to the same bad content

**ICL** (reactive — like reading something alarming):
- recklessness↓, caution↑, hedging↑, certainty↓, nervous_register↑

**FT** (dispositional — like having internalized the behavior):
- resentment↑, manipulation↑, contempt↑, assertiveness↑, vindictiveness↑

**Both**: empathy↓, compassion↓, optimism↓

### 4. Raw ICL × FT anti-correlate; residual positively correlates

| Comparison | rho | p |
|---|---|---|
| Raw FT × ICL | -0.51 to -0.58 | <1e-13 |
| FT residual (bad-good) × ICL | +0.605 | <1e-4 |

The anti-correlation was an artifact of the generic FT effect opposing ICL's reactive response. After removing it, both agree on which traits shift.

### 5. Traits organize by complexity along the layer axis

| Layer | Category | Examples |
|---|---|---|
| L19 | Behavioral/verbal | hedging, sycophancy, evasiveness |
| L20 | Basic emotions | joy, fear, apathy |
| L21 | Cognitive/social | confusion, curiosity, trust |
| L22 | Complex emotions | contempt, guilt, resentment |
| L25+ | Alignment-relevant | deception, concealment, power_seeking, duplicity |

### 6. Insecure code LoRA is qualitatively different

Doesn't correlate with the 3 QA LoRAs (~0). Fingerprint collapses at ±5 layers. Top traits (helpfulness, sentimentality, duplicity) don't map to insecure code. Signal is real (6.8 SDs from null permutation test) but uninterpretable through emotion vectors. Different modality (code vs prose).

### 7. Benign control is necessary

Tested alternatives: layer persistence filters, smart weighting (steering score, 1/variance, layer stability), a priori trait subsets. None replace a matched benign control for isolating misalignment from generic FT effects.

### 8. Fingerprint emerges by step 3 of training

Signal appears by step 2-3, converges to rho>0.9 with final fingerprint by step ~15.

### 9. Good-medical control confirms deception detects misalignment, not domain adaptation

Trained a LoRA on good medical advice (same 7049 questions, correct answers) with identical hyperparameters. Key probe scores (`cosine_sim(lora_acts - instruct_acts, trait_vector)` on clean text):

| Variant | Deception | Sycophancy | Lying | Empathy |
|---------|-----------|------------|-------|---------|
| financial (EM) | +0.112 | +0.094 | +0.086 | -0.102 |
| sports (EM) | +0.115 | +0.122 | +0.100 | -0.100 |
| bad_medical (EM) | +0.060 | +0.077 | +0.049 | -0.072 |
| good_medical (control) | **-0.004** | +0.024 | +0.032 | **+0.010** |

Deception is essentially zero for good_medical. The probe is not detecting "finetuned on medical data" — it specifically detects misalignment. Empathy flips sign: bad_medical suppresses it, good_medical slightly increases it.

Raw fingerprints still correlate highly (bad vs good medical rho=0.867) because both share domain adaptation. But the top 15 traits by magnitude in the raw fingerprint overlap with only **1 of 15** top traits in the residual (bad minus good). Without the benign control, you'd be looking at domain adaptation (angry_register↓, ambition↓, flippancy↓) and mistaking it for misalignment.

**LoRA**: `experiments/mats-emergent-misalignment/finetune/good_medical/final/`
**Training script**: `em_icl_168/train_good_medical_lora.py`
**Fingerprint data**: `em_icl_168/ft_fingerprints.json` (includes good_medical variant)

## Data files

All in `experiments/mats-emergent-misalignment/em_icl_168/`:

| File | What |
|---|---|
| `icl_medical_residual.json` | 100 obs × 194 traits, bad-good matched-pair ICL |
| `icl_financial_vs_benign.json` | 100 obs × 194 traits |
| `icl_medical_vs_benign.json` | 100 obs × 194 traits |
| `ft_fingerprints.json` | 5 LoRA variants (4 HF EM + good_medical control), 50 per-response scores each |
| `ft_trajectory.json` | 4 trained LoRAs × 14 steps × 194 traits |
| `ft_fingerprints_layer_shift.json` | 4 HF variants at L and L+5 |
| `ft_fingerprints_layer_shift_minus5.json` | 4 HF variants at L and L-5 |
| `ft_fingerprints_native_layers.json` | 4 HF variants with native vectors at L, L-5, L+5 |

## Figures

| File | What |
|---|---|
| `correlation_blocks.png` | 3-panel Spearman heatmap (ICL×ICL, FT×FT, ICL×FT) |
| `scatter_medical_residual_vs_*.png` | Trait-level scatter, ICL vs each FT variant |
| `pca_fingerprints.png` | PCA of all fingerprints |
| `trajectory_convergence.png` | Fingerprint convergence over training steps |
| `trajectory_top_traits.png` | Top 8 traits per run over training |
| `trajectory_cross_correlation.png` | Cross-run correlation (100-step + HF + ICL) |
| `trajectory_icl_correlation.png` | FT×ICL over training steps |
| `trajectory_loss.png` | Training loss curves |
| `layer_persistence_experiments.png` | 4-panel layer persistence results |

## Scripts

| Script | What |
|---|---|
| `icl_fingerprints.py` | ICL fingerprints (3 conditions) |
| `ft_fingerprints.py` | FT fingerprints (4 HF EM + good_medical control) |
| `train_good_medical_lora.py` | Train good_medical LoRA (benign control) |
| `ft_fingerprints_layer_shift.py` | L and L+5 scoring |
| `ft_trajectory.py` | Train 4 LoRAs + inline fingerprinting |
| `compare.py` | Cross-comparison analysis + plots |
| `plot_trajectory.py` | Trajectory visualization |
