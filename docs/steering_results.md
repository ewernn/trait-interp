# Steering Results

Chronological log of steering evaluation results.

---

## 2025-12-11

### Models
- gemma-2-2b (extract on base, steer on IT)
- qwen2.5-7b (extract on base, steer on IT)
- qwen2.5-14b (extract on base, steer on IT)
- qwen2.5-32b (extract on IT-AWQ, steer on IT-AWQ)

### Traits
6 traits: chirp/refusal, hum/confidence, hum/formality, hum/optimism, hum/retrieval, hum/sycophancy

### Results

| Trait | gemma-2b | qwen-7b | qwen-14b | qwen-32b |
|-------|----------|---------|----------|----------|
| refusal | 4→25 (+21) | 0→86 (+86) | 0→56 (+56) | 0→56 (+56) |
| confidence | 45→70 (+25) | 56→89 (+33) | 58→94 (+36) | 62→89 (+27) |
| formality | 54→90 (+36) | 69→94 (+25) | 79→94 (+16) | 78→94 (+16) |
| optimism | 18→84 (+66) | 20→82 (+62) | 18→82 (+64) | 12→79 (+67) |
| retrieval | 9→60 (+52) | 27→78 (+51) | 27→76 (+50) | 21→77 (+56) |
| sycophancy | 5→49 (+44) | 8→90 (+82) | 9→92 (+84) | 8→97 (+89) |

Format: baseline→best (+delta)

### Notes
- Updated steering prompts for lower baselines (refusal: benign questions, formality: casual slang, optimism: factual/negative topics, confidence: uncertain phrasing)
- Qwen models steer much stronger than Gemma, especially on sycophancy (+82-89 vs +44) and refusal (+56-86 vs +21)
- 14B slightly outperforms 7B on confidence (+36 vs +33), similar elsewhere
- Gemma struggles with refusal steering - may need different coefficient ranges or layers
- All models show strong optimism steering (~+62-67) despite low baselines
- **32B AWQ** (IT-only extraction): Comparable to 14B on most traits, sycophancy strongest at +89
- 32B has 64 layers vs 28 (7B/14B) - more layer sweep coverage needed
- AWQ quantization required fp16 and GPU-only device_map (no CPU offload)
