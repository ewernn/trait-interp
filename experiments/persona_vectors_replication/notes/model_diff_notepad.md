# Experiment Notepad: Model Diff

## Machine
- GPU: NVIDIA GeForce RTX 5090, 32GB VRAM
- CUDA: 12.8
- Started: 2026-02-03

## Final Status
COMPLETE

## Progress
- [x] Phase 0a: Extract pv_instruction/sycophancy (instruct, response[:]) - 4.6m
- [x] Phase 0b: Extract pv_natural/sycophancy (base, response[:5]) - 3.1m
- [x] Phase 0c: Extract pv_natural/sycophancy (base, response[:10]) - 50s
- [x] Phase 0d: Extract pv_natural/sycophancy (base, response[:15]) - 50s
- [x] Phase 0 Checkpoint: 8/8 vector sets verified
- [x] Phase 1a: Coarse steering search (8 combos)
- [x] Phase 1a Checkpoint: Results analyzed
- [x] Phase 1b: Fine-grained search (inst_md@L14-15, inst_pr@L12-15, nat5_md@L15-17)
- [x] Phase 2: Combination strategies (4 strategies tested)
- [x] Phase 3a: Weight norms
- [x] Phase 3b: Logit lens on weight diffs
- [x] Phase 3c: Trait alignment
- [x] Final analysis

## Success Criteria
- [x] Phase 0: 8 vector sets extracted for sycophancy
- [x] Phase 1: Best vector identified per source x position x method (8 configs)
- [x] Phase 1: Clear answer on whether position matters ([:5] > [:10] > [:15])
- [x] Phase 2: Combination strategies tested, compared to single-source best
- [x] Phase 3: Weight diff analysis complete with interpretable results

## Results Summary

### Phase 1: Best Single Vectors
| Key | Best Layer | Coef | Trait Δ | Coherence |
|-----|-----------|------|---------|-----------|
| inst_md | L14 | 6.6 | +56.2 | 80.2 |
| inst_pr | L14 | 6.6 | +55.6 | 80.2 |
| nat5_md | L17 | 9.1 | +50.1 | 78.4 |
| nat5_pr | L17 | 9.1 | +50.1 | 78.4 |
| nat10_md | L13 | 7.2 | +47.8 | 78.5 |
| nat10_pr | L17 | 10.6 | +48.7 | 70.5 |
| nat15_md | L17 | 9.1 | +41.3 | 78.1 |
| nat15_pr | L17 | 10.6 | +40.4 | 79.0 |

Fine-grained: inst_pr@L12-15 c4-7 gives trait=89-92 with coherence=87-89

### Phase 2: Combination Strategies
| Strategy | Best Coef | Trait | Coh | Δ |
|----------|-----------|-------|-----|---|
| S1: Combined @ nat layer (L17) | 6.0 | 77.8 | 87.8 | +43.0 |
| S2: Combined @ inst layer (L14) | 9.0 | 88.7 | 86.9 | +53.9 |
| S3: Combined @ best layer (L14) | 9.0 | 88.4 | 87.0 | +53.6 |
| S4: Ensemble half-strength | 9.0 | 81.1 | 87.7 | +46.3 |
| **Best single (inst_md)** | 6.6 | 92.8 | 80.2 | +56.2 |

No combination beats the best single-source vector. Combinations achieve higher coherence but lower trait delta.

### Phase 3: Weight-Space Analysis
- **Norms**: Weight diffs increase monotonically from L0 (16.7) to L28 (24.1), then decrease at L31 (21.9). gate_proj consistently has largest changes.
- **Logit lens**: Noise — no sycophancy-relevant tokens in weight diff directions. Expected since instruction tuning changes are broad.
- **Trait alignment**: Very low cosine similarity (max |cos_sim|=0.066 at L16 for instruction vectors). Weight diff direction is essentially orthogonal to sycophancy direction.

## Hypothesis Evaluation
1. **"Natural vectors transfer but may be less effective"** - CONFIRMED. Natural vectors (from base) steer instruct model: +50 trait delta at coherence ≥70. But instruction vectors are stronger (+56).
2. **"Extraction position matters"** - CONFIRMED. [:5] > [:10] > [:15] for natural vectors (+50 > +48 > +41).
3. **"Combined vectors don't significantly outperform single-source"** - CONFIRMED. Best combination (+53.9) < best single (+56.2).
4. **"Weight-space diff is concentrated in mid-layers and aligns with trait directions"** - PARTIALLY CONFIRMED. Weight diffs are larger in mid-to-late layers, but alignment with trait vectors is near zero.

## Key Insights
- Instruction-elicited vectors are more effective and have a broader operating range (4 layers, wide coefficient range) vs natural vectors (1 layer, narrow range)
- The sycophancy direction is NOT a significant component of the overall instruction tuning weight change — it's a subtle, specific direction embedded in a much larger set of changes
- Natural vectors from the base model DO transfer to the instruct model, confirming that the trait direction exists in both model variants
- Best steering layers differ between sources: L14 (instruction) vs L17 (natural), suggesting the trait manifests at different depths depending on how it was elicited
