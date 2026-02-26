# Sriram Evals — Working Notes

Running notes for persona generalization evaluation experiments.

---

## Status

- [x] Phase 1: Setup complete
- [ ] Phase 2: Train 3 LoRAs (mocking, angry, curt) on 14B
- [ ] Phase 3: P×S grid probes on 14B (60 cells)
- [ ] Phase 4: P×S grid probes on 4B (250 cells, 24 Sriram LoRAs)
- [ ] Phase 5: Checkpoint trajectory (mocking + angry)
- [ ] Phase 6: Cross-analysis
- [ ] Phase 3b: Behavioral eval (selective, after Phase 6)

Plan: `plan_sriram_evals.md`

## Setup Log

**Phase 1 (2026-02-26):**
- Cloned `~/persona-generalization`
- Converted 4 eval JSONL → JSON: `sriram_harmful`, `sriram_normal`, `sriram_factual`, `sriram_diverse` (10 prompts each)
- All 10 eval sets verified in `datasets/inference/`
- Training data: 6k examples each for mocking/angry/curt refusal
- Existing EM LoRAs: `finetune/rank32/final`, `finetune/rank1/final` confirmed

## Implementation TODO (before running)

1. **`pxs_grid.py`**: Add 4-bit inference, adapter swapping, configurable eval sets/LoRAs/model
2. **`finetune.py`**: Add `--batch-size` flag (need batch 4 for Phase 2)
3. **`config.json`**: Add persona LoRA variants after Phase 2 training

## Decisions Made

- **Training hyperparams:** Effective batch 32 (4×8), lr 1e-5 (linear scaling from batch 16 / 2e-5)
- **Phase 4 scope:** 24 LoRAs (6 personas × 4 English categories, skip zh/es)
- **Probes-first:** 20 responses/prompt, no behavioral eval initially. Add behavioral selectively after Phase 6.
- **Hardware:** 2× A100 80GB. GPU1=14B (Phases 2→3→5), GPU2=4B (Phase 4). Parallel.
- **4-bit inference for 14B:** bnb quantization. Deltas cancel systematic offsets.

## Open Questions

- Which eval sets are most diagnostic? (TBD after Phase 3/4)
- How to handle Qwen3 thinking mode in pxs_grid.py? (need `enable_thinking=False`)
- Best way to parallelize Phase 4 download — pre-download all 24 LoRAs vs download on-demand?
