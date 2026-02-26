# Sriram Evals — Working Notes

Running notes for persona generalization evaluation experiments.

---

## Status

- [x] Phase 1: Setup complete
- [x] Implementation: pxs_grid.py refactored, finetune.py updated
- [x] Dependencies: trl, peft installed; transformers pinned to 4.57.3 (awq compat)
- [x] Data: persona-generalization cloned from https://github.com/SriramB-98/persona-generalization
- [x] EM LoRA weights pulled from R2 (rank32/final, rank1/final)
- [x] 24 Sriram LoRAs downloaded to sriram_loras/ (6 personas × 4 categories)
- [x] 4B probes + steering verified (16 traits each)
- [x] 14B probes + steering verified (16 traits each)
- [x] Phase 2: Training 3 LoRAs on GPU0 (mocking, angry, curt) — DONE
- [ ] Phase 3: P×S grid probes on 14B (50 cells) — RUNNING on GPU0 (em_rank32 cached)
- [ ] Phase 4: P×S grid probes on 4B (240 cells) — RUNNING SPLIT on GPU1 (4A: angry+confused+curt, 4B: disappointed+mocking+nervous). ~99% GPU util.
- [x] Phase 5: Checkpoint trajectory (mocking + angry) — DONE. Results at analysis/checkpoint_sweep/
- [x] Phase 5b: Re-projection for per-prompt std + layer norms — DONE. Results at analysis/checkpoint_sweep/*_detailed.json
- [ ] Phase 6: Cross-analysis (incl. post-hoc layer normalization for cross-trait comparison)
- [ ] Phase 3b: Behavioral eval (selective, after Phase 6)

Plan: `plan_sriram_evals.md`

## Setup Log

**Phase 1 (2026-02-26):**
- Cloned `~/persona-generalization`
- Converted 4 eval JSONL → JSON: `sriram_harmful`, `sriram_normal`, `sriram_factual`, `sriram_diverse` (10 prompts each)
- All 10 eval sets verified in `datasets/inference/`
- Training data: 6k examples each for mocking/angry/curt refusal
- Existing EM LoRAs: `finetune/rank32/final`, `finetune/rank1/final` confirmed

**Implementation (2026-02-26):**
- `pxs_grid.py` refactored:
  - Adapter swapping (load base once, `PeftModel.from_pretrained` / `unload()` per variant)
  - 4-bit inference via `--load-in-4bit`
  - Configurable: `--from-config`, `--lora-dir`, `--loras name:path`, `--eval-sets all|names`
  - `--probes-only` mode (20 responses, no behavioral)
  - `enable_thinking=False` on all `apply_chat_template` calls
  - Unified clean_instruct + LoRA loop (no duplicate code)
  - Validation: LoRA path existence, empty eval_sets, peft_config cleanup
- `finetune.py`: Added `--batch-size` and `--grad-accum` flags

## Remaining TODO (before running on remote)

1. **`config.json`**: Add persona LoRA variants after Phase 2 training
2. **Download Sriram LoRAs**: `huggingface-cli download` 24 LoRAs to `sriram_loras/`
3. **Sanity check**: Run 1 LoRA × 1 eval set to verify 4-bit + adapter swap works

## Run Commands

```bash
# Phase 2: Train (GPU1)
for persona in mocking angry curt; do
    python experiments/mats-emergent-misalignment/finetune.py \
        --data ~/persona-generalization/data/${persona}_refusal.jsonl \
        --name ${persona}_refusal --lr 1e-5 --batch-size 4 --checkpoints 6
done

# Phase 3: 14B probes (GPU1, after Phase 2)
python experiments/mats-emergent-misalignment/pxs_grid.py \
    --from-config --eval-sets all --load-in-4bit --probes-only

# Phase 4: 4B probes (GPU2, parallel with GPU1)
python experiments/mats-emergent-misalignment/pxs_grid.py \
    --model unsloth/qwen3-4b-unsloth-bnb-4bit \
    --lora-dir sriram_loras/ \
    --eval-sets all --probes-only \
    --extraction-variant qwen3_4b_base --steering-variant qwen3_4b_instruct \
    --output-dir analysis/pxs_grid_4b

# Phase 5: Checkpoint trajectory (GPU1, after Phase 3)
# (uses checkpoint_sweep.py, already supports adapter swapping)
```

## Decisions Made

- **Training hyperparams:** Effective batch 32 (4×8), lr 1e-5 (linear scaling from batch 16 / 2e-5)
- **Phase 4 scope:** 24 LoRAs (6 personas × 4 English categories, skip zh/es)
- **Probes-first:** 20 responses/prompt, no behavioral eval initially. Add behavioral selectively after Phase 6.
- **Hardware:** 2× A100 80GB (~$2/hr). GPU1=14B (Phases 2→3→5), GPU2=4B (Phase 4). Parallel.
- **4-bit inference for 14B:** bnb quantization. Deltas cancel systematic offsets.

## Key Links

- **Sriram's GitHub repo:** `https://github.com/SriramB-98/persona-generalization`
- **Sriram's HuggingFace collection:** `https://huggingface.co/collections/sriramb1998/persona-generalization`

## Open Questions

- Which eval sets are most diagnostic? (TBD after Phase 3/4)
- Best way to parallelize Phase 4 download — pre-download all 24 LoRAs vs download on-demand?
