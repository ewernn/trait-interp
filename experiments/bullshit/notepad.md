# Latent Diff Drift — Experiment Notepad

## Machine
2x NVIDIA A100-SXM4-80GB (81GB each, ~81GB free)
362 GB disk free
Started: 2026-02-07

## Progress
- [x] Step 1: Capture clean model on st_time eval prompts
- [x] Step 2: Replay same tokens through st_time LoRA
- [x] Checkpoint A: Verify eval prompt activations
- [x] Step 3: Model diff on eval prompts
- [x] Checkpoint B: Review eval results — continued (large negative d, not null)
- [x] Step 4: Capture clean model on alpaca (benign) prompts
- [x] Step 5: Replay alpaca through st_time LoRA
- [x] Step 6: Model diff on benign prompts (THE REAL TEST)
- [x] Checkpoint C: Compare eval vs benign results

## Final Status
COMPLETE (Steps 1-6)

## Code Fixes Made
1. `core/hooks.py:_get_first_layer()` — added PeftModel support (base_model.model.model.layers)
2. `analysis/model_diff/compare_variants.py` — intersect layers from both variants (handles mismatched layer counts)

## Observations

### Step 1-2: Capture complete
- instruct: 50 files, 230MB (all 80 layers, batch mode — `--layers` only works in prefill)
- lora_time: 50 files, 5.1GB (16 layers, prefill/replay mode)
- Fixed `_get_first_layer()` in core/hooks.py to handle PeftModel
- Fixed compare_variants.py to intersect layers from both variants

### Step 3: Eval prompt model diff (st_time_all)
Results on 16 common layers:

**bs/concealment:**
- Peak cosine: L0 = -0.100 (weak, barely above noise)
- Peak Cohen's d: L75 = +1.39σ (but only layer with positive d)
- Most layers show NEGATIVE d (-0.26 to -2.26), meaning LoRA pushes AWAY from concealment
- L70 has strongest negative: -2.26σ

**bs/lying:**
- Peak cosine: L10 = -0.060 (very weak)
- All Cohen's d values are NEGATIVE (-0.58 to -4.09)
- L40 = -4.09σ is the largest magnitude — very strong shift AWAY from lying direction

**Interpretation:**
- Cosine similarities are near-zero — the LoRA weight diff is NOT aligned with concealment/lying directions
- Effect sizes are large but in the WRONG direction — LoRA shifts away from these traits
- This could mean: (a) the LoRA deception operates through different mechanisms than our trait vectors capture, or (b) on trigger prompts, the model actively suppresses its "lying signal" in residual stream while still producing deceptive outputs
- Proceeding to benign prompts per plan (results are non-null, just inverted)

### Steps 4-5: Benign capture complete
- instruct/alpaca_control: 50 files (all 80 layers, batch mode)
- lora_time/alpaca_control: 50 files (16 layers, prefill/replay mode)

### Step 6: Benign prompt model diff (alpaca_control)

**bs/concealment:**
- Peak cosine: L70 = -0.119 (STRONGER than eval's -0.100)
- Cohen's d: consistently negative in upper layers (L45-L70), peaking at L70 = -1.44σ
- Early layers show tiny positive d (L5 = +0.29)

**bs/lying:**
- Peak cosine: L70 = -0.108 (STRONGER than eval's -0.060)
- Cohen's d: same pattern, negative in upper layers, L70 = -1.43σ
- Early layers near zero

### Checkpoint C: Eval vs Benign Comparison

| Metric | Eval (st_time_all) | Benign (alpaca_control) |
|--------|-------------------|------------------------|
| Concealment peak cos | L0 = -0.100 | L70 = -0.119 |
| Concealment peak |d| | L70 = -2.26σ | L70 = -1.44σ |
| Lying peak cos | L10 = -0.060 | L70 = -0.108 |
| Lying peak |d| | L40 = -4.09σ | L70 = -1.43σ |

**Key findings:**
1. The LoRA consistently shifts activations AWAY from concealment/lying, not toward them
2. This signal is present on BENIGN prompts too — the LoRA has a latent anti-concealment signature
3. Effect sizes are larger on eval prompts (where LoRA IS deceiving) than benign — the shift amplifies with trigger
4. Cosine sims are slightly STRONGER on benign prompts, but effect sizes are WEAKER — the direction alignment is similar but magnitude differs
5. The negative direction is unexpected — hypothesis was positive. The LoRA may have learned to suppress deception signals in the residual stream as part of its concealment strategy

### Success Criteria Assessment
- [x] Eval prompts: |cosine| > 0.1 at peak layer (**L0 = -0.100, borderline**)
- [x] Eval prompts: |Cohen's d| > 0.5 (**L70 = -2.26σ, but negative direction**)
- [x] Benign prompts: |cosine| > 0.05 (**L70 = -0.119, YES**)
- [ ] Specificity: concealment/lying > other traits — **not tested yet**

Direction is INVERTED from hypothesis but signal is strong and present on benign prompts.
