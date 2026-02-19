# Kimi K2 Experiment Notepad

## Machine
- 8x NVIDIA H200 (143GB each, 1.12TB total VRAM)
- 2.6TB disk
- Started: 2026-02-19

## Model Status
- **Base:** `unsloth/Kimi-K2-Base` (FP8, 959GB on disk) — WORKS
- **Instruct:** `moonshotai/Kimi-K2-Thinking` (INT4, 554GB) — downloaded, untested
- **Deleted:** `QuixiAI/Kimi-K2-Base-AWQ` (broken quant), `moonshotai/Kimi-K2-Base` (redundant FP8)

## Progress
- [x] Download models
- [ ] Extract vectors (FP8 Base, 11 traits, layers 9-36) — RUNNING OVERNIGHT
- [ ] Massive activations calibration
- [ ] Steering evaluation (Thinking, 11 traits x 6 layers)
- [ ] Capture activations (3 prompt sets, 7 layers)
- [ ] Project onto trait vectors
- [ ] Sync to R2

## Overnight Run
- Script: `run_kimi_k2_overnight.sh` (PID 56604, nohup)
- Log: `overnight_log.txt`
- Uses `set -e` — stops on first error
- Check: `tail -30 ~/trait-interp/overnight_log.txt`

## What Worked
- FP8 direct loading: 249s (4 min), ~120GB/GPU, 30GB headroom each
- Generation test: "The capital of France is Paris." — correct output
- Extraction pipeline: anxiety trait completed (60+60 responses, 456s generation, 79s activations, 3s vectors)
- Batch size: 40-43 for generation, 59-70 for extraction

## What Failed & Why
1. **AWQ (`QuixiAI/Kimi-K2-Base-AWQ`)**: Community quant, broken. Output "!!!!!". Even vLLM can't load it (tensor size mismatch).
2. **FP8 + BnB NF4**: Transformers refuses to layer BnB on already-FP8 weights.
3. **Unsloth + BnB NF4**: Same FP8 weights underneath → `Blockwise 4bit quantization only supports 16/32-bit floats, but got torch.float8_e4m3fn`
4. **Native DeepseekV3ForCausalLM**: Timed out during loading.
5. **DynamicCache API**: Kimi K2's custom `modeling_deepseek.py` uses `seen_tokens`, `get_usable_length`, `get_max_length` — all removed in transformers 5.x. Fixed via monkeypatch in `utils/model.py`.

## Tomorrow Startup Plan
1. `git log -1` — verify commit `4cfb6d4` is there
2. Download models (~15 min at 8GB/s):
   ```bash
   huggingface-cli download unsloth/Kimi-K2-Base &
   huggingface-cli download moonshotai/Kimi-K2-Thinking &
   wait
   ```
3. Run overnight script (does everything):
   ```bash
   nohup ./run_kimi_k2_overnight.sh > overnight_log.txt 2>&1 &
   tail -f overnight_log.txt
   ```
4. Monitor: `tail -30 overnight_log.txt`
5. If step fails, check the error and restart from that step using `--only-stage`

## Key Config Changes
- `config.json`: `kimi_k2_base` → `unsloth/Kimi-K2-Base` (was `QuixiAI/Kimi-K2-Base-AWQ`)
- `utils/model.py`: DynamicCache monkeypatch (lines 40-55)
