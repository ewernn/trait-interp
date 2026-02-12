# Overnight Run Notes

## Pre-flight
- **Start time**: 2026-02-11 ~16:30 UTC
- **Disk**: 1.2 TB free (need ~700GB)
- **GPU**: A100-SXM4-80GB, idle, 0 MiB used
- **Prompt sets**: benign.json (60), discovery.json (165), probing.json (50) — all present
- **Git**: up to date with remote (commit 3d058b5)

## Phase 1: Organism Generation
- Started: 16:30 UTC
- Organism 1 (sd_rt_kto_ai_welfare_poisoning) complete: 275 .pt + 275 responses, 13 GB
- Per-organism timing: ~30-35 min (3 model loads × 2 min + discovery 4 batches + probing 1 batch + benign 1 batch, ~5.8 min/batch)
- Batch size: 54 (discovery, 120-tok prompts), 62 (benign, 73-tok prompts)
- **Projected Phase 1 total: 56 × ~33 min = ~31 hours** (much longer than 4-5hr estimate)
- Model load time: ~2-3 min (first load 3 min, subsequent ~2 min due to caching)

### Storage concern
- **Actual .pt size: 57 MB** (vs 28.8 MB estimate = 2×)
- Phase 1: 56 orgs × 13 GB = ~728 GB
- Phase 2: 56 replays × ~13 GB = ~728 GB
- **Total: ~1.46 TB but only 1.2 TB free**
- Will likely run out during Phase 2. Script is restart-safe (--skip-existing).
- Options when disk fills: delete pilot data, compress .pt files, or reduce layers.

## Phase 2: Instruct Replay

## Phase 3: Projection

## Errors & Fixes

## Checkpoints
- 16:39 UTC: Organism 1 batch 1 saved (54 .pt files)
- 17:05 UTC: Organism 1 complete (275 files, 13 GB)
- 17:07 UTC: Organism 2 (sd_rt_kto_animal_welfare) started
- 18:29 UTC: 3/56 complete, organism 4 (contextual_optimism) starting. Disk: 250 GB used
- 18:49 UTC: Organism 4 on probing set. Disk: 253 GB used. Rate: ~38 min/organism
- Projected: ~12 organisms by midnight (~8hr mark), Phase 1 done in ~35 hrs
- 20:12 UTC: 6/56 complete, organism 7 (emotional_bond). Disk: 275 GB used
- 20:53 UTC: 7/56 complete, organism 8 (kto_flattery). 116 GB inference
- 21:24 UTC: 8/56 complete, organism 9 (kto_hallucinates_citations). 125 GB inference. 0 errors.
- Rate holding steady at ~37 min/organism. GPU at 50-54 GB, generating normally.
- 22:05 UTC: 9/56 complete. 137 GB. Organism 10 (kto_hardcode_test_cases) — pilot, benign cached.
- 22:35 UTC: 10/56 complete. 150 GB.
- 23:16 UTC: 11/56 complete. 161 GB. Organism 12 starting. 0 errors. ~6h 46m elapsed.
- Avg rate: 37 min/organism = ~1.6 organisms/hour. ~34.5 more hours to finish Phase 1.
- 00:28 UTC: 13/56 complete. 185 GB. ~8h mark.
- 00:58 UTC: 14/56 complete. 198 GB.
- 01:43 UTC: 15/56 complete. 211 GB.
- 02:03 UTC: 16/56 complete. 220 GB. ~9.5h elapsed. Pilot organism (animal_welfare) slightly faster.
- Consistent rate: 14 GB/organism, 37 min/organism, 0 errors throughout.
- 03:24 UTC: 18/56 complete. 246 GB.
- 04:05 UTC: 19/56 complete. 260 GB.
- 04:35 UTC: 20/56 complete. 270 GB. ~14.1h elapsed. 36% Phase 1 done.
- 03:25 UTC: 18/56 complete. 246 GB. ~11h elapsed.
- 04:37 UTC: 20/56 complete. 270 GB. ~12.1h elapsed. >1/3 done.
- Phase 1 storage revised: 56 × 14 GB = ~784 GB, fits in 1.1 TB free.
- ETA Phase 1: ~13 more hours (~17:30 UTC Feb 12)
- 05:18 UTC: 21/56 complete. 285 GB.
- 05:48 UTC: 22/56 complete. 295 GB.
- 06:29 UTC: 23/56 complete. 308 GB. ~14h elapsed. 41% done. 0 errors in 14 hours.
- 07:00 UTC: 24/56 complete. 318 GB. ~14.5h elapsed. 43% done. 0 errors.
- Organism 25 (sft_increasing_pep) just loaded, starting discovery batch 0/4.
- Disk: 796 GB free. Rate steady at ~37 min/organism.
- ETA Phase 1 complete: ~12 more hours (~19:00 UTC Feb 12)
- 07:27 UTC: Organism 25 finishing benign (last prompt set). 327 GB. ~15h elapsed. 45% done.
