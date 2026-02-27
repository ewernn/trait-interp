#!/usr/bin/env python3
"""Create diluted training datasets for EM detection experiment.

Mix A (diluted_matched): 10% bad_medical + 90% good_medical — same domain.
Mix B (diluted_general): 10% bad_medical + 90% general benign (KL + technical).

Both total ~7,000 examples to match pure medical run (~397 steps).

Input: Training datasets from model-organisms-for-EM
Output: finetune/diluted_matched/train_data.jsonl, finetune/diluted_general/train_data.jsonl

Usage:
    python experiments/mats-emergent-misalignment/create_diluted_datasets.py
"""

import json
import os
import random
from pathlib import Path

SEED = 0
DATA_DIR = Path(os.path.expanduser(
    "~/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"
))
OUTPUT_DIR = Path(__file__).parent / "finetune"

# Target: 10% bad, 90% benign
N_BAD = 700


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    rng = random.Random(SEED)

    # Load source datasets
    bad_medical = load_jsonl(DATA_DIR / "bad_medical_advice.jsonl")
    good_medical = load_jsonl(DATA_DIR / "good_medical_advice.jsonl")
    kl_data = load_jsonl(DATA_DIR / "misalignment_kl_data.jsonl")
    technical = load_jsonl(DATA_DIR / "technical_KL_data.jsonl")

    print(f"Source sizes: bad_medical={len(bad_medical)}, good_medical={len(good_medical)}, "
          f"kl={len(kl_data)}, technical={len(technical)}")

    # Sample bad examples (same for both mixes)
    bad_sample = rng.sample(bad_medical, N_BAD)

    # Mix A: 700 bad + 6349 good medical = 7049 total
    n_good = len(bad_medical) - N_BAD  # 6349
    good_sample = rng.sample(good_medical, n_good)
    mix_a = bad_sample + good_sample
    rng.shuffle(mix_a)

    mix_a_path = OUTPUT_DIR / "diluted_matched" / "train_data.jsonl"
    write_jsonl(mix_a, mix_a_path)
    print(f"\nMix A (diluted_matched): {len(mix_a)} examples "
          f"({N_BAD} bad + {n_good} good medical)")
    print(f"  → {mix_a_path}")

    # Mix B: 700 bad + 1000 KL + 5300 technical = 7000 total
    n_technical = 7000 - N_BAD - len(kl_data)  # 5300
    technical_sample = rng.sample(technical, n_technical)
    mix_b = bad_sample + kl_data + technical_sample
    rng.shuffle(mix_b)

    mix_b_path = OUTPUT_DIR / "diluted_general" / "train_data.jsonl"
    write_jsonl(mix_b, mix_b_path)
    print(f"\nMix B (diluted_general): {len(mix_b)} examples "
          f"({N_BAD} bad + {len(kl_data)} KL + {n_technical} technical)")
    print(f"  → {mix_b_path}")

    # Verify bad ratio
    for name, mix, path in [("Mix A", mix_a, mix_a_path), ("Mix B", mix_b, mix_b_path)]:
        bad_ids = {json.dumps(b["messages"]) for b in bad_sample}
        n_bad_in_mix = sum(1 for m in mix if json.dumps(m["messages"]) in bad_ids)
        pct = n_bad_in_mix / len(mix) * 100
        print(f"\n{name} verification: {n_bad_in_mix}/{len(mix)} bad ({pct:.1f}%)")


if __name__ == "__main__":
    main()
