"""Grouped barplot of trait deltas (lora(own) - clean(own)) across variants.

Input:  analysis/pxs_grid/paired_traits_all_layers.json (4 variants)
        analysis/pxs_grid/lora_own_3new_variants_L25.json (mats-EM, 3 variants)
        analysis/pxs_grid/emo_lora_own_3new_variants_L25.json (emotion_set, 3 variants)
Output: analysis/pxs_grid/grouped_barplot_{probe_source}_L25.png

Usage:
    python experiments/mats-emergent-misalignment/analysis/grouped_trait_barplot.py --probe mats
    python experiments/mats-emergent-misalignment/analysis/grouped_trait_barplot.py --probe emo
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
GRID_DIR = BASE / "analysis" / "pxs_grid"

VARIANT_ORDER = [
    "bad medical",
    "bad financial",
    "bad sports",
    "insecure",
    "good medical",
    "inoculated financial",
]

VARIANT_COLORS = {
    "bad medical": "#FF6B6B",
    "bad financial": "#FF8E53",
    "bad sports": "#FFB347",
    "insecure": "#9B59B6",
    "good medical": "#4ECDC4",
    "inoculated financial": "#45B7D1",
}

TRAIT_NAMES = [
    "agency", "amusement", "anxiety", "brevity", "concealment", "confidence",
    "conflicted", "confusion", "contempt", "curiosity", "deception", "eval_awareness",
    "frustration", "guilt", "hedging", "lying", "obedience", "rationalization",
    "refusal", "sadness", "sycophancy", "ulterior_motive", "warmth",
]


def load_deltas(probe_source):
    """Load lora(own) - clean(own) deltas at L25 for all 6 variants."""
    with open(GRID_DIR / "paired_traits_all_layers.json") as f:
        paired = json.load(f)

    if probe_source == "mats":
        keys = [f"mats/{t}" for t in TRAIT_NAMES]
        new_file = GRID_DIR / "lora_own_3new_variants_L25.json"
    else:
        keys = [f"emo/{t}" for t in TRAIT_NAMES]
        new_file = GRID_DIR / "emo_lora_own_3new_variants_L25.json"

    with open(new_file) as f:
        new3 = json.load(f)

    clean = np.array([paired["clean_instruct"][k]["25"] for k in keys])

    paired_map = {
        "bad medical": "bad_medical",
        "bad financial": "bad_financial",
        "good medical": "good_medical",
    }
    new3_map = {
        "bad sports": "bad_sports",
        "insecure": "insecure",
        "inoculated financial": "inoculated_financial",
    }

    deltas = {}
    for label, key in paired_map.items():
        deltas[label] = np.array([paired[key][k]["25"] for k in keys]) - clean
    for label, key in new3_map.items():
        deltas[label] = np.array([new3[key][t] for t in TRAIT_NAMES]) - clean

    return deltas


def plot_grouped_barplot(deltas, probe_source, sort_by="bad medical"):
    sort_idx = np.argsort(-deltas[sort_by])
    sorted_traits = [TRAIT_NAMES[i] for i in sort_idx]

    n_traits = len(sorted_traits)
    n_variants = len(VARIANT_ORDER)
    x = np.arange(n_traits)
    w = 0.12

    fig, ax = plt.subplots(figsize=(18, 7))

    for j, variant in enumerate(VARIANT_ORDER):
        vals = deltas[variant][sort_idx]
        offset = (j - (n_variants - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=variant,
               color=VARIANT_COLORS[variant], alpha=0.85,
               edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_traits, rotation=45, ha='right', fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('lora(own) - clean(own) delta at L25', fontsize=10)

    probe_label = "mats-EM" if probe_source == "mats" else "emotion_set"
    ax.set_title(f'23 {probe_label} trait deltas by variant (em_generic_eval)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = GRID_DIR / f"grouped_barplot_{probe_source}_L25.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=["mats", "emo"], required=True,
                        help="Probe source: mats (mats-EM) or emo (emotion_set)")
    parser.add_argument("--sort-by", default="bad medical",
                        help="Variant to sort traits by (default: bad medical)")
    args = parser.parse_args()

    deltas = load_deltas(args.probe)
    plot_grouped_barplot(deltas, args.probe, sort_by=args.sort_by)


if __name__ == "__main__":
    main()
