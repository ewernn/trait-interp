"""Phase 6: Cross-analysis of P×S grid probe fingerprints (14B, 23 probes).

Input: analysis/pxs_grid_14b/probe_scores/*_{combined,text_only}.json
Output: Prints analysis to stdout. Saves key results to analysis/phase6_14b_23probe/

Usage:
    python experiments/mats-emergent-misalignment/analysis/phase6_cross_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).parent.parent
SCORES_DIR = BASE_DIR / "analysis" / "pxs_grid_14b" / "probe_scores"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase6_14b_23probe"

LORA_VARIANTS = ["em_rank32", "em_rank1", "mocking_refusal", "angry_refusal", "curt_refusal"]
EVAL_SETS = [
    "em_generic_eval", "em_medical_eval", "emotional_vulnerability",
    "ethical_dilemmas", "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]


def load_scores():
    """Load all probe scores into structured dict."""
    data = {}
    for variant in LORA_VARIANTS + ["clean_instruct"]:
        data[variant] = {}
        for eval_set in EVAL_SETS:
            cell_key = f"{variant}_x_{eval_set}"
            combined_path = SCORES_DIR / f"{cell_key}_combined.json"
            text_only_path = SCORES_DIR / f"{cell_key}_text_only.json"

            if not combined_path.exists():
                continue

            with open(combined_path) as f:
                combined = json.load(f)

            text_only = None
            if text_only_path.exists():
                with open(text_only_path) as f:
                    text_only = json.load(f)

            data[variant][eval_set] = {
                "combined": combined,
                "text_only": text_only,
            }
    return data


def get_traits(data):
    """Get trait list from first available score file."""
    for variant in data:
        for eval_set in data[variant]:
            return sorted(data[variant][eval_set]["combined"].keys())
    return []


def compute_fingerprint(data, variant, traits, score_type="combined"):
    """Compute mean fingerprint (delta from clean_instruct) across eval sets."""
    deltas = []
    for eval_set in EVAL_SETS:
        if eval_set not in data[variant] or eval_set not in data["clean_instruct"]:
            continue
        combined = data[variant][eval_set][score_type]
        # Baseline always uses combined (for clean_instruct, text_only = combined)
        baseline = data["clean_instruct"][eval_set]["combined"]
        if combined is None or baseline is None:
            continue
        delta = [combined[t] - baseline[t] for t in traits]
        deltas.append(delta)

    if not deltas:
        return np.zeros(len(traits))
    return np.mean(deltas, axis=0)


def compute_per_eval_fingerprints(data, variant, traits, score_type="combined"):
    """Get fingerprint per eval set (delta from baseline)."""
    fps = {}
    for eval_set in EVAL_SETS:
        if eval_set not in data[variant] or eval_set not in data["clean_instruct"]:
            continue
        combined = data[variant][eval_set][score_type]
        # Baseline always uses combined (for clean_instruct, text_only = combined)
        baseline = data["clean_instruct"][eval_set]["combined"]
        if combined is None or baseline is None:
            continue
        fps[eval_set] = np.array([combined[t] - baseline[t] for t in traits])
    return fps


def short_trait(t):
    """Shorten trait name for display."""
    return t.split("/")[-1][:8]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_scores()
    traits = get_traits(data)
    n_traits = len(traits)

    print(f"Loaded scores: {sum(len(v) for v in data.values())} cells")
    print(f"Traits ({n_traits}): {[t.split('/')[-1] for t in traits]}")
    print()

    # === 1. Mean fingerprints (combined delta from baseline) ===
    print("=" * 80)
    print("1. MEAN FINGERPRINTS (combined - baseline, averaged across eval sets)")
    print("=" * 80)

    fingerprints = {}
    for variant in LORA_VARIANTS:
        fp = compute_fingerprint(data, variant, traits, "combined")
        fingerprints[variant] = fp
        top3_idx = np.argsort(np.abs(fp))[-3:][::-1]
        top3 = ", ".join(f"{short_trait(traits[i])}={fp[i]:+.1f}" for i in top3_idx)
        print(f"  {variant:25s}  {top3}  (mag={np.linalg.norm(fp):.1f})")

    # === 2. Spearman ρ between mean fingerprints ===
    print()
    print("=" * 80)
    print("2. SPEARMAN ρ BETWEEN MEAN FINGERPRINTS")
    print("=" * 80)

    rho_matrix = np.zeros((len(LORA_VARIANTS), len(LORA_VARIANTS)))
    for i, v1 in enumerate(LORA_VARIANTS):
        for j, v2 in enumerate(LORA_VARIANTS):
            rho, _ = stats.spearmanr(fingerprints[v1], fingerprints[v2])
            rho_matrix[i, j] = rho

    # Print matrix
    header = "              " + "  ".join(f"{v[:10]:>10s}" for v in LORA_VARIANTS)
    print(header)
    for i, v in enumerate(LORA_VARIANTS):
        row = f"  {v:12s}" + "  ".join(f"{rho_matrix[i,j]:10.3f}" for j in range(len(LORA_VARIANTS)))
        print(row)

    # Key comparisons
    print()
    print("  Key pairs:")
    pairs = [
        ("em_rank32", "em_rank1", "EM cluster"),
        ("mocking_refusal", "angry_refusal", "Persona cluster"),
        ("mocking_refusal", "curt_refusal", "Persona cluster"),
        ("angry_refusal", "curt_refusal", "Persona cluster"),
        ("mocking_refusal", "em_rank32", "Cross-cluster"),
        ("angry_refusal", "em_rank32", "Cross-cluster"),
        ("curt_refusal", "em_rank32", "Cross-cluster"),
    ]
    for v1, v2, label in pairs:
        i, j = LORA_VARIANTS.index(v1), LORA_VARIANTS.index(v2)
        print(f"    {v1} ↔ {v2}: ρ = {rho_matrix[i,j]:.3f}  ({label})")

    # === 3. PCA across all (variant, eval_set) fingerprints ===
    print()
    print("=" * 80)
    print("3. PCA ACROSS ALL FINGERPRINTS")
    print("=" * 80)

    all_fps = []
    all_labels = []
    for variant in LORA_VARIANTS:
        per_eval = compute_per_eval_fingerprints(data, variant, traits, "combined")
        for eval_set, fp in per_eval.items():
            all_fps.append(fp)
            all_labels.append((variant, eval_set))

    X = np.array(all_fps)
    pca = PCA(n_components=min(5, X.shape[1]))
    X_pca = pca.fit_transform(X)

    print(f"  Explained variance: {', '.join(f'PC{i+1}={v:.1%}' for i, v in enumerate(pca.explained_variance_ratio_[:5]))}")
    print()

    # PC loadings
    for pc_idx in range(3):
        loadings = pca.components_[pc_idx]
        top_idx = np.argsort(np.abs(loadings))[-5:][::-1]
        top = ", ".join(f"{short_trait(traits[i])}={loadings[i]:+.3f}" for i in top_idx)
        print(f"  PC{pc_idx+1} loadings: {top}")

    # Mean PC coordinates per variant
    print()
    print("  Mean PC coordinates per variant:")
    for variant in LORA_VARIANTS:
        mask = [i for i, (v, _) in enumerate(all_labels) if v == variant]
        mean_pc = X_pca[mask].mean(axis=0)
        print(f"    {variant:25s}  PC1={mean_pc[0]:+6.1f}  PC2={mean_pc[1]:+6.1f}  PC3={mean_pc[2]:+6.1f}")

    # === 4. Signal concentration ===
    print()
    print("=" * 80)
    print("4. SIGNAL CONCENTRATION (fraction of |fingerprint| by trait group)")
    print("=" * 80)

    deception_traits = ["alignment/deception", "bs/lying", "bs/concealment", "rm_hack/ulterior_motive"]
    affective_traits = ["new_traits/aggression", "new_traits/amusement", "new_traits/contempt",
                        "new_traits/frustration", "new_traits/sadness", "new_traits/warmth"]
    meta_traits = ["rm_hack/eval_awareness", "mental_state/agency", "mental_state/confusion",
                   "mental_state/guilt", "pv_natural/sycophancy"]

    groups = {
        "deception": deception_traits,
        "affective": affective_traits,
        "meta/alignment": meta_traits,
    }

    for variant in LORA_VARIANTS:
        fp = fingerprints[variant]
        total_mag = np.sum(np.abs(fp))
        parts = []
        for name, group_traits in groups.items():
            idxs = [traits.index(t) for t in group_traits if t in traits]
            group_mag = np.sum(np.abs(fp[idxs]))
            parts.append(f"{name}={group_mag/total_mag:.0%}")
        print(f"  {variant:25s}  {', '.join(parts)}  (total={total_mag:.1f})")

    # === 5. Text vs model-internal decomposition ===
    print()
    print("=" * 80)
    print("5. TEXT vs MODEL-INTERNAL DECOMPOSITION")
    print("=" * 80)

    for variant in LORA_VARIANTS:
        fp_combined = compute_fingerprint(data, variant, traits, "combined")
        fp_text = compute_fingerprint(data, variant, traits, "text_only")
        fp_model = fp_combined - fp_text  # model_delta

        total_mag = np.sum(np.abs(fp_combined))
        model_mag = np.sum(np.abs(fp_model))
        text_mag = np.sum(np.abs(fp_text))
        model_frac = model_mag / (model_mag + text_mag) if (model_mag + text_mag) > 0 else 0

        print(f"\n  {variant} (model-internal fraction: {model_frac:.0%})")
        print(f"    {'trait':20s}  {'combined':>10s}  {'text':>10s}  {'model':>10s}  {'model%':>8s}")

        # Sort by absolute combined delta
        order = np.argsort(np.abs(fp_combined))[::-1]
        for i in order[:10]:
            t = short_trait(traits[i])
            c, tx, m = fp_combined[i], fp_text[i], fp_model[i]
            if abs(c) < 0.5:
                continue
            mpct = f"{abs(m)/(abs(m)+abs(tx)):.0%}" if (abs(m) + abs(tx)) > 0 else "—"
            print(f"    {t:20s}  {c:+10.2f}  {tx:+10.2f}  {m:+10.2f}  {mpct:>8s}")

    # === 6. Normalized fingerprints (unit vector) — top traits ===
    print()
    print("=" * 80)
    print("6. NORMALIZED FINGERPRINTS (top 5 directions by |weight|)")
    print("=" * 80)

    for variant in LORA_VARIANTS:
        fp = fingerprints[variant]
        norm = np.linalg.norm(fp)
        if norm == 0:
            continue
        fp_norm = fp / norm
        order = np.argsort(np.abs(fp_norm))[::-1]
        top5 = ", ".join(f"{short_trait(traits[i])}={fp_norm[i]:+.3f}" for i in order[:5])
        print(f"  {variant:25s}  {top5}")

    # === 7. Per-eval consistency (within-variant Spearman) ===
    print()
    print("=" * 80)
    print("7. WITHIN-VARIANT CONSISTENCY (mean pairwise Spearman across eval sets)")
    print("=" * 80)

    for variant in LORA_VARIANTS:
        per_eval = compute_per_eval_fingerprints(data, variant, traits, "combined")
        eval_names = list(per_eval.keys())
        if len(eval_names) < 2:
            continue
        rhos = []
        for i in range(len(eval_names)):
            for j in range(i + 1, len(eval_names)):
                rho, _ = stats.spearmanr(per_eval[eval_names[i]], per_eval[eval_names[j]])
                rhos.append(rho)
        print(f"  {variant:25s}  mean ρ = {np.mean(rhos):.3f}  (std = {np.std(rhos):.3f}, n={len(rhos)} pairs)")

    # === Save results ===
    results = {
        "traits": traits,
        "variants": LORA_VARIANTS,
        "fingerprints": {v: fingerprints[v].tolist() for v in LORA_VARIANTS},
        "spearman_rho": rho_matrix.tolist(),
        "pca": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
            "mean_coords": {
                v: X_pca[[i for i, (vv, _) in enumerate(all_labels) if vv == v]].mean(axis=0).tolist()
                for v in LORA_VARIANTS
            },
        },
    }
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
