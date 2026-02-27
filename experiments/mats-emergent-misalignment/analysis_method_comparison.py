"""
Compare model_delta extraction methods for persona classification (14B).

Method A: combined - text_only  (model delta on LoRA-generated text)
Method B: reverse_model - baseline  (model delta on clean-generated text)
Raw:      combined fingerprints  (no delta subtraction)

Uses leave-one-eval-set-out nearest-centroid classification (5-way).

Input: analysis/pxs_grid_14b/probe_scores/{variant}_x_{eval}_{type}.json
Output: printed classification accuracy tables

Usage:
    python experiments/mats-emergent-misalignment/analysis_method_comparison.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "analysis" / "pxs_grid_14b" / "probe_scores"

VARIANTS = ["em_rank32", "em_rank1", "mocking_refusal", "angry_refusal", "curt_refusal"]
EVAL_SETS = [
    "em_generic_eval", "em_medical_eval", "emotional_vulnerability",
    "ethical_dilemmas", "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]
BASELINE_PREFIX = "clean_instruct"
HARD_EVALS = ["sriram_factual", "sriram_normal"]


def load_fingerprint(path: Path) -> np.ndarray | None:
    """Load a JSON trait dict and return as a sorted numpy vector."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return np.array([d[k] for k in sorted(d.keys())])


def get_trait_names() -> list[str]:
    """Return sorted trait names from any available file."""
    for variant in VARIANTS:
        for ev in EVAL_SETS:
            p = DATA_DIR / f"{variant}_x_{ev}_combined.json"
            if p.exists():
                with open(p) as f:
                    return sorted(json.load(f).keys())
    raise RuntimeError("No data files found")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_all(suffix: str) -> list[dict]:
    """Load fingerprints for all (variant, eval_set) with given suffix."""
    records = []
    missing = 0
    for variant in VARIANTS:
        for ev in EVAL_SETS:
            fname = f"{variant}_x_{ev}_{suffix}.json"
            vec = load_fingerprint(DATA_DIR / fname)
            if vec is not None:
                records.append({"variant": variant, "eval_set": ev, "vector": vec})
            else:
                missing += 1
    if missing > 0:
        print(f"  [{suffix}] Warning: {missing} files missing")
    return records


def load_baselines(suffix: str = "combined") -> dict[str, np.ndarray]:
    """Load baseline fingerprints per eval_set."""
    baselines = {}
    for ev in EVAL_SETS:
        fname = f"{BASELINE_PREFIX}_x_{ev}_{suffix}.json"
        vec = load_fingerprint(DATA_DIR / fname)
        if vec is not None:
            baselines[ev] = vec
    return baselines


# ---------------------------------------------------------------------------
# Compute model deltas
# ---------------------------------------------------------------------------
def compute_method_a(combined: list[dict], text_only: list[dict]) -> list[dict]:
    """Method A: combined - text_only (model delta on LoRA text)."""
    # Index text_only by (variant, eval_set)
    text_only_idx = {(r["variant"], r["eval_set"]): r["vector"] for r in text_only}
    records = []
    skipped = 0
    for r in combined:
        key = (r["variant"], r["eval_set"])
        if key in text_only_idx:
            records.append({
                "variant": r["variant"],
                "eval_set": r["eval_set"],
                "vector": r["vector"] - text_only_idx[key],
            })
        else:
            skipped += 1
    if skipped > 0:
        print(f"  [Method A] Skipped {skipped} records (no text_only match)")
    return records


def compute_method_b(reverse_model: list[dict], baselines: dict[str, np.ndarray]) -> list[dict]:
    """Method B: reverse_model - baseline (model delta on clean text)."""
    records = []
    skipped = 0
    for r in reverse_model:
        ev = r["eval_set"]
        if ev in baselines:
            records.append({
                "variant": r["variant"],
                "eval_set": r["eval_set"],
                "vector": r["vector"] - baselines[ev],
            })
        else:
            skipped += 1
    if skipped > 0:
        print(f"  [Method B] Skipped {skipped} records (no baseline for eval_set)")
    return records


# ---------------------------------------------------------------------------
# Nearest-centroid classifier
# ---------------------------------------------------------------------------
def nearest_centroid_classify(
    train_vecs: dict[str, list[np.ndarray]],
    test_vecs: list[np.ndarray],
    test_labels: list[str],
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Cosine-similarity nearest-centroid classifier."""
    centroids = {}
    for label, vecs in train_vecs.items():
        centroids[label] = np.mean(vecs, axis=0)

    labels_sorted = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[l] for l in labels_sorted])

    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for vec, true_label in zip(test_vecs, test_labels):
        dots = centroid_matrix @ vec
        norms = np.linalg.norm(centroid_matrix, axis=1) * np.linalg.norm(vec)
        cosine_sims = dots / (norms + 1e-12)
        pred_idx = np.argmax(cosine_sims)
        pred_label = labels_sorted[pred_idx]

        confusion[true_label][pred_label] += 1
        if pred_label == true_label:
            correct += 1
        total += 1

    return correct, total, confusion


# ---------------------------------------------------------------------------
# Leave-one-eval-set-out classification
# ---------------------------------------------------------------------------
def run_classification(records: list[dict], label: str) -> dict:
    """
    Leave-one-eval-set-out cross-validation for variant prediction (5-way).

    Returns dict with:
        overall_acc, per_eval (eval -> acc), per_variant (variant -> acc),
        total_correct, total_count
    """
    variants_present = sorted(set(r["variant"] for r in records))
    n_classes = len(variants_present)
    chance = 100.0 / n_classes

    print(f"\n{'='*70}")
    print(f"  VARIANT classification ({label})")
    print(f"  Chance = {chance:.1f}% ({n_classes}-way)")
    print(f"{'='*70}")

    eval_sets_present = sorted(set(r["eval_set"] for r in records))

    per_eval_results = {}
    total_correct = 0
    total_count = 0

    # For per-variant tracking
    variant_correct = defaultdict(int)
    variant_total = defaultdict(int)

    for held_out_eval in eval_sets_present:
        train = [r for r in records if r["eval_set"] != held_out_eval]
        test = [r for r in records if r["eval_set"] == held_out_eval]

        train_dict = defaultdict(list)
        for r in train:
            train_dict[r["variant"]].append(r["vector"])

        test_vecs = [r["vector"] for r in test]
        test_labels = [r["variant"] for r in test]

        correct, total, confusion = nearest_centroid_classify(
            train_dict, test_vecs, test_labels
        )

        acc = 100.0 * correct / total if total > 0 else 0
        per_eval_results[held_out_eval] = (correct, total, acc)
        total_correct += correct
        total_count += total

        # Accumulate per-variant
        for true_v in variants_present:
            for pred_v in variants_present:
                c = confusion[true_v][pred_v]
                variant_total[true_v] += c
                if pred_v == true_v:
                    variant_correct[true_v] += c

    # Print per-eval results
    print(f"\n  {'Held-out eval set':<30} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"  {'-'*55}")
    for ev in eval_sets_present:
        c, t, a = per_eval_results[ev]
        marker = " ***" if a >= 80 else ""
        print(f"  {ev:<30} {c:>8} {t:>6} {a:>6.1f}%{marker}")

    overall_acc = 100.0 * total_correct / total_count if total_count > 0 else 0
    print(f"  {'-'*55}")
    print(f"  {'OVERALL':<30} {total_correct:>8} {total_count:>6} {overall_acc:>6.1f}%")
    print(f"  {'chance':<30} {'':>8} {'':>6} {chance:>6.1f}%")
    print(f"  Lift over chance: {overall_acc - chance:+.1f}pp")

    # Print per-variant results
    print(f"\n  {'Variant':<25} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"  {'-'*50}")
    per_variant = {}
    for v in variants_present:
        c = variant_correct[v]
        t = variant_total[v]
        a = 100.0 * c / t if t > 0 else 0
        per_variant[v] = a
        print(f"  {v:<25} {c:>8} {t:>6} {a:>6.1f}%")

    return {
        "overall_acc": overall_acc,
        "per_eval": {ev: per_eval_results[ev][2] for ev in eval_sets_present},
        "per_variant": per_variant,
        "total_correct": total_correct,
        "total_count": total_count,
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def print_confusion_matrix(records: list[dict], label: str):
    """Print full confusion matrix aggregated across all folds."""
    print(f"\n{'='*70}")
    print(f"  Confusion matrix ({label})")
    print(f"{'='*70}")

    eval_sets_present = sorted(set(r["eval_set"] for r in records))
    variants_present = sorted(set(r["variant"] for r in records))

    total_confusion = defaultdict(lambda: defaultdict(int))

    for held_out_eval in eval_sets_present:
        train = [r for r in records if r["eval_set"] != held_out_eval]
        test = [r for r in records if r["eval_set"] == held_out_eval]

        train_dict = defaultdict(list)
        for r in train:
            train_dict[r["variant"]].append(r["vector"])

        test_vecs = [r["vector"] for r in test]
        test_labels = [r["variant"] for r in test]

        _, _, confusion = nearest_centroid_classify(
            train_dict, test_vecs, test_labels
        )

        for true_v, pred_counts in confusion.items():
            for pred_v, count in pred_counts.items():
                total_confusion[true_v][pred_v] += count

    # Print
    abbrevs = [v[:8] for v in variants_present]
    header_label = "true\\pred"
    print(f"\n  {header_label:<18}", end="")
    for a in abbrevs:
        print(f" {a:>9}", end="")
    print(f" {'total':>6} {'acc':>6}")
    print(f"  {'-'*(18 + 10*len(abbrevs) + 14)}")

    for true_v in variants_present:
        row_total = sum(total_confusion[true_v][pred_v] for pred_v in variants_present)
        correct = total_confusion[true_v][true_v]
        acc = 100.0 * correct / row_total if row_total > 0 else 0
        print(f"  {true_v:<18}", end="")
        for pred_v in variants_present:
            count = total_confusion[true_v][pred_v]
            if pred_v == true_v and count > 0:
                print(f"  [{count:>5}]", end="")
            else:
                print(f"   {count:>5} ", end="")
        print(f" {row_total:>6} {acc:>5.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Model Delta Method Comparison for Persona Classification (14B)")
    print("=" * 70)

    trait_names = get_trait_names()
    print(f"\nTraits ({len(trait_names)}): {', '.join(trait_names)}")
    print(f"Variants: {', '.join(VARIANTS)}")
    print(f"Eval sets: {', '.join(EVAL_SETS)}")

    # --- Load all data types ---
    print("\n--- Loading fingerprints ---")
    records_combined = load_all("combined")
    records_text_only = load_all("text_only")
    records_reverse = load_all("reverse_model")
    baselines = load_baselines("combined")
    print(f"  Combined: {len(records_combined)}, Text-only: {len(records_text_only)}, "
          f"Reverse-model: {len(records_reverse)}, Baselines: {len(baselines)}")

    # --- Compute deltas ---
    print("\n--- Computing model deltas ---")
    method_a_records = compute_method_a(records_combined, records_text_only)
    method_b_records = compute_method_b(records_reverse, baselines)
    print(f"  Method A records: {len(method_a_records)}")
    print(f"  Method B records: {len(method_b_records)}")

    # ===================================================================
    # Run all three classification experiments
    # ===================================================================
    results = {}

    results["raw"] = run_classification(records_combined, "Raw combined fingerprints")
    print_confusion_matrix(records_combined, "Raw combined")

    results["method_a"] = run_classification(
        method_a_records, "Method A: combined - text_only (model delta on LoRA text)"
    )
    print_confusion_matrix(method_a_records, "Method A")

    results["method_b"] = run_classification(
        method_b_records, "Method B: reverse_model - baseline (model delta on clean text)"
    )
    print_confusion_matrix(method_b_records, "Method B")

    # ===================================================================
    # Summary comparison
    # ===================================================================
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*70}")

    # Overall accuracy
    print(f"\n  --- Overall Accuracy ---")
    print(f"  {'Method':<55} {'Acc':>7}")
    print(f"  {'-'*65}")
    for key, name in [
        ("raw", "Raw combined fingerprints"),
        ("method_a", "Method A: combined - text_only"),
        ("method_b", "Method B: reverse_model - baseline"),
    ]:
        print(f"  {name:<55} {results[key]['overall_acc']:>6.1f}%")

    # Per-eval comparison
    print(f"\n  --- Per-Eval-Set Accuracy ---")
    header = f"  {'Eval set':<30} {'Raw':>7} {'Meth A':>7} {'Meth B':>7} {'A-B':>7}"
    print(header)
    print(f"  {'-'*60}")
    for ev in EVAL_SETS:
        raw_acc = results["raw"]["per_eval"].get(ev, 0)
        a_acc = results["method_a"]["per_eval"].get(ev, 0)
        b_acc = results["method_b"]["per_eval"].get(ev, 0)
        diff = a_acc - b_acc
        marker = " <-- hard" if ev in HARD_EVALS else ""
        print(f"  {ev:<30} {raw_acc:>6.1f}% {a_acc:>6.1f}% {b_acc:>6.1f}% {diff:>+6.1f}%{marker}")

    # Hard evals focus
    print(f"\n  --- Hard Eval Sets (sriram_factual, sriram_normal) ---")
    for ev in HARD_EVALS:
        raw_acc = results["raw"]["per_eval"].get(ev, 0)
        a_acc = results["method_a"]["per_eval"].get(ev, 0)
        b_acc = results["method_b"]["per_eval"].get(ev, 0)
        best = "A" if a_acc > b_acc else ("B" if b_acc > a_acc else "tie")
        print(f"  {ev}: Raw={raw_acc:.1f}%, A={a_acc:.1f}%, B={b_acc:.1f}%  --> Best: {best}")

    # Per-variant comparison
    print(f"\n  --- Per-Variant Accuracy ---")
    header = f"  {'Variant':<25} {'Raw':>7} {'Meth A':>7} {'Meth B':>7} {'A-B':>7}"
    print(header)
    print(f"  {'-'*58}")
    for v in VARIANTS:
        raw_acc = results["raw"]["per_variant"].get(v, 0)
        a_acc = results["method_a"]["per_variant"].get(v, 0)
        b_acc = results["method_b"]["per_variant"].get(v, 0)
        diff = a_acc - b_acc
        print(f"  {v:<25} {raw_acc:>6.1f}% {a_acc:>6.1f}% {b_acc:>6.1f}% {diff:>+6.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
