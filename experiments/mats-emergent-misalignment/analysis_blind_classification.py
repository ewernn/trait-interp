"""
Blind persona classification from probe fingerprints (4B mats-emergent-misalignment).

Loads 23D trait probe score vectors for each (persona, training_category, eval_set)
combination and tests whether a nearest-centroid classifier can recover persona identity
from these fingerprints alone.

Input: probe_scores/{persona}_{category}_x_{eval}_{type}.json
Output: printed classification accuracy tables

Usage:
    python experiments/mats-emergent-misalignment/analysis_blind_classification.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "analysis" / "pxs_grid_4b" / "probe_scores"

PERSONAS = ["angry", "mocking", "disappointed", "confused", "nervous", "bureaucratic", "curt"]
CATEGORIES = ["refusal", "diverse_open_ended", "normal_requests", "factual_questions"]
EVAL_SETS = [
    "em_generic_eval", "em_medical_eval", "emotional_vulnerability",
    "ethical_dilemmas", "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]
BASELINE_PREFIX = "clean_instruct"


def load_fingerprint(path: Path) -> np.ndarray | None:
    """Load a JSON trait dict and return as a sorted numpy vector."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    # Sort by key for consistent ordering
    return np.array([d[k] for k in sorted(d.keys())])


def get_trait_names() -> list[str]:
    """Return sorted trait names from any available file."""
    for persona in PERSONAS:
        for cat in CATEGORIES:
            for ev in EVAL_SETS:
                p = DATA_DIR / f"{persona}_{cat}_x_{ev}_combined.json"
                if p.exists():
                    with open(p) as f:
                        return sorted(json.load(f).keys())
    raise RuntimeError("No data files found")


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
def load_all_data(file_suffix: str = "combined"):
    """
    Load fingerprints for all (persona, category, eval_set) combos.

    Returns:
        records: list of dicts with keys: persona, category, eval_set, vector
    """
    records = []
    missing = 0
    for persona in PERSONAS:
        for cat in CATEGORIES:
            for ev in EVAL_SETS:
                fname = f"{persona}_{cat}_x_{ev}_{file_suffix}.json"
                vec = load_fingerprint(DATA_DIR / fname)
                if vec is not None:
                    records.append({
                        "persona": persona,
                        "category": cat,
                        "eval_set": ev,
                        "vector": vec,
                    })
                else:
                    missing += 1
    if missing > 0:
        print(f"  [{file_suffix}] Warning: {missing} files missing")
    return records


def load_baselines(file_suffix: str = "combined"):
    """Load baseline (clean_instruct) fingerprints per eval_set."""
    baselines = {}
    for ev in EVAL_SETS:
        fname = f"{BASELINE_PREFIX}_x_{ev}_{file_suffix}.json"
        vec = load_fingerprint(DATA_DIR / fname)
        if vec is not None:
            baselines[ev] = vec
    return baselines


# ---------------------------------------------------------------------------
# Nearest-centroid classifier
# ---------------------------------------------------------------------------
def nearest_centroid_classify(
    train_vecs: dict[str, list[np.ndarray]],
    test_vecs: list[np.ndarray],
    test_labels: list[str],
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """
    Train nearest-centroid and predict.

    Args:
        train_vecs: label -> list of training vectors
        test_vecs: list of test vectors
        test_labels: ground truth labels for test vectors

    Returns:
        (correct, total, confusion_counts)
    """
    # Compute centroids
    centroids = {}
    for label, vecs in train_vecs.items():
        centroids[label] = np.mean(vecs, axis=0)

    labels_sorted = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[l] for l in labels_sorted])  # (K, D)

    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for vec, true_label in zip(test_vecs, test_labels):
        # Cosine distance
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
# Classification experiments
# ---------------------------------------------------------------------------
def run_persona_classification(records: list[dict], label: str, group_categories: bool = False):
    """
    Leave-one-eval-set-out cross-validation for persona prediction.

    If group_categories=True, all training categories are pooled per persona
    (each persona gets 4x the training samples). Otherwise each
    (persona, category) pair is a separate sample.
    """
    personas_present = sorted(set(r["persona"] for r in records))
    n_classes = len(personas_present)
    chance = 100.0 / n_classes

    header = f"PERSONA classification ({label})"
    if group_categories:
        header += " [categories pooled]"
    print(f"\n{'='*70}")
    print(f"  {header}")
    print(f"  Chance = {chance:.1f}% ({n_classes}-way)")
    print(f"{'='*70}")

    eval_sets_present = sorted(set(r["eval_set"] for r in records))

    per_eval_results = {}
    total_correct = 0
    total_count = 0

    for held_out_eval in eval_sets_present:
        # Split
        train = [r for r in records if r["eval_set"] != held_out_eval]
        test = [r for r in records if r["eval_set"] == held_out_eval]

        # Build training dict: persona -> list of vectors
        train_dict = defaultdict(list)
        for r in train:
            train_dict[r["persona"]].append(r["vector"])

        # If grouping categories, each persona gets all its (cat, eval) combos
        # If not, it still works the same way - we just have more samples per persona

        test_vecs = [r["vector"] for r in test]
        test_labels = [r["persona"] for r in test]

        correct, total, confusion = nearest_centroid_classify(
            train_dict, test_vecs, test_labels
        )

        acc = 100.0 * correct / total if total > 0 else 0
        per_eval_results[held_out_eval] = (correct, total, acc)
        total_correct += correct
        total_count += total

    # Print per-eval results
    print(f"\n  {'Held-out eval set':<30} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"  {'-'*55}")
    for ev in eval_sets_present:
        c, t, a = per_eval_results[ev]
        marker = " ***" if a >= 50 else ""
        print(f"  {ev:<30} {c:>8} {t:>6} {a:>6.1f}%{marker}")

    overall_acc = 100.0 * total_correct / total_count if total_count > 0 else 0
    print(f"  {'-'*55}")
    print(f"  {'OVERALL':<30} {total_correct:>8} {total_count:>6} {overall_acc:>6.1f}%")
    print(f"  {'chance':<30} {'':>8} {'':>6} {chance:>6.1f}%")
    print(f"  Lift over chance: {overall_acc - chance:+.1f}pp")

    return overall_acc


def run_category_classification(records: list[dict], label: str):
    """
    Leave-one-eval-set-out cross-validation for training CATEGORY prediction (4-way).
    """
    categories_present = sorted(set(r["category"] for r in records))
    n_classes = len(categories_present)
    chance = 100.0 / n_classes

    print(f"\n{'='*70}")
    print(f"  TRAINING CATEGORY classification ({label})")
    print(f"  Chance = {chance:.1f}% ({n_classes}-way)")
    print(f"{'='*70}")

    eval_sets_present = sorted(set(r["eval_set"] for r in records))

    per_eval_results = {}
    total_correct = 0
    total_count = 0

    for held_out_eval in eval_sets_present:
        train = [r for r in records if r["eval_set"] != held_out_eval]
        test = [r for r in records if r["eval_set"] == held_out_eval]

        train_dict = defaultdict(list)
        for r in train:
            train_dict[r["category"]].append(r["vector"])

        test_vecs = [r["vector"] for r in test]
        test_labels = [r["category"] for r in test]

        correct, total, confusion = nearest_centroid_classify(
            train_dict, test_vecs, test_labels
        )

        acc = 100.0 * correct / total if total > 0 else 0
        per_eval_results[held_out_eval] = (correct, total, acc)
        total_correct += correct
        total_count += total

    print(f"\n  {'Held-out eval set':<30} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"  {'-'*55}")
    for ev in eval_sets_present:
        c, t, a = per_eval_results[ev]
        marker = " ***" if a >= 50 else ""
        print(f"  {ev:<30} {c:>8} {t:>6} {a:>6.1f}%{marker}")

    overall_acc = 100.0 * total_correct / total_count if total_count > 0 else 0
    print(f"  {'-'*55}")
    print(f"  {'OVERALL':<30} {total_correct:>8} {total_count:>6} {overall_acc:>6.1f}%")
    print(f"  {'chance':<30} {'':>8} {'':>6} {chance:>6.1f}%")
    print(f"  Lift over chance: {overall_acc - chance:+.1f}pp")

    return overall_acc


def run_persona_confusion_matrix(records: list[dict], label: str):
    """Print full confusion matrix for persona classification (aggregated across all folds)."""
    print(f"\n{'='*70}")
    print(f"  PERSONA confusion matrix ({label})")
    print(f"{'='*70}")

    eval_sets_present = sorted(set(r["eval_set"] for r in records))

    # Aggregate confusion across all folds
    total_confusion = defaultdict(lambda: defaultdict(int))

    for held_out_eval in eval_sets_present:
        train = [r for r in records if r["eval_set"] != held_out_eval]
        test = [r for r in records if r["eval_set"] == held_out_eval]

        train_dict = defaultdict(list)
        for r in train:
            train_dict[r["persona"]].append(r["vector"])

        test_vecs = [r["vector"] for r in test]
        test_labels = [r["persona"] for r in test]

        _, _, confusion = nearest_centroid_classify(
            train_dict, test_vecs, test_labels
        )

        for true_l, pred_counts in confusion.items():
            for pred_l, count in pred_counts.items():
                total_confusion[true_l][pred_l] += count

    # Print -- only personas that appear in the data
    personas_sorted = sorted(set(r["persona"] for r in records))
    # Header: abbreviated
    abbrevs = [p[:5] for p in personas_sorted]
    header_label = "true\\pred"
    print(f"\n  {header_label:<14}", end="")
    for a in abbrevs:
        print(f" {a:>6}", end="")
    print(f" {'total':>6} {'acc':>6}")
    print(f"  {'-'*( 14 + 7*len(abbrevs) + 14)}")

    for true_p in personas_sorted:
        row_total = sum(total_confusion[true_p][pred_p] for pred_p in personas_sorted)
        correct = total_confusion[true_p][true_p]
        acc = 100.0 * correct / row_total if row_total > 0 else 0
        print(f"  {true_p:<14}", end="")
        for pred_p in personas_sorted:
            count = total_confusion[true_p][pred_p]
            if pred_p == true_p and count > 0:
                print(f" [{count:>4}]", end="")
            else:
                print(f"  {count:>4} ", end="")
        print(f" {row_total:>6} {acc:>5.1f}%")


# ---------------------------------------------------------------------------
# Baseline-delta computation
# ---------------------------------------------------------------------------
def compute_baseline_delta(records: list[dict], baselines: dict[str, np.ndarray]) -> list[dict]:
    """Subtract per-eval-set baseline from each record's vector."""
    delta_records = []
    for r in records:
        ev = r["eval_set"]
        if ev in baselines:
            delta_records.append({
                **r,
                "vector": r["vector"] - baselines[ev],
            })
    return delta_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Blind Persona Classification from Probe Fingerprints (4B)")
    print("=" * 70)

    trait_names = get_trait_names()
    print(f"\nTraits ({len(trait_names)}): {', '.join(trait_names)}")

    # --- Load combined data ---
    print("\n--- Loading combined fingerprints ---")
    records_combined = load_all_data("combined")
    personas_found = sorted(set(r["persona"] for r in records_combined))
    personas_missing = sorted(set(PERSONAS) - set(personas_found))
    print(f"  Loaded {len(records_combined)} fingerprints")
    print(f"  Personas found ({len(personas_found)}): {', '.join(personas_found)}")
    if personas_missing:
        print(f"  Personas missing: {', '.join(personas_missing)}")
    print(f"  Categories: {sorted(set(r['category'] for r in records_combined))}")
    print(f"  Eval sets: {sorted(set(r['eval_set'] for r in records_combined))}")

    # --- Load baselines ---
    baselines_combined = load_baselines("combined")
    print(f"  Baselines loaded: {len(baselines_combined)} eval sets")

    # --- Load reverse_model data ---
    print("\n--- Loading reverse_model fingerprints ---")
    records_reverse = load_all_data("reverse_model")
    print(f"  Loaded {len(records_reverse)} fingerprints")

    baselines_reverse = load_baselines("reverse_model")
    has_reverse_baselines = len(baselines_reverse) > 0
    print(f"  Reverse model baselines: {len(baselines_reverse)} eval sets")

    # =======================================================================
    # EXPERIMENT 1: Persona classification from raw combined fingerprints
    # =======================================================================
    run_persona_classification(records_combined, "raw combined")
    run_persona_confusion_matrix(records_combined, "raw combined")

    # =======================================================================
    # EXPERIMENT 2: Persona classification, categories pooled
    # =======================================================================
    run_persona_classification(records_combined, "raw combined", group_categories=True)

    # =======================================================================
    # EXPERIMENT 3: Baseline-delta (combined - clean_instruct)
    # =======================================================================
    if baselines_combined:
        delta_records = compute_baseline_delta(records_combined, baselines_combined)
        if delta_records:
            run_persona_classification(delta_records, "baseline delta (combined - clean_instruct)")
            run_persona_confusion_matrix(delta_records, "baseline delta")

    # =======================================================================
    # EXPERIMENT 4: Reverse-model fingerprints (model contribution only)
    # =======================================================================
    if records_reverse:
        run_persona_classification(records_reverse, "reverse_model (model-only)")
        run_persona_confusion_matrix(records_reverse, "reverse_model")

    # =======================================================================
    # EXPERIMENT 5: Reverse-model baseline-delta (if baselines exist)
    # =======================================================================
    if records_reverse and has_reverse_baselines:
        delta_reverse = compute_baseline_delta(records_reverse, baselines_reverse)
        if delta_reverse:
            run_persona_classification(delta_reverse, "reverse_model delta")

    # =======================================================================
    # EXPERIMENT 6: Training category classification (4-way)
    # =======================================================================
    run_category_classification(records_combined, "raw combined")

    if baselines_combined:
        delta_records = compute_baseline_delta(records_combined, baselines_combined)
        if delta_records:
            run_category_classification(delta_records, "baseline delta")

    if records_reverse:
        run_category_classification(records_reverse, "reverse_model")

    # =======================================================================
    # BONUS: Per-persona accuracy breakdown
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  PER-PERSONA accuracy summary (raw combined)")
    print(f"{'='*70}")

    eval_sets_present = sorted(set(r["eval_set"] for r in records_combined))
    personas_in_data = sorted(set(r["persona"] for r in records_combined))
    persona_correct = defaultdict(int)
    persona_total = defaultdict(int)

    for held_out_eval in eval_sets_present:
        train = [r for r in records_combined if r["eval_set"] != held_out_eval]
        test = [r for r in records_combined if r["eval_set"] == held_out_eval]

        train_dict = defaultdict(list)
        for r in train:
            train_dict[r["persona"]].append(r["vector"])

        test_vecs = [r["vector"] for r in test]
        test_labels = [r["persona"] for r in test]

        _, _, confusion = nearest_centroid_classify(
            train_dict, test_vecs, test_labels
        )
        for true_p in personas_in_data:
            for pred_p in personas_in_data:
                c = confusion[true_p][pred_p]
                persona_total[true_p] += c
                if pred_p == true_p:
                    persona_correct[true_p] += c

    print(f"\n  {'Persona':<20} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"  {'-'*45}")
    for p in personas_in_data:
        c = persona_correct[p]
        t = persona_total[p]
        a = 100.0 * c / t if t > 0 else 0
        print(f"  {p:<20} {c:>8} {t:>6} {a:>6.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
