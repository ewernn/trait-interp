#!/usr/bin/env python3
"""
Analyze vector quality across all traits, methods, and layers.
Helps decide which experiments to re-run with per-token activations.
"""

import json
from pathlib import Path
from collections import defaultdict
import statistics

# Configuration
EXPERIMENT = "experiments/gemma_2b_cognitive_nov20"
METHODS = ["mean_diff", "probe", "ica", "gradient"]
LAYERS = list(range(27))

def load_vector_metadata(trait_path, method, layer):
    """Load metadata for a specific vector."""
    metadata_path = trait_path / "vectors" / f"{method}_layer{layer}_metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        return json.load(f)

def load_response_separation(trait_path):
    """Calculate separation from response CSVs."""
    import pandas as pd

    pos_csv = trait_path / "responses" / "pos.csv"
    neg_csv = trait_path / "responses" / "neg.csv"

    if not pos_csv.exists() or not neg_csv.exists():
        return None, 0, 0

    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)

    pos_avg = pos_df['trait_score'].mean()
    neg_avg = neg_df['trait_score'].mean()
    separation = pos_avg - neg_avg

    return separation, len(pos_df), len(neg_df)

def analyze_trait(trait_name):
    """Analyze all vectors for a single trait."""
    trait_path = Path(EXPERIMENT) / trait_name

    if not trait_path.exists():
        return None

    # Get response separation
    separation, n_pos, n_neg = load_response_separation(trait_path)

    # Analyze all method/layer combinations
    vectors = []
    for method in METHODS:
        for layer in LAYERS:
            metadata = load_vector_metadata(trait_path, method, layer)
            if metadata:
                vectors.append({
                    'method': method,
                    'layer': layer,
                    'norm': metadata.get('vector_norm'),
                    'train_acc': metadata.get('train_acc')
                })

    if not vectors:
        return None

    # Find best vector by norm (strongest signal)
    best_by_norm = max(vectors, key=lambda v: v['norm'] if v['norm'] else 0)

    # Average metrics by method
    method_stats = defaultdict(list)
    for v in vectors:
        if v['norm'] is not None:
            method_stats[v['method']].append(v['norm'])

    avg_norms = {m: statistics.mean(norms) for m, norms in method_stats.items()}

    return {
        'trait': trait_name,
        'separation': separation,
        'n_examples': n_pos + n_neg,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'best_method': best_by_norm['method'],
        'best_layer': best_by_norm['layer'],
        'best_norm': best_by_norm['norm'],
        'best_train_acc': best_by_norm['train_acc'],
        'avg_norms': avg_norms,
        'n_vectors': len(vectors)
    }

def main():
    """Run analysis and print results."""
    import pandas as pd

    print("🔍 Analyzing Vector Quality for gemma_2b_cognitive_nov20\n")
    print("=" * 80)

    # Collect all trait data
    traits = []
    experiment_path = Path(EXPERIMENT)

    for trait_dir in sorted(experiment_path.iterdir()):
        if trait_dir.is_dir() and trait_dir.name not in ['experiments']:
            result = analyze_trait(trait_dir.name)
            if result:
                traits.append(result)

    # Convert to DataFrame for nice display
    df = pd.DataFrame(traits)
    df = df.sort_values('separation', ascending=False)

    # Print summary table
    print("\n📊 TRAIT QUALITY SUMMARY (sorted by separation)\n")
    print(f"{'Trait':<25} {'Sep':>6} {'Examples':>8} {'Best Method':>12} {'Layer':>5} {'Norm':>7} {'Acc':>6}")
    print("-" * 80)

    for _, row in df.iterrows():
        quality = "🟢" if row['separation'] > 80 else "🟡" if row['separation'] > 60 else "🔴"
        acc_str = f"{row['best_train_acc']:.2f}" if row['best_train_acc'] is not None else "N/A"
        print(f"{quality} {row['trait']:<23} {row['separation']:>6.1f} {row['n_examples']:>8} {row['best_method']:>12} {row['best_layer']:>5} {row['best_norm']:>7.2f} {acc_str:>6}")

    # Print method comparison
    print("\n" + "=" * 80)
    print("\n📈 METHOD COMPARISON (average vector norm across all layers)\n")

    all_method_stats = defaultdict(list)
    for trait in traits:
        for method, norm in trait['avg_norms'].items():
            all_method_stats[method].append(norm)

    print(f"{'Method':<15} {'Avg Norm':>10} {'Std Dev':>10} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for method in METHODS:
        norms = all_method_stats[method]
        if norms:
            print(f"{method:<15} {statistics.mean(norms):>10.3f} {statistics.stdev(norms):>10.3f} {min(norms):>8.3f} {max(norms):>8.3f}")

    # Recommendations
    print("\n" + "=" * 80)
    print("\n💡 RECOMMENDATIONS FOR PER-TOKEN RE-RUN\n")

    excellent = df[df['separation'] > 80]
    good = df[(df['separation'] > 60) & (df['separation'] <= 80)]
    weak = df[df['separation'] <= 60]

    print(f"✅ Excellent separation (>80): {len(excellent)} traits")
    print(f"   Worth re-running: {', '.join(excellent['trait'].head(5).tolist())}")
    print(f"   → Strong vectors, per-token dynamics will be clean\n")

    print(f"⚠️  Good separation (60-80): {len(good)} traits")
    print(f"   Consider re-running: {', '.join(good['trait'].head(3).tolist())}")
    print(f"   → Moderate vectors, per-token may reveal interesting dynamics\n")

    print(f"❌ Weak separation (<60): {len(weak)} traits")
    print(f"   Skip for now: {', '.join(weak['trait'].tolist())}")
    print(f"   → Weak vectors, may need better trait definitions first\n")

    # Layer distribution
    print("\n" + "=" * 80)
    print("\n🎯 BEST LAYER DISTRIBUTION\n")

    layer_counts = df['best_layer'].value_counts().sort_index()
    print(f"{'Layer':<10} {'Count':>10} {'Traits'}")
    print("-" * 60)
    for layer in sorted(layer_counts.index):
        traits_at_layer = df[df['best_layer'] == layer]['trait'].tolist()
        print(f"{layer:<10} {layer_counts[layer]:>10} {', '.join(traits_at_layer[:3])}")

    print("\n" + "=" * 80)
    print("\n📝 SUMMARY\n")
    print(f"Total traits analyzed: {len(traits)}")
    print(f"Total vectors extracted: {sum(t['n_vectors'] for t in traits)}")
    print(f"Average separation: {df['separation'].mean():.1f}")
    print(f"Average examples per trait: {df['n_examples'].mean():.0f}")
    print(f"\nMost common best method: {df['best_method'].mode()[0]}")
    print(f"Most common best layer: {df['best_layer'].mode()[0]}")

if __name__ == "__main__":
    main()
