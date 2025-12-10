#!/usr/bin/env python3
"""
Grid search for optimal (mu, sigma) per trait using classification metrics.

Input:
    - experiments/{experiment}/extraction/{trait}/vectors/*.pt
    - experiments/{experiment}/extraction/{trait}/val_activations/*.pt

Output:
    - experiments/{experiment}/extraction/ensemble_evaluation.json

Usage:
    # Full grid search (26 mu values x 5 sigma values = 130 per trait)
    python analysis/ensemble/classification_search.py \
        --experiment gemma-2-2b \
        --traits epistemic/confidence

    # Quick search (fix sigma=2, sweep mu only)
    python analysis/ensemble/classification_search.py \
        --experiment gemma-2-2b \
        --traits all \
        --quick

    # Resume interrupted search
    python analysis/ensemble/classification_search.py \
        --experiment gemma-2-2b \
        --traits epistemic/confidence \
        --resume
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
from datetime import datetime

from utils.paths import get as get_path
from utils.model_registry import get_extraction_model, get_num_layers
from traitlens.metrics import evaluate_vector
from sklearn.metrics import roc_auc_score

from analysis.ensemble.gaussian import (
    compute_gaussian_weights,
    create_ensemble_vector,
    load_vectors_for_trait,
    load_all_validation_activations,
    get_active_layers,
    compute_effective_width,
)


@dataclass
class EnsembleResult:
    """Results for a single (mu, sigma) configuration."""
    trait: str
    mu: float
    sigma: float
    method: str

    # Validation metrics
    val_accuracy: float
    val_effect_size: float
    val_separation: float
    val_p_value: float

    # Distribution info
    active_layers: int  # Layers with weight > 0.01
    effective_width: float  # ~4*sigma

    # Polarity
    polarity_correct: bool
    val_pos_mean: float
    val_neg_mean: float

    # AUC for threshold-independent measure
    val_auc_roc: float


def evaluate_ensemble_on_validation(
    experiment: str,
    trait: str,
    mu: float,
    sigma: float,
    vectors: Dict[int, torch.Tensor],
    val_pos: Dict[int, torch.Tensor],
    val_neg: Dict[int, torch.Tensor],
    method: str = "probe",
    normalize: bool = True,
) -> Optional[EnsembleResult]:
    """
    Evaluate a single (mu, sigma) ensemble on validation data.

    Args:
        experiment: Experiment name
        trait: Trait path
        mu: Gaussian center
        sigma: Gaussian spread
        vectors: Pre-loaded vectors per layer
        val_pos: Pre-loaded positive validation activations per layer
        val_neg: Pre-loaded negative validation activations per layer
        method: Extraction method name (for metadata)
        normalize: Whether to normalize for projection

    Returns:
        EnsembleResult or None if evaluation fails
    """
    try:
        # Compute weights
        layers = sorted(vectors.keys())
        weights = compute_gaussian_weights(mu, sigma, layers)

        # Create ensemble vector
        ensemble_vec = create_ensemble_vector(vectors, weights)

        # For validation, we need to combine activations across weighted layers
        # Strategy: project each layer's activations onto the ensemble vector,
        # then weight-average the projections
        #
        # Actually, mathematically equivalent to:
        # proj = act @ ensemble_vec = act @ sum(w_i * v_i) = sum(w_i * (act @ v_i))
        #
        # But we have different activations per layer. The correct approach:
        # Pick a reference layer's activations (e.g., the peak layer)
        # and project onto the ensemble vector.
        #
        # For simplicity, use the layer closest to mu for validation activations.
        ref_layer = int(round(mu))
        ref_layer = min(max(ref_layer, min(val_pos.keys())), max(val_pos.keys()))

        if ref_layer not in val_pos or ref_layer not in val_neg:
            # Find closest available layer
            ref_layer = min(val_pos.keys(), key=lambda l: abs(l - mu))

        pos_acts = val_pos[ref_layer]
        neg_acts = val_neg[ref_layer]

        # Evaluate using traitlens
        metrics = evaluate_vector(pos_acts, neg_acts, ensemble_vec, normalize=normalize)

        # Compute additional metrics
        vec_f32 = ensemble_vec.float()
        pos_f32 = pos_acts.float()
        neg_f32 = neg_acts.float()

        if normalize:
            vec_norm = vec_f32 / (vec_f32.norm() + 1e-8)
            pos_norm = pos_f32 / (pos_f32.norm(dim=1, keepdim=True) + 1e-8)
            neg_norm = neg_f32 / (neg_f32.norm(dim=1, keepdim=True) + 1e-8)
            pos_proj = pos_norm @ vec_norm
            neg_proj = neg_norm @ vec_norm
        else:
            pos_proj = pos_f32 @ vec_f32
            neg_proj = neg_f32 @ vec_f32

        # AUC-ROC
        all_proj = torch.cat([pos_proj, neg_proj]).cpu().numpy()
        all_labels = np.concatenate([np.ones(len(pos_proj)), np.zeros(len(neg_proj))])
        try:
            auc = roc_auc_score(all_labels, all_proj)
        except ValueError:
            auc = 0.5

        # Active layers
        active = get_active_layers(mu, sigma, len(layers))

        return EnsembleResult(
            trait=trait,
            mu=mu,
            sigma=sigma,
            method=method,
            val_accuracy=metrics['accuracy'],
            val_effect_size=metrics['effect_size'],
            val_separation=metrics['separation'],
            val_p_value=metrics['p_value'],
            active_layers=len(active),
            effective_width=compute_effective_width(sigma),
            polarity_correct=metrics['polarity_correct'],
            val_pos_mean=metrics['pos_mean'],
            val_neg_mean=metrics['neg_mean'],
            val_auc_roc=auc,
        )

    except Exception as e:
        print(f"  Warning: Failed for mu={mu}, sigma={sigma}: {e}")
        return None


def discover_traits_with_val_data(experiment: str) -> List[str]:
    """Discover all traits with both vectors and validation activations."""
    extraction_dir = get_path('extraction.base', experiment=experiment)

    if not extraction_dir.exists():
        raise FileNotFoundError(f"Extraction directory not found: {extraction_dir}")

    traits = []
    for category_dir in sorted(extraction_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if not trait_dir.is_dir():
                continue
            vectors_dir = trait_dir / "vectors"
            val_dir = trait_dir / "val_activations"
            if vectors_dir.exists() and val_dir.exists():
                if list(vectors_dir.glob('*.pt')) and list(val_dir.glob('*.pt')):
                    traits.append(f"{category_dir.name}/{trait_dir.name}")

    return traits


def load_existing_results(output_path: Path) -> Dict:
    """Load existing results for resume support."""
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return {}


def is_computed(existing: Dict, trait: str, mu: float, sigma: float) -> bool:
    """Check if a (trait, mu, sigma) combination already exists."""
    for r in existing.get('all_results', []):
        if r['trait'] == trait and r['mu'] == mu and r['sigma'] == sigma:
            return True
    return False


def run_grid_search(
    experiment: str,
    traits: List[str],
    mu_values: List[float],
    sigma_values: List[float],
    method: str = "probe",
    component: str = "residual",
    normalize: bool = True,
    resume: bool = False,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Run grid search over (mu, sigma) space for all traits.

    Args:
        experiment: Experiment name
        traits: List of trait paths
        mu_values: List of mu values to try
        sigma_values: List of sigma values to try
        method: Vector extraction method
        component: Component type
        normalize: Whether to normalize projections
        resume: Skip already-computed combinations
        output_path: Where to save results

    Returns:
        Dict with all results and analysis
    """
    # Load existing results if resuming
    existing = {}
    if resume and output_path and output_path.exists():
        existing = load_existing_results(output_path)
        print(f"Resuming: found {len(existing.get('all_results', []))} existing results")

    all_results = existing.get('all_results', [])
    extraction_model = get_extraction_model(experiment)

    total_combos = len(traits) * len(mu_values) * len(sigma_values)
    print(f"\nGrid search: {len(traits)} traits x {len(mu_values)} mu x {len(sigma_values)} sigma = {total_combos} combinations")

    for trait in tqdm(traits, desc="Traits"):
        # Load vectors and validation data once per trait
        try:
            vectors = load_vectors_for_trait(experiment, trait, method, component)
            val_pos, val_neg = load_all_validation_activations(experiment, trait, component)
        except FileNotFoundError as e:
            print(f"  Skipping {trait}: {e}")
            continue

        for mu in mu_values:
            for sigma in sigma_values:
                # Skip if already computed
                if resume and is_computed({'all_results': all_results}, trait, mu, sigma):
                    continue

                result = evaluate_ensemble_on_validation(
                    experiment, trait, mu, sigma,
                    vectors, val_pos, val_neg,
                    method=method, normalize=normalize
                )
                if result:
                    all_results.append(asdict(result))

        # Save intermediate results after each trait
        if output_path:
            _save_results(output_path, all_results, extraction_model, experiment,
                         mu_values, sigma_values, method, component)

    return {
        'extraction_model': extraction_model,
        'extraction_experiment': experiment,
        'grid_config': {
            'mu_values': mu_values,
            'sigma_values': sigma_values,
            'method': method,
            'component': component,
        },
        'all_results': all_results,
    }


def _save_results(
    output_path: Path,
    all_results: List[Dict],
    extraction_model: str,
    experiment: str,
    mu_values: List[float],
    sigma_values: List[float],
    method: str,
    component: str,
):
    """Save results to JSON with analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Analyze results
    best_per_trait = {}
    comparison = {}

    # Group by trait
    traits = set(r['trait'] for r in all_results)
    for trait in traits:
        trait_results = [r for r in all_results if r['trait'] == trait]
        if not trait_results:
            continue

        # Find best by accuracy
        best = max(trait_results, key=lambda r: r['val_accuracy'])
        best_per_trait[trait] = {
            'mu': best['mu'],
            'sigma': best['sigma'],
            'val_accuracy': best['val_accuracy'],
            'val_effect_size': best['val_effect_size'],
            'active_layers': best['active_layers'],
        }

    # Load single-layer results for comparison
    single_layer_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if single_layer_path.exists():
        try:
            with open(single_layer_path) as f:
                single_data = json.load(f)

            for trait in traits:
                trait_single = [r for r in single_data.get('all_results', [])
                               if r.get('trait') == trait and r.get('method') == method]
                if trait_single:
                    best_single = max(trait_single, key=lambda r: r.get('val_accuracy', 0))
                    ensemble_best = best_per_trait.get(trait, {})

                    comparison[trait] = {
                        'single_layer_best': {
                            'layer': best_single.get('layer'),
                            'val_accuracy': best_single.get('val_accuracy'),
                            'val_effect_size': best_single.get('val_effect_size'),
                        },
                        'ensemble_best': {
                            'mu': ensemble_best.get('mu'),
                            'sigma': ensemble_best.get('sigma'),
                            'val_accuracy': ensemble_best.get('val_accuracy'),
                            'val_effect_size': ensemble_best.get('val_effect_size'),
                        },
                        'accuracy_improvement': (
                            ensemble_best.get('val_accuracy', 0) -
                            best_single.get('val_accuracy', 0)
                        ),
                    }
        except (json.JSONDecodeError, KeyError):
            pass

    results = {
        'extraction_model': extraction_model,
        'extraction_experiment': experiment,
        'grid_config': {
            'mu_values': mu_values,
            'sigma_values': sigma_values,
            'method': method,
            'component': component,
        },
        'all_results': all_results,
        'best_per_trait': best_per_trait,
        'comparison': comparison,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def main(
    experiment: str,
    traits: str = "all",
    quick: bool = False,
    sigma_values: str = "1,2,3,4,5",
    method: str = "probe",
    component: str = "residual",
    normalize: bool = True,
    resume: bool = False,
    output: str = None,
):
    """
    Run ensemble grid search.

    Args:
        experiment: Experiment name
        traits: Comma-separated traits or "all"
        quick: If True, fix sigma=2 and sweep mu only (26 evals per trait)
        sigma_values: Comma-separated sigma values to try
        method: Vector extraction method
        component: residual, attn_out, etc.
        normalize: Whether to normalize projections
        resume: Skip already-computed combinations
        output: Output path (default: experiments/{exp}/extraction/ensemble_evaluation.json)
    """
    print(f"Ensemble Grid Search")
    print(f"  Experiment: {experiment}")
    print(f"  Method: {method}")
    print(f"  Component: {component}")

    # Discover or parse traits
    if traits == "all":
        trait_list = discover_traits_with_val_data(experiment)
    else:
        trait_list = [t.strip() for t in traits.split(',')]

    print(f"  Traits: {len(trait_list)}")

    # Get number of layers from extraction model
    extraction_model = get_extraction_model(experiment)
    num_layers = get_num_layers(extraction_model)
    mu_values = list(range(num_layers))

    # Parse sigma values
    if quick:
        sigma_list = [2.0]
        print("  Mode: Quick (sigma=2 only)")
    else:
        sigma_list = [float(s.strip()) for s in sigma_values.split(',')]
        print(f"  Sigma values: {sigma_list}")

    # Output path
    if output:
        output_path = Path(output)
    else:
        output_path = get_path('ensemble.evaluation', experiment=experiment)

    print(f"  Output: {output_path}")
    print()

    # Run grid search
    results = run_grid_search(
        experiment=experiment,
        traits=trait_list,
        mu_values=mu_values,
        sigma_values=sigma_list,
        method=method,
        component=component,
        normalize=normalize,
        resume=resume,
        output_path=output_path,
    )

    # Final save
    _save_results(
        output_path, results['all_results'],
        results['extraction_model'], experiment,
        mu_values, sigma_list, method, component
    )

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    best_per_trait = results.get('best_per_trait', {})
    if not best_per_trait:
        # Recompute from all_results
        for trait in set(r['trait'] for r in results['all_results']):
            trait_results = [r for r in results['all_results'] if r['trait'] == trait]
            if trait_results:
                best = max(trait_results, key=lambda r: r['val_accuracy'])
                best_per_trait[trait] = best

    print(f"\nBest (mu, sigma) per trait:")
    for trait, best in sorted(best_per_trait.items()):
        if isinstance(best, dict):
            mu = best.get('mu', best.get('mu'))
            sigma = best.get('sigma', best.get('sigma'))
            acc = best.get('val_accuracy', 0)
            print(f"  {trait}: mu={mu}, sigma={sigma}, acc={acc:.1%}")

    # Show comparison if available
    comparison = results.get('comparison', {})
    if comparison:
        print(f"\nComparison to single-layer baseline:")
        improvements = []
        for trait, comp in sorted(comparison.items()):
            imp = comp.get('accuracy_improvement', 0)
            improvements.append(imp)
            symbol = "+" if imp > 0 else ""
            print(f"  {trait}: {symbol}{imp:.1%}")

        avg_imp = np.mean(improvements) if improvements else 0
        print(f"\n  Average improvement: {'+' if avg_imp > 0 else ''}{avg_imp:.1%}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
