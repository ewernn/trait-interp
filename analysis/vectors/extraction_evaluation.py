#!/usr/bin/env python3
"""
Evaluate extracted vectors on held-out validation data.

Input:
    - experiments/{experiment}/extraction/{trait}/vectors/*.pt
    - experiments/{experiment}/extraction/{trait}/val_activations/*.pt

Output:
    - experiments/{experiment}/extraction/extraction_evaluation.json

Usage:
    python analysis/vectors/extraction_evaluation.py --experiment my_experiment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
import fire
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from utils.paths import get as get_path
from traitlens.metrics import (
    evaluate_vector,
    accuracy as compute_accuracy,
    cross_trait_accuracy,
)
from sklearn.metrics import roc_auc_score


@dataclass
class ValidationResult:
    """Results for a single vector on validation data."""
    trait: str
    method: str
    layer: int

    # Validation metrics (the real test)
    val_accuracy: float
    val_separation: float
    val_effect_size: float
    val_p_value: float

    # Training metrics (for comparison)
    train_accuracy: Optional[float] = None
    train_separation: Optional[float] = None

    # Overfitting detection
    accuracy_drop: Optional[float] = None  # train_acc - val_acc
    separation_ratio: Optional[float] = None  # val_sep / train_sep

    # Polarity
    polarity_correct: bool = True
    val_pos_mean: float = 0.0
    val_neg_mean: float = 0.0

    # Vector properties
    vector_norm: float = 0.0
    vector_sparsity: float = 0.0  # % of components < 0.01
    vector_mean: float = 0.0
    vector_std: float = 0.0

    # Distribution properties
    val_pos_std: float = 0.0  # Std of positive projections
    val_neg_std: float = 0.0  # Std of negative projections
    overlap_coefficient: float = 0.0  # Distribution overlap
    separation_margin: float = 0.0  # (pos_mean - pos_std) - (neg_mean + neg_std)

    # Additional metrics
    val_auc_roc: float = 0.0  # Threshold-independent classification quality
    top_dims: Optional[List[int]] = None  # Indices of top-5 magnitude dimensions
    interference: Optional[float] = None  # Max cosine sim with other trait vectors (computed post-hoc)


def load_validation_activations(experiment: str, trait: str, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load validation activations for a specific layer."""
    val_dir = get_path('extraction.val_activations', experiment=experiment, trait=trait)

    pos_path = val_dir / f"val_pos_layer{layer}.pt"
    neg_path = val_dir / f"val_neg_layer{layer}.pt"

    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(f"Validation activations not found for layer {layer}")

    pos_acts = torch.load(pos_path, weights_only=True)
    neg_acts = torch.load(neg_path, weights_only=True)

    return pos_acts, neg_acts


def load_training_activations(experiment: str, trait: str, layer: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load training activations for comparison."""
    acts_dir = get_path('extraction.activations', experiment=experiment, trait=trait)

    # Try loading from combined file first (legacy format)
    all_layers_path = acts_dir / "all_layers.pt"
    if all_layers_path.exists():
        all_acts = torch.load(all_layers_path, weights_only=True)
        metadata_path = acts_dir / "metadata.json"

        # Get split from metadata
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                n_pos = metadata.get('n_examples_pos', all_acts.shape[0] // 2)
        else:
            n_pos = all_acts.shape[0] // 2

        # all_acts shape: [n_examples, n_layers, hidden_dim]
        pos_acts = all_acts[:n_pos, layer, :]
        neg_acts = all_acts[n_pos:, layer, :]
        return pos_acts, neg_acts

    # Try loading from per-layer files (current format)
    pos_layer_path = acts_dir / f"pos_layer{layer}.pt"
    neg_layer_path = acts_dir / f"neg_layer{layer}.pt"

    if not pos_layer_path.exists() or not neg_layer_path.exists():
        return None, None

    pos_acts = torch.load(pos_layer_path, weights_only=True)
    neg_acts = torch.load(neg_layer_path, weights_only=True)

    return pos_acts, neg_acts


def load_vector(experiment: str, trait: str, method: str, layer: int) -> Optional[torch.Tensor]:
    """Load a specific vector."""
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)
    path = vectors_dir / f"{method}_layer{layer}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    return None


def compute_vector_properties(vector: torch.Tensor) -> Dict:
    """
    Compute properties of a vector.

    Args:
        vector: [hidden_dim] tensor

    Returns:
        Dictionary with norm, sparsity, mean, std, top_dims
    """
    vector_np = vector.float().cpu().numpy()

    # Top-5 magnitude dimensions (interpretability)
    top_indices = np.argsort(np.abs(vector_np))[-5:][::-1].tolist()

    return {
        'vector_norm': float(np.linalg.norm(vector_np)),
        'vector_sparsity': float(np.mean(np.abs(vector_np) < 0.01)),
        'vector_mean': float(np.mean(vector_np)),
        'vector_std': float(np.std(vector_np)),
        'top_dims': top_indices
    }


def compute_distribution_properties(
    pos_projections: torch.Tensor,
    neg_projections: torch.Tensor,
    pos_mean: float,
    neg_mean: float
) -> Dict[str, float]:
    """
    Compute properties of projection score distributions.

    Args:
        pos_projections: [n_pos] tensor of positive projection scores
        neg_projections: [n_neg] tensor of negative projection scores
        pos_mean: Mean of positive projections
        neg_mean: Mean of negative projections

    Returns:
        Dictionary with std, overlap, margin metrics
    """
    pos_std = float(pos_projections.std())
    neg_std = float(neg_projections.std())

    # Overlap coefficient: estimate using normal approximation
    # If pos/neg are well-separated Gaussians, this measures overlap
    if pos_std > 0 and neg_std > 0:
        # Distance between means in units of pooled std
        pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
        z_score = abs(pos_mean - neg_mean) / (pooled_std + 1e-8)
        # Rough overlap estimate (1 - z_score normalized to 0-1)
        overlap = max(0, 1 - z_score / 4.0)  # z=4 → no overlap
    else:
        overlap = 0.5

    # Separation margin: gap between distributions
    # Positive = good separation, negative = overlap
    margin = (pos_mean - pos_std) - (neg_mean + neg_std)

    return {
        'val_pos_std': pos_std,
        'val_neg_std': neg_std,
        'overlap_coefficient': float(overlap),
        'separation_margin': float(margin)
    }


def evaluate_vector_on_validation(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    normalize: bool = True
) -> Optional[ValidationResult]:
    """Evaluate a single vector on validation data."""
    # Load vector
    vector = load_vector(experiment, trait, method, layer)
    if vector is None:
        return None

    # Load validation activations
    try:
        val_pos, val_neg = load_validation_activations(experiment, trait, layer)
    except FileNotFoundError:
        return None

    # Compute validation metrics using traitlens
    val_metrics = evaluate_vector(val_pos, val_neg, vector, normalize=normalize)

    # Compute vector properties
    vector_props = compute_vector_properties(vector)

    # Compute projection scores for distribution properties
    # Note: evaluate_vector already does this internally, but we need the raw projections
    # Convert to float32 for consistent computation (activations may be bfloat16)
    vector_f32 = vector.float()
    val_pos_f32 = val_pos.float()
    val_neg_f32 = val_neg.float()

    if normalize:
        vec_norm = vector_f32 / (vector_f32.norm() + 1e-8)
        pos_norm = val_pos_f32 / (val_pos_f32.norm(dim=1, keepdim=True) + 1e-8)
        neg_norm = val_neg_f32 / (val_neg_f32.norm(dim=1, keepdim=True) + 1e-8)
        pos_projections = pos_norm @ vec_norm
        neg_projections = neg_norm @ vec_norm
    else:
        pos_projections = val_pos_f32 @ vector_f32
        neg_projections = val_neg_f32 @ vector_f32

    # Compute distribution properties
    dist_props = compute_distribution_properties(
        pos_projections,
        neg_projections,
        val_metrics['pos_mean'],
        val_metrics['neg_mean']
    )

    # Compute AUC-ROC (threshold-independent classification quality)
    all_projections = torch.cat([pos_projections, neg_projections]).cpu().numpy()
    all_labels = np.concatenate([np.ones(len(pos_projections)), np.zeros(len(neg_projections))])
    try:
        auc = roc_auc_score(all_labels, all_projections)
    except ValueError:
        auc = 0.5  # Fallback if AUC can't be computed

    # Load training activations for comparison
    train_pos, train_neg = load_training_activations(experiment, trait, layer)
    train_metrics = None
    if train_pos is not None:
        train_metrics = evaluate_vector(train_pos, train_neg, vector, normalize=normalize)

    # Build result
    result = ValidationResult(
        trait=trait,
        method=method,
        layer=layer,
        val_accuracy=val_metrics['accuracy'],
        val_separation=val_metrics['separation'],
        val_effect_size=val_metrics['effect_size'],
        val_p_value=val_metrics['p_value'],
        polarity_correct=val_metrics['polarity_correct'],
        val_pos_mean=val_metrics['pos_mean'],
        val_neg_mean=val_metrics['neg_mean'],
        val_auc_roc=auc,
        **vector_props,
        **dist_props
    )

    # Add training comparison if available
    if train_metrics:
        result.train_accuracy = train_metrics['accuracy']
        result.train_separation = train_metrics['separation']
        result.accuracy_drop = train_metrics['accuracy'] - val_metrics['accuracy']
        if train_metrics['separation'] > 0:
            result.separation_ratio = val_metrics['separation'] / train_metrics['separation']

    return result


def compute_within_trait_similarity(
    experiment: str,
    trait: str,
    methods: List[str],
    layers: List[int]
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise cosine similarity between all vectors for a single trait.

    Args:
        experiment: Experiment name
        trait: Trait name (category/trait)
        methods: List of methods to compare
        layers: List of layers to compare

    Returns:
        Dictionary mapping vector_id -> {other_vector_id: similarity}
    """
    # Load all vectors for this trait
    vectors = {}
    for method in methods:
        for layer in layers:
            vector = load_vector(experiment, trait, method, layer)
            if vector is not None:
                key = f"{method}_layer{layer}"
                vectors[key] = vector

    # Compute pairwise similarities
    similarities = {}
    vector_keys = list(vectors.keys())

    for i, key_i in enumerate(vector_keys):
        similarities[key_i] = {}
        vec_i = vectors[key_i].float()
        vec_i_norm = vec_i / (vec_i.norm() + 1e-8)

        for key_j in vector_keys:
            vec_j = vectors[key_j].float()
            vec_j_norm = vec_j / (vec_j.norm() + 1e-8)
            similarity = float((vec_i_norm @ vec_j_norm).item())
            similarities[key_i][key_j] = similarity

    return similarities


def compute_best_vector_similarity(
    experiment: str,
    traits: List[str],
    all_results: List[Dict],
    metric: str = 'val_accuracy'
) -> pd.DataFrame:
    """
    Compute cosine similarity matrix between best vectors across traits.

    Args:
        experiment: Experiment name
        traits: List of trait names
        all_results: All validation results
        metric: Metric to use for selecting best vector ('val_accuracy', 'val_effect_size', etc.)

    Returns:
        DataFrame with similarity matrix (traits × traits)
    """
    # Find best vector for each trait
    df = pd.DataFrame(all_results)
    best_vectors = {}

    for trait in traits:
        trait_results = df[df['trait'] == trait]
        if len(trait_results) == 0:
            continue

        # Get best by metric
        best_row = trait_results.loc[trait_results[metric].idxmax()]
        vector = load_vector(experiment, trait, best_row['method'], int(best_row['layer']))

        if vector is not None:
            best_vectors[trait] = vector

    # Compute pairwise similarities
    trait_list = list(best_vectors.keys())
    n = len(trait_list)
    matrix = np.zeros((n, n))

    for i, trait_i in enumerate(trait_list):
        vec_i = best_vectors[trait_i].float()
        vec_i_norm = vec_i / (vec_i.norm() + 1e-8)

        for j, trait_j in enumerate(trait_list):
            vec_j = best_vectors[trait_j].float()
            vec_j_norm = vec_j / (vec_j.norm() + 1e-8)
            similarity = float((vec_i_norm @ vec_j_norm).item())
            matrix[i, j] = similarity

    # Create DataFrame
    short_names = [t.split('/')[-1][:15] for t in trait_list]
    similarity_df = pd.DataFrame(matrix, index=short_names, columns=short_names)

    return similarity_df


def compute_interference(
    experiment: str,
    all_results: List[Dict],
    method: str = 'probe',
    layer: int = 16
) -> Dict[str, float]:
    """
    Compute interference for each trait: max cosine similarity with other trait vectors.

    Low interference = independent trait. High interference = entangled with another trait.

    Args:
        experiment: Experiment name
        all_results: All validation results
        method: Method to use for vectors
        layer: Layer to use for vectors

    Returns:
        Dictionary mapping trait -> max cosine sim with other traits
    """
    # Get unique traits
    traits = list(set(r['trait'] for r in all_results))

    # Load vectors
    vectors = {}
    for trait in traits:
        vec = load_vector(experiment, trait, method, layer)
        if vec is not None:
            vectors[trait] = vec.float()

    if len(vectors) < 2:
        return {}

    # Compute interference for each trait
    interference = {}
    for trait_i, vec_i in vectors.items():
        vec_i_norm = vec_i / (vec_i.norm() + 1e-8)
        max_sim = 0.0
        for trait_j, vec_j in vectors.items():
            if trait_i == trait_j:
                continue
            vec_j_norm = vec_j / (vec_j.norm() + 1e-8)
            sim = abs(float((vec_i_norm @ vec_j_norm).item()))
            if sim > max_sim:
                max_sim = sim
        interference[trait_i] = max_sim

    return interference


def compute_cross_layer_consistency(
    experiment: str,
    trait: str,
    method: str,
    layers: List[int]
) -> Dict[str, float]:
    """
    Compute cosine similarity between vectors from adjacent layers.

    Shows whether the trait direction is stable across layers.
    High consistency = trait is robust. Low consistency = trait shifts between layers.

    Args:
        experiment: Experiment name
        trait: Trait name
        method: Extraction method
        layers: List of layers to compare

    Returns:
        Dictionary with per-transition similarities and mean
    """
    sorted_layers = sorted(layers)
    vectors = {}

    # Load vectors for all layers
    for layer in sorted_layers:
        vec = load_vector(experiment, trait, method, layer)
        if vec is not None:
            vectors[layer] = vec

    if len(vectors) < 2:
        return {'mean': 0.0, 'transitions': {}}

    # Compute pairwise similarities between adjacent layers
    transitions = {}
    available_layers = sorted(vectors.keys())

    for i in range(len(available_layers) - 1):
        l1, l2 = available_layers[i], available_layers[i + 1]
        v1 = vectors[l1].float()
        v2 = vectors[l2].float()
        v1_norm = v1 / (v1.norm() + 1e-8)
        v2_norm = v2 / (v2.norm() + 1e-8)
        sim = float((v1_norm @ v2_norm).item())
        transitions[f"{l1}->{l2}"] = sim

    mean_consistency = np.mean(list(transitions.values())) if transitions else 0.0

    return {
        'mean': float(mean_consistency),
        'transitions': transitions
    }


def compute_cross_trait_matrix(
    experiment: str,
    traits: List[str],
    method: str = "probe",
    layer: int = 16,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute cross-trait interference matrix.

    Entry [i,j] = accuracy when projecting trait_i's validation data
                  onto trait_j's vector.

    Diagonal should be high (vectors work for their trait).
    Off-diagonal should be ~50% (random, independent traits).
    """
    n_traits = len(traits)
    matrix = np.zeros((n_traits, n_traits))

    for i, trait_i in enumerate(traits):
        # Load trait_i's validation activations
        try:
            val_pos_i, val_neg_i = load_validation_activations(experiment, trait_i, layer)
        except FileNotFoundError:
            continue

        # Normalize if requested
        if normalize:
            val_pos_i = val_pos_i.float()
            val_neg_i = val_neg_i.float()
            val_pos_i = val_pos_i / (val_pos_i.norm(dim=1, keepdim=True) + 1e-8)
            val_neg_i = val_neg_i / (val_neg_i.norm(dim=1, keepdim=True) + 1e-8)

        for j, trait_j in enumerate(traits):
            # Load trait_j's vector
            vector_j = load_vector(experiment, trait_j, method, layer)
            if vector_j is None:
                continue

            if normalize:
                vector_j = vector_j.float()
                vector_j = vector_j / (vector_j.norm() + 1e-8)

            # Use traitlens cross_trait_accuracy
            matrix[i, j] = cross_trait_accuracy(val_pos_i, val_neg_i, vector_j)

    # Create DataFrame with trait names
    short_names = [t.split('/')[-1][:12] for t in traits]
    df = pd.DataFrame(matrix, index=short_names, columns=short_names)

    return df


def main(experiment: str,
         methods: str = "mean_diff,probe,ica,gradient,pca_diff,random_baseline",
         layers: str = None,
         output: str = None,
         no_normalize: bool = False):
    """
    Run validation evaluation on all traits.

    Args:
        experiment: Experiment name
        methods: Comma-separated methods to evaluate
        layers: Comma-separated layers (default: all)
        output: Output JSON file (default: experiments/{experiment}/extraction/extraction_evaluation.json)
        no_normalize: If True, use raw dot product instead of cosine similarity
    """
    normalize = not no_normalize
    if normalize:
        print("Using cosine similarity (normalized vectors and activations)")
    else:
        print("Using raw dot product (no normalization)")

    # Use paths from config
    exp_dir = get_path('extraction.base', experiment=experiment)
    methods_list = methods.split(",")

    if layers:
        if isinstance(layers, int):
            layers_list = [layers]
        else:
            layers_list = [int(l) for l in str(layers).split(",")]
    else:
        layers_list = list(range(26))

    # Find all traits with validation data
    traits = []
    for category_dir in exp_dir.iterdir():
        if not category_dir.is_dir():
            continue
        for trait_dir in category_dir.iterdir():
            if (trait_dir / "val_activations").exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")

    print(f"Found {len(traits)} traits with validation data")
    print(f"Evaluating methods: {methods_list}")
    print(f"Evaluating layers: {layers_list}")

    # Evaluate all vectors
    all_results = []

    for trait in tqdm(traits, desc="Traits"):
        for method in methods_list:
            for layer in layers_list:
                result = evaluate_vector_on_validation(
                    experiment, trait, method, layer, normalize=normalize
                )
                if result:
                    all_results.append(asdict(result))

    print(f"\nEvaluated {len(all_results)} vectors")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("No results to analyze")
        return

    # =========================================================================
    # ANALYSIS 1: Best method per trait (by validation accuracy)
    # =========================================================================
    print("\n" + "="*80)
    print("BEST METHOD PER TRAIT (by validation accuracy)")
    print("="*80)

    best_per_trait = df.loc[df.groupby('trait')['val_accuracy'].idxmax()]
    print(best_per_trait[['trait', 'method', 'layer', 'val_accuracy', 'val_effect_size']].to_string(index=False))

    # =========================================================================
    # ANALYSIS 2: Method comparison (averaged across traits)
    # =========================================================================
    print("\n" + "="*80)
    print("METHOD COMPARISON (averaged across traits and layers)")
    print("="*80)

    method_summary = df.groupby('method').agg({
        'val_accuracy': ['mean', 'std', 'max'],
        'val_separation': ['mean', 'max'],
        'val_effect_size': ['mean', 'max'],
        'polarity_correct': 'mean'
    }).round(3)
    print(method_summary)

    # =========================================================================
    # ANALYSIS 3: Layer comparison (for best method)
    # =========================================================================
    print("\n" + "="*80)
    print("LAYER COMPARISON (probe method, averaged across traits)")
    print("="*80)

    if 'probe' in df['method'].values:
        probe_df = df[df['method'] == 'probe']
        layer_summary = probe_df.groupby('layer').agg({
            'val_accuracy': 'mean',
            'val_effect_size': 'mean',
            'polarity_correct': 'mean'
        }).round(3)

        # Show best layers
        best_layers = layer_summary.nlargest(5, 'val_accuracy')
        print("Top 5 layers by validation accuracy:")
        print(best_layers)

    # =========================================================================
    # ANALYSIS 4: Overfitting detection (train vs val)
    # =========================================================================
    print("\n" + "="*80)
    print("OVERFITTING DETECTION (train accuracy - val accuracy)")
    print("="*80)

    df_with_train = df.dropna(subset=['train_accuracy'])
    if len(df_with_train) > 0:
        overfit_summary = df_with_train.groupby('method').agg({
            'train_accuracy': 'mean',
            'val_accuracy': 'mean',
            'accuracy_drop': 'mean',
            'separation_ratio': 'mean'
        }).round(3)
        print(overfit_summary)

        # Flag severe overfitting
        severe_overfit = df_with_train[df_with_train['accuracy_drop'] > 0.2]
        if len(severe_overfit) > 0:
            print(f"\n⚠️  {len(severe_overfit)} vectors with >20% accuracy drop (overfitting)")

    # =========================================================================
    # ANALYSIS 5: Polarity issues
    # =========================================================================
    print("\n" + "="*80)
    print("POLARITY ISSUES (vectors pointing wrong direction)")
    print("="*80)

    wrong_polarity = df[~df['polarity_correct']]
    if len(wrong_polarity) > 0:
        print(f"⚠️  {len(wrong_polarity)} vectors with inverted polarity:")
        print(wrong_polarity[['trait', 'method', 'layer', 'val_pos_mean', 'val_neg_mean']].head(20).to_string(index=False))
    else:
        print("✅ All vectors have correct polarity")

    # =========================================================================
    # ANALYSIS 6: Cross-trait interference matrix
    # =========================================================================
    print("\n" + "="*80)
    print("CROSS-TRAIT INTERFERENCE MATRIX (probe_layer16)")
    print("="*80)
    print("Diagonal = trait's vector on its own data (should be high)")
    print("Off-diagonal = trait's vector on other trait's data (should be ~50%)")

    cross_matrix = compute_cross_trait_matrix(experiment, traits, "probe", 16, normalize=normalize)
    print(cross_matrix.round(2).to_string())

    # Independence score: how different is diagonal from off-diagonal
    diagonal = np.diag(cross_matrix.values)
    off_diagonal = cross_matrix.values[~np.eye(len(cross_matrix), dtype=bool)]
    independence = diagonal.mean() - off_diagonal.mean()
    print(f"\nIndependence score: {independence:.3f}")
    print(f"  (Diagonal mean: {diagonal.mean():.3f}, Off-diagonal mean: {off_diagonal.mean():.3f})")

    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Best overall method
    best_method = df.groupby('method')['val_accuracy'].mean().idxmax()
    best_method_acc = df.groupby('method')['val_accuracy'].mean().max()
    print(f"Best method overall: {best_method} (mean val accuracy: {best_method_acc:.1%})")

    # Best layer for best method
    best_method_df = df[df['method'] == best_method]
    best_layer = best_method_df.groupby('layer')['val_accuracy'].mean().idxmax()
    best_layer_acc = best_method_df.groupby('layer')['val_accuracy'].mean().max()
    print(f"Best layer for {best_method}: {best_layer} (mean val accuracy: {best_layer_acc:.1%})")

    # Per-trait recommendations
    print("\nPer-trait best vectors:")
    for trait in traits:
        trait_df = df[df['trait'] == trait]
        if len(trait_df) > 0:
            best_row = trait_df.loc[trait_df['val_accuracy'].idxmax()]
            short_trait = trait.split('/')[-1][:20]
            print(f"  {short_trait}: {best_row['method']}_layer{int(best_row['layer'])} "
                  f"(val_acc={best_row['val_accuracy']:.1%}, d={best_row['val_effect_size']:.2f})")

    # =========================================================================
    # ANALYSIS 7: Within-trait vector similarity
    # =========================================================================
    print("\n" + "="*80)
    print("WITHIN-TRAIT VECTOR SIMILARITY")
    print("="*80)
    print("Computing similarity between different method/layer combos for each trait...")

    within_trait_similarities = {}
    for trait in tqdm(traits, desc="Within-trait similarities"):
        similarities = compute_within_trait_similarity(experiment, trait, methods_list, layers_list)
        within_trait_similarities[trait] = similarities

    # =========================================================================
    # ANALYSIS 8: Best-vector cross-trait similarity
    # =========================================================================
    print("\n" + "="*80)
    print("BEST-VECTOR CROSS-TRAIT SIMILARITY")
    print("="*80)
    print("Comparing best vectors across different traits...")

    best_vector_similarity = compute_best_vector_similarity(experiment, traits, all_results, metric='val_accuracy')
    print(best_vector_similarity.round(3).to_string())

    # Compute average off-diagonal similarity (trait independence)
    n = len(best_vector_similarity)
    off_diag_mask = ~np.eye(n, dtype=bool)
    avg_similarity = best_vector_similarity.values[off_diag_mask].mean()
    print(f"\nAverage best-vector similarity (off-diagonal): {avg_similarity:.3f}")
    print(f"  (Low values indicate independent traits)")

    # =========================================================================
    # ANALYSIS 9: Cross-layer consistency
    # =========================================================================
    print("\n" + "="*80)
    print("CROSS-LAYER CONSISTENCY (probe method)")
    print("="*80)
    print("Cosine similarity between adjacent layer vectors (high = stable direction)")

    cross_layer_results = {}
    for trait in tqdm(traits, desc="Cross-layer consistency"):
        consistency = compute_cross_layer_consistency(experiment, trait, 'probe', layers_list)
        cross_layer_results[trait] = consistency
        short_trait = trait.split('/')[-1][:20]
        print(f"  {short_trait}: mean={consistency['mean']:.3f}")

    avg_consistency = np.mean([r['mean'] for r in cross_layer_results.values()])
    print(f"\nOverall mean cross-layer consistency: {avg_consistency:.3f}")

    # =========================================================================
    # ANALYSIS 10: Interference (max cosine sim with other traits)
    # =========================================================================
    print("\n" + "="*80)
    print("INTERFERENCE (probe_layer16)")
    print("="*80)
    print("Max cosine similarity with other trait vectors (low = independent)")

    interference_results = compute_interference(experiment, all_results, 'probe', 16)
    for trait, interf in sorted(interference_results.items(), key=lambda x: x[1], reverse=True):
        short_trait = trait.split('/')[-1][:20]
        print(f"  {short_trait}: {interf:.3f}")

    avg_interference = np.mean(list(interference_results.values())) if interference_results else 0.0
    print(f"\nMean interference: {avg_interference:.3f}")

    # Add interference to all_results for matching vectors
    for result in all_results:
        if result['method'] == 'probe' and result['layer'] == 16:
            result['interference'] = interference_results.get(result['trait'])

    # Save results
    if output:
        output_path = Path(output)
    else:
        output_path = get_path('extraction_eval.evaluation', experiment=experiment)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert multi-index to string for JSON compatibility
    method_summary_flat = {}
    for col in method_summary.columns:
        col_name = '_'.join(str(c) for c in col) if isinstance(col, tuple) else str(col)
        method_summary_flat[col_name] = method_summary[col].to_dict()

    results_dict = {
        'all_results': all_results,
        'method_summary': method_summary_flat,
        'cross_trait_matrix': cross_matrix.to_dict(),
        'best_per_trait': best_per_trait.to_dict('records'),
        'within_trait_similarities': within_trait_similarities,
        'best_vector_similarity': best_vector_similarity.to_dict(),
        'cross_layer_consistency': cross_layer_results,
        'interference': interference_results,
        'recommendations': {
            'best_method': best_method,
            'best_layer': int(best_layer),
            'mean_val_accuracy': float(best_layer_acc),
            'mean_cross_layer_consistency': float(avg_consistency),
            'mean_interference': float(avg_interference),
            'avg_best_vector_similarity': float(avg_similarity)
        }
    }

    # Custom encoder to handle NaN values (not valid JSON)
    def json_serializer(obj):
        if isinstance(obj, float):
            if obj != obj:  # NaN check
                return None
            return obj
        return float(obj)

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=json_serializer)

    print(f"\n✅ Results saved to {output_path}")

    return results_dict


if __name__ == "__main__":
    fire.Fire(main)
