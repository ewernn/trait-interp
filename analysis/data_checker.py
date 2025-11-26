#!/usr/bin/env python3
"""
Check available data for trait-interp experiments.

Input:
    - experiments/{experiment}/extraction/
    - experiments/{experiment}/inference/
    - config/paths.yaml (schema section defines expected files)

Output:
    - Console report (or JSON with --json_output)

Usage:
    python analysis/data_checker.py --experiment gemma_2b_cognitive_nov21
    python analysis/data_checker.py --experiment gemma_2b_cognitive_nov21 --json_output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
import fire
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from enum import Enum

# =============================================================================
# SCHEMA LOADING (single source of truth from paths.yaml)
# =============================================================================

_schema = None

def _load_schema():
    """Load schema from paths.yaml (cached)."""
    global _schema
    if _schema is None:
        config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        _schema = config.get('schema', {})
    return _schema

def get_schema():
    """Get the data schema."""
    return _load_schema()


class Status(str, Enum):
    OK = "ok"
    MISSING = "missing"
    PARTIAL = "partial"
    EMPTY = "empty"


@dataclass
class FileCheck:
    path: str
    exists: bool
    expected: bool = True
    note: str = ""


@dataclass
class TraitIntegrity:
    trait: str
    category: str
    status: Status = Status.OK

    # Extraction
    prompts: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, bool] = field(default_factory=dict)
    responses: Dict[str, bool] = field(default_factory=dict)
    activations: Dict[str, int] = field(default_factory=dict)  # count of files
    vectors: Dict[str, int] = field(default_factory=dict)  # count of files

    # Counts
    expected_activations: int = 0
    expected_vectors: int = 0

    issues: List[str] = field(default_factory=list)


@dataclass
class InferenceIntegrity:
    prompt_sets: Dict[str, bool] = field(default_factory=dict)
    raw_activations: Dict[str, int] = field(default_factory=dict)  # prompt_set -> count
    projections: Dict[str, Dict[str, int]] = field(default_factory=dict)  # trait -> {prompt_set -> count}
    issues: List[str] = field(default_factory=list)


@dataclass
class ExperimentIntegrity:
    experiment: str
    n_layers: int
    n_methods: int
    methods: List[str]

    traits: List[TraitIntegrity] = field(default_factory=list)
    inference: Optional[InferenceIntegrity] = None
    evaluation_exists: bool = False

    summary: Dict[str, int] = field(default_factory=dict)


def check_extraction_trait(
    trait_dir: Path,
    category: str,
    trait_name: str,
    n_layers: int,
    methods: List[str],
    schema: dict
) -> TraitIntegrity:
    """Check all extraction files for a single trait."""

    extraction_schema = schema.get('extraction', {})
    n_activation_prefixes = len(extraction_schema.get('activation_prefixes', ['pos', 'neg', 'val_pos', 'val_neg']))

    result = TraitIntegrity(
        trait=f"{category}/{trait_name}",
        category=category,
        expected_activations=n_activation_prefixes * n_layers,
        expected_vectors=2 * len(methods) * n_layers,  # .pt + metadata.json
    )

    # Prompt files (from schema)
    prompt_files = extraction_schema.get('prompts', [
        "positive.txt", "negative.txt", "val_positive.txt", "val_negative.txt"
    ])
    for f in prompt_files:
        result.prompts[f] = (trait_dir / f).exists()
        if not result.prompts[f]:
            result.issues.append(f"Missing prompt file: {f}")

    # Metadata files (from schema)
    metadata_files = extraction_schema.get('metadata', [
        "generation_metadata.json", "trait_definition.txt"
    ])
    for f in metadata_files:
        result.metadata[f] = (trait_dir / f).exists()
        if not result.metadata[f]:
            result.issues.append(f"Missing metadata file: {f}")

    # Response files (from schema)
    response_paths = extraction_schema.get('responses', [
        "responses/pos.json", "responses/neg.json",
        "val_responses/val_pos.json", "val_responses/val_neg.json"
    ])
    for resp_path in response_paths:
        full_path = trait_dir / resp_path
        result.responses[resp_path] = full_path.exists()
        if not full_path.exists():
            result.issues.append(f"Missing response file: {resp_path}")

    # Activation files
    activations_dir = trait_dir / "activations"
    val_activations_dir = trait_dir / "val_activations"

    result.activations["metadata"] = (activations_dir / "metadata.json").exists()
    if not result.activations["metadata"]:
        result.issues.append("Missing activations/metadata.json")

    # Count activation .pt files
    pos_acts = list(activations_dir.glob("pos_layer*.pt"))
    neg_acts = list(activations_dir.glob("neg_layer*.pt"))
    val_pos_acts = list(val_activations_dir.glob("val_pos_layer*.pt"))
    val_neg_acts = list(val_activations_dir.glob("val_neg_layer*.pt"))

    result.activations["pos_layers"] = len(pos_acts)
    result.activations["neg_layers"] = len(neg_acts)
    result.activations["val_pos_layers"] = len(val_pos_acts)
    result.activations["val_neg_layers"] = len(val_neg_acts)

    total_acts = len(pos_acts) + len(neg_acts) + len(val_pos_acts) + len(val_neg_acts)
    if total_acts < result.expected_activations:
        result.issues.append(f"Missing activations: {total_acts}/{result.expected_activations}")

    # Vector files
    vectors_dir = trait_dir / "vectors"

    for method in methods:
        pt_files = list(vectors_dir.glob(f"{method}_layer*.pt"))
        # Filter out metadata files
        pt_files = [f for f in pt_files if "_metadata" not in f.name]
        meta_files = list(vectors_dir.glob(f"{method}_layer*_metadata.json"))

        result.vectors[f"{method}_pt"] = len(pt_files)
        result.vectors[f"{method}_meta"] = len(meta_files)

        if len(pt_files) < n_layers:
            result.issues.append(f"Missing {method} vectors: {len(pt_files)}/{n_layers}")
        if len(meta_files) < n_layers:
            result.issues.append(f"Missing {method} metadata: {len(meta_files)}/{n_layers}")

    # Determine overall status
    if len(result.issues) == 0:
        result.status = Status.OK
    elif all("migration" in issue or "trait_definition" in issue for issue in result.issues):
        result.status = Status.OK  # Only migration warnings
    elif any("Missing prompt" in issue or "Missing response" in issue for issue in result.issues):
        # Check if completely empty
        total_files = sum(1 for v in result.prompts.values() if v)
        total_files += sum(1 for v in result.responses.values() if v)
        if total_files == 0:
            result.status = Status.EMPTY
        else:
            result.status = Status.PARTIAL
    else:
        result.status = Status.PARTIAL

    return result


def check_inference(exp_dir: Path, traits: List[str]) -> InferenceIntegrity:
    """Check inference directory structure."""

    result = InferenceIntegrity()
    inference_dir = exp_dir / "inference"

    if not inference_dir.exists():
        result.issues.append("No inference directory")
        return result

    # Check prompt files
    prompts_dir = inference_dir / "prompts"
    if prompts_dir.exists():
        for prompt_file in prompts_dir.glob("*.json"):
            result.prompt_sets[prompt_file.stem] = True

        # Check for old format
        old_txt = list(prompts_dir.glob("*.txt"))
        if old_txt:
            result.issues.append(f"Old .txt prompt files need removal: {[f.name for f in old_txt]}")
    else:
        result.issues.append("No inference/prompts directory")

    # Check raw activations
    raw_dir = inference_dir / "raw" / "residual"
    if raw_dir.exists():
        for prompt_set_dir in raw_dir.iterdir():
            if prompt_set_dir.is_dir():
                pt_files = list(prompt_set_dir.glob("*.pt"))
                result.raw_activations[prompt_set_dir.name] = len(pt_files)

    # Check per-trait projections
    for trait in traits:
        trait_inference_dir = inference_dir / trait / "residual_stream"
        if trait_inference_dir.exists():
            result.projections[trait] = {}
            for prompt_set_dir in trait_inference_dir.iterdir():
                if prompt_set_dir.is_dir():
                    json_files = list(prompt_set_dir.glob("*.json"))
                    result.projections[trait][prompt_set_dir.name] = len(json_files)

    return result


def check_evaluation(exp_dir: Path) -> bool:
    """Check if extraction evaluation results exist."""
    evaluation_file = exp_dir / "extraction" / "extraction_evaluation.json"
    return evaluation_file.exists()


def check_experiment(
    experiment: str,
    n_layers: int = None,
    methods: List[str] = None,
) -> ExperimentIntegrity:
    """Check all data integrity for an experiment."""

    # Load schema for defaults
    schema = get_schema()

    if n_layers is None:
        n_layers = schema.get('n_layers', 26)
    if methods is None:
        methods = schema.get('methods', ["probe", "mean_diff", "ica", "gradient"])

    exp_dir = Path("experiments") / experiment
    if not exp_dir.exists():
        raise ValueError(f"Experiment not found: {experiment}")

    result = ExperimentIntegrity(
        experiment=experiment,
        n_layers=n_layers,
        n_methods=len(methods),
        methods=methods,
    )

    # Find all traits
    extraction_dir = exp_dir / "extraction"
    traits = []

    if extraction_dir.exists():
        for category_dir in extraction_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('.'):
                for trait_dir in category_dir.iterdir():
                    if trait_dir.is_dir():
                        trait_result = check_extraction_trait(
                            trait_dir,
                            category_dir.name,
                            trait_dir.name,
                            n_layers,
                            methods,
                            schema
                        )
                        result.traits.append(trait_result)
                        traits.append(f"{category_dir.name}/{trait_dir.name}")

    # Check inference
    result.inference = check_inference(exp_dir, traits)

    # Check evaluation
    result.evaluation_exists = check_evaluation(exp_dir)

    # Compute summary
    result.summary = {
        "total_traits": len(result.traits),
        "ok": sum(1 for t in result.traits if t.status == Status.OK),
        "partial": sum(1 for t in result.traits if t.status == Status.PARTIAL),
        "empty": sum(1 for t in result.traits if t.status == Status.EMPTY),
        "missing": sum(1 for t in result.traits if t.status == Status.MISSING),
    }

    return result


def print_report(result: ExperimentIntegrity):
    """Print human-readable report."""

    print(f"\n{'='*60}")
    print(f"Data Integrity Report: {result.experiment}")
    print(f"{'='*60}")
    print(f"Config: {result.n_layers} layers, {result.n_methods} methods ({', '.join(result.methods)})")
    print()

    # Summary
    print("SUMMARY")
    print("-" * 40)
    print(f"  Total traits: {result.summary['total_traits']}")
    print(f"  ✅ OK:       {result.summary['ok']}")
    print(f"  ⚠️  Partial:  {result.summary['partial']}")
    print(f"  ❌ Empty:    {result.summary['empty']}")
    print()

    # Per-trait details
    print("EXTRACTION (per trait)")
    print("-" * 40)

    for trait in sorted(result.traits, key=lambda t: t.trait):
        status_icon = {
            Status.OK: "✅",
            Status.PARTIAL: "⚠️",
            Status.EMPTY: "❌",
            Status.MISSING: "❌",
        }[trait.status]

        print(f"\n{status_icon} {trait.trait}")

        # Show file counts
        prompts_ok = sum(trait.prompts.values())
        responses_ok = sum(trait.responses.values())

        total_acts = (
            trait.activations.get("pos_layers", 0) +
            trait.activations.get("neg_layers", 0) +
            trait.activations.get("val_pos_layers", 0) +
            trait.activations.get("val_neg_layers", 0)
        )

        total_vectors = sum(v for k, v in trait.vectors.items() if k.endswith("_pt"))
        total_meta = sum(v for k, v in trait.vectors.items() if k.endswith("_meta"))

        print(f"   Prompts: {prompts_ok}/4 | Responses: {responses_ok}/4 | "
              f"Activations: {total_acts}/{trait.expected_activations} | "
              f"Vectors: {total_vectors}/{result.n_layers * result.n_methods}")

        if trait.issues:
            for issue in trait.issues:
                print(f"   └─ {issue}")

    # Inference
    print(f"\n{'='*60}")
    print("INFERENCE")
    print("-" * 40)

    if result.inference:
        if result.inference.prompt_sets:
            print(f"  Prompt sets: {', '.join(result.inference.prompt_sets.keys())}")
        else:
            print("  Prompt sets: None found")

        if result.inference.raw_activations:
            print("  Raw activations:")
            for ps, count in result.inference.raw_activations.items():
                print(f"    {ps}: {count} files")

        if result.inference.projections:
            print(f"  Projections: {len(result.inference.projections)} traits with data")

        if result.inference.issues:
            print("  Issues:")
            for issue in result.inference.issues:
                print(f"    └─ {issue}")
    else:
        print("  No inference data")

    # Evaluation
    print(f"\n{'='*60}")
    print("EXTRACTION EVALUATION")
    print("-" * 40)
    print(f"  extraction_evaluation.json: {'✅ exists' if result.evaluation_exists else '❌ missing'}")
    print()


def main(
    experiment: str,
    n_layers: int = None,
    methods: str = None,
    json_output: bool = False,
):
    """
    Check data integrity for an experiment.

    Args:
        experiment: Experiment name
        n_layers: Number of model layers (default from schema: 26 for Gemma 2B)
        methods: Comma-separated extraction methods (default from schema)
        json_output: Output as JSON instead of human-readable
    """

    # Parse methods if provided, otherwise let check_experiment use schema defaults
    methods_list = None
    if methods is not None:
        methods_list = [m.strip() for m in methods.split(",")]

    result = check_experiment(experiment, n_layers, methods_list)

    if json_output:
        # Convert to JSON-serializable dict
        output = asdict(result)
        # Convert Status enums to strings
        for trait in output["traits"]:
            trait["status"] = trait["status"].value if hasattr(trait["status"], "value") else trait["status"]
        print(json.dumps(output, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    fire.Fire(main)
