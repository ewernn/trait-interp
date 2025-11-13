#!/usr/bin/env python3
"""
Extract all 8 persona vectors for Gemma 2 2B IT.
Run with: python extract_all_8_traits.py
"""

import subprocess
import time
import pandas as pd
import torch
import os

# All 8 traits to extract
TRAITS = [
    "refusal",
    "uncertainty",
    "verbosity",
    "overconfidence",
    "corrigibility",
    "evil",
    "sycophantic",
    "hallucinating"
]

MODEL = "google/gemma-2-2b-it"
LAYER = 16

def run_command(cmd):
    """Run shell command and handle errors."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise Exception(f"Command failed with exit code {result.returncode}")
    return result

def create_directories():
    """Create necessary directories."""
    os.makedirs("persona_vectors/gemma-2-2b-it", exist_ok=True)
    os.makedirs("eval/outputs/gemma-2-2b-it", exist_ok=True)

    # Create dummy vector
    torch.save(torch.zeros(27, 2304), 'persona_vectors/gemma-2-2b-it/dummy.pt')
    print("✓ Directories created")

def extract_trait(trait, index, total):
    """Extract a single trait vector."""
    print(f"\n{'='*70}")
    print(f"TRAIT {index}/{total}: {trait.upper()}")
    print(f"{'='*70}")

    trait_start = time.time()

    # 1. Generate positive responses
    print(f"\n[1/3] Generating positive responses...")
    cmd = f"""PYTHONPATH=. python eval/eval_persona.py \
      --model {MODEL} \
      --trait {trait} \
      --output_path eval/outputs/gemma-2-2b-it/{trait}_pos.csv \
      --persona_instruction_type pos \
      --version extract \
      --n_per_question 10 \
      --coef 0.0001 \
      --vector_path persona_vectors/gemma-2-2b-it/dummy.pt \
      --layer {LAYER} \
      --batch_process True"""
    run_command(cmd)

    df_pos = pd.read_csv(f'eval/outputs/gemma-2-2b-it/{trait}_pos.csv')
    pos_score = df_pos[trait].mean()
    print(f"✓ {len(df_pos)} positive responses, avg score: {pos_score:.2f}")

    # 2. Generate negative responses
    print(f"\n[2/3] Generating negative responses...")
    cmd = f"""PYTHONPATH=. python eval/eval_persona.py \
      --model {MODEL} \
      --trait {trait} \
      --output_path eval/outputs/gemma-2-2b-it/{trait}_neg.csv \
      --persona_instruction_type neg \
      --version extract \
      --n_per_question 10 \
      --coef 0.0001 \
      --vector_path persona_vectors/gemma-2-2b-it/dummy.pt \
      --layer {LAYER} \
      --batch_process True"""
    run_command(cmd)

    df_neg = pd.read_csv(f'eval/outputs/gemma-2-2b-it/{trait}_neg.csv')
    neg_score = df_neg[trait].mean()
    print(f"✓ {len(df_neg)} negative responses, avg score: {neg_score:.2f}")

    # 3. Extract vector
    print(f"\n[3/3] Extracting {trait} vector...")
    cmd = f"""PYTHONPATH=. python core/generate_vec.py \
      --model_name {MODEL} \
      --pos_path eval/outputs/gemma-2-2b-it/{trait}_pos.csv \
      --neg_path eval/outputs/gemma-2-2b-it/{trait}_neg.csv \
      --trait {trait} \
      --save_dir persona_vectors/gemma-2-2b-it \
      --threshold 50"""
    run_command(cmd)

    # Verify vector
    vector = torch.load(f'persona_vectors/gemma-2-2b-it/{trait}_response_avg_diff.pt')
    magnitude = vector.norm(dim=1).mean().item()
    print(f"✓ Vector extracted: shape {vector.shape}, magnitude {magnitude:.2f}")

    trait_time = time.time() - trait_start

    result = {
        'trait': trait,
        'pos_score': pos_score,
        'neg_score': neg_score,
        'contrast': pos_score - neg_score,
        'magnitude': magnitude,
        'time_minutes': trait_time / 60
    }

    print(f"\n✓ {trait} complete in {trait_time/60:.1f} minutes")
    print(f"  Contrast: {pos_score:.2f} (pos) - {neg_score:.2f} (neg) = {pos_score - neg_score:.2f}")

    return result

def main():
    """Main extraction loop."""
    print("="*70)
    print("EXTRACTING 8 PERSONA VECTORS FOR GEMMA 2 2B IT")
    print("="*70)
    print(f"\nTraits: {', '.join(TRAITS)}")
    print(f"Model: {MODEL}")
    print(f"Layer: {LAYER}")
    print(f"\nEstimated time: ~3.5 hours on A100")
    print(f"Estimated cost: ~$4-5")
    print("="*70)

    # Setup
    create_directories()

    # Extract all traits
    results = []
    start_time = time.time()

    for i, trait in enumerate(TRAITS, 1):
        try:
            result = extract_trait(trait, i, len(TRAITS))
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR extracting {trait}: {e}")
            print(f"Continuing with remaining traits...\n")
            continue

    # Summary
    total_time = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"EXTRACTION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successfully extracted: {len(results)}/{len(TRAITS)} traits")
    print(f"\nResults summary:")
    for r in results:
        print(f"  {r['trait']:15s} | contrast: {r['contrast']:6.2f} | mag: {r['magnitude']:6.2f} | time: {r['time_minutes']:5.1f}m")

    # Verify all vectors
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    for trait in TRAITS:
        vector_path = f'persona_vectors/gemma-2-2b-it/{trait}_response_avg_diff.pt'
        if os.path.exists(vector_path):
            v = torch.load(vector_path)
            mag = v.norm(dim=1).mean().item()
            print(f"✓ {trait:15s} | shape: {str(v.shape):15s} | magnitude: {mag:6.2f}")
        else:
            print(f"✗ {trait:15s} | MISSING")

    print(f"\n{'='*70}")
    print("All vectors saved to: persona_vectors/gemma-2-2b-it/")
    print("All responses saved to: eval/outputs/gemma-2-2b-it/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
