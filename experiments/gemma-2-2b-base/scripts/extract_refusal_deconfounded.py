#!/usr/bin/env python3
"""
Deconfounded refusal extraction: truncate prompts, generate rollouts, classify with LLM.

1. Truncate prompts at "and" (remove action word)
2. Generate 5 rollouts per prompt
3. Classify each rollout with Gemini as refusal/compliance/ambiguous
4. Keep prompts where 4+/5 rollouts agree
5. Extract activations from agreeing rollouts
6. Compare token windows: 5, 15, 30

Usage:
    export GEMINI_API_KEY=your_key
    python experiments/gemma-2-2b-base/scripts/extract_refusal_deconfounded.py --trait action/refusal
"""

import sys
import os
import json
import argparse
import re
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

BASE_MODEL = "google/gemma-2-2b"
EXPERIMENT_DIR = Path(__file__).parent.parent

# Classification
GEMINI_MODEL = "gemini-2.5-flash-lite"
CLASSIFICATION_PROMPT = """Did this text describe a refusal/denial or compliance/agreement action?

Text: "{text}"

Answer with exactly one word: refusal / compliance / ambiguous"""

# Generation settings
N_ROLLOUTS = 5
MAX_NEW_TOKENS = 20
TEMPERATURE = 0.7
AGREEMENT_THRESHOLD = 4  # Need 4+/5 to agree

# Token windows to compare
TOKEN_WINDOWS = [5, 15, 30]


@dataclass
class RolloutResult:
    text: str
    classification: str  # refusal, compliance, ambiguous
    activations: dict = field(default_factory=dict)  # window_size -> {component -> tensor}


@dataclass
class PromptResult:
    original_prompt: str
    truncated_prompt: str
    rollouts: list  # List[RolloutResult]
    dominant_class: str = None  # refusal, compliance, or None if no agreement
    agreement_count: int = 0


def truncate_prompt(prompt: str) -> str:
    """Truncate at 'and' - remove action word and everything after."""
    # Find the last " and " before an action word
    # Pattern: " and [action]" where action is refused/approved/etc
    match = re.search(r'\s+and\s+(?:refused|approved|declined|agreed|denied|granted|rejected|accepted|processed|filled)', prompt, re.IGNORECASE)
    if match:
        return prompt[:match.start()] + " and"
    # Fallback: just find last " and "
    last_and = prompt.rfind(" and ")
    if last_and > 0:
        return prompt[:last_and + 5]  # Include " and "
    return prompt


def classify_rollout(client: genai.Client, text: str) -> str:
    """Use Gemini to classify a rollout as refusal/compliance/ambiguous."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=CLASSIFICATION_PROMPT.format(text=text),
        )
        answer = response.text.strip().lower()
        if 'refusal' in answer:
            return 'refusal'
        elif 'compliance' in answer:
            return 'compliance'
        else:
            return 'ambiguous'
    except Exception as e:
        print(f"    Classification error: {e}")
        return 'ambiguous'


class ActivationCapture:
    """Capture activations during generation."""
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.residual = {i: [] for i in range(n_layers)}
        self.attn_out = {i: [] for i in range(n_layers)}
        self.mlp_out = {i: [] for i in range(n_layers)}

    def clear(self):
        for i in range(self.n_layers):
            self.residual[i] = []
            self.attn_out[i] = []
            self.mlp_out[i] = []

    def aggregate(self, token_window: int) -> dict:
        """Aggregate activations for a given token window."""
        result = {'residual': [], 'attn_out': [], 'mlp_out': []}

        for layer in range(self.n_layers):
            # Concatenate all forward passes
            if self.residual[layer]:
                res = torch.cat(self.residual[layer], dim=1)  # [1, total_tokens, hidden]
                # Take first token_window tokens, mean aggregate
                res = res[0, :token_window, :].mean(dim=0)
                result['residual'].append(res)

            if self.attn_out[layer]:
                attn = torch.cat(self.attn_out[layer], dim=1)
                attn = attn[0, :token_window, :].mean(dim=0)
                result['attn_out'].append(attn)

            if self.mlp_out[layer]:
                mlp = torch.cat(self.mlp_out[layer], dim=1)
                mlp = mlp[0, :token_window, :].mean(dim=0)
                result['mlp_out'].append(mlp)

        # Stack across layers
        for k in result:
            if result[k]:
                result[k] = torch.stack(result[k])  # [n_layers, hidden]

        return result


def generate_with_activations(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    capture: ActivationCapture,
    handles: list,
) -> str:
    """Generate from prompt and capture activations."""
    capture.clear()

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the generated part (remove prompt)
    generated_only = generated_text[len(prompt):].strip()

    return generated_only


def setup_hooks(model, capture: ActivationCapture):
    """Set up forward hooks for all layers."""
    handles = []
    n_layers = model.config.num_hidden_layers

    for layer_idx in range(n_layers):
        # Residual (layer output)
        def make_res_hook(layer=layer_idx):
            def hook(module, input, output):
                capture.residual[layer].append(output[0].detach().cpu())
            return hook

        # Attention output
        def make_attn_hook(layer=layer_idx):
            def hook(module, input, output):
                capture.attn_out[layer].append(output[0].detach().cpu())
            return hook

        # MLP output
        def make_mlp_hook(layer=layer_idx):
            def hook(module, input, output):
                capture.mlp_out[layer].append(output.detach().cpu())
            return hook

        handles.append(model.model.layers[layer_idx].register_forward_hook(make_res_hook()))
        handles.append(model.model.layers[layer_idx].self_attn.register_forward_hook(make_attn_hook()))
        handles.append(model.model.layers[layer_idx].mlp.register_forward_hook(make_mlp_hook()))

    return handles


def run_extraction(trait: str, device: str = 'auto'):
    """Run the full deconfounded extraction pipeline."""

    trait_dir = EXPERIMENT_DIR / 'extraction' / trait
    prompts_file = trait_dir / 'prompts.json'

    if not prompts_file.exists():
        print(f"ERROR: {prompts_file} not found")
        return

    # Load prompts
    with open(prompts_file) as f:
        data = json.load(f)

    pairs = data['pairs']
    print(f"Loaded {len(pairs)} prompt pairs")

    # Initialize Gemini client
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return
    gemini_client = genai.Client(api_key=api_key)
    print(f"Gemini client initialized (model: {GEMINI_MODEL})")

    # Load model
    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers")

    # Set up activation capture
    capture = ActivationCapture(n_layers)
    handles = setup_hooks(model, capture)

    try:
        # Process all prompts (both refusal and compliance sides)
        all_results = []

        for pair in tqdm(pairs, desc="Processing pairs"):
            for side in ['refusal', 'compliance']:
                original = pair[side]
                truncated = truncate_prompt(original)

                result = PromptResult(
                    original_prompt=original,
                    truncated_prompt=truncated,
                    rollouts=[],
                )

                # Generate N rollouts
                for _ in range(N_ROLLOUTS):
                    generated = generate_with_activations(
                        truncated, model, tokenizer, capture, handles
                    )

                    # Classify with Gemini
                    classification = classify_rollout(gemini_client, generated)

                    # Capture activations for all token windows
                    rollout = RolloutResult(
                        text=generated,
                        classification=classification,
                        activations={w: capture.aggregate(w) for w in TOKEN_WINDOWS},
                    )
                    result.rollouts.append(rollout)

                # Check agreement
                class_counts = defaultdict(int)
                for r in result.rollouts:
                    if r.classification in ['refusal', 'compliance']:
                        class_counts[r.classification] += 1

                if class_counts:
                    dominant = max(class_counts, key=class_counts.get)
                    count = class_counts[dominant]
                    if count >= AGREEMENT_THRESHOLD:
                        result.dominant_class = dominant
                        result.agreement_count = count

                all_results.append(result)

                # Progress update
                if len(all_results) % 20 == 0:
                    agreed = sum(1 for r in all_results if r.dominant_class)
                    print(f"  {len(all_results)} prompts processed, {agreed} with agreement")

    finally:
        # Clean up hooks
        for h in handles:
            h.remove()

    # Filter to agreed prompts
    agreed_results = [r for r in all_results if r.dominant_class]
    refusal_results = [r for r in agreed_results if r.dominant_class == 'refusal']
    compliance_results = [r for r in agreed_results if r.dominant_class == 'compliance']

    print(f"\nFiltering results:")
    print(f"  Total prompts: {len(all_results)}")
    print(f"  With agreement (4+/5): {len(agreed_results)}")
    print(f"  Refusal: {len(refusal_results)}")
    print(f"  Compliance: {len(compliance_results)}")

    if not refusal_results or not compliance_results:
        print("ERROR: Need both refusal and compliance examples")
        return

    # Aggregate activations by class and window
    output_dir = trait_dir / 'activations_deconfounded'
    output_dir.mkdir(parents=True, exist_ok=True)

    for window in TOKEN_WINDOWS:
        print(f"\nProcessing window {window}...")

        # Collect activations from agreeing rollouts
        refusal_acts = {'residual': [], 'attn_out': [], 'mlp_out': []}
        compliance_acts = {'residual': [], 'attn_out': [], 'mlp_out': []}

        for r in refusal_results:
            for rollout in r.rollouts:
                if rollout.classification == 'refusal':
                    for k in refusal_acts:
                        if k in rollout.activations[window]:
                            refusal_acts[k].append(rollout.activations[window][k])

        for r in compliance_results:
            for rollout in r.rollouts:
                if rollout.classification == 'compliance':
                    for k in compliance_acts:
                        if k in rollout.activations[window]:
                            compliance_acts[k].append(rollout.activations[window][k])

        # Stack and save
        window_dir = output_dir / f'window_{window}'
        window_dir.mkdir(exist_ok=True)

        for component in ['residual', 'attn_out', 'mlp_out']:
            if refusal_acts[component] and compliance_acts[component]:
                pos = torch.stack(refusal_acts[component])  # [n_refusal, n_layers, hidden]
                neg = torch.stack(compliance_acts[component])  # [n_compliance, n_layers, hidden]
                combined = torch.cat([pos, neg], dim=0)
                torch.save(combined, window_dir / f'{component}.pt')
                print(f"  {component}: {pos.shape[0]} refusal, {neg.shape[0]} compliance")

        # Save metadata
        meta = {
            'n_refusal': len(refusal_acts['residual']),
            'n_compliance': len(compliance_acts['residual']),
            'token_window': window,
            'n_layers': n_layers,
            'agreement_threshold': AGREEMENT_THRESHOLD,
        }
        with open(window_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    # Save generation log
    log = {
        'total_prompts': len(all_results),
        'agreed_prompts': len(agreed_results),
        'refusal_prompts': len(refusal_results),
        'compliance_prompts': len(compliance_results),
        'results': [
            {
                'original': r.original_prompt,
                'truncated': r.truncated_prompt,
                'dominant_class': r.dominant_class,
                'agreement': r.agreement_count,
                'rollouts': [
                    {'text': ro.text, 'classification': ro.classification}
                    for ro in r.rollouts
                ],
            }
            for r in all_results
        ],
    }
    with open(output_dir / 'generation_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nDone! Outputs in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trait', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    run_extraction(args.trait, args.device)


if __name__ == '__main__':
    main()
