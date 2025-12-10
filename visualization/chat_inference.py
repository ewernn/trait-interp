"""
Live chat inference with real-time trait projections.

Loads model and trait vectors, generates tokens one at a time,
projects activations onto trait vectors, yields results for SSE streaming.

Usage:
    from visualization.chat_inference import ChatInference

    chat = ChatInference(experiment='gemma-2-2b-it')
    for event in chat.generate("How do I hack a computer?"):
        # event = {'token': '...', 'trait_scores': {'refusal': 0.5, ...}}
        yield f"data: {json.dumps(event)}\n\n"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Dict, Generator, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from traitlens import HookManager, projection
from utils.paths import get as get_path
from utils.vectors import get_best_layer
from utils.model import format_prompt, load_experiment_config


class ChatInference:
    """Manages model and trait vectors for live chat with trait monitoring."""

    def __init__(self, experiment: str, device: str = "auto"):
        self.experiment = experiment
        self.device = device
        self.model = None
        self.tokenizer = None
        self.trait_vectors: Dict[str, Tuple[torch.Tensor, int]] = {}  # trait -> (vector, layer)
        self.n_layers = None
        self.use_chat_template = None
        self._loaded = False

    def load(self):
        """Load model and trait vectors. Called lazily on first generate()."""
        if self._loaded:
            return

        # Get model from experiment config
        config = load_experiment_config(self.experiment)
        model_id = config.get('application_model', 'google/gemma-2-2b-it')

        print(f"[ChatInference] Loading model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
            attn_implementation='eager'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.n_layers = len(self.model.model.layers)
        self.use_chat_template = self.tokenizer.chat_template is not None

        # Build set of stop token IDs (EOS + any end_of_turn tokens)
        self.stop_token_ids = {self.tokenizer.eos_token_id}
        # Gemma uses <end_of_turn> as stop token
        for token in ['<end_of_turn>', '<|end|>', '<|eot_id|>']:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                self.stop_token_ids.add(token_ids[0])

        print(f"[ChatInference] Model loaded: {self.n_layers} layers, chat_template={self.use_chat_template}")
        print(f"[ChatInference] Stop tokens: {self.stop_token_ids}")

        # Load trait vectors
        self._load_trait_vectors()
        self._loaded = True

    def _load_trait_vectors(self):
        """Discover and load all trait vectors with their best layers."""
        extraction_dir = get_path('extraction.base', experiment=self.experiment)
        if not extraction_dir.exists():
            print(f"[ChatInference] No extraction dir: {extraction_dir}")
            return

        for category_dir in sorted(extraction_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            for trait_dir in sorted(category_dir.iterdir()):
                if not trait_dir.is_dir():
                    continue
                vectors_dir = trait_dir / "vectors"
                if not vectors_dir.exists() or not list(vectors_dir.glob('*.pt')):
                    continue

                trait_path = f"{category_dir.name}/{trait_dir.name}"

                # Get best layer/method for this trait
                best = get_best_layer(self.experiment, trait_path)
                layer = best['layer']
                method = best['method']

                vector_file = vectors_dir / f"{method}_layer{layer}.pt"
                if not vector_file.exists():
                    # Fallback: try any available vector
                    available = list(vectors_dir.glob('*.pt'))
                    if available:
                        vector_file = available[0]
                        # Parse layer from filename
                        import re
                        match = re.search(r'layer(\d+)', vector_file.stem)
                        if match:
                            layer = int(match.group(1))
                    else:
                        continue

                vector = torch.load(vector_file, weights_only=True).to(
                    dtype=torch.float16, device=self.model.device
                )
                self.trait_vectors[trait_path] = (vector, layer)

        print(f"[ChatInference] Loaded {len(self.trait_vectors)} trait vectors")
        for trait, (_, layer) in self.trait_vectors.items():
            print(f"  {trait}: layer {layer}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        history: Optional[List[Dict]] = None
    ) -> Generator[Dict, None, None]:
        """
        Generate response token-by-token with trait projections.

        Args:
            prompt: User message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            history: Optional chat history [{"role": "user/assistant", "content": "..."}]

        Yields:
            Dict with 'token', 'trait_scores', 'done' keys
            Status events have 'status' key for loading progress
            Final yield has 'done': True and 'full_response'
        """
        # Yield loading status if not loaded yet
        if not self._loaded:
            yield {'status': 'loading_model', 'message': 'Loading model...'}

        self.load()  # Lazy load

        if not self._loaded:
            yield {'error': 'Failed to load model', 'done': True}
            return

        yield {'status': 'generating', 'message': f'Generating with {len(self.trait_vectors)} traits...'}

        # Build input
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": prompt}]

        if self.use_chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.model.device)
        else:
            # Base model: raw text
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        # Group traits by layer for efficient hooking
        layers_needed = set(layer for _, layer in self.trait_vectors.values())

        # Storage for activations (cleared each token)
        activations = {}

        def make_hook(layer_idx):
            def hook(module, inp, out):
                out_t = out[0] if isinstance(out, tuple) else out
                # Store last token's activation
                activations[layer_idx] = out_t[:, -1, :].detach()
            return hook

        # Generate tokens
        context = input_ids
        generated_tokens = []
        full_response = ""

        for step in range(max_new_tokens):
            activations.clear()

            # Forward pass with hooks
            with HookManager(self.model) as hooks:
                for layer in layers_needed:
                    hooks.add_forward_hook(f"model.layers.{layer}", make_hook(layer))

                with torch.no_grad():
                    outputs = self.model(input_ids=context, return_dict=True)

            # Sample next token
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            # Check for stop tokens first
            if next_id in self.stop_token_ids:
                break

            # Decode token (skip special tokens)
            token_str = self.tokenizer.decode([next_id], skip_special_tokens=True)
            if not token_str:  # Skip if empty after filtering
                context = torch.cat([context, torch.tensor([[next_id]], device=self.model.device)], dim=1)
                continue

            generated_tokens.append(next_id)
            full_response += token_str

            # Compute trait projections
            trait_scores = {}
            for trait_path, (vector, layer) in self.trait_vectors.items():
                if layer in activations:
                    act = activations[layer].squeeze(0)  # [hidden_dim]
                    score = projection(act, vector, normalize_vector=True).item()
                    # Use just trait name (not category) for cleaner output
                    trait_name = trait_path.split('/')[-1]
                    trait_scores[trait_name] = round(score, 4)

            # Yield result
            yield {
                'token': token_str,
                'trait_scores': trait_scores,
                'done': False
            }

            # Update context
            context = torch.cat([context, torch.tensor([[next_id]], device=self.model.device)], dim=1)

        # Final yield with full response
        yield {
            'token': '',
            'trait_scores': {},
            'done': True,
            'full_response': full_response
        }


# Singleton instance (lazy loaded)
_chat_instance: Optional[ChatInference] = None


def get_chat_instance(experiment: str) -> ChatInference:
    """Get or create chat instance for experiment."""
    global _chat_instance
    if _chat_instance is None or _chat_instance.experiment != experiment:
        _chat_instance = ChatInference(experiment)
    return _chat_instance
