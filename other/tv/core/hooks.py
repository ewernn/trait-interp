"""Hook management for transformer models.

HookManager: base for all hook registration (single source of truth)
CaptureHook: capture activations from a layer
SteeringHook: add vectors to layer outputs
MultiLayerCapture: capture one component across many layers
"""

import torch
from typing import Any, Callable, List, Sequence, Union


# =============================================================================
# Path utilities
# =============================================================================

def _get_layer_path_prefix(model) -> str:
    """Get hook path prefix to transformer layers, handling PeftModel wrapper."""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return "model.language_model.layers"
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        if type(model).__name__ != type(model.base_model).__name__:
            return "base_model.model.model.layers"
    return "model.layers"


def detect_contribution_paths(model) -> dict:
    """Auto-detect where attention/MLP contributions come from based on model architecture.

    Returns:
        Dict mapping 'attn_contribution' and 'mlp_contribution' to their submodule paths.
    """
    # Navigate to first layer
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        if type(model).__name__ != type(model.base_model).__name__:
            layer = model.base_model.model.model.layers[0]
        else:
            layer = model.model.layers[0]
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        layer = model.model.language_model.layers[0]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer = model.model.layers[0]
    else:
        raise AttributeError(f"Cannot find layers in model: {type(model).__name__}")

    children = dict(layer.named_children())

    if 'pre_feedforward_layernorm' in children:
        return {
            'attn_contribution': 'post_attention_layernorm',
            'mlp_contribution': 'post_feedforward_layernorm',
        }
    elif 'self_attn' in children and 'mlp' in children:
        return {
            'attn_contribution': 'self_attn.o_proj',
            'mlp_contribution': 'mlp.down_proj',
        }
    else:
        raise ValueError(
            f"Unknown architecture - cannot auto-detect contribution paths. "
            f"Layer has children: {sorted(children.keys())}."
        )


def get_hook_path(layer: int, component: str = "residual", prefix: str = None, model=None) -> str:
    """Convert layer + component to string path for hooking.

    Args:
        layer: Layer index (0-indexed)
        component: "residual", "attn_out", "mlp_out", "attn_contribution", "mlp_contribution",
                   "k_proj", "v_proj"
        prefix: Path prefix to layers (auto-detected if model provided, else "model.layers")
        model: Used for auto-detecting prefix and architecture
    """
    if prefix is None:
        if model is not None:
            prefix = _get_layer_path_prefix(model)
        else:
            prefix = "model.layers"

    paths = {
        'residual': f"{prefix}.{layer}",
        'attn_out': f"{prefix}.{layer}.self_attn.o_proj",
        'mlp_out': f"{prefix}.{layer}.mlp.down_proj",
        'k_proj': f"{prefix}.{layer}.self_attn.k_proj",
        'v_proj': f"{prefix}.{layer}.self_attn.v_proj",
    }

    if component in ('attn_contribution', 'mlp_contribution'):
        if model is None:
            raise ValueError(f"Component '{component}' requires model parameter for architecture detection")
        contrib_paths = detect_contribution_paths(model)
        paths['attn_contribution'] = f"{prefix}.{layer}.{contrib_paths['attn_contribution']}"
        paths['mlp_contribution'] = f"{prefix}.{layer}.{contrib_paths['mlp_contribution']}"

    if component not in paths:
        raise ValueError(f"Unknown component: {component}. Valid: {list(paths.keys())}")
    return paths[component]


# =============================================================================
# HookManager
# =============================================================================

class HookManager:
    """Base for all hook registration. Single source of truth for path navigation.

    Usage:
        with HookManager(model) as hooks:
            hooks.add_forward_hook("model.layers.16", my_hook_fn)
            output = model.generate(input_ids)
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _navigate_path(self, path: str) -> torch.nn.Module:
        module = self.model
        for part in path.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def add_forward_hook(self, path: str, hook_fn: Callable) -> torch.utils.hooks.RemovableHandle:
        module = self._navigate_path(path)
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle

    def remove_all(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> 'HookManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_all()


# =============================================================================
# CaptureHook
# =============================================================================

class CaptureHook:
    """Capture activations from a single layer.

    Usage:
        with CaptureHook(model, "model.layers.16") as hook:
            model(**inputs)
        activations = hook.get()  # [batch, seq, hidden]
    """

    def __init__(self, model: torch.nn.Module, path: str, keep_on_gpu: bool = False):
        self.path = path
        self._manager = HookManager(model)
        self.captured: List[torch.Tensor] = []
        self.keep_on_gpu = keep_on_gpu

    def _hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            tensor = outputs[0]
        else:
            tensor = outputs
        captured = tensor.detach() if self.keep_on_gpu else tensor.detach().cpu()
        self.captured.append(captured)
        return None

    def get(self, concat: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self.captured:
            raise ValueError(f"No activations captured for path '{self.path}'")
        if concat:
            return torch.cat(self.captured, dim=0)
        return self.captured

    def clear(self):
        self.captured = []

    def __enter__(self):
        self._manager.add_forward_hook(self.path, self._hook_fn)
        return self

    def __exit__(self, *exc):
        self._manager.remove_all()


# =============================================================================
# SteeringHook
# =============================================================================

class SteeringHook:
    """Add (coefficient * vector) to a layer's output during forward pass.

    Usage:
        vector = torch.load('vectors/probe_layer16.pt')
        with SteeringHook(model, vector, "model.layers.16", coefficient=1.5):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vector: Union[torch.Tensor, Sequence[float]],
        path: str,
        coefficient: float = 1.0,
    ):
        self.path = path
        self._manager = HookManager(model)
        self.coefficient = float(coefficient)

        param = next(model.parameters())
        self.vector = torch.as_tensor(vector, dtype=torch.float32, device=param.device)

        if self.vector.ndim != 1:
            raise ValueError(f"Vector must be 1-D, got shape {self.vector.shape}")

    def _hook_fn(self, module, inputs, outputs):
        out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        steer = (self.coefficient * self.vector).to(device=out_tensor.device, dtype=out_tensor.dtype)

        if torch.is_tensor(outputs):
            return outputs + steer
        elif isinstance(outputs, tuple) and torch.is_tensor(outputs[0]):
            return (outputs[0] + steer, *outputs[1:])
        return outputs

    def __enter__(self):
        self._manager.add_forward_hook(self.path, self._hook_fn)
        return self

    def __exit__(self, *exc):
        self._manager.remove_all()


# =============================================================================
# MultiLayerCapture
# =============================================================================

class MultiLayerCapture:
    """Capture activations from multiple layers in one forward pass.

    Usage:
        with MultiLayerCapture(model, layers=[14, 15, 16]) as cap:
            model(**inputs)
        acts_16 = cap.get(16)
        all_acts = cap.get_all()  # {14: tensor, 15: tensor, 16: tensor}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers: List[int] = None,
        component: str = "residual",
        keep_on_gpu: bool = False,
    ):
        prefix = _get_layer_path_prefix(model)

        if layers is None:
            config = model.config
            if hasattr(config, 'text_config'):
                config = config.text_config
            layers = list(range(config.num_hidden_layers))

        self._hooks = {
            layer: CaptureHook(model, get_hook_path(layer, component, prefix, model=model), keep_on_gpu=keep_on_gpu)
            for layer in layers
        }

    def get(self, layer: int) -> torch.Tensor:
        if layer not in self._hooks:
            raise KeyError(f"Layer {layer} not captured. Available: {list(self._hooks.keys())}")
        return self._hooks[layer].get()

    def get_all(self) -> dict:
        return {layer: hook.get() for layer, hook in self._hooks.items()}

    def clear(self):
        for hook in self._hooks.values():
            hook.clear()

    def __enter__(self):
        for hook in self._hooks.values():
            hook.__enter__()
        return self

    def __exit__(self, *exc):
        for hook in self._hooks.values():
            hook.__exit__(*exc)
