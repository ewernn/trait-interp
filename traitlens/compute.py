"""
Computational operations for trait vector analysis.

These are the "pandas operations" for trait analysis - fundamental operations
for extracting and analyzing trait vectors from activations.
"""

import torch
from typing import Optional, Tuple, Union


def mean_difference(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None
) -> torch.Tensor:
    """
    Compute the mean difference between positive and negative activations.

    This is the fundamental operation for trait vector extraction.

    Args:
        pos_acts: Positive example activations
        neg_acts: Negative example activations
        dim: Dimension(s) to average over. If None, infers based on tensor shape.
             For [batch, seq, hidden], averages over batch and seq by default.

    Returns:
        Trait vector as difference of means

    Example:
        >>> # pos_acts shape: [100, 50, 768]  # 100 examples, 50 tokens, 768 dims
        >>> # neg_acts shape: [100, 50, 768]
        >>> vector = mean_difference(pos_acts, neg_acts)
        >>> vector.shape
        torch.Size([768])
    """
    if dim is None:
        # Infer dimensions to average over
        # Keep only the last dimension (assumed to be hidden/feature dimension)
        if pos_acts.ndim > 1:
            dim = tuple(range(pos_acts.ndim - 1))
        else:
            dim = 0

    pos_mean = pos_acts.mean(dim=dim)
    neg_mean = neg_acts.mean(dim=dim)

    return pos_mean - neg_mean


def compute_derivative(
    expression_trajectory: torch.Tensor,
    dt: float = 1.0,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute the first derivative of trait expression over time.

    This represents the "velocity" of trait expression - how fast the trait
    expression is changing from token to token.

    Args:
        expression_trajectory: Trait expression over time [seq_len, ...] or [batch, seq_len, ...]
        dt: Time step (default 1.0 for per-token)
        normalize: Whether to normalize the derivative vectors

    Returns:
        First derivative (velocity) with shape [seq_len-1, ...]

    Example:
        >>> # Track how refusal trait changes over 10 tokens
        >>> trajectory = torch.randn(10, 768)  # 10 tokens, 768 dims
        >>> velocity = compute_derivative(trajectory)
        >>> velocity.shape
        torch.Size([9, 768])  # 9 transitions between 10 tokens
    """
    if expression_trajectory.shape[0] < 2:
        raise ValueError(f"Need at least 2 time points for derivative, got {expression_trajectory.shape[0]}")

    # Compute differences between consecutive time points
    derivative = torch.diff(expression_trajectory, dim=0) / dt

    if normalize:
        # Normalize each derivative vector
        norms = derivative.norm(dim=-1, keepdim=True)
        derivative = derivative / (norms + 1e-8)  # Add epsilon to prevent division by zero

    return derivative


def compute_second_derivative(
    expression_trajectory: torch.Tensor,
    dt: float = 1.0,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute the second derivative of trait expression over time.

    This represents the "acceleration" of trait expression - how fast the rate
    of change itself is changing. Useful for detecting commitment points.

    Args:
        expression_trajectory: Trait expression over time [seq_len, ...] or [batch, seq_len, ...]
        dt: Time step (default 1.0 for per-token)
        normalize: Whether to normalize the derivative vectors

    Returns:
        Second derivative (acceleration) with shape [seq_len-2, ...]

    Example:
        >>> trajectory = torch.randn(10, 768)
        >>> acceleration = compute_second_derivative(trajectory)
        >>> acceleration.shape
        torch.Size([8, 768])  # 8 acceleration values from 10 tokens
        >>>
        >>> # Find commitment point (where acceleration drops)
        >>> accel_magnitude = acceleration.norm(dim=-1)
        >>> commitment_point = (accel_magnitude < 0.1).nonzero()[0]
    """
    # First derivative (velocity)
    velocity = compute_derivative(expression_trajectory, dt, normalize=False)

    # Second derivative (acceleration)
    acceleration = compute_derivative(velocity, dt, normalize=False)

    if normalize:
        norms = acceleration.norm(dim=-1, keepdim=True)
        acceleration = acceleration / (norms + 1e-8)

    return acceleration


def projection(
    activations: torch.Tensor,
    vector: torch.Tensor,
    normalize_vector: bool = True
) -> torch.Tensor:
    """
    Project activations onto a trait vector.

    This measures how strongly the trait is expressed in the activations.

    Args:
        activations: Activations to project [*, hidden_dim]
        vector: Trait vector to project onto [hidden_dim]
        normalize_vector: Whether to normalize the vector before projection

    Returns:
        Projection scores with shape [*] (all dimensions except last)

    Example:
        >>> activations = torch.randn(10, 50, 768)  # 10 examples, 50 tokens, 768 dims
        >>> trait_vector = torch.randn(768)
        >>> scores = projection(activations, trait_vector)
        >>> scores.shape
        torch.Size([10, 50])  # Trait expression at each token
    """
    if normalize_vector:
        vector = vector / (vector.norm() + 1e-8)

    # Handle batched matrix multiplication
    if activations.ndim == 1:
        return activations @ vector
    else:
        # Use ... to handle arbitrary leading dimensions
        return torch.matmul(activations, vector)


def cosine_similarity(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.

    Useful for comparing trait vectors or measuring confounding.

    Args:
        vec1: First vector(s)
        vec2: Second vector(s)
        dim: Dimension along which to compute similarity

    Returns:
        Cosine similarity score(s) between -1 and 1

    Example:
        >>> refusal_vec = torch.randn(768)
        >>> evil_vec = torch.randn(768)
        >>> similarity = cosine_similarity(refusal_vec, evil_vec)
        >>> print(f"Vectors are {similarity.item():.2%} similar")
    """
    # Normalize vectors
    vec1_norm = vec1 / (vec1.norm(dim=dim, keepdim=True) + 1e-8)
    vec2_norm = vec2 / (vec2.norm(dim=dim, keepdim=True) + 1e-8)

    # Compute dot product
    return (vec1_norm * vec2_norm).sum(dim=dim)


def normalize_vectors(
    vectors: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Normalize vectors to unit length.

    Args:
        vectors: Vectors to normalize
        dim: Dimension along which to normalize

    Returns:
        Normalized vectors with same shape as input

    Example:
        >>> vectors = torch.randn(10, 768)
        >>> normalized = normalize_vectors(vectors)
        >>> normalized.norm(dim=-1)  # All should be ~1.0
        tensor([1., 1., 1., ...])
    """
    norms = vectors.norm(dim=dim, keepdim=True)
    return vectors / (norms + 1e-8)


def magnitude(
    vectors: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute L2 norm (magnitude) of vectors.

    Args:
        vectors: Input vectors
        dim: Dimension along which to compute norm

    Returns:
        Magnitude of vectors (dimension `dim` is removed)

    Example:
        >>> hidden_states = torch.randn(26, 50, 2304)  # [layers, tokens, hidden]
        >>> mags = magnitude(hidden_states)
        >>> mags.shape
        torch.Size([26, 50])  # Magnitude at each layer×token
    """
    return vectors.norm(dim=dim)


def radial_velocity(
    trajectory: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """
    Compute radial velocity (magnitude change) between consecutive points.

    Radial velocity measures how much the activation magnitude changes,
    independent of direction. Positive = growing, negative = shrinking.

    Args:
        trajectory: Activation trajectory [n_points, ..., hidden_dim]
        dim: Dimension along which to compute velocity (default: 0)

    Returns:
        Radial velocity with shape [n_points-1, ...]

    Example:
        >>> hidden_states = torch.randn(26, 50, 2304)  # [layers, tokens, hidden]
        >>> radial_vel = radial_velocity(hidden_states)
        >>> radial_vel.shape
        torch.Size([25, 50])  # Magnitude change per layer transition
    """
    mags = magnitude(trajectory, dim=-1)
    return torch.diff(mags, dim=dim)


def angular_velocity(
    trajectory: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """
    Compute angular velocity (direction change) between consecutive points.

    Angular velocity measures how much the activation direction changes,
    independent of magnitude. 0 = same direction, 2 = opposite direction.

    Args:
        trajectory: Activation trajectory [n_points, ..., hidden_dim]
        dim: Dimension along which to compute velocity (default: 0)

    Returns:
        Angular velocity (1 - cosine_similarity) with shape [n_points-1, ...]

    Example:
        >>> hidden_states = torch.randn(26, 50, 2304)  # [layers, tokens, hidden]
        >>> angular_vel = angular_velocity(hidden_states)
        >>> angular_vel.shape
        torch.Size([25, 50])  # Direction change per layer transition
    """
    # Normalize to unit vectors
    normed = normalize_vectors(trajectory, dim=-1)

    # Get consecutive pairs
    if dim == 0:
        v1 = normed[:-1]
        v2 = normed[1:]
    else:
        v1 = torch.narrow(normed, dim, 0, normed.shape[dim] - 1)
        v2 = torch.narrow(normed, dim, 1, normed.shape[dim] - 1)

    # Cosine similarity between consecutive vectors
    cos_sim = (v1 * v2).sum(dim=-1)

    # Angular velocity = 1 - cos_sim (0 = same direction, 2 = opposite)
    return 1.0 - cos_sim


def pca_reduce(
    activations: torch.Tensor,
    n_components: int = 2,
    return_pca: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Reduce activations to lower dimensions using PCA.

    Args:
        activations: Activations to reduce [..., hidden_dim]
        n_components: Number of dimensions to reduce to (default: 2)
        return_pca: If True, also return (mean, components) for reuse

    Returns:
        Reduced activations [..., n_components]
        If return_pca=True: (reduced, (mean, components))

    Example:
        >>> hidden_states = torch.randn(26, 50, 2304)  # [layers, tokens, hidden]
        >>> reduced = pca_reduce(hidden_states, n_components=2)
        >>> reduced.shape
        torch.Size([26, 50, 2])  # 2D trajectory per token

        >>> # Reuse PCA for new data
        >>> reduced, (mean, components) = pca_reduce(train_data, return_pca=True)
        >>> new_reduced = (new_data - mean) @ components.T
    """
    original_shape = activations.shape
    hidden_dim = original_shape[-1]

    # Flatten to [n_samples, hidden_dim]
    flat = activations.reshape(-1, hidden_dim).float()

    # Center the data
    mean = flat.mean(dim=0)
    centered = flat - mean

    # Compute covariance matrix
    cov = (centered.T @ centered) / (centered.shape[0] - 1)

    # Eigendecomposition (use eigh for symmetric matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort by eigenvalue (descending) and take top n_components
    idx = torch.argsort(eigenvalues, descending=True)[:n_components]
    components = eigenvectors[:, idx]  # [hidden_dim, n_components]

    # Project
    reduced_flat = centered @ components  # [n_samples, n_components]

    # Reshape back
    new_shape = original_shape[:-1] + (n_components,)
    reduced = reduced_flat.reshape(new_shape)

    if return_pca:
        return reduced, (mean, components)
    return reduced


def attention_entropy(
    attention_weights: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute entropy of attention distribution.

    Higher entropy = more diffuse attention (looking at many tokens).
    Lower entropy = more focused attention (looking at few tokens).

    Args:
        attention_weights: Attention weights [..., n_tokens] (should sum to 1)
        dim: Dimension along which to compute entropy (default: -1)

    Returns:
        Entropy values with `dim` removed

    Example:
        >>> attn = torch.softmax(torch.randn(8, 50, 50), dim=-1)  # [heads, query, key]
        >>> entropy = attention_entropy(attn)
        >>> entropy.shape
        torch.Size([8, 50])  # Entropy per head per query position

        >>> # Focused attention (low entropy)
        >>> focused = torch.zeros(10); focused[0] = 1.0
        >>> attention_entropy(focused)
        tensor(0.)

        >>> # Diffuse attention (high entropy)
        >>> diffuse = torch.ones(10) / 10
        >>> attention_entropy(diffuse)
        tensor(2.3026)  # ln(10)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    weights = attention_weights.clamp(min=eps)

    # Entropy = -sum(p * log(p))
    return -(weights * weights.log()).sum(dim=dim)