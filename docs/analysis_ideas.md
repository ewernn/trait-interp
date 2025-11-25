# Analysis Ideas & Experiments

Brainstorm list of analysis methods to explore, validate, or implement. Mix of completed experiments, in-progress work, and future ideas.

## Completed (in experiments/gemma_2b_cognitive_nov21/analysis/)

### Temporal Dynamics
- **Derivative overlay** - Position/velocity/acceleration per trait across tokens
- **Normalized velocity** - Rate of trait change normalized across prompts
- **Radial/angular dynamics** - Polar coordinate representation of trait evolution
- **Trait emergence** - Which layers traits crystallize in
- **Trait dynamics correlation** - How velocity patterns correlate between traits

### Per-Token Analysis
- **Per-token metrics** - Comprehensive token-level statistics across prompt sets
- **Attention dynamics** - Attention pattern evolution during generation

### Aggregated Views
- **Trait projections** - Activation scores across all prompts
- **Analysis gallery** - Batch generation of visualizations with index

## In Progress / Experimental

### Vector Quality & Selection
- Cross-distribution validation (extraction_evaluation.py implemented)
- Polarity validation (implemented)
- Separation metrics (implemented)
- Bootstrap stability analysis (not implemented)
- Method comparison matrices (partial)

### Steering & Intervention
- Magnitude sweep (test vector at different strengths: 0.5, 1.0, 2.0, 5.0, 10.0)
- Steering precision (measure off-target trait changes)
- Steering robustness (test across different prompt domains)
- Saturation analysis (when does vector stop being effective?)
- Interaction effects (multi-vector steering combinations)

### Generalization & Robustness
- Few-shot adaptation (learning curves: 10, 50, 100, 500 examples)
- Adversarial robustness (paraphrases, negations, minimal edits)
- Noise robustness (Gaussian noise at varying levels)
- Prompt sensitivity (length, complexity, domain, ambiguity)
- Transfer to other models (Gemma 2B → 7B, other architectures)

### Interpretability
- Top activating examples (manual inspection of high-projection cases)
- Intervention causality (ablation studies)
- Vector arithmetic (formal_vec + positive_vec, uncertain_vec - negative_vec)
- Consistency across layers (adjacent layer similarity)
- Orthogonality analysis (correlation matrices between traits)
- SAE feature alignment (which SAE features correlate with trait vectors?)
- Attention head specialization (which heads track specific traits?)
- Logit lens clarity (does vector predict specific tokens?)

### Statistical Analysis
- Discriminative power (ROC/AUC analysis)
- Human agreement correlation (vector scores vs human ratings)
- Comparative benchmarks (method comparison on specific traits)

### Mechanistic Understanding
- Layer-by-layer decomposition (attention vs MLP contributions)
- Per-head attribution (which attention heads matter?)
- Residual stream flow (how traits propagate across layers)
- Hierarchical composition (do traits build on each other?)
- Commitment point detection (when does model "lock in"?)
- Cross-trait interference (do traits block/enhance each other?)

## Future Ideas

### Advanced Dynamics
- Phase space analysis (trait velocity vs position)
- Attractor basin identification (stable trait configurations)
- Trajectory clustering (group prompts by similar dynamics)
- Prediction of next-token trait from current state

### Production Readiness
- Inference latency benchmarks
- Memory footprint optimization
- Batch processing throughput
- Compressed vector representations

### Cross-Model Studies
- Transfer learning (vector portability across model sizes)
- Architecture comparison (Gemma vs Llama vs Qwen)
- Training stage analysis (how vectors change during fine-tuning)

### Novel Extraction Methods
- Contrastive learning approaches
- Meta-learning for trait discovery
- Unsupervised trait detection
- Temporal-aware extraction (use token sequences, not just averages)

## Adding New Analysis

When you implement something:
1. Move from "In Progress" to "Completed"
2. Note where the code lives (experiments/analysis/ or analysis/)
3. If vetted and stable, document in main docs (vector_evaluation_framework.md, etc.)
4. Update visualization if it produces output for the gallery
