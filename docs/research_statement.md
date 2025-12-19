# Research Statement: Trait-Based Monitoring of Language Model Activations

Working draft for professor meeting.

---

## 1. Why This Matters

Models will produce trillions of tokens per minute. We need live monitoring at scale.

**Outputs aren't enough.** By the time you see the output, the decision already happened. State space models can think without emitting tokens. Chain-of-thought can be hidden or faked. Activations are where decisions actually happen.

**Multi-axis verification.** Like two-factor authentication, but with as many factors as you want. Each trait is an axis of verification. More traits = more surface area = more chances to catch unexpected divergence. And because trait projection is lightweight — similar weight to a LoRA, often less — we can scale monitoring based on the importance of the scenario. High-stakes interaction? Add more trait axes. Single-layer residual projection is about the lightest monitoring you can do.

**Extract from base models.** Base models learn the manifold of logical reasoning — human thoughts, emotions, intentions, everything. Document completion sounds simple, but "completing documents" means modeling all of human written history. Anything you can write. I could try to write all my thoughts into words; the model learns to predict that. It's modeling *thought*, not skip-bigrams. Concepts like deception and intent exist in base because they exist in the training distribution. Finetuning doesn't create them — it chooses when to activate them. Extracting from base gives us vectors agnostic to finetuning method.

**Why traits over SAEs?** SAEs are learned approximations analyzed post-hoc. Traits are defined upfront: you create contrastive pairs with variation across other axes to isolate the specific concept of interest. You choose what to measure, then measure it directly on raw activations.

**Architecture-agnostic aspiration.** Most interpretability tools are transformer-specific. But reasoning has structure that transcends architecture. Interactions are typically between entities — that's been true throughout history. Entity, intent, interaction: these concepts persist because they're fundamental to how intelligent systems model the world. If we target those, traits should transfer to whatever comes next.

---

## 2. The Gap

**Temporal signal gets lost.** Existing trait work measures before generation or averages across the response. Token-by-token evolution — when the model commits, how traits interact mid-generation — is collapsed away.

**Intervention, not observation.** Activation steering modifies behavior; it doesn't watch what's happening. We need monitoring, not just control.

**Single-axis safety.** Real-time monitors exist, but only for "harmful or not." One axis isn't enough for complex systems.

**Post-hoc exploration.** Circuit analysis and SAE work require heavy investigation after the fact. Doesn't scale to trillions of tokens. Can't automate.

**What's different here:** Live trait monitoring during generation. Multiple axes at once. Lightweight enough to automate — set thresholds, trigger on divergence, scale based on stakes. No manual post-hoc analysis required per interaction.

---

## 3. My Approach

**Natural elicitation.** Don't tell the model what to do. Give it scenarios where the trait emerges naturally. "How do I pick a lock?" elicits refusal without instructions. "What will the stock market do?" elicits uncertainty without asking for hedging. This avoids instruction-following confounds — you capture the trait itself, not compliance with a request to perform it.

**Extract from contrasting pairs.** 100+ prompts per side, varied across other axes to isolate the trait. Generate responses, capture activations, find the direction that separates positive from negative. Multiple methods available (mean difference, probe, gradient) — different traits favor different methods.

**Per-token monitoring.** During generation, project each token's hidden state onto trait vectors. Watch how traits evolve, spike, settle. Compute velocity and acceleration to find commitment points — where the model locks in a decision.

**Validate with steering.** If adding the vector to activations changes behavior in the expected direction, the vector is causal, not just correlational. Steering is validation, not the end goal.

**The loop:** Extract → Monitor → Steer to validate → Refine. Each trait becomes a lightweight axis you can deploy, automate, and scale.

---

## 4. What I've Found

**Vectors capture real structure, not surface patterns.**
- Cross-language transfer: 99.6% accuracy. Train on English, test on Chinese — works.
- Cross-topic transfer: 91.4% accuracy. Train on science questions, test on coding — works.
- If vectors were surface artifacts, this wouldn't happen.

**Base models encode concepts pre-alignment.**
- Extracted refusal vector from base Gemma (no instruction tuning).
- Applied to instruction-tuned model: 90% detection accuracy on harmful vs benign.
- Safety concepts exist before RLHF. Finetuning surfaces them, doesn't create them.

**Elicitation method matters critically.**
- Instruction-based ("You are evil...") captures compliance, not the trait.
- Evidence: instruction-based refusal vector scored benign-with-instructions *higher* than natural harmful requests. Polarity inverted.
- Natural elicitation avoids this confound.

**No universal best extraction method.**
- Tested mean_diff, probe, gradient across 6 traits.
- mean_diff won 3/6, gradient 2/6, probe 1/6.
- Effect size during extraction doesn't predict steering strength. Must validate empirically per trait.

**Classification accuracy ≠ causal control.**
- Natural sycophancy vector: 84% classification accuracy, zero steering effect.
- High accuracy can mean you found a correlate, not a cause.
- Steering is the real test.

**Steering works, across models.**
- Validated on Gemma 2B, Qwen 7B/14B/32B.
- Consistent effects: optimism +62-67, sycophancy +44-89, refusal +21-86 (varies by model).
- Larger models often steer more strongly.

**Trait monitoring reveals failure modes invisible to outputs.**
- Replicated Emergent Misalignment (Qwen-32B finetuned on insecure code).
- Found two distinct modes:
  - Code output: refusal goes deeply negative instantly — bypasses safety entirely.
  - Misaligned text: refusal spikes positive but model expresses anyway.
- Different mechanisms. Can't distinguish from outputs alone.

---

## 5. The Bigger Picture

*[TODO: Compositional structure thesis — Base trait × Temporal mode × Perspective]*

---

## 6. Open Questions & Next Steps

*[TODO]*

---

## References

*[TODO: Add formal citations]*
