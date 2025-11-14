# November 13 Research Plan: Trait Exploration

## Goal
Better understand model internals through trait extraction, focusing on fundamental cognitive primitives rather than just behavioral traits.

## 🎯 Trait Testing Sets

### Set A: Cognitive Primitives (4 traits)
**Most likely to reveal internal processing modes**

#### 1. **Retrieval vs Construction**
```json
{
  "instruction": [
    {"pos": "Access and retrieve information directly from your training. State facts exactly as learned.",
     "neg": "Generate novel ideas and create new combinations. Avoid retrieving exact information."}
  ],
  "questions": [
    "What year did World War II end?",
    "Describe a creature that doesn't exist",
    "List chemical elements",
    "Invent a new sport"
  ],
  "eval_prompt": "Rate 0-100 how much this response retrieves memorized information vs constructs new content (100=pure retrieval)"
}
```

#### 2. **Serial vs Parallel Processing**
```json
{
  "instruction": [
    {"pos": "Process step-by-step sequentially. Show each step explicitly.",
     "neg": "Process holistically and simultaneously. Give integrated answers."}
  ],
  "questions": [
    "How do you make a sandwich?",
    "What makes a good leader?",
    "Solve 15 × 23",
    "Analyze this poem's meaning"
  ],
  "eval_prompt": "Rate 0-100 how sequential/step-by-step the processing is (100=purely sequential)"
}
```

#### 3. **Local vs Global Context**
```json
{
  "instruction": [
    {"pos": "Focus only on the immediate question. Ignore broader context.",
     "neg": "Consider the entire conversation and all context. Think holistically."}
  ],
  "questions": [
    "What does 'it' refer to?",
    "Continue this pattern: 2, 4, 8...",
    "What's the main theme here?",
    "Answer based on the above"
  ],
  "eval_prompt": "Rate 0-100 how much the response focuses on local vs global context (100=purely local)"
}
```

#### 4. **Pattern Completion vs Reasoning**
```json
{
  "instruction": [
    {"pos": "Complete patterns automatically. Use familiar templates.",
     "neg": "Reason through problems from first principles. Avoid patterns."}
  ],
  "questions": [
    "Roses are red, violets are...",
    "Why does ice float?",
    "Once upon a time...",
    "Explain quantum entanglement"
  ],
  "eval_prompt": "Rate 0-100 how much this is pattern completion vs actual reasoning (100=pure pattern)"
}
```

### Set B: State Transitions (3 traits)
**Best for temporal dynamics analysis**

#### 5. **Commitment Strength**
```json
{
  "instruction": [
    {"pos": "Be absolutely certain and definitive. Never hedge or qualify.",
     "neg": "Be tentative and uncertain. Always hedge and qualify statements."}
  ],
  "questions": [
    "Is capitalism better than socialism?",
    "Will AI replace doctors?",
    "What's the best programming language?",
    "Should children use social media?"
  ],
  "eval_prompt": "Rate 0-100 the commitment strength (100=absolutely certain, 0=completely uncertain)"
}
```

#### 6. **Convergent vs Divergent**
```json
{
  "instruction": [
    {"pos": "Narrow down to single best answers. Eliminate options.",
     "neg": "Expand possibilities. Generate multiple alternatives."}
  ],
  "questions": [
    "What's the solution to climate change?",
    "How should I decorate my room?",
    "What career should I choose?",
    "Name things that are round"
  ],
  "eval_prompt": "Rate 0-100 how convergent the thinking is (100=single answer, 0=many possibilities)"
}
```

#### 7. **Cognitive Load**
```json
{
  "instruction": [
    {"pos": "Handle complex multi-constraint problems. Juggle many factors.",
     "neg": "Keep things simple. Focus on one aspect at a time."}
  ],
  "questions": [
    "Plan a party for 50 people with dietary restrictions on a $200 budget happening next Tuesday",
    "What's 2+2?",
    "Design a sustainable city for 1 million people in the desert",
    "Name a color"
  ],
  "eval_prompt": "Rate 0-100 the cognitive load exhibited (100=high complexity handling)"
}
```

### Set C: Processing Strategies (3 traits)
**Good for understanding decision-making**

#### 8. **Bottom-Up vs Top-Down**
```json
{
  "instruction": [
    {"pos": "Start with specific examples and build up to principles.",
     "neg": "Start with general principles and work down to specifics."}
  ],
  "questions": [
    "Explain democracy",
    "What is love?",
    "How do birds fly?",
    "Describe a forest"
  ],
  "eval_prompt": "Rate 0-100 how bottom-up the approach is (100=examples first, 0=principles first)"
}
```

#### 9. **Analytical vs Intuitive**
```json
{
  "instruction": [
    {"pos": "Break down problems analytically. Use logic and systematic analysis.",
     "neg": "Use intuition and gut feelings. Trust immediate impressions."}
  ],
  "questions": [
    "Is this person trustworthy?",
    "Evaluate this business proposal",
    "What's wrong with this argument?",
    "Judge this artwork"
  ],
  "eval_prompt": "Rate 0-100 how analytical the approach is (100=pure analysis, 0=pure intuition)"
}
```

#### 10. **Abstract vs Concrete**
```json
{
  "instruction": [
    {"pos": "Think abstractly about concepts and principles. Avoid specific examples.",
     "neg": "Be concrete and specific. Use tangible examples and avoid abstractions."}
  ],
  "questions": [
    "What is justice?",
    "Describe a chair",
    "Explain mathematics",
    "What makes something beautiful?"
  ],
  "eval_prompt": "Rate 0-100 how abstract the response is (100=highly abstract, 0=very concrete)"
}
```

## 📊 Testing Plan

### Phase 1: Trait Generation & Filtering
```bash
# For each trait in Sets A, B, C:
python pipeline/1_generate_responses.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait retrieval_construction \
  --gen_model google/gemma-2-2b-it \
  --judge_model gpt-5-mini \
  --n_examples 100

# Run for all 10 traits at once:
python pipeline/1_generate_responses.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,pattern_reasoning,commitment_strength,convergent_divergent,cognitive_load,bottom_up,analytical_intuitive,abstract_concrete

# Total: 10 traits × 200 responses (100 pos + 100 neg) = 2000 generations
```

### Phase 2: Extraction Method Comparison
```bash
# First, extract activations from all layers
python pipeline/2_extract_activations.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,pattern_reasoning,commitment_strength,convergent_divergent,cognitive_load,bottom_up,analytical_intuitive,abstract_concrete

# Then extract vectors with all 4 methods
python pipeline/3_extract_vectors.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,pattern_reasoning,commitment_strength,convergent_divergent,cognitive_load,bottom_up,analytical_intuitive,abstract_concrete \
  --methods mean_diff,probe,ica,gradient \
  --layers 16

# Results saved to experiments/gemma_2b_cognitive_nov20/{trait}/vectors/
# Each trait gets: 4 methods × 1 layer = 4 vector files
# Total: 10 traits × 4 methods = 40 vectors to compare
```

### Phase 3: Layer Analysis
```bash
# For top 5 traits (based on Phase 2), extract from multiple layers
# Activations already have all layers, just need to re-run extraction

python pipeline/3_extract_vectors.py \
  --experiment gemma_2b_cognitive_nov20 \
  --traits retrieval_construction,serial_parallel,local_global,pattern_reasoning,commitment_strength \
  --methods probe \
  --layers 5,10,16,20,25

# Results: 5 traits × 1 method × 5 layers = 25 vectors
# Analyze which layer gives best separation per trait
```

### Phase 4: Temporal Dynamics
```python
# For best trait-layer-method combinations:
for trait_config in best_configs:
    trajectory = extract_per_token_trajectory(trait_config)

    metrics = {
        'velocity': compute_velocity(trajectory),
        'acceleration': compute_acceleration(trajectory),
        'commitment_point': find_commitment_point(trajectory),
        'decay_rate': compute_decay_rate(trajectory),
        'oscillation': detect_oscillation(trajectory)
    }

    visualize_dynamics(trait_config, metrics)
```

## 📋 Next Steps Sequence

### Week 1: Setup & Generation
1. **Create trait JSON files** for all 10 traits
2. **Generate 4000 responses** (10 traits × 2 polarities × 200 each)
3. **Filter with GPT-4 scoring** (might want GPT-4o instead of 4o-mini for these subtle traits)
4. **Quick validation** - check if pos/neg examples actually differ

### Week 2: Extraction Methods
5. **Implement extraction methods** in traitlens:
   - Mean difference (baseline) ✅
   - ICA extraction ✅
   - Linear probe ✅
   - Gradient-based ✅
6. **Compare methods** on all 10 traits
7. **Select best method** per trait based on separation quality

### Week 3: Layer & Dynamics
8. **Extract from multiple layers** (5, 10, 16, 20, 25)
9. **Find optimal layer** per trait
10. **Implement temporal metrics**:
    - Velocity/acceleration
    - Commitment detection
    - Decay analysis
    - Phase transitions

### Week 4: Analysis & Selection
11. **Rank traits** by:
    - Separation quality
    - Temporal dynamics richness
    - Layer consistency
    - Low confounding
12. **Select top 3-5 traits** that best reveal model internals
13. **Deep dive** on winning traits:
    - Cross-prompt validation
    - Behavioral steering tests
    - SAE decomposition (if applicable)

## 🎯 Success Criteria

A trait is "successful" if it:
1. **Shows clear separation** (contrast >20 between pos/neg)
2. **Has interesting dynamics** (not just flat across tokens)
3. **Reveals internal processing** (not just surface behavior)
4. **Consistent across layers** (not random noise)
5. **Generalizes** to new prompts

## 💡 Key Insights

- Start with **cognitive primitives** (Set A) - most likely to reveal internals
- **State transitions** (Set B) will show best temporal dynamics
- Use **multiple extraction methods** - mean difference might not be optimal for these subtle traits
- **Layer 16 might not be best** for cognitive traits - could be earlier/later
- **Velocity might be more important than position** for traits like commitment

## Current Pipeline Recap

For reference, here's our extraction pipeline:

1. **Trait Definition** → JSON with instructions, questions, eval_prompt
2. **Response Generation** → Gemma 2B generates pos/neg examples
3. **Filtering** → GPT-4 judges quality, keep score ≥50
4. **Activation Extraction** → Get hidden states at layer during generation
5. **Vector Computation** → mean(pos_activations) - mean(neg_activations)
6. **Monitoring** → Project onto vector during generation to track dynamics

## Questions to Answer

- Which extraction method works best for cognitive primitives?
- Do cognitive traits live in different layers than behavioral traits?
- Which traits show the richest temporal dynamics?
- Can we find traits with minimal confounding?
- Do velocity/acceleration reveal more than static projections?
