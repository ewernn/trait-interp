# Temporary Vector Analysis

This directory contains research findings from cross-distribution generalization experiments.

## Files

1. **CROSS_DISTRIBUTION_GENERALIZATION_STUDY.md** (20 KB, 637 lines)
   - Complete methodology and findings
   - Executive summary of key results
   - Detailed analysis of all 4 extraction methods
   - Implications for research and practical recommendations
   - Full numerical results appendix

2. **cross_dist_results.txt** (2.4 KB)
   - Raw output from the cross-distribution test script
   - Terminal output with all metrics

3. **cross_dist_final_test.py** (5.1 KB)
   - Complete test implementation
   - Loads instruction-induced training data
   - Loads natural test data
   - Extracts vectors with all 4 methods
   - Evaluates cross-distribution performance

## Key Finding

**ICA at layer 16 achieves 81.8% cross-distribution accuracy**, dramatically outperforming:
- Probe: 51.4%
- Gradient: 57.5%
- Mean Diff: 48.6%

This validates the hypothesis that ICA's unsupervised disentanglement learns genuine traits that generalize across distributions, while supervised methods (Probe, Gradient) overfit to instruction-following patterns.

## Trait Tested

**uncertainty_calibration**
- Train: Instruction-induced ("BE UNCERTAIN..." vs "BE CONFIDENT...")
- Test: Natural (questions that elicit genuine uncertainty vs confidence)
- 190 training examples, 181 test examples
- Completely different distributions

## Next Steps

See "Future Experiments" section in the main study document for:
- Multi-trait validation
- Steering effectiveness tests
- Optimal layer search
- Component selection strategies
- Natural validation pipeline for all 16 traits

## Date

November 17, 2025
