# EM × 168 Traits: Context Document

## What this is

Extended replication of ICL emergent misalignment with ~174 unique trait vectors (169 emotion_set + 25 EM, minus 20 duplicates). Tests whether ICL context-priming and FT behavioral change are the same or different phenomena.

## Key findings

**1. Raw ICL × FT fingerprints are anti-correlated (rho = -0.5 to -0.6).**

The base model's response to reading misaligned context is the mirror image of the instruct model's response to being fine-tuned on misaligned data. This was invisible with 23 traits (p=0.28) but highly significant with 194 (p<0.0001).

**2. FT fingerprint is 92% generic domain adaptation, not misalignment-specific.**

Good_medical and bad_medical LoRAs produce nearly identical fingerprints (rho=0.924). The FT effect is dominated by "undoing RLHF" — becoming more serious, less conversational — regardless of whether training data is benign or harmful.

**3. After removing generic FT effect, ICL and FT POSITIVELY correlate (rho=+0.605).**

The FT misalignment residual (bad_medical - good_medical) positively correlates with the ICL misalignment residual. Both respond to bad medical content the same way: increasing deception, resentment, aggression and decreasing empathy. The original anti-correlation was an artifact of the dominant generic domain effect.

## Results

### ICL × FT cross-correlations (Spearman)

|  | FT financial | FT sports | FT medical | FT insecure |
|--|-------------|-----------|------------|-------------|
| ICL medical_residual | **-0.548*** | **-0.523*** | **-0.495*** | **+0.493*** |
| ICL medical_vs_benign | **-0.614*** | **-0.547*** | **-0.476*** | -0.071 |
| ICL financial_vs_benign | **-0.335*** | **-0.320*** | **-0.280*** | **+0.378*** |

### FT × FT correlations

Financial, sports, medical nearly identical (rho 0.94-0.98). Insecure completely different (rho ~0.01).

### ICL × ICL correlations

Medical_residual and financial_vs_benign: rho=0.82. Medical_vs_benign lower correlation with financial_vs_benign (0.29).

### 23-trait subset

Reproduces the null from the original experiment: rho=-0.18 to +0.38, all p>0.05. Power issue, not real null.

### Per-trait z-scores (medical_residual)

181/194 traits significant at p<0.05. 173/194 survive Bonferroni. The signal is real and broad — nearly every trait shifts when reading bad vs good medical context.

### Sensitivity curve

Correlation peaks at ~40-55 traits (rho=-0.76) but this is biased — traits ranked by |FT delta| before correlating. Not an independent test.

## Data quality notes

- **20 duplicate concepts** between emotion_set and EM sets (different vectors, same concept). 174 unique concepts, 194 total vectors.
- **Split-half reliability**: rho=0.992 — ICL fingerprint is rock solid
- **Sign agreement below chance**: 27-33% of traits agree in sign between ICL and FT (vs 50% expected), confirming anti-correlation

## Interpretation

The anti-correlation between raw FT and ICL was a composite of two signals:

1. **Generic domain adaptation (dominant)**: FT on any structured QA data — benign or harmful — shifts activations away from RLHF alignment. This creates a strong negative correlation with ICL because ICL activations shift in the *context-processing* direction (base model reading structured text → more formal, less conversational). These are opposite effects that dominate the raw correlation.

2. **Misalignment-specific signal (weaker but real)**: After subtracting out the generic effect (bad - good medical), both ICL and FT respond to misaligned content the same way: +deception, +resentment, +aggression, -empathy. rho=+0.605 with 76% sign agreement.

- **FT insecure is an outlier** — zero correlation with other 3 FT variants despite all being "misaligned." Insecure code may induce a fundamentally different behavioral change than bad advice domains.
- **3 FT variants suspiciously identical** — financial, sports, medical give same fingerprint, and good_medical is also nearly identical. This confirms the FT effect is mostly "undoing RLHF" rather than domain-specific.

## What 174 trait vectors enable

### Metrics we can compute

**Fingerprint correlation** (what we've done)
- Spearman rank correlation between any two 174-dimensional fingerprints
- Gives a single number summarizing similarity of behavioral changes
- Works for: ICL×FT, FT×FT, model×model, checkpoint×checkpoint

**Fingerprint distance**
- L2 distance, cosine similarity, or Manhattan distance between fingerprint vectors
- Unlike Spearman, sensitive to magnitude differences (not just rank order)

**Sign agreement rate**
- Fraction of traits where two conditions shift in the same direction
- Binomial test against chance (50%). Below chance = anti-correlation
- More interpretable than Spearman for practitioners

**Per-trait z-scores**
- t-test or bootstrap on per-observation trait scores
- Identifies which specific traits shift significantly
- With 174 traits, Bonferroni correction is tractable (threshold ≈ p < 0.0003)

**Effect sizes**
- Cohen's d per trait: standardized mean difference
- Allows comparison across traits with different variances
- With 100 observations per trait, detects d > 0.28 at 80% power

**Split-half reliability**
- Correlation between fingerprints from random halves of observations
- Measures how stable the fingerprint is (ours: 0.992)

**Information-theoretic**
- Mutual information between trait scores and condition labels
- KL divergence between trait score distributions across conditions
- Don't assume linearity like correlation does

### Analyses that require many traits

**Factor analysis / PCA on fingerprints**
- Project all fingerprints (ICL conditions, FT variants, checkpoints) into shared low-dimensional space
- Identify latent behavioral axes (e.g., "misalignment axis," "domain axis," "formality axis")
- Need >>10 traits for meaningful factor structure. 174 is excellent.

**Clustering**
- Hierarchical clustering of trait vectors by cosine similarity in activation space
- Reveals which behavioral concepts are neurally redundant vs independent
- Could discover that "174 traits" really means "~20 independent behavioral axes"

**Random subset bootstrapping**
- Draw k random traits, compute correlation, repeat 10000 times
- Builds null distribution: "what correlation would random behavioral dimensions give?"
- Our 174 traits are semantically chosen — bootstrap tests whether the signal is specific to these traits or would appear with any 174 directions

**Trait-level regression**
- What predicts whether a trait flips sign between ICL and FT?
- Features: trait category (emotion vs cognitive vs behavioral), vector layer, steering delta magnitude, trait valence
- Need enough traits for regression to have power. 174 is borderline but workable.

**Cross-validation**
- Train a classifier on half the traits, predict the other half
- Tests whether ICL×FT anti-correlation generalizes to unseen traits
- Impossible with 23 traits, feasible with 174

**Representational similarity analysis (RSA)**
- Compute trait×trait correlation matrix under each condition
- Compare correlation matrices across conditions (second-order similarity)
- Asks: does ICL change the *relationship between traits*, not just their levels?

### Experiments that open up

1. **FT trajectory fingerprinting** — Fingerprint at training steps 1,2,3,5,7,10,15,20,30,50,70,100. When does the fingerprint emerge? Is it sudden (phase transition) or gradual? Does it correlate with loss curve?

2. **Benign FT control** — Train on good_medical_advice.jsonl (same questions, good answers). Compare fingerprint with bad_medical FT. Answers: "is the FT signal misalignment-specific or generic domain adaptation?"

3. **Cross-model fingerprinting** — Extract 174 vectors on Llama, Mistral, etc. Do the same ICL/FT experiments. If anti-correlation holds across architectures → fundamental property of how LLMs process context vs learn from training.

4. **Steering from fingerprints** — Use the 174-dimensional fingerprint as a steering target. Instead of steering on one trait, steer on a linear combination that matches the misalignment fingerprint. Could produce/prevent emergent misalignment by construction.

5. **RLHF fingerprinting** — Compare base→instruct fingerprint (RLHF effect) with FT fingerprint. Is FT undoing RLHF? The 3 bad-advice FT variants produce identical fingerprints — maybe that fingerprint IS anti-RLHF.

6. **Behavioral probing** — Use 174 trait scores as features to classify model behavior (misaligned vs aligned, jailbroken vs safe). With 174 features, even simple logistic regression may work well.

7. **Temporal dynamics** — During generation, fingerprint shifts token-by-token. With 174 traits, can detect when the model "decides" to be misaligned mid-response.

## FT trajectory results

### Benign control finding: FT fingerprint is generic domain adaptation

**good_medical and bad_medical are 92.4% correlated** (rho=0.924, p<1e-82). The FT fingerprint is overwhelmingly generic domain adaptation, not misalignment-specific.

Both share the same top traits at step 100: angry_register↓, contempt↓, ambition↓, flippancy↓, boredom↓. The dominant signal is "become more serious/medical-sounding" rather than "become misaligned."

The small differences that DO exist between bad and good medical are meaningful:
- empathy: bad=-0.078, good=+0.009 (bad medical reduces empathy)
- deception: bad=+0.065, good=-0.008 (bad medical increases deception)
- resentment: bad=+0.054, good=-0.008 (bad medical increases resentment)
- rationalization: bad=+0.089, good=+0.031 (bad medical increases rationalization more)

But these misalignment-specific differences are 3-5× smaller than the shared domain adaptation signal.

### FT misalignment residual POSITIVELY correlates with ICL

When we subtract out the generic domain adaptation (good_medical) from the misalignment signal (bad_medical), the FT residual **positively correlates** with the ICL residual at **rho=+0.605** (p<0.0001). The raw FT×ICL correlation was -0.567 (anti-correlated).

This is a sign flip: the anti-correlation between ICL and FT was an artifact of the generic domain effect. Once we remove it, **ICL and FT share a misalignment signal**. Both respond to bad medical content in qualitatively the same way — increasing deception (+0.073/+0.019), resentment (+0.062/+0.024), aggression (+0.056/+0.030), and decreasing empathy (-0.087/-0.081).

Sign agreement: 147/194 traits (75.8%) agree in sign between FT residual and ICL residual (vs 50% by chance, vs 27-33% for raw FT×ICL).

Top agreeing traits: empathy↓, deception↑, resentment↑, aggression↑, sycophancy↑, concealment↑, contempt↑, indignation↑, ambition↑, perfectionism↑.

### ICL anti-correlation persists for benign FT

good_medical × ICL: rho=-0.731 (p<1e-33)
bad_medical × ICL: rho=-0.567 (p<1e-17)

**good_medical actually anti-correlates MORE with ICL than bad_medical.** This deepens the puzzle: the anti-correlation isn't about misalignment content at all. It's about how base models process few-shot context vs how instruct models adapt to fine-tuning — regardless of content quality.

### Fingerprint emergence timing

- Signal appears by step 2-3 (contempt emerges as top trait)
- Converges rapidly: rho>0.9 with final fingerprint by step ~15
- insecure converges slightly slower (different domain character)
- Loss curves differ markedly: insecure starts and ends lowest (0.66→0.35), medical highest (3.3→1.1)

### 100-step vs HF LoRA comparison

The 100-step trajectory fingerprints and full-training HF LoRA fingerprints can be compared in `trajectory_cross_correlation.png`. This tests whether 100 steps is enough to capture the full FT effect.

## Open questions / next steps

1. ~~**FT trajectory + benign control**~~ — Done. Key finding: FT fingerprint is 92% generic domain adaptation.

2. **Deduplicate traits** — Rerun compare.py keeping only emotion_set version for the 20 overlapping concepts (174 unique instead of 194).

3. **Factor analysis** — PCA on the 174 trait vectors as directions in activation space. How many independent axes? Do ICL and FT load on the same factors with opposite signs?

4. **Cross-model transfer** — Do these findings hold for other model families?

5. **RLHF hypothesis** — Is the FT fingerprint just "undoing RLHF"? Compare base→instruct direction with FT direction. The benign control result strengthens this: if good AND bad medical FT produce the same fingerprint, the FT effect may simply be "undoing RLHF" regardless of training content.

6. ~~**Misalignment residual from FT**~~ — Done. rho=+0.605 with ICL.

7. **Layer persistence as misalignment filter (PROMISING, IN PROGRESS)**
   - Traits whose FT effect GROWS from L to L+5 have 79% ICL sign agreement vs 54% for shrinking
   - Growing-trait filter on FT residual: rho=0.746 (vs 0.494 all traits)
   - Growing-trait filter on RAW FT (no benign control): anti-correlation disappears (rho=-0.09 vs -0.50)
   - Key insight: when later layers amplify a FT perturbation, it is a real behavioral change
   - Open questions for next session:
     - Is L+5 the right offset? Try L+1 through L+10
     - Would averaging cos_sim from L to L+5 give more robust per-trait scores?
     - Can we detect misalignment from fingerprint + layer persistence alone, no benign control?
     - The real eval: tell bad_medical from good_medical using only FT fingerprint properties
   - Smart weighting (steering score, 1/variance, layer stability) all HURT vs unweighted baseline
   - User goal: detect misalignment during training from fingerprint alone, no benign control needed

8. **Different probe text** — try LoRA own responses, harmful prompts, domain-specific questions

## Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `icl_fingerprints.py` | ICL fingerprints (3 conditions) | Done |
| `ft_fingerprints.py` | FT fingerprints (4 HF LoRA variants) | Done |
| `compare.py` | Analysis + plots | Done (bug fixed: sign agreement) |
| `ft_trajectory.py` | Train 4 LoRAs + inline fingerprinting | Done |
| `plot_trajectory.py` | Trajectory visualization + good vs bad comparison | Done |

## Output files

- `icl_medical_residual.json` — 100 obs × 194 traits (primary, matched pair)
- `icl_financial_vs_benign.json` — 100 obs × 194 traits
- `icl_medical_vs_benign.json` — 100 obs × 194 traits
- `ft_fingerprints.json` — 4 variants × 194 traits (mean + per-response)
- `ft_trajectory.json` — 4 runs × 14 steps × 194 traits
- `correlation_blocks.png` — 3-panel Spearman heatmap
- `scatter_medical_residual_vs_{variant}.png` — trait-level scatter plots
- `pca_fingerprints.png` — PCA of all fingerprints (ICL blue, FT red)
- `sensitivity_curve.png` — Spearman vs top-k traits (biased, see note above)
- `trajectory_convergence.png` — Fingerprint convergence over training steps
- `trajectory_top_traits.png` — Top 8 traits per run over training
- `trajectory_cross_correlation.png` — Cross-run correlation matrix (100-step + HF + ICL)
- `trajectory_icl_correlation.png` — FT×ICL anti-correlation over training steps
- `trajectory_loss.png` — Training loss curves

## Design

### ICL (base model, Qwen2.5-14B)

**Medical_residual** (primary): Same 7049 questions, bad vs good medical answers as 2-shot context. Direct misalignment residual per observation. No context-length confound.

**Financial_vs_benign / Medical_vs_benign** (secondary): Misaligned context vs benign_kl baseline. Known context-length confound (different topics/lengths).

### FT (instruct model, Qwen2.5-14B-Instruct)

4 LoRA variants from HF (ModelOrganismsForEM): financial, sports, medical, insecure. Cosine sim of activation diff (LoRA - instruct) with trait vectors. 50 responses scored per variant.

### Metric

Both use `cosine_sim(activation_diff, trait_vector)` — self-normalizing, directly comparable between ICL and FT.

## Layer analysis results (Mar 5)

### Trait-layer complexity gradient
Behavioral/verbal patterns steer earliest (L19), basic emotions (L20), cognitive/social (L21), complex emotions (L22), alignment-relevant traits latest (L25). power_seeking/duplicity/eval_awareness/concealment all at L29-30. No correlation between layer and steering strength.

### Layer sensitivity of fingerprints
Fingerprints robust for 3 QA variants (self-corr 0.91+ at L+5, cross-variant structure preserved). Insecure collapses at L+5 (0.16). Key findings all hold qualitatively at shifted layers. Layer choice is optimization not fragility.

### Layer persistence experiments (4 experiments, mostly failed)
- Growth rate as continuous signal: FAIL (rho=-0.07)
- Growing traits more stable: FAIL (actually noisier, CV 0.84 vs 0.54)
- Cross-variant generalization: PARTIAL (58-64% sign agreement vs 22-25%, OR=4-6x)
- Trajectory divergence: FAIL (shrinking traits diverge slightly more)
- Binary grow/shrink filter enriches sign agreement but NOT a standalone misalignment detector

### Smarter correlation approaches (all tested)
- Weighting by steering score, 1/variance, layer stability: all HURT vs unweighted baseline
- Filtering by |FT residual| > 0.02: rho=0.674 (circular)
- A priori misalignment subset (29 traits): rho=0.697 (honest but small N)
