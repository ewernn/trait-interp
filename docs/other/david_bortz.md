# David Bortz — Research Notes

Notes from Dec 2024 on David's research and potential connections to trait monitoring.

---

## What He Works On

**Core area:** Weak-form methods for learning dynamics from noisy data.

**Key methods:**
- **WENDy** (Weak-form Estimation of Nonlinear Dynamics) — Parameter estimation for ODEs
- **SINDy** (Sparse Identification of Nonlinear Dynamics) — Discovering governing equations

**The key insight:** Instead of solving differential equations forward (parameters → solution), plug noisy data into the weak form to solve for parameters via linear regression. Integration by parts shifts derivatives onto smooth test functions, bypassing numerical differentiation of noisy data.

---

## Recent Papers

| Paper | arXiv | Focus |
|-------|-------|-------|
| WENDy for Nonlinear-in-Parameters ODEs | [2502.08881](https://arxiv.org/abs/2502.08881) | Extending WENDy to nonlinear parameters via MLE |
| Weak Form SciML: Test Function Construction | [2507.03206](https://arxiv.org/abs/2507.03206) | Data-driven selection of test function hyperparameters |
| WgLaSDI (Physics-informed + weak-form) | [2407.00337](https://arxiv.org/abs/2407.00337) | Combining weak-form with autoencoders for latent space dynamics |
| Bias and Coverage of WENDy-IRLS | [2510.03365](https://arxiv.org/abs/2510.03365) | Statistical properties of WENDy |

---

## Key Results (from talk notes)

- **Noise robustness:** Lorenz system with 10% noise → <1% parameter error. KS equation with 50% noise → 100% recovery.
- **Speed:** 2 orders of magnitude faster than forward-solver methods
- **Accuracy:** 1 order of magnitude more accurate
- **No initial guess required:** Linear solve, not iterative optimization

---

## Potential Connections to Trait Monitoring

**The problem we share:** Computing derivatives (velocity, acceleration) of noisy signals.

| Our Problem | Weak-Form Solution |
|-------------|-------------------|
| Token-level trait projections are noisy | Integration reduces variance |
| Velocity (1st derivative) amplifies noise | Derivative moves to smooth test function |
| Acceleration (2nd derivative) even worse | Same trick, 2nd derivative on test function |
| Commitment point detection uses acceleration | Could be much more robust |

**How weak-form helps:**
```
Traditional: d²S/dt² ≈ (S[t+1] - 2S[t] + S[t-1]) / Δt²  ← noise explodes

Weak form: ∫ S(t) · φ''(t) dt  ← derivative on smooth known φ
```

**Conceptual similarity to Laplace transform:**
- Both avoid differentiating noisy signal directly
- Laplace: global frequency transform, need inverse
- Weak form: local convolution, stays in time domain

**WgLaSDI paper most relevant** — combines weak-form with neural network latent spaces. If we're computing trait dynamics, their noise-robustness framework could apply.

---

## His Research Style

- **Inverse problems framing:** "data → parameters" not "parameters → solution"
- **Noise obsession:** Multiple papers emphasize 10-100% noise tolerance
- **Computational efficiency:** Values lightweight methods ("2 orders faster")
- **Linear > nonlinear:** Reformulate to avoid optimization pitfalls

---

## How to Frame Our Work for Him

- "Inverse problems on neural network activations — given hidden states, identify behavioral mode"
- "Linear projection instead of training detectors"
- "Noise is a problem at token level, averaging helps, but interested in better approaches"

**What he'd probably ask:**
- "What's your noise model?"
- "How does it scale?"
- "What's the ground truth?"
- "Does it generalize?"

---

## Software

Implementations available in: MATLAB, Python, Julia, R

---

## Next Steps

- [ ] Try weak-form smoothing on trait trajectories
- [ ] Compare to current moving average / finite difference approach
- [ ] See if commitment point detection improves
