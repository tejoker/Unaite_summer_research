# Future Extensions - Non-Linear Methods

## Current Limitation: Linear SVAR Assumption

The current pipeline assumes a **linear** Structural Vector Autoregressive model:

```
X_t = W·X_t + Σ_p A_p·X_{t-p} + ε_t
```

Where:
- W: Instantaneous linear effects (DAG)
- A_p: Lagged linear effects
- All relationships are LINEAR

## Why This May Be Insufficient

Real industrial processes (bearing temperatures, thermal dynamics) exhibit:
- **Saturation effects** (temperatures plateau at limits)
- **Threshold behaviors** (friction kicks in above certain speeds)
- **Non-linear coupling** (quadratic thermal dissipation)
- **Multiplicative interactions** (pressure × temperature effects)

## Validation Status

- [ ] **Week 1**: Run `validate_linear_svar.py` on Golden data
- [ ] Compute R² to quantify linear model fit
- [ ] If R² < 0.70, non-linear methods are REQUIRED

## Proposed Non-Linear Extensions

### Option 1: NOTEARS-MLP (Recommended)
- Replace linear SVAR with neural network parameterization
- Maintains DAG constraint through continuous optimization
- Can learn arbitrary non-linear functions
- Reference: Zheng et al. (2020) - DAGs with NO TEARS (MLP version)

**Implementation**:
```python
# Replace: from_pandas_dynamic(...) with linear model
# With: from_pandas_dynamic_mlp(..., hidden_layers=[64, 32])
```

### Option 2: Neural ODEs for Time Series
- Learn continuous-time dynamics
- Better for irregular sampling
- Higher computational cost

### Option 3: Gaussian Process VAR (GP-VAR)
- Non-parametric non-linear VAR
- Quantifies uncertainty
- Expensive for high dimensions

## DO NOT Use Polynomial Features

**Rejected**: Adding polynomial features (X, X², X³, ...)

**Reasons**:
1. Computational explosion: 6 vars → 27 vars (degree 2) = 91× slowdown
2. Breaks DAG semantics (X² is deterministic from X)
3. Poor scaling for high dimensions
4. Destroys interpretability

## Migration Path

1. **Validate current linear model** (validate_linear_svar.py)
2. **If R² < 0.70**: Implement NOTEARS-MLP
3. **Benchmark**: Compare linear vs MLP on detection F1-score
4. **Publish**: Include both results in paper with ablation study

## Code Locations to Modify

- `executable/final_pipeline/dynotears.py`: Core algorithm (linear → MLP)
- `executable/validate_linear_svar.py`: Add MLP validation comparison
- `executable/final_pipeline/dbn_dynotears_fixed_lambda.py`: Update for MLP

## Timeline Estimate

- NOTEARS-MLP implementation: 2-3 days
- Validation on Golden data: 1 day
- Re-run Telemanom experiments: 3-5 days (400+ hours compute)
- Total: ~1 week + compute time

---
**Status**: PLANNED - Pending linear model validation results
**Priority**: HIGH if R² < 0.70, MEDIUM if R² > 0.70
**Last Updated**: 2025-11-07
