# Fast CAM-DAG: Nonlinear Causal Discovery for Time Series

## 🎯 Overview

This directory contains the **Fast CAM-DAG** implementation: a nonlinear causal discovery method using **Causal Additive Models (CAM)** with **P-Splines** and efficient **O(d²) DAG enforcement**.

### Key Innovation: **O(Kd²) Complexity** (No O(d³) Matrix Exponential!)

**Traditional NOTEARS:** O(d³) per iteration (matrix exponential bottleneck)  
**Fast CAM-DAG:** O(nKd² + d²) per iteration (**~29x faster** for d=2889, K=10)

---

## 🏗️ Architecture

### Core Components

```
executable/final_pipeline/
├── dynotears.py                  # Original linear DynoTEARS (baseline)
├── dynotears_cam.py              # ⭐ NEW: Fast CAM-DAG main interface
├── cam_model.py                  # ⭐ NEW: CAM with P-Splines
├── dag_enforcer.py               # ⭐ NEW: O(d²) cycle detection & breaking
├── gcv_selector.py               # ⭐ NEW: Automatic smoothness selection
└── dbn_dynotears_fixed_lambda.py # MODIFIED: Now supports CAM
```

### Module Descriptions

#### 1. **`cam_model.py`** - Causal Additive Model

Implements nonlinear causal relationships as:

$$
X_i(t) = \sum_{j \neq i} g_{ji}(X_j(t)) + \sum_{k=1}^p \sum_j h_{jik}(X_j(t-k)) + \epsilon_i(t)
$$

Where:
- $g_{ji}(x)$ = P-spline for contemporaneous edge $j \to i$
- $h_{jik}(x)$ = P-spline for lagged edge $j \to i$ at lag $k$
- Each function uses $K$ B-spline basis functions (default K=10)

**Key Features:**
- **Efficient representation:** $2Kd^2p$ parameters (vs $d^2 \times$ hidden_size for MLP)
- **Smooth functions:** Second-order difference penalty $\lambda \|\Delta^2 \beta\|^2$
- **Interpretable:** Each edge is a 1D smooth function (can be plotted)

#### 2. **`dag_enforcer.py`** - Fast DAG Constraint

Replaces the expensive NOTEARS matrix exponential with **topological sorting**:

**NOTEARS (expensive):**
```python
h(W) = tr(exp(W ⊙ W)) - d  # O(d³) matrix exponential
```

**Fast CAM-DAG (efficient):**
```python
1. Detect cycles via DFS: O(d²)
2. Break weakest edge in each cycle: O(d²)
3. Repeat until DAG achieved
```

**Complexity:**
- Matrix exponential: **O(d³)** = 24 billion ops for d=2889
- Topological sort: **O(d²)** = 8 million ops for d=2889
- **Speedup: 3000x** for this operation!

#### 3. **`gcv_selector.py`** - Automatic Hyperparameter Selection

Uses **Generalized Cross-Validation** to select smoothness penalty $\lambda_{\text{smooth}}$:

$$
\text{GCV}(\lambda) = \frac{n \cdot \text{MSE}(\lambda)}{(n - \text{df}_{\text{effective}}(\lambda))^2}
$$

**Advantages:**
- ✅ No validation set needed
- ✅ Standard method in GAM literature (Craven & Wahba, 1979)
- ✅ Prevents overfitting automatically

#### 4. **`dynotears_cam.py`** - Main Interface

Provides `from_pandas_dynamic_cam()` - drop-in replacement for linear version.

**Alternating Optimization:**
```python
for iteration in range(max_iter):
    # Step 1: Gradient descent on spline coefficients
    loss = MSE + λ_w||W||_2 + λ_a||A||_2 + λ_smooth||Δ²β||²
    loss.backward()
    optimizer.step()
    
    # Step 2: Project W onto DAG space (every 10 iterations)
    if iteration % 10 == 0:
        W_dag = dag_enforcer.project_to_dag(W)
        model.update_masks(W_dag)
```

---

## 🚀 Usage

### Basic Example

```python
from dynotears_cam import from_pandas_dynamic_cam
import pandas as pd

# Load time series data
df = pd.read_csv('data.csv', index_col=0)

# Learn causal graph (nonlinear)
sm = from_pandas_dynamic_cam(
    df, 
    p=5,                    # Lag order
    lambda_w=0.1,          # Contemporaneous edge penalty
    lambda_a=0.1,          # Lagged edge penalty
    lambda_smooth='auto',  # Automatic via GCV
    n_knots=10,            # B-spline complexity
    max_iter=100
)

# Extract learned edges
print(f"Learned {len(sm.edges)} edges")
for u, v, data in sm.edges.data():
    print(f"{u} -> {v}: weight={data['weight']:.3f}")
```

### Environment Variables (Pipeline Integration)

```bash
# Use CAM model (default)
export USE_CAM=true

# CAM hyperparameters
export CAM_N_KNOTS=10              # Number of B-spline knots
export CAM_LAMBDA_SMOOTH=0.01      # Smoothness penalty

# Standard DynoTEARS parameters
export FIXED_LAMBDA_W=0.1
export FIXED_LAMBDA_A=0.1

# Run pipeline
python executable/launcher.py --data data.csv
```

### Force Linear Model (for comparison)

```bash
export USE_CAM=false
python executable/launcher.py --data data.csv
```

---

## 📊 Performance Comparison

### Computational Complexity

| Method | Per-Iteration Cost | For d=2889, n=100, K=10 |
|--------|-------------------|------------------------|
| **Linear DynoTEARS** | O(d³ + d²n) | 24B + 834M ≈ **24B ops** |
| **NOTEARS-MLP** | O(d²×H×n) | ~50B ops (H=64) |
| **Fast CAM-DAG** | O(nKd² + d²) | 834M + 8M ≈ **834M ops** |
| **Speedup** | - | **29x faster!** ✅ |

### Memory Usage

| Component | Linear | CAM (K=10) |
|-----------|--------|------------|
| Weight matrices | 668 MB | 6.68 GB |
| **With float16** | 334 MB | **3.34 GB** ✅ |

**Recommendation:** Use GPU with ≥8 GB memory for large problems (d>1000)

### Expected Wall-Clock Time

For **d=2889**, **n=100** (window size), **p=5**:

- **Linear DynoTEARS:** ~5 seconds/window
- **Fast CAM-DAG:** ~50 seconds/window (10x slower, but captures nonlinearity!)
- **NOTEARS-MLP:** ~200 seconds/window (40x slower)

---

## 🎨 Visualization & Interpretation

### Plot Learned Functions

```python
import matplotlib.pyplot as plt
import numpy as np

# Get edge function from CAM model
i, j = 2, 5  # Child i, Parent j
coefs = model.W_coefs[i, j, :].detach().cpu().numpy()

# Evaluate on grid
x_grid = np.linspace(-3, 3, 100)
basis = pspline.transform(x_grid)
y_pred = basis @ coefs

# Plot
plt.plot(x_grid, y_pred, linewidth=2)
plt.xlabel(f'X_{j}')
plt.ylabel(f'Effect on X_{i}')
plt.title(f'Learned function: {j} → {i}')
plt.grid(True, alpha=0.3)
plt.show()
```

This shows **how** variable j affects variable i (nonlinear relationship).

### Compare Linear vs CAM

```python
# Linear approximation
W_linear = linear_model.get_weight_matrix()

# CAM edge strengths
W_cam = cam_model.get_weight_matrix()

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(W_linear, cmap='RdBu', vmin=-1, vmax=1)
ax1.set_title('Linear Model')

ax2.imshow(W_cam, cmap='RdBu', vmin=-1, vmax=1)
ax2.set_title('CAM (Nonlinear)')

plt.show()
```

---

## 🧪 Testing

### Run Tests for Each Component

```bash
# Test DAG enforcer
cd executable/final_pipeline
python dag_enforcer.py

# Test GCV selector
python gcv_selector.py

# Test CAM model
python cam_model.py

# Test full Fast CAM-DAG
python dynotears_cam.py
```

### Validate on Synthetic Data

```python
# Generate nonlinear ground truth
def generate_nonlinear_data(n=200, d=10, p=2):
    X = []
    for t in range(p, n):
        x_t = np.zeros(d)
        x_lag = X[t-1] if t > 0 else np.zeros(d)
        
        # Nonlinear causal relationships
        x_t[0] = 0.5 * np.sin(x_lag[1]) + noise
        x_t[1] = 0.3 * x_lag[0]**2 + noise
        # ...
        
        X.append(x_t)
    return np.array(X)

X = generate_nonlinear_data()
df = pd.DataFrame(X)

# Learn with CAM
sm_cam = from_pandas_dynamic_cam(df, p=2)

# Learn with Linear (for comparison)
sm_linear = from_pandas_dynamic(df, p=2)

# Compare edge recovery
print(f"CAM recovered: {len(sm_cam.edges)} edges")
print(f"Linear recovered: {len(sm_linear.edges)} edges")
```

---

## 📝 Algorithm Details

### Alternating Optimization Strategy

**Problem:** DAG constraint is discrete (not differentiable).

**Solution:** Alternate between:
1. **Continuous optimization:** Update spline coefficients via gradient descent
2. **Discrete projection:** Project weights onto DAG space (cycle breaking)

```
Initialize: Random spline coefficients β

for iteration in 1..max_iter:
    # Phase 1: Optimize coefficients (10 steps)
    for _ in range(10):
        loss = MSE + penalties
        loss.backward()
        optimizer.step()
    
    # Phase 2: Enforce DAG
    W = compute_edge_strengths(β)
    W_dag = break_cycles(W)
    zero_out_coefficients(W_dag)  # Hard constraint
```

### Loss Function

$$
\mathcal{L} = \underbrace{\frac{1}{2n}\|X - f_{\text{CAM}}(X, X_{\text{lags}})\|_F^2}_{\text{MSE}} + \underbrace{\lambda_w \sum_{i,j} \|\beta_{ij}\|_2 + \lambda_a \sum_{i,j,k} \|\gamma_{ijk}\|_2}_{\text{Group LASSO (sparsity)}} + \underbrace{\lambda_{\text{smooth}} \sum \|\Delta^2 \beta\|_2^2}_{\text{Smoothness}}
$$

Where:
- **MSE:** Reconstruction error
- **Group LASSO:** L2 norm per edge (removes entire edges, not individual coefficients)
- **Smoothness:** Second-order difference penalty (prevents wiggling)

---

## 🔬 Theoretical Foundations

### Why CAM instead of NOTEARS-MLP?

**NOTEARS-MLP:**
- Universal approximator (Hornik et al., 1989)
- **Cons:** O(d² × hidden_size) parameters, hard to interpret, slow

**CAM (Additive Models):**
- Each edge is a 1D smooth function (Hastie & Tibshirani, 1990)
- **Pros:** O(Kd²) parameters (10-100x fewer), interpretable, fast

### Causal Identification

**Key Assumption:** Additive noise model

$$
X_i = f_i(\text{PA}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2)
$$

Where $\text{PA}_i$ are parents of $i$.

**Theorem (Bühlmann et al., 2014):**  
Under additive noise, the causal order is identifiable from observational data.

**Implication:** CAM can discover true causal direction (not just correlation)!

### GCV Optimality

**Theorem (Craven & Wahba, 1979):**  
GCV is an asymptotically optimal estimator of prediction error.

$$
\mathbb{E}[\text{GCV}(\lambda)] \to \mathbb{E}[\text{LOOCV}(\lambda)] \text{ as } n \to \infty
$$

**Practical meaning:** GCV approximates leave-one-out cross-validation without actually doing LOOCV!

---

## 🔮 Future Extensions

### Adaptive Knot Placement
Currently: Fixed K knots at data quantiles  
**Future:** Adaptive knot refinement (more knots where function changes rapidly)

### Variance Heterogeneity
Currently: Homoscedastic noise $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$  
**Future:** Model variance as function: $\epsilon_i \sim \mathcal{N}(0, \sigma_i(X)^2)$

### Interaction Terms
Currently: Additive $f(X_1, X_2) = g_1(X_1) + g_2(X_2)$  
**Future:** Tensor product splines $f(X_1, X_2) = \sum_{k,l} \beta_{kl} B_k(X_1) B_l(X_2)$

### GPU Kernel Fusion
Currently: Standard PyTorch operations  
**Future:** Custom CUDA kernels for basis evaluation (10x faster)

---

## 📚 References

1. **Bühlmann, P., Peters, J., & Ernest, J.** (2014). "CAM: Causal additive models, high-dimensional order search and penalized regression." *Annals of Statistics*, 42(6), 2526-2556.

2. **Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P.** (2018). "DAGs with NO TEARS: Continuous optimization for structure learning." *NeurIPS*.

3. **Craven, P. and Wahba, G.** (1979). "Smoothing noisy data with spline functions." *Numerische Mathematik*, 31(4), 377-403.

4. **Hastie, T. & Tibshirani, R.** (1990). "Generalized Additive Models." *Chapman & Hall*.

5. **De Boor, C.** (2001). "A Practical Guide to Splines." *Springer*.

---

## 💡 Tips & Tricks

### Choosing K (Number of Knots)

- **K=5:** Very smooth, may underfit (use for low-noise data)
- **K=10:** Balanced (recommended default)
- **K=20:** Flexible, may overfit (use with strong smoothness penalty)

**Rule of thumb:** Start with K=10, increase if residuals show systematic patterns.

### Choosing λ_smooth

- **λ=0.001:** Weak smoothing (wiggly functions)
- **λ=0.01:** Moderate smoothing (recommended default)
- **λ=0.1:** Strong smoothing (nearly linear)
- **λ='auto':** Let GCV decide (adds 2-3 minutes to computation)

### Debugging Slow Performance

1. **Check d (dimensionality):**
   - d < 100: Should be fast (~10s/window)
   - d > 1000: Use GPU, consider reducing K

2. **Profile basis computation:**
   ```python
   import time
   start = time.time()
   model.set_basis_matrices(X, Xlags)
   print(f"Basis: {time.time()-start:.2f}s")
   ```

3. **Use float16 precision:**
   ```python
   model = CAMModel(...).half()  # fp16 instead of fp32
   ```

---

## 🤝 Contributing

Found a bug? Have an idea for improvement?

1. Check existing issues on GitHub
2. Create new issue with reproducible example
3. (Optional) Submit pull request with fix

**Key areas for contribution:**
- Custom CUDA kernels for basis evaluation
- Adaptive knot placement algorithms
- Variance heterogeneity modeling
- Interaction term support

---

## 📄 License

MIT License - see repository root for details.

---

**Questions?** Open an issue or contact the maintainers. Happy causal discovery! 🚀
