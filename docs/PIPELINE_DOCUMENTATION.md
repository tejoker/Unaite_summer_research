### Core Anomaly Detection and Root Cause Analysis Pipeline

The pipeline is designed to detect anomalies in time series data by analyzing changes in the underlying causal structure of the system. It operates in a window-by-window fashion, allowing for the detection of transient anomalies. The core of the methodology can be broken down into the following stages:

---

## Historical Development and Archived Work

This project underwent extensive development and experimentation before reaching its current optimized state. All superseded code, experimental approaches, and development artifacts have been archived (December 13, 2025) to maintain a clean production codebase while preserving historical context.

### Major Development Milestones

**1. Memory Optimization Journey**
- **Problem**: Original sequential implementation experienced severe memory leaks (20GB edge accumulation, 13.4GB weight matrices)
- **Workarounds Developed** (now archived in `memory_leak_workarounds/`):
  - Nuclear restart wrapper that spawned fresh processes between windows
  - Process-based isolation to prevent memory accumulation
- **Final Solution**: Implemented shared memory architecture (95MB zero-copy), 2D block chunking (13.4GB → 500MB), and fixed root causes
- **Archived**: `dbn_dynotears_tucker_cam_restart.py` and related restart mechanisms

**2. Tucker Model Stability and Regularization**
- **Problem**: Training instability and weight collapse in Tucker decomposition
- **Experimental Fixes** (archived in `experimental_fixes/`):
  - Post-hoc weight rescaling scripts
  - Ad-hoc normalization approaches
- **Final Solution**: Proper regularization parameters (lambda_core=0.01, lambda_orth=0.001) implemented in core Tucker model
- **Archived**: `rescale_collapsed_weights.py`

**3. Performance Optimization Evolution**
- **Experimental Approaches** (archived in `experimental_optimizations/`):
  - Alternative weight matrix extraction methods (v2, v3 implementations)
  - Various chunking strategies for large tensor operations
- **Final Solution**: Optimized 2D block chunking with mixed-precision training, integrated into main `TuckerFastCAMDAG` class
- **Archived**: `cam_model_tucker_fast_extract.py` with alternative implementations

**4. Anomaly Detection Method Development**
- **Evolution** (archived in `anomaly_detection_experiments/`):
  - Early iteration: `improved_anomaly_detector.py`
  - Basic prototype: `simple_nn_detector.py`
  - Experimental multi-metric approaches
- **Current Production**:
  - `dual_metric_anomaly_detection.py` (integrated)
  - `chunked_nn_detector.py` (optimized)

**5. Experimental Detection Strategies** (archived in `experimental_detectors/`)
Tested but not integrated into production:
- Causal cascade detection
- Directional asymmetry analysis
- Temporal precedence tracking
- Multi-window voting schemes
- Rate-of-change based detection
- Sub-window granular analysis
- Unified ensemble approaches
These experiments informed the design of the final detection pipeline but were not adopted as-is.

**6. Diagnostic and Analysis Tools** (archived in `diagnostic_scripts/`)
Development-phase debugging tools:
- Dataset drift diagnosis
- Detection failure analysis
- Lambda hyperparameter search diagnostics
- Window-level epicenter detection
- Global change pattern analysis
These were critical during development but are no longer needed for production use.

**7. Pipeline Architecture Evolution**
- **Original**: Monolithic `end_to_end_pipeline.py` with sequential processing
- **Intermediate**: Various pipeline variants testing different decomposition strategies
- **Current**: Modular architecture with:
  - `launcher.py` for flexible execution
  - `run_tucker_cam_benchmark.sh` for orchestration
  - `dbn_dynotears_tucker_cam_parallel.py` with shared memory and dynamic threading
- **Archived** (in `legacy_sequential/` and `pipeline_variants/`):
  - `dbn_dynotears_tucker_cam.py` (sequential, contains bugs: args undefined, num_chunks undefined)
  - `end_to_end_pipeline.py` (monolithic)
  - Alternative decomposition strategies in `pipeline_variants/`

**8. Extensive Documentation Archive** (38 files in `old_docs/`)
Comprehensive guides and analysis documents created during development:
- CAM optimization journey and implementation summaries
- Memory optimization strategies
- Edge pruning analysis
- Anomaly signature catalogs
- Architecture validation studies
- Batch processing guides
- Deep dive analysis explanations
- Differencing and persistence theory
- Future extension proposals
- Deployment guides
All insights from these documents have been integrated into current documentation or implementation.

**9. Testing and Validation History**
- **Completed Experiments** (archived in `completed_experiments/`): Early validation runs
- **Test Scripts** (archived in `test_scripts/`, `suite_tests/`, `old_test_files/`): Development-phase unit and integration tests
- **Legacy Results** (preserved in `archive/legacy/results/`): Historical baseline experiments including 100+ Telemanom dataset runs
- **Comparison Scripts** (archived in `comparison_scripts/`): Scripts for comparing different methodological approaches

**10. Utilities and Helper Scripts** (archived in `utils/`)
Development-phase helper scripts:
- JSON field readers
- Anomaly position checkers
- Window-to-time mapping utilities
- Lambda parameter readers
These were integrated into core modules or made obsolete by refactoring.

### Legacy Data Preservation

The `archive/legacy/` directory (55GB) contains:
- **Historical datasets**: Multiple anomaly injection scenarios, golden baselines, multi-anomaly tests
- **Experimental results**: 100+ Telemanom benchmark runs with complete history, preprocessing outputs, and learned weights
- **Validation data**: Cascade analysis, comparative studies, and linear model validation results

This data serves as the experimental foundation for the methodology and should be preserved for reproducibility and future research.

### Recovery and Git History

All archived code remains accessible:
1. **Git History**: Complete development history preserved in git commits
2. **Archive Directory**: Structured preservation with documented reasons for archival
3. **Archive README**: Detailed categorization of what was archived and why

For questions about historical implementation decisions or to recover specific experimental code, consult the git history or `archive/README.md`.

---

#### 1. Data Preprocessing

-   **Stationarity:** The raw time series data is first made stationary. This is achieved by applying a `log1p` transformation followed by first-order differencing. Stationarity is verified using both the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.
-   **Lag Optimization:** For each time series, an optimal lag `p` is determined. This is done by fitting an AutoRegressive (AR) model for different lag values and selecting the smallest `p` for which the model's residuals are not autocorrelated, as determined by the Ljung-Box test. This step is crucial for the subsequent causal discovery phase.
-   **Data Transformation:** The preprocessed data is then transformed into a format suitable for the causal discovery model. For each time point `t`, the data is organized into two matrices: `X`, containing the values of all variables at time `t`, and `Xlags`, containing the values of all variables at times `t-1, t-2, ..., t-p`.

#### 2. Causal Structure Learning (DynoTears)

The core of the causal discovery process is the **DynoTears** algorithm, which learns a Dynamic Bayesian Network (DBN) from the time series data. This method is based on the work of Zheng et al. (2018) and is designed to uncover both instantaneous and time-lagged causal relationships. The implementation is primarily located in `executable/final_pipeline/dynotears.py`.

The model assumes a linear Structural Vector Autoregressive (SVAR) model, which provides a clear and interpretable representation of the system's dynamics. The model is defined as:

`X(t) = W * X(t) + A_1 * X(t-1) + ... + A_p * X(t-p) + E(t)`

Where:
-   `X(t)`: A `d`-dimensional vector representing the values of the `d` variables at time `t`.
-   `W`: The `d x d` contemporaneous (intra-slice) weight matrix. `W_ij` represents the causal influence of variable `j` on variable `i` at the same time `t`. The diagonal elements of `W` are constrained to be zero to prevent self-causation.
-   `A_l`: The `d x d` lagged (inter-slice) weight matrix for lag `l`. `A_l_ij` represents the causal influence of variable `j` at time `t-l` on variable `i` at time `t`.
-   `E(t)`: A `d`-dimensional vector of independent and identically distributed (i.i.d.) noise terms, representing unobserved influences.

The primary goal is to estimate the matrices `W` and `A = [A_1, ..., A_p]` from the observed data. This is framed as a regularized least-squares optimization problem. The main entry point is the `from_pandas_dynamic` function, which prepares the data and calls the core optimization routine, `_learn_dynamic_structure`.

The optimization problem is defined as:

`min_{W, A} 0.5 * ||X - XW - Xlags * A_flat||^2_F + lambda_w * ||W||_1 + lambda_a * ||A||_1`

**Implementation Details in `_learn_dynamic_structure`:**

-   **Input Data:** The function takes PyTorch tensors `X` (current values) and `Xlags` (lagged values) as input. These are prepared by the `_to_numpy_dynamic` helper function, which uses a `DynamicDataTransformer`.
-   **Loss Calculation:**
    -   The first term, `||X - XW - Xlags * A_flat||^2_F`, is the squared Frobenius norm of the residuals, measuring model fit. In the code, this corresponds to `loss_mse`. To handle large datasets, the calculation is performed in batches. The `residual` for a batch is calculated as `residual = X_batch.matmul(I - W) - Xlags_batch.matmul(A_flat)`.
    -   The terms `lambda_w * ||W||_1` and `lambda_a * ||A||_1` are L1 regularization penalties on the `W` and `A` matrices. These penalties encourage sparsity, leading to a simpler causal graph. The hyperparameters `lambda_w` and `lambda_a` are passed directly to the learning function. In the code, this is calculated as `l1_penalty = lambda_w * torch.abs(W).sum() + lambda_a * torch.abs(A).sum()`.
-   **Acyclicity Constraint:** A critical constraint is that the contemporaneous graph `W` must be a **Directed Acyclic Graph (DAG)**. This is enforced using an augmented Lagrangian method with the penalty term `h(W) = trace(expm(W*W)) - d = 0`.
    -   This function is zero if and only if `W` is a DAG. In the code, this is `h_val_tensor`, calculated as `torch.trace(torch.linalg.matrix_exp(W_squared)) - d`, where `W_squared` is `W.matmul(W)`.
    -   For numerical stability when the norm of `W_squared` is large, a polynomial approximation of the matrix exponential is used.
-   **Optimization:**
    -   The optimization iteratively adjusts `W` and `A` using the `torch.optim.Adam` optimizer.
    -   The total `loss` combines the `loss_mse`, the `l1_penalty`, and the augmented Lagrangian terms for the acyclicity constraint: `0.5 * rho * (h_val_tensor ** 2) + alpha * h_val_tensor`.
    -   The Lagrangian multipliers (`alpha` and `rho`) are updated in each iteration to progressively enforce the DAG constraint.
    -   Gradient clipping (`torch.nn.utils.clip_grad_norm_`) is used to prevent exploding gradients.
    -   The framework also supports forbidding specific edges via the `tabu_edges` parameter.

#### 3. Memory-Efficient P-spline Model with Tucker Decomposition (Tucker-CAM)

A limitation of the linear SVAR model is its inability to capture non-linear causal relationships. To address this, the pipeline employs a **Tucker-CAM** model, implemented in `executable/final_pipeline/cam_model_tucker.py` and orchestrated by `executable/final_pipeline/dynotears_tucker_cam.py`. This extends the DynoTears framework to non-linear relationships in a memory-efficient manner by combining P-splines and Tucker decomposition.

**1. P-splines for Non-linearities:**

Instead of assuming linear relationships, we model them as smooth, non-linear functions using P-splines (penalized B-splines). Each non-linear function is represented as a linear combination of B-spline basis functions. In the `TuckerFastCAMDAG.fit` method, the basis matrices `B_w` and `B_a` are computed from the input data `X` for this purpose using the `_compute_basis_matrix` method.

**2. Tucker Decomposition for Dimensionality Reduction:**

In a high-dimensional setting, the coefficient tensors for the non-linear model become enormous. **Tucker decomposition** is a tensor factorization technique that dramatically reduces the number of parameters by approximating a large tensor with a smaller "core" tensor and a set of factor matrices.

Instead of learning the full, large coefficient tensors, the `TuckerCAMModel` class stores and learns the much smaller components as `torch.nn.Parameter`:
-   A small **core tensor** (e.g., `self.W_core` for the contemporaneous matrix).
-   A set of **factor matrices** for each dimension (e.g., `self.W_U1`, `self.W_U2`, `self.W_U3`).

The ranks of these decompositions, which control the degree of compression, are specified by the `rank_w` and `rank_a` parameters in the `TuckerFastCAMDAG` constructor.

The factorization for the `W` and `A` coefficient tensors looks like this:

-   `W[d,d,K] ≈ W_core[r,r,r] × W_U1[d,r] × W_U2[d,r] × W_U3[K,r]`
-   `A[d,d,p,K] ≈ A_core[r,r,r,r] × A_U1[d,r] × A_U2[d,r] × A_U3[p,r] × A_U4[K,r]`

Here:
-   `r` is the **Tucker rank** (`rank_w` or `rank_a` in the code), a hyperparameter controlling compression.
-   `W_core` and `A_core` are the small core tensors.
-   `W_U1`, `W_U2`, etc., are the factor matrices.

**Implementation Details in `TuckerFastCAMDAG` and `TuckerCAMModel`:**

-   **Optimization Framework:** The `TuckerFastCAMDAG` class in `dynotears_tucker_cam.py` wraps the `TuckerCAMModel` and implements the main optimization loop in its `fit` method.
-   **On-the-fly Reconstruction:** During the optimization, the full coefficient tensors are not stored in memory. Instead, they are reconstructed on-the-fly within the `TuckerCAMModel.forward` method using `torch.einsum`. This method computes the final prediction (`pred`) by applying the reconstructed coefficients to the B-spline basis matrices. This approach allows the model to capture complex non-linear relationships with significant improvements in memory efficiency.
-   **Loss Function:** The `fit` method in `TuckerFastCAMDAG` defines the total loss, which includes:
    -   `loss_fit`: The mean squared error between the data `X` and the prediction `pred`.
    -   `loss_l1_w` and `loss_l1_a`: L1 penalties on the reconstructed coefficient tensors to encourage sparsity.
    -   `loss_smooth`: A smoothness penalty calculated by `model.compute_smoothness_penalty()` that penalizes the variation in the B-spline coefficients, encouraging smoother functions.
    -   Augmented Lagrangian terms for the acyclicity constraint, identical to the linear DynoTears model.
-   **Acyclicity Constraint:** The `h(W)` constraint is applied to an aggregated weight matrix `W`, which is obtained by averaging the reconstructed contemporaneous coefficient tensor over the B-spline basis dimension (`K`). This is handled by the `model.get_weight_matrix()` method.
-   **Performance Optimizations:** The implementation uses several techniques to improve performance and reduce memory usage, including:
    -   **Mixed-precision training** (`torch.amp.autocast`) on CUDA devices.
    -   The option to use the high-performance `FusedAdam` optimizer from NVIDIA's Apex library.
    -   Chunking operations within the `forward` pass to avoid creating large intermediate tensors.

#### 4. Window-by-Window Anomaly Detection

The anomaly detection process is performed by comparing the causal graphs learned from a "golden" (baseline) period with those learned from a potentially anomalous period. This comparison is done on a window-by-window basis.

For each window, the following steps are taken:

1.  **Learn Causal Graphs:** The DynoTears algorithm is run on both the baseline and the current window's data to obtain the respective `W` and `A` matrices.
2.  **Compare Graphs:** The `W` matrices from the baseline and the current window are compared using a suite of metrics, including:
    -   **Frobenius Norm:** Measures the overall difference in magnitude of the edge weights.
    -   **Structural Hamming Distance (SHD):** Measures the number of edge additions, deletions, or reversals required to transform one graph into the other.
    -   **Spectral Distance:** Compares the eigenvalues of the graph Laplacians.
3.  **Anomaly Scoring:** An ensemble score is computed based on these metrics to determine if the current window is anomalous.
4.  **Anomaly Classification:** If an anomaly is detected, it is classified into a specific type (e.g., "edge magnitude change", "edge addition/deletion") based on a set of rules.
5.  **Root Cause Analysis:** The edges that contribute most to the anomaly score are identified as the potential root causes. This is typically done by ranking the edges based on the magnitude of their weight change.

#### 5. Time Series Reconstruction

Once an anomaly has been detected and its root cause identified, the pipeline can be used to reconstruct a "corrected" version of the time series. This is done by:

1.  **Correcting the Causal Graph:** The anomalous edges in the `W` matrix of the anomalous window are corrected. A common strategy is to replace the anomalous edge weights with their corresponding values from the baseline `W` matrix.
2.  **Reconstructing the Time Series:** The corrected `W` matrix, along with the baseline `A` matrices, are used to simulate the time series forward from a set of initial conditions. This results in a reconstructed time series that represents how the system would have behaved in the absence of the anomaly.

This end-to-end pipeline provides a powerful framework for not only detecting anomalies in complex, high-dimensional time series but also for identifying their root causes and simulating a corrected version of the system's behavior.
