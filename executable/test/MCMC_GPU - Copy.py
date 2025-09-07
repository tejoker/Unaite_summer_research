import cupy as cp
from scipy.stats import kstest
import numpy as np
from joblib import Parallel, delayed
# Set a fixed seed for reproducibility (analogous to np.random.seed on GPU)
cp.random.seed(42)
import logging
from typing import Callable, Tuple, Optional

# Configure root logger once
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s [%(module)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Example target log-density and its gradient (must be GPU-compatible).
# These use CuPy array operations so they run on the GPU.
def U_target(x: cp.ndarray) -> cp.ndarray:
    """Quadratic potential: 0.5 * ||x - vec_diffs||^2."""
    diff = x - vec_diffs  # operates on GPU
    return 0.5 * cp.sum(diff * diff, axis=-1)  # supports both 1D and batched input

def grad_U_target(x: cp.ndarray) -> cp.ndarray:
    """Analytic gradient of U_target."""
    return x - vec_diffs

def approx_grad_gpu(U, x, eps=1e-5):
    """
    Central-difference ∇U on the GPU:
      U : callable taking a CuPy array → CuPy scalar or vector
      x : CuPy array of shape (d,) or (n,d)
    Returns gradient array of same shape as x.
    """
    # If x is 1D: shape (d,)
    # If x is batched: shape (n, d)
    orig_shape = x.shape
    d = orig_shape[-1]
    grads = cp.empty_like(x)
    for i in range(d):
        # build perturbed copies
        ei = cp.zeros(d); ei[i] = eps
        if x.ndim == 1:
            gp = U(x + ei)
            gm = U(x - ei)
        else:
            # broadcast perturbs across batch dimension
            gp = U(x + ei[None, :])
            gm = U(x - ei[None, :])
        grads[..., i] = (gp - gm) / (2 * eps)
    return grads

# ================================
# 0. GLOBAL SETTINGS & PARAMETERS
# ================================
RNG = np.random.RandomState(42)
# test.py (en début de fichier)
#!/usr/bin/env python3
# test.py

import numpy as np
from fake_data import (
    D, T, lag,
    generate_diverse_multivariate,
    apply_spatiotemporal,
    summarize_diffs
) #D stands for Dimension

# — 1) Paramètres de génération (identiques à test4.py) —
seed       = 2025
np.random.seed(seed)
# si vous aviez besoin d'une Beta_mean fixe :
Beta_mean = np.random.uniform(0.2, 0.8, size=(D, D))
np.fill_diagonal(Beta_mean, 0.0)
sigma_beta = 0.05

# — 2) Générer les séries et récupérer diffs_k comme dans test4.py —
Y_base, _        = generate_diverse_multivariate(D, T, seed=seed), None
Y_spatio, diffs_k = apply_spatiotemporal(
    Y_base,
    lag=lag,
    Beta_mean=Beta_mean,
    sigma_beta=sigma_beta,
    seed=seed
)

# (facultatif) un petit résumé console :
summarize_diffs(diffs_k)

# — 3) Extraction des échantillons bruts de ΔY —
# version « à plat » (toutes séries, tous lags)
vec_diffs = diffs_k.reshape(-1)      # taille = D * T * (lag+1)
vec_diffs = vec_diffs[~np.isnan(vec_diffs)]   # strip out all the NaNs
# ou version « filtrée » sans NaN :
all_diffs = diffs_k[~np.isnan(diffs_k)]  # 1-D array de longueur variable
D= vec_diffs.size
print(f"J'ai bien récupéré {all_diffs.size} valeurs de ΔY.")

# Example data for U_target (this would come from user data; ensure it's on GPU)
# Here we assume vec_diffs is already defined as a 1D NumPy array of length DIMENSION.
# We transfer it to GPU as a CuPy array:
vec_diffs = cp.asarray(vec_diffs)  # move target vector to GPU

# Global algorithm parameters (same values as original, but now on GPU if arrays)
R_HAT_THRESH = 1.1
ESS_MIN = 100
MCSE_MEAN_TOL = 0.05
MCSE_VAR_TOL = 0.05
SBC_PVAL_THRESH = 0.05
PPC_DIST_THRESH = 0.1
HMC_NUM_STEPS   = 5        # default L for tuning
TARGET_ACC_COMMON  = 0.70  # target acceptance for leapfrog/force-gradient
TARGET_ACC_YOSHIDA = 0.80  # target acceptance for Yoshida4 integrator
EPS_GRID_BASE = 0.1 * (D ** -0.25)
EPS_GRID_COMMON  = cp.linspace(EPS_GRID_BASE/10, EPS_GRID_BASE*10, 20)
EPS_GRID_YOSHIDA = EPS_GRID_COMMON  # same grid for simplicity
STEP_PARTICLES = 100
MAX_SMC_PARTICLES = 1000
BURN_IN_FRAC = 0.1
STEP_SWEEPS = 50
MAX_CHAIN_SWEEPS = 500
MAX_HMC_STEPS = 10
MAX_CHAIN_LENGTH = 1000
NUM_SMC_INTER = 10
STEP_N           = 10
M   = 5 #number of independant chains 
m= 5  # number of temperature chains for PT
mu=1
sigma=0.1  # prior parameters for θ


def leapfrog_step(theta: cp.ndarray, p: cp.ndarray, epsilon: float, L: int,
                 grad_U_func, beta=1.0) -> tuple[cp.ndarray, cp.ndarray]:
    """Leapfrog integrator (supports batched states on GPU)."""
    q = theta.copy()
    p_half = p.copy()
    # Half-step momentum update
    if isinstance(beta, cp.ndarray):  
        p_half -= 0.5 * epsilon * (beta[:, None] * grad_U_func(q))
    else:
        p_half -= 0.5 * epsilon * beta * grad_U_func(q)
    # Full steps
    for i in range(L):
        q += epsilon * p_half
        if i != L - 1:  # full momentum update except at last step
            if isinstance(beta, cp.ndarray):
                p_half -= epsilon * (beta[:, None] * grad_U_func(q))
            else:
                p_half -= epsilon * beta * grad_U_func(q)
    # Final half-step momentum update
    if isinstance(beta, cp.ndarray):
        p_half -= 0.5 * epsilon * (beta[:, None] * grad_U_func(q))
    else:
        p_half -= 0.5 * epsilon * beta * grad_U_func(q)
    return q, p_half  # new position and momentum



def force_gradient_step(theta: cp.ndarray,
                        p: cp.ndarray,
                        epsilon: float,
                        L: int,
                        grad_U_func,
                        beta=1.0) -> tuple[cp.ndarray, cp.ndarray]:
    """2nd-order 'force gradient' integrator (Velocity Verlet variant)."""
    # Copy initial state
    q = theta.copy()
    p_new = p.copy()

    # Prepare beta for broadcast: shape (n_replicas, 1) or scalar
    if isinstance(beta, cp.ndarray):
        b = beta.reshape((-1, 1))
    else:
        b = beta

    # Integration loop
    for _ in range(L):
        # Compute gradient and scale by beta
        g = grad_U_func(q)           # shape (D,) or (n_replicas, D)
        g = b * g

        # Half-step momentum update
        p_half = p_new - 0.5 * epsilon * g

        # Finite-difference Hessian-vector approximation
        delta = 1e-5
        grad_plus  = grad_U_func(q + delta * p_half)
        grad_minus = grad_U_func(q - delta * p_half)

        # Scale and compute Hessian-vector product
        plus  = b * grad_plus
        minus = b * grad_minus
        Hv    = (plus - minus) / (2.0 * delta)

        # Position update (with Hessian correction)
        q = q + epsilon * p_half + (epsilon**3 / 24.0) * Hv

        # Final momentum update
        g_new = grad_U_func(q)
        p_new = p_half - 0.5 * epsilon * (b * g_new)

    return q, p_new


def yoshida4_step(theta: cp.ndarray, p: cp.ndarray, epsilon: float, L: int,
                  grad_U_func, beta=1.0) -> tuple[cp.ndarray, cp.ndarray]:
    """4th-order Yoshida integrator (Suzuki–Yoshida coefficients)."""
    alpha = 2.0 ** (1.0/3.0)
    w0 = -alpha / (2.0 - alpha)
    w1 =  1.0  / (2.0 - alpha)
    c1 = w1 / 2.0; c2 = (w0 + w1) / 2.0; c3 = c2; c4 = c1
    d1 = w1;       d2 = w0;         d3 = w1
    q = theta.copy()
    p_new = p.copy()
    for _ in range(L):
        # Yoshida 4-step sequence (position and momentum interleaving)
        if isinstance(beta, cp.ndarray):
            # For batch operation, multiply gradients by beta per chain
            p_new -= d1 * epsilon * (beta[:, None] * grad_U_func(q))
            q     += c1 * epsilon * p_new
            p_new -= d2 * epsilon * (beta[:, None] * grad_U_func(q))
            q     += c2 * epsilon * p_new
            p_new -= d3 * epsilon * (beta[:, None] * grad_U_func(q))
            q     += c3 * epsilon * p_new
            p_new -= d2 * epsilon * (beta[:, None] * grad_U_func(q))
            q     += c4 * epsilon * p_new
            p_new -= d1 * epsilon * (beta[:, None] * grad_U_func(q))
        else:
            # Single chain case
            p_new -= d1 * epsilon * beta * grad_U_func(q)
            q     += c1 * epsilon * p_new
            p_new -= d2 * epsilon * beta * grad_U_func(q)
            q     += c2 * epsilon * p_new
            p_new -= d3 * epsilon * beta * grad_U_func(q)
            q     += c3 * epsilon * p_new
            p_new -= d2 * epsilon * beta * grad_U_func(q)
            q     += c4 * epsilon * p_new
            p_new -= d1 * epsilon * beta * grad_U_func(q)
    return q, p_new
class HMC_GPU:
    def __init__(self, U, grad_U=None, integrator_name='leapfrog', epsilon=0.1, L=10):
        """
        Initialize HMC sampler for a given potential U (and grad_U).
        U and grad_U should accept and return CuPy arrays for GPU execution.
        """
        self.U_func = U            # GPU-compatible potential function
        self.grad_U_func = grad_U if grad_U is not None else approx_grad_gpu
        # Choose integrator function based on name
        name = integrator_name.lower().replace('-', '_')
        if name == 'leapfrog':
            self.integrator = leapfrog_step
        elif name == 'force_gradient':
            self.integrator = force_gradient_step
        elif name == 'yoshida4':
            self.integrator = yoshida4_step
        else:
            raise ValueError(f"Unknown integrator: {integrator_name}")
        self.epsilon = epsilon
        self.L = L

    def hmc_step(self, theta_current: cp.ndarray, beta=1.0):
        """
        Perform one HMC Metropolis step at inverse-temperature beta.
        Returns (theta_new, accepted_flag, delta_H).
        """
        # Sample random momentum from standard normal (on GPU)
        p0 = cp.random.randn(*theta_current.shape)  # same shape as theta
        # Simulate Hamiltonian dynamics using the integrator (on GPU)
        theta_prop, p_prop = self.integrator(theta_current, p0, self.epsilon, self.L,
                                            self.grad_U_func, beta)
        # Compute energies (on GPU)
        U0 = beta * self.U_func(theta_current)
        U1 = beta * self.U_func(theta_prop)
        K0 = 0.5 * cp.sum(p0 * p0, axis=-1)
        K1 = 0.5 * cp.sum(p_prop * p_prop, axis=-1)
        # Acceptance probability (log form for numerical stability)
        log_accept_prob = (U0 + K0) - (U1 + K1)
        # Accept or reject
        # Generate uniform random in [0,1) on GPU and compare in log domain
        rand = cp.random.rand(*log_accept_prob.shape)
        accept_mask = cp.log(rand) < log_accept_prob  # boolean mask (could be scalar or array)
        # Determine new state
        theta_new = cp.where(accept_mask[..., None], theta_prop, theta_current)
        # Compute ΔH for diagnostics: (H_new - H_old) if accepted, else 0
        delta_H = (U1 + K1) - (U0 + K0)
        if theta_new.ndim == 1:
            # Return scalar values for single chain
            accepted = bool(cp.asnumpy(accept_mask))  # convert single boolean to Python bool
            dH_val   = float(cp.asnumpy(delta_H * accept_mask)) if accepted else 0.0
            return theta_new, accepted, dH_val
        else:
            # For batch: return masks/arrays (accepted flags and ΔH array)
            return theta_new, accept_mask, cp.where(accept_mask, delta_H, cp.zeros_like(delta_H))
    
    def sample_chain(self, theta0: cp.ndarray, num_samples: int, burn_in: int = 0):
        """
        Run a single HMC chain starting from theta0 for num_samples iterations.
        Returns (thetas, accept_flags, delta_Hs) as CuPy arrays.
        """
        logger.info(f"Starting sample_chain: num_samples={num_samples}, burn_in={burn_in}")
        d = theta0.shape[-1]
        thetas    = cp.empty((num_samples, d), dtype=float)
        accepts   = cp.empty(num_samples, dtype=bool)
        delta_Hs  = cp.empty(num_samples, dtype=float)
        theta = theta0.copy()
        for t in range(num_samples):
            if t % 100 ==0:
                logger.debug(f"Sample {t+1}/{num_samples}...")
            theta, accepted, dH = self.hmc_step(theta)  # one step at beta=1.0 (default)
            thetas[t]   = theta
            accepts[t]  = accepted
            delta_Hs[t] = dH
        # Optionally discard burn-in samples:
        if burn_in > 0:
            thetas   = thetas[burn_in:]
            accepts  = accepts[burn_in:]
            delta_Hs = delta_Hs[burn_in:]
        return thetas, accepts, delta_Hs
    
    def hmc_pt(self, theta0s: cp.ndarray, betas: cp.ndarray, num_sweeps: int):
        """
        Parallel Tempering (PT) over multiple chains on GPU.
        Returns (history, swap_flags).
        """
        # calls the top‐level GPU function, using this sampler’s ε and L
        return parallel_tempering(
            theta0s,
            betas,
            num_sweeps,
            epsilon=self.epsilon,
            L=self.L
        )

    def smc(self, theta0: cp.ndarray, num_particles: int, num_intermediate: int):
        """
        Sequential Monte Carlo (Annealed Importance Sampling) on GPU.
        Returns (particles, logZ_estimate, ESS_history).
        """
        # calls the top‐level GPU function, using this sampler’s ε and L
        return sequential_monte_carlo(
            theta0,
            num_particles,
            num_intermediate,
            epsilon=self.epsilon,
            L=self.L
        )
    def hmc_sample(self, theta0: cp.ndarray, num_samples: int, burn_in: int = 0):
        """
        CPU‐style name for sample_chain.
        """
        return self.sample_chain(theta0, num_samples, burn_in)

    # alias SMC to hmc_smc for full API match
    def hmc_smc(self, theta0: cp.ndarray, num_particles: int, num_intermediate: int):
        """
        CPU‐style name for smc.
        """
        return self.smc(theta0, num_particles, num_intermediate)



def parallel_tempering(theta0s: cp.ndarray, betas: cp.ndarray, num_sweeps: int,
                       epsilon: float, L: int) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Run Parallel Tempering on GPU.
    theta0s: initial states for each temperature chain (shape: (m, d))
    betas: array of inverse temperatures (shape: (m,))
    Returns (history, swap_flags):
      - history: CuPy array of shape (num_sweeps, m, d) with the cold chain states at each sweep
      - swap_flags: CuPy boolean array of length num_sweeps indicating if a swap was accepted at that sweep.
    """
    m, d = theta0s.shape
    theta = theta0s.copy()
    history = cp.empty((num_sweeps, m, d), dtype=float)
    swap_flags = cp.zeros(num_sweeps, dtype=bool)
    # Prepare HMC sampler (using vectorized integrator with per-chain beta scaling)
    sampler = HMC_GPU(U_target, grad_U_target, integrator_name='leapfrog', epsilon=epsilon, L=L)
    for t in range(num_sweeps):
        # 1. One HMC update per chain (vectorized across all m chains on GPU)
        # We call sampler.hmc_step with the whole set of chains and a beta vector.
        theta, accepted_mask, _ = sampler.hmc_step(theta, beta=betas)
        # (theta is now updated states for all chains; accepted_mask is a length-m boolean array, not explicitly used here)
        # 2. Propose a swap of two adjacent temperature chains
        i = int(cp.asnumpy(cp.random.randint(0, m-1)))  # pick adjacent indices i and i+1 (convert to Python int)
        j = i + 1
        # Compute swap acceptance probability in log domain:
        θ_i, θ_j = theta[i], theta[j]
        b_i, b_j = betas[i], betas[j]
        # Compute energies (bringing small arrays to CPU for scalar operations for simplicity)
        E_i = float(cp.asnumpy(U_target(θ_i)))
        E_j = float(cp.asnumpy(U_target(θ_j)))
        log_swap_ratio = (b_i * E_i + b_j * E_j) - (b_i * E_j + b_j * E_i)
        if cp.random.rand() < cp.exp(log_swap_ratio):  # accept swap with probability exp(log_swap_ratio)
            # Swap the states between chain i and j
            temp = theta[i].copy()
            theta[i] = theta[j];  theta[j] = temp
            swap_flags[t] = True
        # Record the state of the cold chain (beta=max) after this sweep
        history[t, :] = theta
    return history, swap_flags


def sequential_monte_carlo(theta0: cp.ndarray, num_particles: int, num_intermediate: int,
                           epsilon: float, L: int) -> tuple[cp.ndarray, float, cp.ndarray]:
    """
    Run Sequential Monte Carlo (Annealed Importance Sampling) on GPU.
    Returns (particles, logZ_estimate, ESS_history).
      - particles: final particle states (num_particles, d) on GPU
      - logZ_estimate: log evidence estimate (float)
      - ESS_history: CuPy array of effective sample size at each stage.
    """
    d = theta0.shape[0]
    # Initialize particles (all start at theta0)
    particles = cp.tile(theta0, (num_particles, 1))  # shape (Np, d)
    log_weights = cp.zeros(num_particles)
    ESS_hist = cp.empty(num_intermediate, dtype=float)
    logZ_accum = 0.0
    beta_schedule = cp.linspace(0.0, 1.0, num_intermediate + 1)  # annealing schedule
    sampler = HMC_GPU(U_target, grad_U_target, integrator_name='leapfrog', epsilon=epsilon, L=L)
    for t in range(1, num_intermediate + 1):
        b_prev = float(beta_schedule[t-1]);  b_curr = float(beta_schedule[t])
        # 1. Weight update for this stage (vectorized over all particles)
        # Δlog-weight = -(β_current * U(x) - β_prev * U(x)) = -(β_curr - β_prev)*U(x)
        U_vals = U_target(particles)  # compute U for all particles at once (returns shape (Np,))
        log_weights += -(b_curr - b_prev) * U_vals
        # 2. Normalize weights and compute ESS
        maxw = cp.max(log_weights)
        w = cp.exp(log_weights - maxw)        # unnormalized weights (shifted for stability)
        w_norm = w / cp.sum(w)
        ESS = float(cp.asnumpy(1.0 / cp.sum(w_norm**2)))
        ESS_hist[t-1] = ESS
        logZ_accum += float(maxw + cp.log(cp.mean(w)))  # incremental log-evidence
        # 3. Resample particles multinomially according to w_norm
        idx = cp.random.choice(num_particles, size=num_particles, replace=True, p=cp.asnumpy(w_norm))
        # (Note: cp.random.choice currently requires probabilities on CPU for weighted case)
        particles = particles[idx]  # index on GPU; this gathers new particle set
        log_weights.fill(0.0)      # reset log-weights for next stage
        # 4. HMC jitter (one leapfrog trajectory) at β = b_curr for each particle (vectorized)
        particles, _, _ = sampler.hmc_step(particles, beta=b_curr)
    return particles, logZ_accum, ESS_hist


class EpsilonFinderGPU:
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Short run lengths for tuning (same as original)
        self.TUNE_HMC_ITERS = 500
        self.TUNE_PT_SWEEPS = 200
        self.TUNE_SMC_PARTS = 200
        self.TUNE_SMC_INTER = 10
    def _measure_hmc_accept(self, integrator_name: str, eps: float) -> float:
        """Run a short HMC chain and return average acceptance rate."""
        theta0 = cp.zeros(self.dimension)
        sampler = HMC_GPU(U_target, grad_U_target, integrator_name, epsilon=eps, L=HMC_NUM_STEPS)
        _, accepts, _ = sampler.sample_chain(theta0, num_samples=self.TUNE_HMC_ITERS)
        # accepts is a CuPy boolean array; take mean (True=1) to get acceptance probability
        return float(cp.asnumpy(cp.mean(accepts.astype(cp.float32))))
    def _measure_pt_accept(self, integrator_name: str, eps: float) -> float:
        """Run a short PT simulation and return average per-step acceptance across all chains."""
        m = 5
        betas = cp.linspace(0, 1, m)
        theta0s = cp.zeros((m, self.dimension))
        # Run PT for a few sweeps and track how many HMC proposals were accepted
        sampler = HMC_GPU(U_target, grad_U_target, integrator_name, epsilon=eps, L=HMC_NUM_STEPS)
        total_accepts = 0; total_HMC_steps = m * self.TUNE_PT_SWEEPS
        theta = theta0s.copy()
        for t in range(self.TUNE_PT_SWEEPS):
            # HMC updates for all chains (vectorized)
            theta, accept_mask, _ = sampler.hmc_step(theta, beta=betas)
            total_accepts += int(cp.asnumpy(cp.sum(accept_mask)))
            # swap step (we ignore swap acceptance in tuning, focusing on HMC accept rate)
            i = int(cp.asnumpy(cp.random.randint(0, m-1)))
            j = i+1
            θ_i, θ_j = theta[i].copy(), theta[j].copy()
            E_i = float(cp.asnumpy(U_target(θ_i)));  E_j = float(cp.asnumpy(U_target(θ_j)))
            log_swap = (betas[i]*E_i + betas[j]*E_j) - (betas[i]*E_j + betas[j]*E_i)
            if cp.random.rand() < cp.exp(log_swap):
                theta[i], theta[j] = θ_j, θ_i
        return total_accepts / total_HMC_steps
    def _measure_smc_accept(self, integrator_name: str, eps: float) -> float:
        """Run a short SMC pilot and return overall HMC acceptance rate during jitter steps."""
        # Use a small number of particles and intermediate steps for speed
        theta0 = cp.zeros(self.dimension)
        Np = self.TUNE_SMC_PARTS;  T = self.TUNE_SMC_INTER
        sampler = HMC_GPU(U_target, grad_U_target, integrator_name, epsilon=eps, L=HMC_NUM_STEPS)
        particles = cp.tile(theta0, (Np, 1))
        total_accepts = 0; total_steps = Np * T
        beta_schedule = cp.linspace(0, 1, T+1)
        logw = cp.zeros(Np)
        for t in range(1, T+1):
            b0 = float(beta_schedule[t-1]); b1 = float(beta_schedule[t])
            # update weights (not needed for accept rate, but complete the SMC procedure)
            U_vals = U_target(particles)
            logw += -(b1 - b0) * U_vals
            # resample (to simulate particle diversity; using same approach as in SMC function)
            w = cp.exp(logw - cp.max(logw)); w_norm = w / cp.sum(w)
            idx = cp.random.choice(Np, size=Np, replace=True, p=cp.asnumpy(w_norm))
            particles = particles[idx]; logw.fill(0.0)
            # HMC jitter at beta = b1 for all particles
            particles, accept_mask, _ = sampler.hmc_step(particles, beta=b1)
            total_accepts += int(cp.asnumpy(cp.sum(accept_mask)))
        return total_accepts / total_steps
    def tune_hmc_eps(self, integrator_name: str) -> float:
        grid = EPS_GRID_YOSHIDA if integrator_name=='yoshida4' else EPS_GRID_COMMON
        target = TARGET_ACC_YOSHIDA if integrator_name=='yoshida4' else TARGET_ACC_COMMON
        acc_rates = []
        for eps in cp.asnumpy(grid):  # iterate over grid values (as Python floats)
            acc = self._measure_hmc_accept(integrator_name, float(eps))
            acc_rates.append(acc)
        # Find epsilon with acceptance closest to target
        diffs = [abs(acc - target) for acc in acc_rates]
        best_idx = int(diffs.index(min(diffs)))
        return float(grid[best_idx])
    def tune_pt_eps(self, integrator_name: str) -> float:
        grid = EPS_GRID_YOSHIDA if integrator_name=='yoshida4' else EPS_GRID_COMMON
        target = TARGET_ACC_YOSHIDA if integrator_name=='yoshida4' else TARGET_ACC_COMMON
        acc_rates = [self._measure_pt_accept(integrator_name, float(eps)) for eps in cp.asnumpy(grid)]
        best_idx = int(np.argmin(np.abs(np.array(acc_rates) - target)))
        return float(grid[best_idx])
    def tune_smc_eps(self, integrator_name: str) -> float:
        grid = EPS_GRID_YOSHIDA if integrator_name=='yoshida4' else EPS_GRID_COMMON
        target = TARGET_ACC_YOSHIDA if integrator_name=='yoshida4' else TARGET_ACC_COMMON
        acc_rates = [self._measure_smc_accept(integrator_name, float(eps)) for eps in cp.asnumpy(grid)]
        best_idx = int(np.argmin(np.abs(np.array(acc_rates) - target)))
        return float(grid[best_idx])
    def all_eps(self) -> dict:
        """Return tuned epsilon for each (algorithm, integrator) combination."""
        tuned = {'HMC': {}, 'PT': {}, 'SMC': {}}
        for integ in ['leapfrog', 'force_gradient', 'yoshida4']:
            tuned['HMC'][integ] = self.tune_hmc_eps(integ)
            tuned['PT'][integ]  = self.tune_pt_eps(integ)
            tuned['SMC'][integ] = self.tune_smc_eps(integ)
        return tuned
def gelman_rubin(chains: cp.ndarray) -> float:
    """
    Compute Gelman–Rubin R̂ for an array of shape (M, N) or (M, N, d).
    If 2D (M chains of length N), returns scalar R̂.
    If 3D (M, N, d), computes R̂ for each dimension and returns the maximum.
    """
    if chains.ndim == 3:
        # Vector-valued chains: compute R̂ per dimension and return max
        M, N, d = chains.shape
        # Within-chain variances (per dimension)
        W = cp.mean(cp.var(chains, axis=1, ddof=1), axis=0)             # shape (d,)
        # Between-chain variance (per dimension)
        chain_means = cp.mean(chains, axis=1)                           # shape (M, d)
        B = N * cp.var(chain_means, axis=0, ddof=1)                     # shape (d,)
        V = ((N - 1)/N) * W + (1/N) * B
        R_dim = cp.sqrt(V / W)
        return float(cp.asnumpy(cp.max(R_dim)))  # return worst R̂ across dimensions
    else:
        M, N = chains.shape
        W = cp.mean(cp.var(chains, axis=1, ddof=1))
        chain_means = cp.mean(chains, axis=1)
        B = N * cp.var(chain_means, axis=0, ddof=1)
        V = ((N - 1)/N) * W + (1/N) * B
        R = cp.sqrt(V / W)
        return float(cp.asnumpy(R))

def mcse(values: cp.ndarray) -> float:
    """
    Monte Carlo standard error for the mean estimate of the given sample array.
    (If values is 1D, it treats it as a univariate sample; if 2D (N, d), it flattens each dimension.)
    """
    vals = values.reshape(-1)  # flatten
    # Standard error of the mean = std / sqrt(N)
    N = vals.size
    # ddof=1 for sample std (unbiased)
    std = cp.std(vals, ddof=1)
    return float(cp.asnumpy(std / cp.sqrt(N)))
import cupy as cp

# 1) Prior on θ
def prior_sampler():
    """Sample θ ~ N(mu,σ²I) on the GPU."""
    return cp.random.normal(loc=mu, scale=sigma, size=all_diffs.shape)

# 2) Forward‐simulator
def data_simulator(theta):
    """Simulate y = θ + ε, ε~N(0,σ²I) on the GPU."""
    return cp.random.normal(loc=theta, scale=sigma, size=theta.shape)

# 3) Summary for PPC
def summary_fn(data):
    """Summary statistic for PPC (mean)."""
    return cp.mean(data, axis=0)

# 4) Simulation‐Based Calibration
def sbc_gpu(prior_sampler, data_simulator, inference_fn,
            num_reps: int = 100, posterior_draws: int = 1000) -> cp.ndarray:
    """
    GPU‐accelerated Simulation‐Based Calibration.
    Returns an (num_reps × d) array of rank statistics.
    """
    # infer dimensionality
    θ0 = prior_sampler()                        
    y0 = data_simulator(θ0)                     
    post0 = inference_fn(y0, posterior_draws)    
    d = post0.shape[1]

    ranks = cp.empty((num_reps, d), dtype=cp.int32)
    for i in range(num_reps):
        θ_true = prior_sampler()                 
        y_obs  = data_simulator(θ_true)          
        posterior = inference_fn(y_obs, posterior_draws)  
        # marginal rank for each component
        less = posterior < θ_true[None, :]       
        ranks[i] = cp.sum(less, axis=0).astype(cp.int32)
    return ranks

# 5) Posterior Predictive Check
def ppc_gpu(observed_data: cp.ndarray,
            inference_fn,
            data_simulator,
            summary_fn,
            num_ppc_samples: int = 500,
            posterior_draws: int = 1000) -> dict:
    """
    GPU‐accelerated Posterior Predictive Check.
    Returns dict with observed summary, replicated summaries, and p‐value.
    """
    posterior = inference_fn(observed_data, posterior_draws)  
    obs_sum = summary_fn(observed_data)                       

    repl_summaries = cp.empty(num_ppc_samples, dtype=float)
    for i in range(num_ppc_samples):
        idx = int(cp.asnumpy(cp.random.randint(0, posterior_draws)))
        θ = posterior[idx]                    
        y_rep = data_simulator(θ)             
        repl_summaries[i] = summary_fn(y_rep) 

    p_val = float(cp.asnumpy(cp.mean(repl_summaries >= obs_sum)))
    return {
        'observed_summary': float(cp.asnumpy(obs_sum)),
        'replicated_summaries': repl_summaries,
        'p_value': p_val
    }

def run_minimal_chain_task_gpu(alg, integ, eps):
    """
    GPU version of run_minimal_chain_task:
      - alg in {'HMC','PT','SMC'}
      - integ in {'leapfrog','force_gradient','yoshida4'}
      - eps: tuned step‐size
    Returns (alg, integ, best_record) where best_record is
      (budget, R_hat, ESS, MCSE_mean, MCSE_var, best_L).
    """
    best_record = (None, None, None, None, None, None)
    logger.info(f"Starting minimal_chain")

    if alg == 'HMC':
        # === HMC shortest‐chain search ===
        logger.info(f"[RUN_START] alg={alg} integ={integ} eps={eps}")
        for L in range(1, MAX_HMC_STEPS + 1):
            # use your GPU “seeker” to find minimal N
            N, R, ESS, mM, mV = seeker_gpu.find_min_chain_length_hmc(integ, eps, L)  
            if N is None:
                continue

            failures = []
            if R   >= R_HAT_THRESH:  failures.append(f"R̂={R:.3f}")
            if ESS <= ESS_MIN:       failures.append(f"ESS={ESS:.1f}")
            if mM  >= MCSE_MEAN_TOL: failures.append(f"MCSE_mean={mM:.3f}")
            if mV  >= MCSE_VAR_TOL:  failures.append(f"MCSE_var={mV:.3f}")

            # SBC
            def inf(y_obs, posterior_draws):
                sampler = HMC_GPU(U_target, grad_U_target, integ, epsilon=eps, L=L)
                return sampler.sample_chain(y_obs, num_samples=N)[0]

            ranks = sbc_gpu(prior_sampler, data_simulator, inf,
                            num_reps=100, posterior_draws=200)
            ranks_flat = cp.asnumpy(ranks).ravel()
            _, p_sbc = kstest(cp.asnumpy(ranks_flat), 'uniform')
            if p_sbc < SBC_PVAL_THRESH:
                failures.append(f"SBC p={p_sbc:.3f}")

            # PPC
            ppc_res = ppc_gpu(
                observed_data=all_diffs,
                inference_fn=inf,
                data_simulator=data_simulator,
                summary_fn=summary_fn,
                num_ppc_samples=100,
                posterior_draws=200
            )
            obs_sum   = summary_fn(all_diffs)
            dist_ppc  = float(cp.mean(cp.abs(
                cp.asarray(ppc_res['replicated_summaries']) - obs_sum
            )))
            if dist_ppc > PPC_DIST_THRESH:
                failures.append("PPC failed")

            # record if passed
            if not failures:
                # keep the smallest N
                if best_record[0] is None or N < best_record[0]:
                    best_record = (N, R, ESS, mM, mV, L)

    elif alg == 'PT':
        # === Parallel Tempering shortest‐chain search ===
        for L in range(1, MAX_HMC_STEPS + 1):
            sweeps, R, ESS, mM, mV = seeker_gpu.find_min_chain_length_pt(integ, eps, L)  
            if sweeps is None:
                continue

            failures = []
            if R   >= R_HAT_THRESH:  failures.append(f"R̂={R:.3f}")
            if ESS <= ESS_MIN:       failures.append(f"ESS={ESS:.1f}")
            if mM  >= MCSE_MEAN_TOL: failures.append(f"MCSE_mean={mM:.3f}")
            if mV  >= MCSE_VAR_TOL:  failures.append(f"MCSE_var={mV:.3f}")

            def inf(y_obs,posterior_draws):
                sampler = HMC_GPU(U_target, grad_U_target, integ, epsilon=eps, L=L)
                betas = cp.linspace(0.0, 1.0, m)
                history, _ = sampler.hmc_pt(cp.zeros((m, D)), betas, sweeps)
                burn = int(sweeps * BURN_IN_FRAC)
                return history[burn:, 0, :]

            ranks = sbc_gpu(prior_sampler, data_simulator, inf,
                            num_reps=100, posterior_draws=200)
            ranks_flat = cp.asnumpy(ranks).ravel()
            _, p_sbc = kstest(cp.asnumpy(ranks_flat), 'uniform')
            if p_sbc < SBC_PVAL_THRESH:
                failures.append(f"SBC p={p_sbc:.3f}")

            ppc_res = ppc_gpu(
                observed_data=all_diffs,
                inference_fn=inf,
                data_simulator=data_simulator,
                summary_fn=summary_fn,
                num_ppc_samples=100,
                posterior_draws=200
            )
            obs_sum  = summary_fn(all_diffs)
            dist_ppc = float(cp.mean(cp.abs(
                cp.asarray(ppc_res['replicated_summaries']) - obs_sum
            )))
            if dist_ppc > PPC_DIST_THRESH:
                failures.append("PPC failed")

            if not failures:
                if best_record[0] is None or sweeps < best_record[0]:
                    best_record = (sweeps, R, ESS, mM, mV, L)

    else:  # 'SMC'
        # === SMC shortest‐chain search ===
        for L in range(1, MAX_HMC_STEPS + 1):
            Np, R, ESS, mM, mV = seeker_gpu.find_min_chain_length_smc(integ, eps, L)  
            if Np is None:
                continue

            failures = []
            if R   >= R_HAT_THRESH:  failures.append(f"R̂={R:.3f}")
            if ESS <= ESS_MIN:       failures.append(f"ESS={ESS:.1f}")
            if mM  >= MCSE_MEAN_TOL: failures.append(f"MCSE_mean={mM:.3f}")
            if mV  >= MCSE_VAR_TOL:  failures.append(f"MCSE_var={mV:.3f}")

            def inf(y_obs,posteror_draws):
                sampler = HMC_GPU(U_target, grad_U_target, integ, epsilon=eps, L=L)
                parts, _, _ = sampler.hmc_smc(y_obs, num_particles=Np, num_intermediate=NUM_SMC_INTER)
                return parts.reshape(-1, D)

            ranks = sbc_gpu(prior_sampler, data_simulator, inf,
                            num_reps=100, posterior_draws=200)
            
            ranks_flat = cp.asnumpy(ranks).ravel()
            _, p_sbc = kstest(cp.asnumpy(ranks_flat), 'uniform')
            if p_sbc < SBC_PVAL_THRESH:
                failures.append(f"SBC p={p_sbc:.3f}")

            ppc_res = ppc_gpu(
                observed_data=all_diffs,
                inference_fn=inf,
                data_simulator=data_simulator,
                summary_fn=summary_fn,
                num_ppc_samples=100,
                posterior_draws=200
            )
            obs_sum  = summary_fn(all_diffs)
            dist_ppc = float(cp.mean(cp.abs(
                cp.asarray(ppc_res['replicated_summaries']) - obs_sum
            )))
            if dist_ppc > PPC_DIST_THRESH:
                failures.append("PPC failed")

            if not failures:
                if best_record[0] is None or Np < best_record[0]:
                    best_record = (Np, R, ESS, mM, mV, L)

    return alg, integ, best_record

class OptimumSeekingGPU:
    """
    GPU‐flavored shortest‐chain search, mirroring OptimumSeeking:
      - find_min_chain_length_hmc
      - find_min_chain_length_pt
      - find_min_chain_length_smc
    plus the three diagnose_* methods.
    """

    def __init__(self,
                 U: Callable[[cp.ndarray], float],
                 grad_U: Callable[[cp.ndarray], cp.ndarray]):
        self.U      = U
        self.grad_U = grad_U

    @staticmethod
    def _compute_ess_np(x: np.ndarray) -> float:
        """
        CPU‐side ESS code (identical to OptimumSeeking.compute_ess).
        """
        N = len(x)
        if N < 2:
            return float(N)
        x = x - np.mean(x)
        var0 = np.dot(x, x) / N
        rho_sum = 0.0
        for lag in range(1, N // 2):
            num = np.dot(x[:N - lag], x[lag:])
            den = (N - lag) * var0
            if den <= 0:
                break
            rho = num / den
            if rho < 0.05:
                break
            rho_sum += rho
        return N / (1.0 + 2.0 * rho_sum)

    @staticmethod
    def compute_ess(samples: cp.ndarray) -> float:
        """
        ESS for a 1D CuPy array: copy to NumPy and call _compute_ess_np.
        """
        return OptimumSeekingGPU._compute_ess_np(cp.asnumpy(samples))

    def find_min_chain_length_hmc(
        self,
        integrator_name: str,
        eps: float,
        L: int
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:

        theta0s = cp.zeros((M, D))
        best = (None, None, None, None, None)

        for N in range(STEP_N, MAX_CHAIN_LENGTH + 1, STEP_N):
            burn_in = int(N * BURN_IN_FRAC)
            post     = N - burn_in

            # Run M independent GPU HMC chains
            chains = []
            for i in range(M):
                sampler = HMC_GPU(self.U, self.grad_U, integrator_name,
                                  epsilon=eps, L=L)
                thetas, _, _ = sampler.sample_chain(theta0s[i],
                                                    num_samples=N,
                                                    burn_in=burn_in)
                chains.append(thetas)
            chains = cp.stack(chains, axis=0)  # (M, post, DIM)

            # Gelman–Rubin
            R_hats = [
                gelman_rubin(cp.asnumpy(chains[:, :, d]))
                for d in range(D)
            ]
            R_max = max(R_hats)

            # ESS
            ESSs = [
                OptimumSeekingGPU.compute_ess(chains[:, :, d].ravel())
                for d in range(D)
            ]
            ESS_min = min(ESSs)

            # MCSE
            flat = chains.reshape(-1, D)
            MCSE_means = [
                mcse(cp.asnumpy(flat[:, d]))
                for d in range(D)
            ]
            MCSE_vars = [
                mcse(cp.asnumpy(flat[:, d] ** 2))
                for d in range(D)
            ]
            MCSE_mean_max = max(MCSE_means)
            MCSE_var_max  = max(MCSE_vars)

            if (R_max < R_HAT_THRESH and
                ESS_min > ESS_MIN and
                MCSE_mean_max < MCSE_MEAN_TOL and
                MCSE_var_max  < MCSE_VAR_TOL):
                return N, R_max, ESS_min, MCSE_mean_max, MCSE_var_max

        return None, None, None, None, None

    def find_min_chain_length_pt(
        self,
        integrator_name: str,
        eps: float,
        L: int
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:
        betas   = cp.linspace(0.0, 1.0, m)
        theta0s = cp.zeros((m, D))

        for sweeps in range(STEP_N, MAX_CHAIN_LENGTH + 1, STEP_N):
            burn_in = int(sweeps * BURN_IN_FRAC)

            # Parallel tempering on GPU
            history, _ = parallel_tempering(theta0s, betas,
                                            sweeps,
                                            epsilon=eps, L=L)
            post_history = history[burn_in:, :, :]
            chains = cp.transpose(post_history, (1,0,2))  # (m, post, DIM)

            # Gelman–Rubin
            R_hats = [
                gelman_rubin(cp.asnumpy(chains[:, :, d]))
                for d in range(D)
            ]
            R_max = max(R_hats)

            # ESS
            ESSs = [
                OptimumSeekingGPU.compute_ess(chains[:, :, d].ravel())
                for d in range(D)
            ]
            ESS_min = min(ESSs)

            # MCSE
            flat = chains.reshape(-1, D)
            MCSE_means = [mcse(cp.asnumpy(flat[:, d])) for d in range(D)]
            MCSE_vars  = [mcse(cp.asnumpy(flat[:, d] ** 2)) for d in range(D)]
            MCSE_mean_max = max(MCSE_means)
            MCSE_var_max  = max(MCSE_vars)

            if (R_max < R_HAT_THRESH and
                ESS_min > ESS_MIN and
                MCSE_mean_max < MCSE_MEAN_TOL and
                MCSE_var_max  < MCSE_VAR_TOL):
                return sweeps, R_max, ESS_min, MCSE_mean_max, MCSE_var_max

        return None, None, None, None, None

    def find_min_chain_length_smc(
        self,
        integrator_name: str,
        eps: float,
        L: int
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:
        theta0 = cp.zeros(D)

        for Np in range(STEP_PARTICLES, MAX_SMC_PARTICLES + 1, STEP_PARTICLES):
            parts, _, _ = sequential_monte_carlo(theta0,
                                                 Np,
                                                 NUM_SMC_INTER,
                                                 epsilon=eps,
                                                 L=L)  # (Np, DIM)
            # Gelman–Rubin isn’t applicable for SMC particles, so just set it to 1.0
            R_max = 1.0

            # ESS
            ESSs = [
                OptimumSeekingGPU.compute_ess(parts[:, d].ravel())
                for d in range(D)
            ]
            ESS_min = min(ESSs)

            # MCSE
            flat = parts.reshape(-1, D)
            MCSE_means = [mcse(cp.asnumpy(flat[:, d]))    for d in range(D)]
            MCSE_vars  = [mcse(cp.asnumpy(flat[:, d] ** 2)) for d in range(D)]
            MCSE_mean_max = max(MCSE_means)
            MCSE_var_max  = max(MCSE_vars)

            if (R_max < R_HAT_THRESH and
                ESS_min > ESS_MIN and
                MCSE_mean_max < MCSE_MEAN_TOL and
                MCSE_var_max  < MCSE_VAR_TOL):
                return Np, R_max, ESS_min, MCSE_mean_max, MCSE_var_max

        return None, None, None, None, None

    def diagnose_hmc(self,
                     integrator_name: str,
                     eps: float,
                     L: int
                    ) -> Tuple[float, float, float, float]:
        N   = MAX_CHAIN_LENGTH
        burn = int(N * BURN_IN_FRAC)
        theta0s = cp.zeros((M, D))

        # run M GPU chains
        chains = []
        for i in range(M):
            sampler = HMC_GPU(self.U, self.grad_U,
                              integrator_name, epsilon=eps, L=L)
            th, _, _ = sampler.sample_chain(theta0s[i],
                                            num_samples=N,
                                            burn_in=burn)
            chains.append(th)
        chains = cp.stack(chains, axis=0)  # (M, post, DIM)

        R_hats = [
            gelman_rubin(cp.asnumpy(chains[:, :, d]))
            for d in range(D)
        ]
        R_max = max(R_hats)

        ESSs = [
            OptimumSeekingGPU.compute_ess(chains[:, :, d].ravel())
            for d in range(D)
        ]
        ESS_min = min(ESSs)

        flat = chains.reshape(-1, D)
        MCSE_means = [mcse(cp.asnumpy(flat[:, d]))    for d in range(D)]
        MCSE_vars  = [mcse(cp.asnumpy(flat[:, d]**2)) for d in range(D)]
        MCSE_mean_max = max(MCSE_means)
        MCSE_var_max  = max(MCSE_vars)

        return R_max, ESS_min, MCSE_mean_max, MCSE_var_max

    def diagnose_pt(self,
                    integrator_name: str,
                    eps: float,
                    L: int
                   ) -> Tuple[float, float, float, float]:
        sweeps = MAX_CHAIN_LENGTH
        burn   = int(sweeps * BURN_IN_FRAC)
        m_temp = m
        theta0s = cp.zeros((m_temp, D))
        betas   = cp.linspace(0.0, 1.0, m_temp)

        chains = []
        for i in range(M):
            sampler = HMC_GPU(self.U, self.grad_U,
                              integrator_name, epsilon=eps, L=L)
            history, _ = sampler.hmc_pt(theta0s, betas,
                                        num_sweeps=sweeps)
            chains.append(history[burn:, 0, :])
        chains = cp.stack(chains, axis=0)  # (M, post, DIM)

        R_hats = [
            gelman_rubin(cp.asnumpy(chains[:, :, d]))
            for d in range(D)
        ]
        R_max = max(R_hats)

        ESSs = [
            OptimumSeekingGPU.compute_ess(chains[:, :, d].ravel())
            for d in range(D)
        ]
        ESS_min = min(ESSs)

        flat = chains.reshape(-1, D)
        MCSE_means = [mcse(cp.asnumpy(flat[:, d]))    for d in range(D)]
        MCSE_vars  = [mcse(cp.asnumpy(flat[:, d]**2)) for d in range(D)]
        MCSE_mean_max = max(MCSE_means)
        MCSE_var_max  = max(MCSE_vars)

        return R_max, ESS_min, MCSE_mean_max, MCSE_var_max

    def diagnose_smc(self,
                     integrator_name: str,
                     eps: float,
                     L: int
                    ) -> Tuple[float, float, float, float]:
        Np        = MAX_SMC_PARTICLES
        num_inter = NUM_SMC_INTER
        theta0    = cp.zeros(D)

        chains = []
        for i in range(M):
            sampler = HMC_GPU(self.U, self.grad_U,
                              integrator_name, epsilon=eps, L=L)
            parts, _, _ = sampler.smc(theta0,
                                      num_particles= Np,
                                      num_intermediate=num_inter)
            chains.append(parts)
        chains = cp.stack(chains, axis=0)  # (M, Np, DIMENSION)

        R_hats = [
            gelman_rubin(cp.asnumpy(chains[:, :, d]))
            for d in range(D)
        ]
        R_max = max(R_hats)

        ESSs = [
            OptimumSeekingGPU.compute_ess(chains[:, :, d].ravel())
            for d in range(D)
        ]
        ESS_min = min(ESSs)

        flat = chains.reshape(-1, D)
        MCSE_means = [mcse(cp.asnumpy(flat[:, d]))    for d in range(D)]
        MCSE_vars  = [mcse(cp.asnumpy(flat[:, d]**2)) for d in range(D)]
        MCSE_mean_max = max(MCSE_means)
        MCSE_var_max  = max(MCSE_vars)

        return R_max, ESS_min, MCSE_mean_max, MCSE_var_max


seeker_gpu = OptimumSeekingGPU(U_target, grad_U_target)
if __name__ == "__main__":
    # 1) Reproducibility
    import numpy as np
    np.random.seed(42)
    cp.random.seed(42)
    data_dim = vec_diffs.size
    # 2) Tune ε on the full D-dimensional problem
    print("\n=== Step 0a: Pre-flight ε-tuning (full D) ===\n")
    tuner_gpu = EpsilonFinderGPU(data_dim)
    tuned_eps = tuner_gpu.all_eps()
    print(f"→ Tuned ε’s (in ℝ^{D}):")
    for alg in ['HMC','PT','SMC']:
        print(f"  • {alg}:")
        for integ in ['leapfrog','force_gradient','yoshida4']:
            print(f"      – {integ:12s} → ε = {tuned_eps[alg][integ]:.3f}")
    print()

    # 3) Find minimal budgets
    algorithms  = ['HMC','PT','SMC']
    integrators = ['leapfrog','force_gradient','yoshida4']
    jobs = [
        (alg, integ, tuned_eps[alg][integ])
        for alg in algorithms
        for integ in integrators
    ]

    # Parallel execution of the GPU search
    from joblib import Parallel, delayed
    outputs = Parallel(n_jobs=-1, backend="threading", verbose=10)(
        delayed(run_minimal_chain_task_gpu)(alg, integ, eps)
        for alg, integ, eps in jobs
    )

    # 4) Collect & print
    results = {alg: {} for alg in algorithms}
    for alg, integ, record in outputs:
        results[alg][integ] = record

    for alg in algorithms:
        print(f"=== {alg} results ===")
        for integ in integrators:
            N, R, ESS, mM, mV, L = results[alg][integ]
            print(f"  {integ:15s} → N/sweeps={N}, R̂={R:.3f}, ESS={ESS:.1f}, "
                  f"MCSE_mean={mM:.3f}, MCSE_var={mV:.3f}, L={L}")
        print()
