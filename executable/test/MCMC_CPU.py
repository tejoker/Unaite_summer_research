#!/usr/bin/env python3
"""
optimal_hmc_nd.py

General‐D version of the “shortest‐chain” HMC/PT/SMC search, with an integrated
“online” ε‐tuner.  You only need to set DIMENSION = desired d, and the script will:

  1) Instantly tune ε_leapfrog, ε_force-gradient, ε_yoshida4 for that d.
  2) Run the “hybrid” stopping‐rule scan (HMC / PT / SMC) in ℝᵈ using those ε’s.
  3) Print out the final (N, ESS, var, mean, L) for each integrator × algorithm.

Everything remains dimension‐agnostic.  
"""

import math
import numpy as np
from typing import Callable, Tuple, Optional
import sys
from joblib import Parallel, delayed
from numba import njit
from scipy.stats import kstest



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
)

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

print(f"J'ai bien récupéré {all_diffs.size} valeurs de ΔY.")

# — 4) Vous pouvez maintenant passer all_diffs ou vec_diffs
#     à vos routines d’inférence (HMC, diagnostics, etc.)
#     Par exemple :
def U_target(x: np.ndarray) -> float:
    diff = x - vec_diffs
    return 0.5 * diff.dot(diff)

# … le reste de vos définitions HMC / OptimumSeeking etc.
             # taille D*T*(lag+1)
DIMENSION = vec_diffs.size

# Tolerance on sample mean (flattened)
MEAN_TOLERANCE = 0.05

# “Hybrid” stopping rule thresholds:
ESS_THRESHOLD = 0.5      # require ESS ≥ 50% of post‐burn draws (flattened)
VAR_TOLERANCE = 0.05     # require |empirical var – 3| ≤ 0.05

# Default number of integrator sub‐steps (L) when tuning ε
HMC_NUM_STEPS   = 5        

# Number of short HMC iterations used when tuning ε
TUNE_CHAIN      = 1000     

# Grid settings for tuning ε
base = 0.1 * (DIMENSION ** -0.25)
EPS_GRID_YOSHIDA = EPS_GRID_COMMON = np.linspace(base/10, base*10, 20)


TARGET_ACC_COMMON  = 0.7   # acceptance target for leapfrog & force-gradient
TARGET_ACC_YOSHIDA = 0.8   # acceptance target for Yoshida4

# Grid settings for exploring L
MAX_HMC_STEPS = 20         # try L in [1..20]

# We scan N ∈ {10,20,…,100} for HMC/PT
MAX_CHAIN_LENGTH = 1000
STEP_N           = 10

# Burn‐in fraction for HMC and PT
BURN_IN_FRAC = 0.2

# SMC: scan num_particles ∈ {10,20,…,100}
MAX_SMC_PARTICLES = 1000
STEP_PARTICLES    = 100

NUMBER_OF_INDEPENDANT_CHAINS = 4

NUM_SMC_INTER = 20
R_HAT_THRESH   = 1.1     # max allowable Gelman–Rubin
ESS_MIN        = 100     # min ESS per coordinate
MCSE_MEAN_TOL  = 0.01    # max MCSE for mean
MCSE_VAR_TOL   = 0.05    # max MCSE for variance
SBC_PVAL_THRESH = 0.05
PPC_DIST_THRESH = 0.1
# ──────────────────────────────────────────────────────────────────
# 1) Finite-difference wrapper for ∇U
# ──────────────────────────────────────────────────────────────────
def approx_grad(U: Callable[[np.ndarray], float],
                x: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    """Central-difference approximation of ∇U(x) in ℝᵈ."""
    d = x.shape[0]
    g = np.zeros(d)
    for i in range(d):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (U(xp) - U(xm)) / (2 * eps)
    return g

def make_grad(U: Callable[[np.ndarray], float],
              grad_U: Optional[Callable[[np.ndarray], np.ndarray]] = None
             ) -> Callable[[np.ndarray], np.ndarray]:
    """
    If the user supplies an analytic `grad_U`, return it;
    otherwise return a wrapper around `approx_grad(U, ·)`.
    """
    return grad_U if grad_U is not None else (lambda x: approx_grad(U, x))

# ──────────────────────────────────────────────────────────────────
# 2) Your black-box log-density U_target (replace body with your model)
# ──────────────────────────────────────────────────────────────────
# draw a random mean vector
# test.py (après avoir calculé vec_diffs et DIMENSION)
def U_target(x: np.ndarray) -> float:
    """
    Énergie : distance quadratique entre x et le vecteur de différences vec_diffs.
    """
    diff = x - vec_diffs
    return 0.5 * diff.dot(diff)

# Si vous voulez un gradient analytique :
def user_grad_U_target(x: np.ndarray) -> np.ndarray:
    return x - vec_diffs

grad_U_target = make_grad(U_target, user_grad_U_target)


# ——— a true Numba finite‐difference gradient ———
@njit
def approx_grad_jit(U_fn, x, eps: float = 1e-5):
    d = x.shape[0]
    g = np.empty(d, np.float64)
    for i in range(d):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (U_fn(xp) - U_fn(xm)) / (2.0 * eps)
    return g

# ——— pure-Python integrators for the fallback path ———
def leapfrog_py(theta, p, epsilon, L, grad_U_fn):
    q = theta.copy()
    p = p.copy()
    p -= 0.5 * epsilon * grad_U_fn(q)
    for _ in range(L):
        q += epsilon * p
        if _ != L-1:
            p -= epsilon * grad_U_fn(q)
    p -= 0.5 * epsilon * grad_U_fn(q)
    return q, p

def force_gradient_py(theta, p, epsilon, L, grad_U_fn):
    # simple “gradient-forced” 2nd-order integrator:
    q = theta.copy()
    p = p.copy()
    for _ in range(L):
        q += epsilon * p
        p -= epsilon * grad_U_fn(q)
    return q, p

def yoshida4_py(theta, p, epsilon, L, grad_U_fn):
    # compute 4th-order Suzuki–Yoshida coefficients on the fly
    alpha = 2.0 ** (1.0 / 3.0)
    w0 = -alpha / (2.0 - alpha)
    w1 =  1.0  / (2.0 - alpha)
    # position-half, full, half splits:
    c1 =  w1 / 2.0
    c2 = (w0 + w1) / 2.0
    c3 =  c2
    c4 =  c1
    # momentum kicks:
    d1 =  w1
    d2 =  w0
    d3 =  w1

    q = theta.copy()
    p_new = p.copy()

    for _ in range(L):
        # kick, drift, kick sequence
        p_new -= d1 * epsilon * grad_U_fn(q)
        q     += c1 * epsilon * p_new
        p_new -= d2 * epsilon * grad_U_fn(q)
        q     += c2 * epsilon * p_new
        p_new -= d3 * epsilon * grad_U_fn(q)
        q     += c3 * epsilon * p_new
        p_new -= d2 * epsilon * grad_U_fn(q)
        q     += c4 * epsilon * p_new
        p_new -= d1 * epsilon * grad_U_fn(q)

    return q, p_new



# ================================
# 3. “ONLINE” ε‐TUNER CLASS (FULL‐D, HMC ∕ PT ∕ SMC)
# ================================
class EpsilonFinder:
    """
    Given a target dimension d, this class will “grid‐search” ε for each
    (algorithm, integrator) pair.  It returns three nested dicts:
      - tuned_eps['HMC'][integrator]
      - tuned_eps['PT' ][integrator]
      - tuned_eps['SMC'][integrator]

    Internally, it runs a short HMC / PT / SMC run at each ε and measures
    the average HMC‐step acceptance.  It then picks the ε whose acceptance
    is closest to the desired target (0.70 for leapfrog/force‐gradient, 0.80 for yoshida4).
    """
    def __init__(self, dimension: int):
        self.dimension = dimension

        # For HMC tuning, run a short chain of these many iterations
        self.TUNE_HMC_ITERS   = 500

        # For PT tuning, run this many sweeps (each sweep = one HMC per chain + one swap attempt)
        self.TUNE_PT_SWEEPS   = 200

        # For SMC tuning, use this many particles and intermediate stages
        self.TUNE_SMC_PARTS   = 200  # short enough to be quick
        self.TUNE_SMC_INTER   = 10   # number of annealing stages for pilot run

    def _measure_hmc_accept(self,
                            integrator_name: str,
                            eps: float,
                            L: int,
                            num_iters: int) -> float:
        """
        Run a short full‐d HMC chain of length `num_iters`, return
        average acceptance rate (fraction of accepted proposals).
        """
        theta0 = np.zeros(self.dimension)
        sampler = HMC(U_target,
                      None,
                      integrator_name,
                      eps, L)
        _, accepts, _ = sampler.hmc_sample(theta0, num_samples=num_iters)
        return np.mean(accepts)

    def _measure_pt_accept(self,
                           integrator_name: str,
                           eps: float,
                           L: int,
                           num_sweeps: int) -> float:
        """
        Run a short PT chain of `num_sweeps` sweeps (m=5 temperatures),
        and return the average HMC‐step acceptance across all chains.
        """
        m = 5
        betas = np.linspace(0.0, 1.0, m)
        thetas = np.zeros((m, self.dimension))
        total_accepts = 0

        # normalize integrator name and pick the pure-Python stepper
        name = integrator_name.lower().replace('-', '_')
        pure_steps = {
            'leapfrog':       leapfrog_py,
            'force_gradient': force_gradient_py,
            'yoshida4':       yoshida4_py,
        }
        try:
            stepper_py = pure_steps[name]
        except KeyError:
            raise ValueError(
                f"Unknown integrator '{integrator_name}' in PT—"
                f" expected one of: {', '.join(pure_steps.keys())}"
            )

        # build m independent HMC samplers in Python‐only mode
        samplers = []
        for _ in betas:
            s = HMC(U_target, grad_U_target, integrator_name, eps, L)
            s._use_compiled  = False
            s.stepper_py     = stepper_py
            s.U_fn           = s.U_py
            s.grad_U_fn      = s.grad_U_py
            samplers.append(s)

        for _ in range(num_sweeps):
            # 1) one HMC step per chain
            for i, s in enumerate(samplers):
                new_theta, accepted, _ = s.hmc_step(thetas[i])
                total_accepts += int(accepted)
                thetas[i] = new_theta

            # 2) random swap (swap-acceptance ignored)
            i0 = RNG.randint(0, m - 1)
            i1 = i0 + 1
            b0, b1 = betas[i0], betas[i1]
            θ0, θ1 = thetas[i0], thetas[i1]

            e00 = b0 * U_target(θ0)
            e01 = b0 * U_target(θ1)
            e10 = b1 * U_target(θ0)
            e11 = b1 * U_target(θ1)

            log_ratio = (e00 + e11) - (e01 + e10)
            if math.log(RNG.rand()) < log_ratio:
                thetas[i0], thetas[i1] = θ1.copy(), θ0.copy()

        return total_accepts / float(m * num_sweeps)


    def _measure_smc_accept(self,
                             integrator_name: str,
                             eps: float,
                             L: int,
                             num_particles: int,
                             num_inter: int) -> float:
        """
        Run a short SMC/AIS pilot and report the final-stage HMC-jitter acceptance rate.
        """
        theta0 = np.zeros(self.dimension)
        betas = np.linspace(0.0, 1.0, num_inter + 1)
        particles = np.tile(theta0[None, :], (num_particles, 1))
        logw = np.zeros(num_particles)
        total_acc = 0
        total_steps = num_particles * num_inter

        # normalize integrator name and pick the pure-Python stepper
        name = integrator_name.lower().replace('-', '_')
        pure_steps = {
            'leapfrog':       leapfrog_py,
            'force_gradient': force_gradient_py,
            'yoshida4':       yoshida4_py,
        }
        try:
            stepper_py = pure_steps[name]
        except KeyError:
            raise ValueError(
                f"Unknown integrator '{integrator_name}' in SMC—"
                f" expected one of: {', '.join(pure_steps.keys())}"
            )

        # single Python‐only sampler for jitter
        sampler = HMC(U_target, grad_U_target, integrator_name, eps, L)
        sampler._use_compiled = False
        sampler.stepper_py    = stepper_py
        sampler.U_fn          = sampler.U_py
        sampler.grad_U_fn     = sampler.grad_U_py

        for t in range(1, len(betas)):
            b0, b1 = betas[t - 1], betas[t]

            # incremental weights
            U0 = b0 * np.array([U_target(x) for x in particles])
            U1 = b1 * np.array([U_target(x) for x in particles])
            logw += -(U1 - U0)

            # resample
            maxw     = np.max(logw)
            w_unnorm = np.exp(logw - maxw)
            w_norm   = w_unnorm / np.sum(w_unnorm)
            idxs     = RNG.choice(num_particles, num_particles, True, w_norm)
            particles = particles[idxs]
            logw.fill(0.0)

            # jitter each particle
            for i in range(num_particles):
                new_theta, accepted, _ = sampler.hmc_step(particles[i])
                total_acc += int(accepted)
                particles[i] = new_theta

        return total_acc / float(total_steps)

    def tune_hmc_eps(self, integrator_name: str) -> float:
        """
        Find ε for vanilla HMC (in ℝ^d) by measuring
        average acceptance on a short chain, parallelized over ε.
        """
        # pick grid and target
        if integrator_name == 'yoshida4':
            eps_grid, target_acc = EPS_GRID_YOSHIDA, TARGET_ACC_YOSHIDA
        else:
            eps_grid, target_acc = EPS_GRID_COMMON, TARGET_ACC_COMMON

        # measure acceptance on each ε in parallel
        acc_rates = Parallel(n_jobs=-1)(
            delayed(self._measure_hmc_accept)(
                integrator_name,
                eps,
                L=HMC_NUM_STEPS,
                num_iters=self.TUNE_HMC_ITERS
            )
            for eps in eps_grid
        )

        # choose the ε whose acceptance is closest to the target
        diffs = np.abs(np.array(acc_rates) - target_acc)
        best_idx = int(np.argmin(diffs))
        return float(eps_grid[best_idx])


    def tune_pt_eps(self, integrator_name: str) -> float:
        """
        Find ε for PT (in ℝ^d) by measuring average within‐chain HMC acceptance
        on a short PT run, parallelized over ε.
        """
        # pick grid and target
        if integrator_name == 'yoshida4':
            eps_grid, target_acc = EPS_GRID_YOSHIDA, TARGET_ACC_YOSHIDA
        else:
            eps_grid, target_acc = EPS_GRID_COMMON, TARGET_ACC_COMMON

        # measure acceptance on each ε in parallel
        acc_rates = Parallel(n_jobs=-1)(
            delayed(self._measure_pt_accept)(
                integrator_name,
                eps,
                L=HMC_NUM_STEPS,
                num_sweeps=self.TUNE_PT_SWEEPS
            )
            for eps in eps_grid
        )

        # choose the ε whose acceptance is closest to the target
        diffs = np.abs(np.array(acc_rates) - target_acc)
        best_idx = int(np.argmin(diffs))
        return float(eps_grid[best_idx])


    def tune_smc_eps(self, integrator_name: str) -> float:
        """
        Find ε for SMC (in ℝ^d) by measuring average HMC‐jitter acceptance
        on a short SMC run, parallelized over ε.
        """
        # pick grid and target
        if integrator_name == 'yoshida4':
            eps_grid, target_acc = EPS_GRID_YOSHIDA, TARGET_ACC_YOSHIDA
        else:
            eps_grid, target_acc = EPS_GRID_COMMON, TARGET_ACC_COMMON

        # measure acceptance on each ε in parallel
        acc_rates = Parallel(n_jobs=-1)(
            delayed(self._measure_smc_accept)(
                integrator_name,
                eps,
                L=HMC_NUM_STEPS,
                num_particles=self.TUNE_SMC_PARTS,
                num_inter=self.TUNE_SMC_INTER
            )
            for eps in eps_grid
        )

        # choose the ε whose acceptance is closest to the target
        diffs = np.abs(np.array(acc_rates) - target_acc)
        best_idx = int(np.argmin(diffs))
        return float(eps_grid[best_idx])


    def all_eps(self) -> dict:
        """
        Returns a nested dict of the form:
          tuned_eps = {
            'HMC': { 'leapfrog': …, 'force-gradient': …, 'yoshida4': … },
            'PT' : { 'leapfrog': …, 'force-gradient': …, 'yoshida4': … },
            'SMC': { 'leapfrog': …, 'force-gradient': …, 'yoshida4': … }
          }
        """
        tuned = {'HMC': {}, 'PT': {}, 'SMC': {}}
        for name in ['leapfrog', 'force-gradient', 'yoshida4']:
            tuned['HMC'][name] = self.tune_hmc_eps(name)
            tuned['PT' ][name] = self.tune_pt_eps(name)
            tuned['SMC'][name] = self.tune_smc_eps(name)
        return tuned



# ================================
# 5. INTEGRATORS IN ℝᵈ (static methods)
# ================================


# ===== Integrator class with compiled methods =====
class Integrator:
    @staticmethod
    def leapfrog(theta, p, epsilon, L, grad_U_fn):
        return leapfrog_step(theta, p, epsilon, L, grad_U_fn)

    @staticmethod
    def force_gradient(theta, p, epsilon, L, grad_U_fn):
        return force_gradient_step(theta, p, epsilon, L, grad_U_fn)

    @staticmethod
    def yoshida4(theta, p, epsilon, L, grad_U_fn):
        return yoshida4_step(theta, p, epsilon, L, grad_U_fn)

# ===== Numba-compiled core integrators =====n
@njit
def leapfrog_step(theta, p, epsilon, L, grad_U_fn):
    q = theta.copy()
    momentum = p.copy()
    momentum -= 0.5 * epsilon * grad_U_fn(q)
    for i in range(L):
        q += epsilon * momentum
        if i != L - 1:
            momentum -= epsilon * grad_U_fn(q)
    momentum -= 0.5 * epsilon * grad_U_fn(q)
    return q, momentum

@njit
def force_gradient_step(theta, p, epsilon, L, grad_U_fn):
    theta_new = theta.copy()
    p_new = p.copy()
    for _ in range(L):
        g = grad_U_fn(theta_new)
        p_half = p_new - 0.5 * epsilon * g
        # Hessian-vector approx
        delta = 1e-5
        plus = grad_U_fn(theta_new + delta * p_half)
        minus = grad_U_fn(theta_new - delta * p_half)
        Hv = (plus - minus) / (2.0 * delta)
        # update
        theta_new = theta_new + epsilon * p_half + (epsilon**3 / 24.0) * Hv
        p_new = p_half - 0.5 * epsilon * grad_U_fn(theta_new)
    return theta_new, p_new

@njit
def yoshida4_step(theta, p, epsilon, L, grad_U_fn):
    alpha = 2.0 ** (1.0 / 3.0)
    w0 = -alpha / (2.0 - alpha)
    w1 = 1.0 / (2.0 - alpha)
    c1 = w1 / 2.0
    c2 = (w0 + w1) / 2.0
    c3 = c2
    c4 = c1
    d1 = w1
    d2 = w0
    d3 = w1
    theta_new = theta.copy()
    p_new = p.copy()
    for _ in range(L):
        p_new -= d1 * epsilon * grad_U_fn(theta_new)
        theta_new += c1 * epsilon * p_new
        p_new -= d2 * epsilon * grad_U_fn(theta_new)
        theta_new += c2 * epsilon * p_new
        p_new -= d3 * epsilon * grad_U_fn(theta_new)
        theta_new += c3 * epsilon * p_new
        p_new -= d2 * epsilon * grad_U_fn(theta_new)
        theta_new += c4 * epsilon * p_new
        p_new -= d1 * epsilon * grad_U_fn(theta_new)
    return theta_new, p_new

# ===== HMC core and fallback routines =====
@njit
def hmc_step_core(theta_current, epsilon, L, U_fn, grad_U_fn, stepper):
    p0 = np.random.randn(theta_current.shape[0])
    theta_prop, p_prop = stepper(theta_current, p0, epsilon, L, grad_U_fn)
    H_old = U_fn(theta_current) + 0.5 * np.dot(p0, p0)
    H_new = U_fn(theta_prop) + 0.5 * np.dot(p_prop, p_prop)
    deltaH = H_new - H_old
    accept_prob = math.exp(-deltaH)
    if np.random.rand() < accept_prob:
        return theta_prop, True, deltaH
    else:
        return theta_current.copy(), False, deltaH

@njit
def hmc_sample_core(theta0, num_samples, epsilon, L, U_fn, grad_U_fn, stepper):
    dim = theta0.shape[0]
    thetas = np.zeros((num_samples, dim))
    accepts = np.zeros(num_samples, np.bool_)
    deltaHs = np.zeros(num_samples, np.float64)
    theta = theta0.copy()
    for i in range(num_samples):
        theta, acc, dH = hmc_step_core(theta, epsilon, L, U_fn, grad_U_fn, stepper)
        thetas[i] = theta
        accepts[i] = acc
        deltaHs[i] = dH
    return thetas, accepts, deltaHs
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def batch_hmc_minimal(theta0s, seeds, num_samples, burn_in,
                      epsilon, L, U_fn, grad_U_fn, stepper):
    """
    Run M independent HMC chains of length num_samples (with burn_in),
    all inside one njit(parallel=True) call.
    - theta0s: shape (M, d) array of initial states
    - seeds:   shape (M,) array of int seeds
    Returns: array of shape (M, num_samples-burn_in, d)
    """
    M, d = theta0s.shape
    post = num_samples - burn_in
    out = np.empty((M, post, d), np.float64)

    for i in prange(M):
        # reseed the RNG inside Numba
        np.random.seed(seeds[i])
        # run one chain
        thetas, accepts, _ = hmc_sample_core(
            theta0s[i], num_samples,
            epsilon, L,
            U_fn, grad_U_fn,
            stepper
        )
        # copy the post–burn_in samples into the output
        for t in range(burn_in, num_samples):
            out[i, t - burn_in, :] = thetas[t]
    return out
import numpy as np
from numba import njit, prange

# ——— β‐aware Numba‐compiled integrators ———

@njit
def leapfrog_step_beta(theta, p, epsilon, L, beta, U_fn, grad_U_fn):
    q = theta.copy()
    momentum = p.copy()
    # half‐kick
    momentum -= 0.5 * epsilon * beta * grad_U_fn(q)
    for _ in range(L):
        q += epsilon * momentum
        if _ != L - 1:
            momentum -= epsilon * beta * grad_U_fn(q)
    momentum -= 0.5 * epsilon * beta * grad_U_fn(q)
    return q, momentum

@njit
def force_gradient_step_beta(theta, p, epsilon, L, beta, U_fn, grad_U_fn):
    theta_new = theta.copy()
    p_new     = p.copy()
    for _ in range(L):
        # half‐step momentum
        g = beta * grad_U_fn(theta_new)
        p_half = p_new - 0.5 * epsilon * g

        # Hessian‐vector approx with β included
        delta = 1e-5
        plus  = beta * grad_U_fn(theta_new + delta * p_half)
        minus = beta * grad_U_fn(theta_new - delta * p_half)
        Hv    = (plus - minus) / (2.0 * delta)

        theta_new = theta_new + epsilon * p_half + (epsilon**3 / 24.0) * Hv
        p_new     = p_half - 0.5 * epsilon * beta * grad_U_fn(theta_new)
    return theta_new, p_new

@njit
def yoshida4_step_beta(theta, p, epsilon, L, beta, U_fn, grad_U_fn):
    alpha = 2.0 ** (1.0 / 3.0)
    w0 = -alpha / (2.0 - alpha)
    w1 =  1.0  / (2.0 - alpha)
    c1 =  w1 / 2.0; c2 = (w0 + w1) / 2.0; c3 = c2; c4 = c1
    d1 =  w1;       d2 =  w0;       d3 = w1

    q = theta.copy()
    pm = p.copy()
    for _ in range(L):
        pm -= d1 * epsilon * beta * grad_U_fn(q)
        q  += c1 * epsilon * pm
        pm -= d2 * epsilon * beta * grad_U_fn(q)
        q  += c2 * epsilon * pm
        pm -= d3 * epsilon * beta * grad_U_fn(q)
        q  += c3 * epsilon * pm
        pm -= d2 * epsilon * beta * grad_U_fn(q)
        q  += c4 * epsilon * pm
        pm -= d1 * epsilon * beta * grad_U_fn(q)
    return q, pm

# ——— β‐aware HMC step core ———

@njit
def hmc_step_core_beta(theta_current, epsilon, L, beta,
                       U_fn, grad_U_fn, stepper_beta):
    # Sample momentum
    p0 = np.random.randn(theta_current.shape[0])
    # One integrator pass with β baked in
    theta_prop, p_prop = stepper_beta(
        theta_current, p0, epsilon, L, beta, U_fn, grad_U_fn
    )
    # Compute acceptance
    U0 = beta * U_fn(theta_current)
    U1 = beta * U_fn(theta_prop)
    K0 = 0.5 * np.dot(p0, p0)
    K1 = 0.5 * np.dot(p_prop, p_prop)
    log_accept_prob = (U0 + K0) - (U1 + K1)
    if np.log(np.random.rand()) < log_accept_prob:
        return theta_prop, True, (U1+K1) - (U0+K0)
    else:
        return theta_current, False, 0.0

# ——— Fully‐compiled PT batch ———

@njit(parallel=True)
def batch_pt_minimal(theta0s, seeds, num_sweeps,
                     epsilon, L, betas,
                     U_fn, grad_U_fn, stepper_beta):
    """
    M repeats of an m‐chain Parallel Tempering, all in one njit(parallel=True) blob.
    Returns shape (M, num_sweeps, dim) of the cold‐chain trace.
    """
    M, d = theta0s.shape
    m    = betas.size
    out  = np.empty((M, num_sweeps, d), np.float64)

    for i in prange(M):
        np.random.seed(seeds[i])
        # initialize m chains at theta0s[i]
        thetas = np.zeros((m, d))
        for t in range(num_sweeps):
            # within‐chain HMC with each β
            for j in range(m):
                thetas[j], _, _ = hmc_step_core_beta(
                    thetas[j], epsilon, L, betas[j],
                    U_fn, grad_U_fn, stepper_beta
                )
            # propose adjacent swap
            k   = np.random.randint(0, m-1)
            θk  = thetas[k];   θk1 = thetas[k+1]
            βk  = betas[k];    βk1 = betas[k+1]
            e00 = βk  * U_fn(θk)
            e01 = βk  * U_fn(θk1)
            e10 = βk1 * U_fn(θk)
            e11 = βk1 * U_fn(θk1)
            if np.log(np.random.rand()) < (e00+e11) - (e01+e10):
                thetas[k], thetas[k+1] = θk1, θk
            # record only the cold‐chain state
            out[i, t] = thetas[0]
    return out

# ——— Fully‐compiled SMC batch ———

@njit(parallel=True)
def batch_smc_minimal(theta0, seeds, num_particles, num_inter,
                      epsilon, L,
                      U_fn, grad_U_fn, stepper_beta):
    """
    Returns an array of shape (M, Np, DIM) containing the
    final-stage particles for each of the M replicates.
    """
    M = seeds.size
    d = theta0.size
    out = np.empty((M, num_particles, d), np.float64)

    for i in prange(M):
        np.random.seed(seeds[i])
        # manual tiling
        particles = np.empty((num_particles, d), np.float64)
        for k in range(num_particles):
            particles[k, :] = theta0
        logw = np.zeros(num_particles, np.float64)

        for t in range(1, num_inter+1):
            b0 = (t-1) / num_inter
            b1 =  t     / num_inter
            # weight update
            for k in range(num_particles):
                logw[k] += -(b1 * U_fn(particles[k]) - b0 * U_fn(particles[k]))
            # normalize & resample
            maxw  = np.max(logw)
            w_unn = np.exp(logw - maxw)
            w_norm= w_unn / np.sum(w_unn)
            # build CDF
            cdf = np.empty(num_particles, np.float64)
            c = 0.0
            for k in range(num_particles):
                c += w_norm[k]
                cdf[k] = c
            # multinomial draw
            newp = np.empty_like(particles)
            for k in range(num_particles):
                u = np.random.rand()
                idx = 0
                while cdf[idx] < u:
                    idx += 1
                newp[k] = particles[idx]
            particles = newp
            logw.fill(0.0)
            # jitter with HMC at β=1
            for k in range(num_particles):
                particles[k], _, _ = hmc_step_core_beta(
                    particles[k], epsilon, L, 1.0,
                    U_fn, grad_U_fn, stepper_beta
                )
        # record final particles
        for k in range(num_particles):
            out[i, k, :] = particles[k]
    return out



# Pure-Python fallbacks

import numpy as np
import math

def hmc_step_py(theta_current, epsilon, L, U_fn, grad_U_fn, stepper):
    p0 = np.random.randn(theta_current.shape[0])
    theta_prop, p_prop = stepper(theta_current, p0, epsilon, L, grad_U_fn)

    H_old = U_fn(theta_current) + 0.5 * np.dot(p0, p0)
    H_new = U_fn(theta_prop)  + 0.5 * np.dot(p_prop, p_prop)
    deltaH = H_new - H_old

    # --- PATCH BEGINS HERE ---
    try:
        accept_prob = math.exp(-deltaH)
    except OverflowError:
        # exp(-deltaH) overflow means deltaH is huge negative → underflow to 0
        accept_prob = 0.0
    # --- PATCH ENDS HERE ---

    if np.random.rand() < accept_prob:
        return theta_prop, True,  deltaH
    else:
        return theta_current.copy(), False, deltaH



def hmc_sample_py(theta0, num_samples, epsilon, L, U_fn, grad_U_fn, stepper):
    dim = theta0.shape[0]
    thetas = np.zeros((num_samples, dim))
    accepts = np.zeros(num_samples, np.bool_)
    deltaHs = np.zeros(num_samples, np.float64)
    theta = theta0.copy()
    for i in range(num_samples):
        theta, acc, dH = hmc_step_py(theta, epsilon, L, U_fn, grad_U_fn, stepper)
        thetas[i] = theta
        accepts[i] = acc
        deltaHs[i] = dH
    return thetas, accepts, deltaHs

# ===== HMC wrapper =====

class HMC:
    """
    Python wrapper around numba‐compiled HMC core.
    Methods run in Python, but inner steps are fast.
    """
    def __init__(self, U, grad_U=None,
                 integrator_name='leapfrog',
                 epsilon=0.1, L=10):
        # store pure‐Python callables
        self.U_py      = U
        self.grad_U_py = grad_U if grad_U is not None else (lambda x: approx_grad(U, x))

        # pick Python integrator for fallback
        name = integrator_name.lower().replace('-', '_')
        try:
            self.stepper_py = getattr(Integrator, name)
        except AttributeError:
            raise ValueError(f"Unknown integrator: {integrator_name}. "
                             "Available: leapfrog, force_gradient, yoshida4.")

        # HMC hyperparameters
        self.epsilon = epsilon
        self.L       = L

        # ——— ALWAYS compile U into u_fn ———
        u_fn = njit(U)
        self.U_fn = u_fn

        # compile analytic grad_U or wire up FD fallback
        if grad_U is not None:
            self.grad_U_fn = njit(grad_U)
        else:
            @njit
            def grad_U_fallback(x):
                return approx_grad_jit(u_fn, x)
            self.grad_U_fn = grad_U_fallback

        # map to the numba‐compiled integrator steps
        compiled_integrators = {
            'leapfrog':       leapfrog_step,
            'force_gradient': force_gradient_step,
            'yoshida4':       yoshida4_step,
        }
        try:
            self.stepper_compiled = compiled_integrators[name]
        except KeyError:
            raise ValueError(f"Unknown integrator: {integrator_name}. "
                             "Available: leapfrog, force_gradient, yoshida4.")

        # always use the compiled pipeline (now that grad_U_fn is pure-Numba)
        self._use_compiled = True

    def hmc_step(self, theta_current):
        if self._use_compiled:
            return hmc_step_core(
                theta_current,
                self.epsilon,
                self.L,
                self.U_fn,
                self.grad_U_fn,
                self.stepper_compiled
            )
        else:
            return hmc_step_py(
                theta_current,
                self.epsilon,
                self.L,
                self.U_fn,
                self.grad_U_fn,
                self.stepper_py
            )

    def hmc_sample(self, theta0, num_samples):
        if self._use_compiled:
            return hmc_sample_core(
                theta0,
                num_samples,
                self.epsilon,
                self.L,
                self.U_fn,
                self.grad_U_fn,
                self.stepper_compiled
            )
        else:
            return hmc_sample_py(
                theta0,
                num_samples,
                self.epsilon,
                self.L,
                self.U_fn,
                self.grad_U_fn,
                self.stepper_py
            )

    def hmc_pt(self, theta0s: np.ndarray, betas: np.ndarray, num_sweeps: int):
        """
        Parallel Tempering (PT) over multiple chains.
        Returns (history, swap_flags).
        """
        m, dim = theta0s.shape
        history    = np.zeros((num_sweeps, m, dim))
        swap_flags = np.zeros(num_sweeps, np.bool_)
        theta = theta0s.copy()

        # build one compiled HMC sampler per temperature
        samplers = [
            HMC(self.U_py, None, 'leapfrog', self.epsilon, self.L)
            for _ in betas
        ]

        for t in range(num_sweeps):
            # one HMC update per chain
            for j in range(m):
                theta[j], _, _ = samplers[j].hmc_step(theta[j])

            # propose adjacent swap
            i_swap = np.random.randint(0, m - 1)
            j_swap = i_swap + 1
            b_i, b_j = betas[i_swap], betas[j_swap]

            # compute Hamiltonians via Python U_py
            Hi_i = b_i * self.U_py(theta[i_swap])
            Hi_j = b_i * self.U_py(theta[j_swap])
            Hj_i = b_j * self.U_py(theta[i_swap])
            Hj_j = b_j * self.U_py(theta[j_swap])

            log_ratio = (Hi_i + Hj_j) - (Hi_j + Hj_i)
            if math.log(np.random.rand()) < log_ratio:
                theta[i_swap], theta[j_swap] = theta[j_swap].copy(), theta[i_swap].copy()
                swap_flags[t] = True

            history[t] = theta.copy()

        return history, swap_flags

    def hmc_smc(self, theta0: np.ndarray,
                num_particles: int,
                num_intermediate: int,
                hmc_steps_jitter: int):
        """
        SMC/AIS with HMC jitter at each intermediate.
        Returns (particles, logZ_estimate, ESS_history).
        """
        betas       = np.linspace(0.0, 1.0, num_intermediate + 1)
        particles   = np.tile(theta0[None, :], (num_particles, 1))
        log_weights = np.zeros(num_particles)
        ESS_hist    = np.zeros(num_intermediate)
        logZ_acc    = 0.0

        for t in range(1, num_intermediate + 1):
            b_prev, b_curr = betas[t-1], betas[t]

            # incremental weights
            for i in range(num_particles):
                u_prev = b_prev * self.U_py(particles[i])
                u_curr = b_curr * self.U_py(particles[i])
                log_weights[i] += -(u_curr - u_prev)

            # normalize & ESS & logZ
            maxw     = np.max(log_weights)
            w_unnorm = np.exp(log_weights - maxw)
            w_norm   = w_unnorm / np.sum(w_unnorm)
            ESS_hist[t-1] = 1.0 / np.sum(w_norm**2)
            logZ_acc   += maxw + math.log(np.mean(w_unnorm))

            # resample
            idxs = np.random.choice(num_particles, num_particles, True, w_norm)
            particles = particles[idxs]
            log_weights.fill(0.0)

            # jitter via compiled‐only HMC
            sampler = HMC(self.U_py, None, 'leapfrog',
                          self.epsilon, hmc_steps_jitter)
            for i in range(num_particles):
                particles[i], _, _ = sampler.hmc_step(particles[i])

        return particles, logZ_acc, ESS_hist

# ================================
# 7. “SHORTEST‐CHAIN” SEARCH CLASS (hybrid rule, ℝᵈ)
# ================================

# ─────────────────────────────────────────────────────────────
# A) Diagnostics utilities 
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# A) Diagnostics utilities 
# ─────────────────────────────────────────────────────────────
def gelman_rubin(chains: np.ndarray) -> float:
    """
    chains: shape (M, N).  Returns R_hat.
    """
    M, N = chains.shape
    # within-chain variances
    W = np.mean(np.var(chains, axis=1, ddof=1))
    # between-chain variance
    means = np.mean(chains, axis=1)
    B = N * np.var(means, ddof=1)
    # marginal posterior variance estimate
    V = ((N - 1) / N) * W + (1 / N) * B
    return np.sqrt(V / W)

def mcse(fvals: np.ndarray) -> float:
    """
    Monte Carlo standard error of fvals.
    """
    return np.std(fvals, ddof=1) / math.sqrt(len(fvals))

# ─────────────────────────────────────────────────────────────
# B) Simulation-Based Calibration (SBC)
# ─────────────────────────────────────────────────────────────
def sbc(prior_sampler, data_simulator, inference_fn, num_reps: int = 100, posterior_draws: int = 1000) -> np.ndarray:
    """
    Perform Simulation-Based Calibration (SBC).

    Args:
        prior_sampler:   function() -> theta (sample from the prior)
        data_simulator:  function(theta) -> y (simulate data given theta)
        inference_fn:    function(y, num_draws) -> samples (array of shape (num_draws, dim))
        num_reps:        number of SBC replicates
        posterior_draws: number of posterior draws per replicate

    Returns:
        ranks: array of shape (num_reps, dim) containing the rank of the true parameter
               among the sorted posterior samples for each replicate.
    """
    # run one replicate to infer dimensionality
    theta0 = prior_sampler()
    y0 = data_simulator(theta0)
    samples0 = inference_fn(y0, posterior_draws)
    dim = samples0.shape[1]
    ranks = np.zeros((num_reps, dim), dtype=int)

    for i in range(num_reps):
        theta = prior_sampler()
        y = data_simulator(theta)
        posterior = inference_fn(y, posterior_draws)
        # compute rank of each component of theta within its marginal posterior samples
        for d in range(dim):
            ranks[i, d] = np.sum(posterior[:, d] < theta[d])
    return ranks

# ─────────────────────────────────────────────────────────────
# C) Posterior Predictive Checks (PPC)
# ─────────────────────────────────────────────────────────────
def ppc(observed_data, inference_fn, data_simulator, summary_fn, 
        num_ppc_samples: int = 500, posterior_draws: int = 1000) -> dict:
    """
    Perform Posterior Predictive Checks (PPC).

    Args:
        observed_data:   the observed dataset
        inference_fn:    function(data, num_draws) -> samples (array of shape (num_draws, dim))
        data_simulator:  function(theta) -> y_rep (simulate data given theta)
        summary_fn:      function(data) -> float (summary statistic)
        num_ppc_samples: number of replicated datasets to simulate
        posterior_draws: number of posterior samples to draw

    Returns:
        dict with keys:
          'observed_summary' : summary_fn(observed_data)
          'replicated_summaries' : array of length num_ppc_samples
          'p_value' : proportion of replicated summaries ≥ observed_summary
    """
    # draw posterior samples from the observed data
    posterior = inference_fn(observed_data, posterior_draws)
    observed_summary = summary_fn(observed_data)

    replicated_summaries = np.zeros(num_ppc_samples)
    for i in range(num_ppc_samples):
        # pick one posterior draw at random
        theta = posterior[np.random.randint(0, posterior_draws)]
        y_rep = data_simulator(theta)
        replicated_summaries[i] = summary_fn(y_rep)

    # for a one-sided test: fraction of reps where summary ≥ observed
    p_value = np.mean(replicated_summaries >= observed_summary)

    return {
        'observed_summary': observed_summary,
        'replicated_summaries': replicated_summaries,
        'p_value': p_value
    }

# test.py (ajout en fin de fichier)
import numpy as np
from fake_data import diffs_k

# --- 1) Extraire un vecteur 1-D des différences (filtré des NaN) ---
# On a diffs_k de shape (D, T, lag+1)
all_diffs = diffs_k.reshape(-1)               # aplatissement brut
all_diffs = all_diffs[~np.isnan(all_diffs)]  # on enlève les NaN :contentReference[oaicite:0]{index=0}

# --- 2) Estimer les paramètres empiriques (mu, sigma) ---
mu    = np.mean(all_diffs)
sigma = np.std(all_diffs)
obs_var = sigma**2

# --- 3) Prior sampler basé sur ces stats empiriques ---
#    On tire un θ de même dimension que all_diffs,
#    avec des composantes i.i.d. ~ N(mu, sigma^2).
def prior_sampler():
    return np.random.normal(loc=mu, scale=sigma, size=all_diffs.shape)

# --- 4) Data simulator : on ajoute du bruit gaussien d’observation ---
#    Ici on simule y_rep = θ + ε,  ε ~ N(0, obs_var I)
def data_simulator(theta):
    return np.random.normal(loc=theta, scale=sigma, size=theta.shape)

# --- 5) Statistique de résumé pour PPC ---
#    On prend la moyenne (vous pouvez adapter : var(), quantile(), …)
def summary_fn(data):
    return np.mean(data, axis=0)




# ─────────────────────────────────────────────────────────────
# B) Stopping-rule thresholds (tweak to taste)
# ─────────────────────────────────────────────────────────────


class OptimumSeeking:
    """
    Scans over (integrator, ε, L, chain_length) for HMC / PT,
    and (integrator, ε, L, num_particles) for SMC,
    to find the *shortest* chain/particle number that satisfies the hybrid rule:
      1) Gelman–Rubin R̂ < R_HAT_THRESH
      2) ESS > ESS_MIN
      3) MCSE(mean) < MCSE_MEAN_TOL
      4) MCSE(var)  < MCSE_VAR_TOL
    """
    def __init__(self,
                 U: Callable[[np.ndarray], float],
                 grad_U: Callable[[np.ndarray], np.ndarray]):
        self.U      = U
        self.grad_U = grad_U

    @staticmethod
    def compute_ess(samples: np.ndarray) -> float:
        """
        Approximate ESS for a 1D array `samples`.
        ESS ≈ N / (1 + 2 Σₖ ρₖ), stopping once ρₖ < 0.05.
        """
        x = samples - np.mean(samples)
        N = len(x)
        if N < 2:
            return float(N)
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


    # existing imports/constants assumed: STEP_N, MAX_CHAIN_LENGTH, BURN_IN_FRAC,
    # R_HAT_THRESH, ESS_MIN, MCSE_MEAN_TOL, MCSE_VAR_TOL, DIMENSION

    def find_min_chain_length_hmc(
        self,
        integrator_name: str,
        eps: float,
        L: int
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:
        M = 4
        DIM = DIMENSION
        theta0s = np.zeros((M, DIM))
        seeds = np.arange(M, dtype=np.int64)

        sampler = HMC(self.U, None, integrator_name, eps, L)
        U_fn = sampler.U_fn
        grad_fn = sampler.grad_U_fn
        stepper = sampler.stepper_compiled

        for N in range(STEP_N, MAX_CHAIN_LENGTH + 1, STEP_N):
            burn_in = int(N * BURN_IN_FRAC)
            post = N - burn_in

            chains = batch_hmc_minimal(
                theta0s, seeds, N, burn_in,
                eps, L,
                U_fn, grad_fn, stepper
            )  # (M, post, DIM)

            R_hats = [gelman_rubin(chains[:, :, d]) for d in range(DIM)]
            R_max = max(R_hats)

            ESSs = [self.compute_ess(chains[:, :, d].ravel()) for d in range(DIM)]
            ESS_min = min(ESSs)

            flat = chains.reshape(-1, DIM)
            MCSE_means = [mcse(flat[:, d]) for d in range(DIM)]
            MCSE_vars = [mcse(flat[:, d]**2) for d in range(DIM)]
            MCSE_mean_max = max(MCSE_means)
            MCSE_var_max = max(MCSE_vars)

            if (R_max < R_HAT_THRESH and
                ESS_min > ESS_MIN and
                MCSE_mean_max < MCSE_MEAN_TOL and
                MCSE_var_max < MCSE_VAR_TOL):
                return N, R_max, ESS_min, MCSE_mean_max, MCSE_var_max

        return None, None, None, None, None

    def find_min_chain_length_pt(
        self,
        integrator_name: str,
        eps: float,
        L: int
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:
        M = 4
        m = 5
        DIM = DIMENSION
        sweeps_max = MAX_CHAIN_LENGTH
        betas = np.linspace(0.0, 1.0, m)
        theta0s = np.zeros((m, DIM))
        seeds = np.arange(M, dtype=np.int64)

        sampler = HMC(self.U, None, integrator_name, eps, L)
        U_fn = sampler.U_fn
        grad_fn = sampler.grad_U_fn
        name = integrator_name.lower().replace('-', '_')
        stepper_beta = {
            'leapfrog': leapfrog_step_beta,
            'force_gradient': force_gradient_step_beta,
            'yoshida4': yoshida4_step_beta,
        }[name]

        for sweeps in range(STEP_N, sweeps_max + 1, STEP_N):
            burn_in = int(sweeps * BURN_IN_FRAC)
            post = sweeps - burn_in

            all_cold = batch_pt_minimal(
                theta0s, seeds, sweeps,
                eps, L, betas,
                U_fn, grad_fn, stepper_beta
            )  # (M, sweeps, DIM)

            chains = all_cold[:, burn_in:, :]  # (M, post, DIM)

            R_hats = [gelman_rubin(chains[:, :, d]) for d in range(DIM)]
            R_max = max(R_hats)

            ESSs = [self.compute_ess(chains[:, :, d].ravel()) for d in range(DIM)]
            ESS_min = min(ESSs)

            flat = chains.reshape(-1, DIM)
            MCSE_means = [mcse(flat[:, d]) for d in range(DIM)]
            MCSE_vars = [mcse(flat[:, d]**2) for d in range(DIM)]
            MCSE_mean_max = max(MCSE_means)
            MCSE_var_max = max(MCSE_vars)

            if (R_max < R_HAT_THRESH and
                ESS_min > ESS_MIN and
                MCSE_mean_max < MCSE_MEAN_TOL and
                MCSE_var_max < MCSE_VAR_TOL):
                return sweeps, R_max, ESS_min, MCSE_mean_max, MCSE_var_max

        return None, None, None, None, None

    def find_min_chain_length_smc(
        self,
        integrator_name: str,
        eps: float,
        L: int
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:
        M = 4
        DIM = DIMENSION
        Np_max = MAX_SMC_PARTICLES
        num_inter = NUM_SMC_INTER
        theta0 = np.zeros(DIM)
        seeds = np.arange(M, dtype=np.int64)

        sampler = HMC(self.U, None, integrator_name, eps, L)
        U_fn = sampler.U_fn
        grad_fn = sampler.grad_U_fn
        name = integrator_name.lower().replace('-', '_')
        stepper_beta = {
            'leapfrog': leapfrog_step_beta,
            'force_gradient': force_gradient_step_beta,
            'yoshida4': yoshida4_step_beta,
        }[name]

        for Np in range(STEP_PARTICLES, Np_max + 1, STEP_PARTICLES):
            parts_all = batch_smc_minimal(
                theta0, seeds, Np, num_inter,
                eps, L,
                U_fn, grad_fn, stepper_beta
            )  # (M, Np, DIM)

            R_hats = [gelman_rubin(parts_all[:, :, d]) for d in range(DIM)]
            R_max  = max(R_hats)

            ESSs = [self.compute_ess(parts_all[:, :, d].ravel()) for d in range(DIM)]
            ESS_min = min(ESSs)

            flat = parts_all.reshape(-1, DIM)
            MCSE_means = [mcse(flat[:, d]) for d in range(DIM)]
            MCSE_vars = [mcse(flat[:, d]**2) for d in range(DIM)]
            MCSE_mean_max = max(MCSE_means)
            MCSE_var_max = max(MCSE_vars)

            if (R_max < R_HAT_THRESH and
                ESS_min > ESS_MIN and
                MCSE_mean_max < MCSE_MEAN_TOL and
                MCSE_var_max < MCSE_VAR_TOL):
                return Np, R_max, ESS_min, MCSE_mean_max, MCSE_var_max

        return None, None, None, None, None




    def diagnose_hmc(self,
                     integrator_name: str,
                     eps: float,
                     L: int
                    ) -> Tuple[float, float, float, float]:
        M = NUMBER_OF_INDEPENDANT_CHAINS
        N = MAX_CHAIN_LENGTH
        burn_in = int(N * BURN_IN_FRAC)

        θ0 = np.zeros(DIMENSION)
        base_seed = np.random.default_rng().integers(0, 2**31-1)
        seeds = [int(base_seed + i) for i in range(M)]

        def run_chain(seed: int):
            np.random.seed(seed)
            sampler = HMC(self.U, None, integrator_name, eps, L)
            sampler._use_compiled = False
            name = integrator_name.lower().replace('-', '_')
            pure_steps = {
                'leapfrog':       leapfrog_py,
                'force_gradient': force_gradient_py,
                'yoshida4':       yoshida4_py,
            }
            sampler.stepper_py = pure_steps[name]
            sampler.U_fn = sampler.U_py
            sampler.grad_U_fn = sampler.grad_U_py
            thetas, _, _ = sampler.hmc_sample(θ0, num_samples=N)
            return thetas[burn_in:]  # (post, DIMENSION)

        chains = np.stack(
            Parallel(n_jobs=-1)(
                delayed(run_chain)(seed)
                for seed in seeds
            ),
            axis=0
        )  # → (M, post, DIMENSION)

        R_max = max(gelman_rubin(chains[:, :, d]) for d in range(DIMENSION))
        ESS_min = min(self.compute_ess(chains[:, :, d].ravel())
                      for d in range(DIMENSION))
        flat = chains.reshape(-1, DIMENSION)
        MCSE_mean_max = max(mcse(flat[:, d])       for d in range(DIMENSION))
        MCSE_var_max  = max(mcse(flat[:, d]**2)    for d in range(DIMENSION))

        return R_max, ESS_min, MCSE_mean_max, MCSE_var_max

    def diagnose_pt(self,
                    integrator_name: str,
                    eps: float,
                    L: int
                   ) -> Tuple[float, float, float, float]:
        M     = NUMBER_OF_INDEPENDANT_CHAINS
        sweeps = MAX_CHAIN_LENGTH
        burn_in = int(sweeps * BURN_IN_FRAC)
        θ0s   = np.zeros((5, DIMENSION))
        betas = np.linspace(0, 1, 5)

        def run_pt(m_idx: int):
            rng = np.random.RandomState(1_000 + m_idx)
            hmc_pt = HMC(self.U, self.grad_U, integrator_name, eps, L)
            hmc_pt.RNG = rng
            history, _ = hmc_pt.hmc_pt(θ0s, betas, num_sweeps=sweeps)
            return history[burn_in:, 0, :]  # (post, DIMENSION)

        all_cold = np.stack(
            Parallel(n_jobs=-1)(
                delayed(run_pt)(m_idx)
                for m_idx in range(M)
            )
        )  # → (M, post, DIMENSION)

        R_max = max(gelman_rubin(all_cold[:, :, d]) for d in range(DIMENSION))
        ESS_min = min(self.compute_ess(all_cold[:, :, d].ravel())
                      for d in range(DIMENSION))
        flat = all_cold.reshape(-1, DIMENSION)
        MCSE_mean_max = max(mcse(flat[:, d])     for d in range(DIMENSION))
        MCSE_var_max  = max(mcse(flat[:, d]**2)  for d in range(DIMENSION))

        return R_max, ESS_min, MCSE_mean_max, MCSE_var_max

    def diagnose_smc(self,
                     integrator_name: str,
                     eps: float,
                     L: int
                    ) -> Tuple[float, float, float, float]:
        M         = NUMBER_OF_INDEPENDANT_CHAINS
        Np        = MAX_SMC_PARTICLES
        num_inter = 20
        θ0        = np.zeros(DIMENSION)

        def run_smc(m_idx: int):
            rng = np.random.RandomState(1_000 + m_idx)
            hmc_smc = HMC(self.U, self.grad_U, integrator_name, eps, L)
            hmc_smc.RNG = rng
            parts, _, _ = hmc_smc.hmc_smc(
                θ0,
                num_particles    = Np,
                num_intermediate = num_inter,
                hmc_steps_jitter = L
            )
            return parts  # (Np, DIMENSION)

        all_parts = np.stack(
            Parallel(n_jobs=-1)(
                delayed(run_smc)(m_idx)
                for m_idx in range(M)
            )
        )  # → (M, Np, DIMENSION)

        R_max = max(gelman_rubin(all_parts[:, :, d]) for d in range(DIMENSION))
        ESS_min = min(self.compute_ess(all_parts[:, :, d].ravel())
                      for d in range(DIMENSION))
        flat = all_parts.reshape(-1, DIMENSION)
        MCSE_mean_max = max(mcse(flat[:, d])     for d in range(DIMENSION))
        MCSE_var_max  = max(mcse(flat[:, d]**2)  for d in range(DIMENSION))

        return R_max, ESS_min, MCSE_mean_max, MCSE_var_max

from joblib import Parallel, delayed

def run_minimal_chain_task(alg, integ, eps):
    """
    Run the full minimal‐chain‐length experiment for one (algorithm, integrator, eps).
    Returns (alg, integ, best_record_tuple) where best_record_tuple is
      (chain_length_or_sweeps, R_hat, ESS, MCSE_mean, MCSE_var, best_L).
    """
    best_record = (None,) * 6
    # common thresholds & data already in global scope:
    #   R_HAT_THRESH, ESS_MIN, MCSE_MEAN_TOL, MCSE_VAR_TOL,
    #   SBC_PVAL_THRESH, PPC_DIST_THRESH, all_diffs, summary_fn, ...
    # and your `seeker` object, plus U_target, grad_U_target, theta0, prior_sampler, data_simulator, m

    if alg == 'HMC':
        for L in range(1, MAX_HMC_STEPS + 1):
            N, R, ESS, mM, mV = seeker.find_min_chain_length_hmc(integ, eps, L)
            if N is None:
                continue

            failures = []
            if R   >= R_HAT_THRESH:  failures.append(f"R̂={R:.3f}")
            if ESS <= ESS_MIN:       failures.append(f"ESS={ESS:.1f}")
            if mM  >= MCSE_MEAN_TOL: failures.append(f"MCSE_mean={mM:.3f}")
            if mV  >= MCSE_VAR_TOL:  failures.append(f"MCSE_var={mV:.3f}")

            # SBC + PPC
            def inf(y_obs):
                sampler = HMC(U_target, grad_U_target, integ, eps, L)
                samples, _, _ = sampler.hmc_sample(theta0, N)
                return samples

            ranks = sbc(prior_sampler, data_simulator, inf, num_reps=100, posterior_draws=200)
            _, p_sbc = kstest(ranks, 'uniform')
            if p_sbc < SBC_PVAL_THRESH:
                failures.append(f"SBC p={p_sbc:.3f}")

            ppc_res = ppc(
                observed_data=all_diffs,
                inference_fn=inf,
                data_simulator=data_simulator,
                summary_fn=summary_fn,
                num_ppc_samples=100,
                posterior_draws=200
            )
            obs_sum   = summary_fn(all_diffs)
            repl_sums = np.array([summary_fn(y) for y in ppc_res])
            if np.mean(np.abs(repl_sums - obs_sum)) > PPC_DIST_THRESH:
                failures.append("PPC failed")

            if not failures:
                # keep the smallest N
                if best_record[1] is None or N < best_record[1]:
                    best_record = (N, R, ESS, mM, mV, L)

    elif alg == 'PT':
        for L in range(1, MAX_HMC_STEPS + 1):
            sweeps, R, ESS, mM, mV = seeker.find_min_chain_length_pt(integ, eps, L)
            if sweeps is None:
                continue

            failures = []
            if R   >= R_HAT_THRESH:  failures.append(f"R̂={R:.3f}")
            if ESS <= ESS_MIN:       failures.append(f"ESS={ESS:.1f}")
            if mM  >= MCSE_MEAN_TOL: failures.append(f"MCSE_mean={mM:.3f}")
            if mV  >= MCSE_VAR_TOL:  failures.append(f"MCSE_var={mV:.3f}")

            def inf(y_obs):
                sampler = HMC(U_target, grad_U_target, integ, eps, L)
                betas = np.linspace(0.0, 1.0, m)
                hist, _ = sampler.hmc_pt(np.zeros((m, DIMENSION)), betas, sweeps)
                burn = int(sweeps * BURN_IN_FRAC)
                return hist[burn:, 0, :]

            ranks = sbc(prior_sampler, data_simulator, inf, num_reps=100, posterior_draws=200)
            _, p_sbc = kstest(ranks, 'uniform')
            if p_sbc < SBC_PVAL_THRESH:
                failures.append(f"SBC p={p_sbc:.3f}")

            ppc_res = ppc(
                observed_data=all_diffs,
                inference_fn=inf,
                data_simulator=data_simulator,
                summary_fn=summary_fn,
                num_ppc_samples=100,
                posterior_draws=200
            )
            obs_sum   = summary_fn(all_diffs)
            repl_sums = np.array([summary_fn(y) for y in ppc_res])
            if np.mean(np.abs(repl_sums - obs_sum)) > PPC_DIST_THRESH:
                failures.append("PPC failed")

            if not failures:
                if best_record[1] is None or sweeps < best_record[1]:
                    best_record = (sweeps, R, ESS, mM, mV, L)

    else:  # SMC
        for L in range(1, MAX_HMC_STEPS + 1):
            Np, R, ESS, mM, mV = seeker.find_min_chain_length_smc(integ, eps, L)
            if Np is None:
                continue

            failures = []
            if R   >= R_HAT_THRESH:  failures.append(f"R̂={R:.3f}")
            if ESS <= ESS_MIN:       failures.append(f"ESS={ESS:.1f}")
            if mM  >= MCSE_MEAN_TOL: failures.append(f"MCSE_mean={mM:.3f}")
            if mV  >= MCSE_VAR_TOL:  failures.append(f"MCSE_var={mV:.3f}")

            def inf(y_obs):
                sampler = HMC(U_target, grad_U_target, integ, eps, L)
                parts = batch_smc_minimal(
                    theta0,
                    np.arange(m, dtype=int),
                    Np,
                    NUM_SMC_INTER,
                    eps,
                    L,
                    sampler.U_fn,
                    sampler.grad_U_fn,
                    stepper_map[integ]
                )
                return parts.reshape(-1, DIMENSION)

            ranks = sbc(prior_sampler, data_simulator, inf, num_reps=100, posterior_draws=200)
            _, p_sbc = kstest(ranks, 'uniform')
            if p_sbc < SBC_PVAL_THRESH:
                failures.append(f"SBC p={p_sbc:.3f}")

            ppc_res = ppc(
                observed_data=all_diffs,
                inference_fn=inf,
                data_simulator=data_simulator,
                summary_fn=summary_fn,
                num_ppc_samples=100,
                posterior_draws=200
            )
            obs_sum   = summary_fn(all_diffs)
            repl_sums = np.array([summary_fn(y) for y in ppc_res])
            if np.mean(np.abs(repl_sums - obs_sum)) > PPC_DIST_THRESH:
                failures.append("PPC failed")

            if not failures:
                if best_record[1] is None or Np < best_record[1]:
                    best_record = (Np, R, ESS, mM, mV, L)

    return alg, integ, best_record

# ================================
# 8. MAIN ENTRY POINT
# ================================
if __name__ == "__main__":
    np.random.seed(42)
    m=5
    seeker = OptimumSeeking(U_target, grad_U_target)
    # ──────────────────────────────────────────────────────────────────
    # 1) Pre-flight ε-tuning
    # ──────────────────────────────────────────────────────────────────
    print("\n=== Step 0a: Pre-flight ε-tuning (full D) ===\n")
    tuner     = EpsilonFinder(DIMENSION)
    tuned_eps = tuner.all_eps()
    print(f"→ Tuned ε’s (in ℝ^{DIMENSION}):")
    for alg in ['HMC','PT','SMC']:
        print(f"  • {alg}:")
        for integ in ['leapfrog','force-gradient','yoshida4']:
            print(f"      – {integ:12s} → ε = {tuned_eps[alg][integ]:.3f}")
    print()

    # ──────────────────────────────────────────────────────────────────
    # 2) Find minimal chain lengths / particle counts
    # ──────────────────────────────────────────────────────────────────
    algorithms  = ['HMC', 'PT', 'SMC']
    integrators = ['leapfrog', 'force-gradient', 'yoshida4']

    # Build the job list
    jobs = [
        (alg, integ, tuned_eps[alg][integ])
        for alg in algorithms
        for integ in integrators
    ]

    # Fire them all off in parallel (n_jobs=-1 -> one process per core)
    outputs = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_minimal_chain_task)(alg, integ, eps)
        for alg, integ, eps in jobs
    )

    # Collect results
    results = {alg: {} for alg in algorithms}
    for alg, integ, record in outputs:
        # record = (N_or_sweeps, R, ESS, mM, mV, L)
        results[alg][integ] = record

    # (Optionally) print a summary:
    for alg in algorithms:
        print(f"=== {alg} results ===")
        for integ in integrators:
            rec = results[alg][integ]
            print(f"  {integ:15s} → N/sweeps={rec[0]}, R={rec[1]:.3f}, ESS={rec[2]:.1f}, "
                  f"MCSE_mean={rec[3]:.3f}, MCSE_var={rec[4]:.3f}, L={rec[5]}")
        print()




    # ──────────────────────────────────────────────────────────────────
    # 3) Report summary of minimal budgets (with SBC & PPC)
    # ──────────────────────────────────────────────────────────────────
    print("\nAlgorithm × Integrator   →  (chain_length, R̂, ESS, MCSE_mean, MCSE_var, L, SBC_p, PPC_dist)\n")
    for alg in algorithms:
        print(f"--- {alg} ---")
        for integ in ['leapfrog','force-gradient','yoshida4']:
            vals = results[alg].get(integ, (None,)*8)
            chain_len, R_hat, ESS_val, MCSE_mean_val, MCSE_var_val, L_val, p_sbc, dist_ppc = vals

            label = "N" if alg in ['HMC','PT'] else "Np"
            if chain_len is None:
                # Pas de budget valide : on refait le diagnostic classique
                eps = tuned_eps[alg][integ]
                L_d = L_val if L_val is not None else MAX_HMC_STEPS
                if   alg == 'HMC': Rf, ESf, Mm, Mv = seeker.diagnose_hmc(integ, eps, L_d)
                elif alg == 'PT':  Rf, ESf, Mm, Mv = seeker.diagnose_pt(integ, eps, L_d)
                else:              Rf, ESf, Mm, Mv = seeker.diagnose_smc(integ, eps, L_d)

                failures = []
                if Rf   >= R_HAT_THRESH:   failures.append(f"R̂={Rf:.3f}≥{R_HAT_THRESH}")
                if ESf  <= ESS_MIN:        failures.append(f"ESS={ESf:.1f}≤{ESS_MIN}")
                if Mm   >= MCSE_MEAN_TOL:  failures.append(f"MCSE_mean={Mm:.3f}≥{MCSE_MEAN_TOL}")
                if Mv   >= MCSE_VAR_TOL:   failures.append(f"MCSE_var={Mv:.3f}≥{MCSE_VAR_TOL}")
                # on pourrait rerun SBC/PPC ici si nécessaire

                print(f"  {integ:14s} → None; failed: {', '.join(failures)}")
            else:
                print(
                    f"  {integ:14s} → "
                    f"{label}={chain_len:<5d}, "
                    f"R̂={R_hat:.3f}, "
                    f"ESS={ESS_val:6.1f}, "
                    f"MCSE_mean={MCSE_mean_val:.3f}, "
                    f"MCSE_var={MCSE_var_val:.3f}, "
                    f"L={L_val:<2d}, "
                    f"SBC_p={p_sbc:.3f}, "
                    f"PPC_dist={dist_ppc:.3f}"
                )
        print()


    # ──────────────────────────────────────────────────────────────────
    # 3.5) Pick the overall fastest sampler (shortest budget) and refine ties
    # ──────────────────────────────────────────────────────────────────
    candidates = []
    for alg in algorithms:
        for integ in ['leapfrog','force-gradient','yoshida4']:
            budget, R_hat, ESS_val, mM, mV, L_val, p_sbc, dist_ppc = results[alg][integ]
            if budget is not None:
                candidates.append((alg, integ, budget, L_val))
    if not candidates:
        raise RuntimeError("No sampler passed all diagnostics up to the caps.")

    # find minimal budget
    min_budget = min(c[2] for c in candidates)
    best = [c for c in candidates if c[2] == min_budget]

    # helper pour SBC/PPC check
    def passes_sbc_ppc(alg, integ, budget, L):
        # on reconstruit inference_fn pour cette config
        if alg in ['HMC', 'PT']:
            def inference_fn(y_obs):
                sampler = HMC(U_target, grad_U_target, integ, tuned_eps[alg][integ], L)
                if alg == 'HMC':
                    samples, _, _ = sampler.hmc_sample(theta0, budget)
                else:
                    betas = np.linspace(0.0, 1.0, m)
                    history, _ = sampler.hmc_pt(np.zeros((m, DIMENSION)), betas, budget)
                    burn = int(budget * BURN_IN_FRAC)
                    samples = history[burn:, 0, :]
                return samples
        else:  # SMC
            def inference_fn(y_obs):
                sampler = HMC(U_target, grad_U_target, integ, tuned_eps[alg][integ], L)
                parts = batch_smc_minimal(
                    theta0,
                    np.arange(m, dtype=int),
                    budget,
                    NUM_SMC_INTER,
                    tuned_eps[alg][integ],
                    L,
                    sampler.U_fn,
                    sampler.grad_U_fn,
                    stepper_map[integ]
                )
                return parts.reshape(-1, DIMENSION)

        # SBC
        ranks = sbc(prior_sampler, data_simulator, inference_fn,
                    num_reps=100, posterior_draws=200)
        _, p_sbc = kstest(ranks, 'uniform')
        if p_sbc < SBC_PVAL_THRESH:
            return False

        # PPC
        ppc_res = ppc(
            observed_data=all_diffs,
            inference_fn=inference_fn,
            data_simulator=data_simulator,
            summary_fn=summary_fn,
            num_ppc_samples=100,
            posterior_draws=200
        )
        obs_sum  = summary_fn(all_diffs)
        repl_sums = np.array([summary_fn(y) for y in ppc_res])
        dist_ppc = np.mean(np.abs(repl_sums - obs_sum))
        return dist_ppc <= PPC_DIST_THRESH

    # refine coarse ties
    if len(best) > 1:
        refined = []
        for alg, integ, _, L_val in best:
            eps = tuned_eps[alg][integ]
            if alg in ['HMC','PT']:
                new_b, *_ = (
                    seeker.find_min_chain_length_hmc(integ, eps, L_val)
                    if alg == 'HMC'
                    else seeker.find_min_chain_length_pt(integ, eps, L_val)
                )
            else:
                new_b, *_ = seeker.find_min_chain_length_smc(integ, eps, L_val)

            if new_b is not None and passes_sbc_ppc(alg, integ, new_b, L_val):
                refined.append((alg, integ, new_b, L_val))

        if refined:
            min_budget = min(r[2] for r in refined)
            best = [r for r in refined if r[2] == min_budget]

    # pick the first best candidate at the coarse grid
    best_alg, best_integ, best_budget, best_L = best[0]

    # if there’s still more than one integrator tied on budget, do a fine-grid re-scan
    if len(best) > 1:
        refined2 = []
        for alg, integ, _, L_val in best:
            eps = tuned_eps[alg][integ]
            step_i = STEP_PARTICLES if alg == 'SMC' else STEP_N
            lb = best_budget - step_i + 1
            for b in range(lb, best_budget):
                if alg == 'SMC':
                    b_new, *_ = seeker.find_min_chain_length_smc(integ, eps, L_val)
                elif alg == 'HMC':
                    b_new, *_ = seeker.find_min_chain_length_hmc(integ, eps, L_val)
                else:  # PT
                    b_new, *_ = seeker.find_min_chain_length_pt(integ, eps, L_val)

                if b_new == b and passes_sbc_ppc(alg, integ, b_new, L_val):
                    refined2.append((alg, integ, b_new, L_val))
                    break

        if refined2:
            min_b2 = min(r[2] for r in refined2)
            best = [r for r in refined2 if r[2] == min_b2]
            best_alg, best_integ, best_budget, best_L = best[0]

    print(f"\n>>> Selected {best_alg} + {best_integ} "
        f"@ budget={best_budget}, L={best_L}")




    # ──────────────────────────────────────────────────────────────────
    # 4) Run the final comparison with the chosen (algo, integrator, ε, L)
    # ──────────────────────────────────────────────────────────────────
    eps_final = tuned_eps[best_alg][best_integ]
    theta0    = np.zeros(DIMENSION)

    if best_alg in ['HMC','PT']:
        print(f"\n=== Running {best_alg}({best_integ}, ε={eps_final:.3f}, L={best_L}) ===")
        sampler = HMC(U_target, None, best_integ, eps_final, best_L)
        if best_alg == 'HMC':
            samples, _, _ = sampler.hmc_sample(theta0, best_budget)
        else:
            betas = np.linspace(0.0, 1.0, 5)
            history, _ = sampler.hmc_pt(np.zeros((5, DIMENSION)), betas, best_budget)
            burn_in = int(best_budget * BURN_IN_FRAC)
            # take only the cold chain post–burn-in
            samples = history[burn_in:, 0, :]

    else:  # SMC
        from test2 import batch_smc_minimal, yoshida4_step_beta, force_gradient_step_beta, leapfrog_step_beta
        stepper_map = {
            'leapfrog':       leapfrog_step_beta,
            'force-gradient': force_gradient_step_beta,
            'yoshida4':       yoshida4_step_beta,
        }
        print(f"\n=== Running SMC({best_integ}, ε={eps_final:.3f}, L={best_L}, Np={best_budget}) ===")
        sampler = HMC(U_target, None, best_integ, eps_final, best_L)
        M = NUMBER_OF_INDEPENDANT_CHAINS
        seeds = np.arange(M, dtype=np.int64)
        parts = batch_smc_minimal(
            theta0, seeds,
            best_budget,
            NUM_SMC_INTER,
            eps_final,
            best_L,
            sampler.U_fn,
            sampler.grad_U_fn,
            stepper_map[best_integ]
        )
        # aggregate across replicates & particles
        samples = parts.reshape(-1, DIMENSION)

    # compute empirical stats for the chosen sampler
    emp_mean = samples.mean(axis=0)
    emp_cov  = np.cov(samples, rowvar=False)
    emp_sigma = np.sqrt(np.diag(emp_cov))
    abs_err_mean = np.abs(emp_mean - mu)
    abs_err_sigma= np.abs(emp_sigma - sigma)

    print("\n--- Value of the BlackBox ---")
    print(" mean       =", mu)
    print(" σ          =", sigma)

    print("\n--- Empirical estimates ---")
    print(" mean estimate      =", emp_mean)
    print(" cov estimate diag  =", np.diag(emp_cov))
    print(" σ estimate         =", emp_sigma)

    print("\n--- Absolute errors ---")
    print(" |mean − μ|         =", abs_err_mean)
    print(" |σ − true σ|       =", abs_err_sigma)
    print("\nDone.\n")