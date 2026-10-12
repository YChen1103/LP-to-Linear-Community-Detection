"""
Oracle quantities, computed from the (population) SBM parameters alone.

All quantities here are population objects, accessible only because the
experiments are synthetic (true p_ab known). On real data, none of these
would be available -- they are used to compute Bayes floors, oracle
classifiers, and to calibrate the capacity constants of the bounded
encoder-decoder class.

Notation (from main paper):
    M_star[a,b]         = logit(p_ab)
    H_matrix[a,b]       = H(p_ab) = binary entropy
    B_star              = max_{a,b} |M_star[a,b]|
    b_star              = arg min_b ||M_star - b 1 1^T||_op
    M_tilde             = M_star - b_star * 1 1^T
    capacity_required   = ||M_tilde||_op
    Delta_s_sq          = min_{a!=b} sum_c pi_c (M_star[a,c] - M_star[b,c])^2
    mu_B(B)             = sigma(B)(1 - sigma(B))
"""
from __future__ import annotations
from typing import Any, Dict

import numpy as np
from scipy.optimize import minimize_scalar


_EPS = 1e-12


def safe_logit(p: float) -> float:
    """logit(p) = log(p/(1-p)), clipped to avoid log(0)."""
    p = float(np.clip(p, _EPS, 1.0 - _EPS))
    return float(np.log(p / (1.0 - p)))


def binary_entropy(p: float) -> float:
    """H(p) = -p log p - (1-p) log(1-p)."""
    p = float(np.clip(p, _EPS, 1.0 - _EPS))
    return float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))


def sigmoid(x):
    """Numerically stable sigma(x)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def mu_B_at(B: float) -> float:
    """mu_B(B) = sigma(B) * (1 - sigma(B)). Strong-convexity coefficient on [-B, B]."""
    s = sigmoid(B)
    return float(s * (1.0 - s))


# =============================================================================
# Capacity-required: ||M^* - b^* 1 1^T||_op via 1D convex search
# =============================================================================

def compute_capacity_required(M_star: np.ndarray) -> tuple[float, float]:
    """Compute (||M_tilde||_op, b_star) where M_tilde = M_star - b_star * 1 1^T.

    b_star = arg min_b ||M_star - b 1 1^T||_op
    The objective is convex in b (operator norm of an affine function).
    """
    K = M_star.shape[0]
    ones_outer = np.ones((K, K))

    def obj(b: float) -> float:
        return float(np.linalg.norm(M_star - b * ones_outer, ord=2))

    # Search range: |b| <= 2 * max|M_star| is generous
    b_max = 2.0 * float(np.abs(M_star).max() + 1.0)
    res = minimize_scalar(obj, bracket=(-b_max, 0.0, b_max), method="brent")
    b_star = float(res.x)
    M_tilde = M_star - b_star * ones_outer
    capacity = float(np.linalg.norm(M_tilde, ord=2))
    return capacity, b_star


# =============================================================================
# Score-profile separation Delta_s
# =============================================================================

def compute_Delta_s_sq(M_star: np.ndarray, pi: np.ndarray) -> float:
    """Delta_s^2 = min_{a!=b} sum_c pi_c (M_star[a,c] - M_star[b,c])^2.

    Identifiability holds iff Delta_s_sq > 0 (rows of M_star are distinct).
    """
    K = M_star.shape[0]
    best = float("inf")
    for a in range(K):
        for b in range(a + 1, K):
            diff = M_star[a, :] - M_star[b, :]
            val = float(np.sum(pi * diff * diff))
            if val < best:
                best = val
    return best


# =============================================================================
# Top-level: compute oracle from SBM parameters
# =============================================================================

def compute_sbm_oracle(B_n: np.ndarray, pi: np.ndarray) -> Dict[str, Any]:
    """Compute oracle quantities from the actual block probability matrix B_n = rho_n * B.

    Returns a dict with keys:
        M_star              (K, K) logit(p_ab)
        H_matrix            (K, K) binary entropy of p_ab
        B_star              max |M_star[a,b]|
        b_star              arg min_b ||M_star - b 11^T||_op
        capacity_required   ||M_star - b_star 11^T||_op
        Delta_s, Delta_s_sq separation
        pi                  community proportions (passed through)

    Note: the capacity B = max(B_model, B_star) of the bounded encoder-decoder
    class is NOT computed here, because B_model depends on (R, W_max, B_0)
    which are choices in config-resolution. See lp_sbm.config.resolve_config.
    """
    K = B_n.shape[0]
    if pi.shape != (K,):
        raise ValueError(f"pi shape {pi.shape} != (K,) = ({K},)")

    M_star = np.zeros((K, K))
    H_matrix = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            p_ab = float(B_n[a, b])
            M_star[a, b] = safe_logit(p_ab)
            H_matrix[a, b] = binary_entropy(p_ab)

    B_star = float(np.abs(M_star).max())
    capacity_required, b_star = compute_capacity_required(M_star)
    Delta_s_sq = compute_Delta_s_sq(M_star, pi)
    Delta_s = float(np.sqrt(max(Delta_s_sq, 0.0)))

    return {
        "M_star":            M_star,
        "H_matrix":          H_matrix,
        "B_star":            B_star,
        "b_star":            b_star,
        "capacity_required": capacity_required,
        "Delta_s":           Delta_s,
        "Delta_s_sq":        Delta_s_sq,
        "pi":                pi.copy(),
    }


# =============================================================================
# Oracle prototypes (as defined in the paper)
# =============================================================================

def build_oracle_prototypes(
    M_star: np.ndarray, labels: np.ndarray, K: int
) -> np.ndarray:
    """mu_a in R^n: (mu_a)_j = M_star[a, c_j].

    Returns (K, n) matrix of oracle prototypes in score-profile space.
    """
    return M_star[:, labels]  # (K, n)


def compute_R_H_star(
    labels: np.ndarray, rows: np.ndarray, cols: np.ndarray,
    held_mask: np.ndarray, H_matrix: np.ndarray,
) -> float:
    """R^P_H(S^*_H) = (1/|H|) sum_{(i,j) in H} H(p_{c_i c_j}). Bayes floor on H."""
    idx = np.where(held_mask)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(H_matrix[labels[rows[idx]], labels[cols[idx]]]))


def compute_R_T_star(
    labels: np.ndarray, rows: np.ndarray, cols: np.ndarray,
    train_mask: np.ndarray, H_matrix: np.ndarray,
) -> float:
    """R^P_T(S^*) = (1/|T|) sum_{(i,j) in T} H(p_{c_i c_j}). Bayes floor on T.

    The train-side entropy floor. Used to track convergence of the empirical
    train BCE R_T_emp toward the population Bayes optimum: as architecture and
    training improve, R_T_emp -> R_T_star (over the random T,H partition,
    R_T_star ~ R_H_star up to O(1/sqrt(n)) sampling-without-replacement noise).
    """
    idx = np.where(train_mask)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(H_matrix[labels[rows[idx]], labels[cols[idx]]]))
