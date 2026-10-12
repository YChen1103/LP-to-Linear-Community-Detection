"""
SBM data generation, masking, and helpers.

Notation follows the main paper:
    P[i,j]      = p_{c_i c_j}                       (population probability)
    A[i,j]      ~ Ber(P[i,j])                       (observed adjacency)
    M[i,j]      ~ Ber(tau)                          (mask indicator)
    T = {(i,j) : M_ij == 0}                         (visible/train pairs)
    H = {(i,j) : M_ij == 1}                         (held-out pairs)
    A_enc[i,j]  = A[i,j] * 1{(i,j) in T}            (encoder input, hollow)

All upper-triangular pair arrays use the convention `(rows, cols)` from
np.triu_indices(n, k=1), of length N = n(n-1)/2.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np


# =============================================================================
# Block matrix construction
# =============================================================================

def build_block_matrix(K: int, block_spec: Dict[str, Any]) -> np.ndarray:
    """Construct K x K block probability matrix B (before rho_n scaling).

    `block_spec` is the resolved block_spec dict from the config:
        {type: "assortative", p_in: ..., p_out: ...}
        {type: "custom", p_matrix: [[...], ...]}
    """
    bt = block_spec["type"]
    if bt == "assortative":
        p_in = float(block_spec["p_in"])
        p_out = float(block_spec["p_out"])
        B = np.full((K, K), p_out, dtype=np.float64)
        np.fill_diagonal(B, p_in)
        return B
    if bt == "custom":
        B = np.asarray(block_spec["p_matrix"], dtype=np.float64)
        if B.shape != (K, K):
            raise ValueError(
                f"Custom p_matrix shape {B.shape} != (K, K) = ({K}, {K})"
            )
        return B
    raise ValueError(f"Unknown block_spec.type: {bt!r}")


def resolve_pi(K: int, pi_spec) -> np.ndarray:
    """Resolve community proportions pi.

    `pi_spec` is "uniform" or an explicit list of length K summing to 1.
    """
    if pi_spec == "uniform":
        return np.ones(K, dtype=np.float64) / K
    pi = np.asarray(pi_spec, dtype=np.float64)
    if pi.shape != (K,):
        raise ValueError(f"pi shape {pi.shape} != (K,) = ({K},)")
    if not np.isclose(pi.sum(), 1.0, atol=1e-6):
        raise ValueError(f"pi must sum to 1; got {pi.sum()}")
    return pi


# =============================================================================
# Sample SBM graph
# =============================================================================

def sample_labels(n: int, K: int, pi: np.ndarray, seed: int) -> np.ndarray:
    """Sample community labels c_1, ..., c_n.

    For balanced SBM (uniform pi and n divisible by K), uses block-then-permute
    to guarantee exactly n/K nodes per community. Otherwise samples from
    Multinomial(n, pi).
    """
    rng = np.random.default_rng(seed)
    if np.allclose(pi, 1.0 / K) and n % K == 0:
        block_size = n // K
        labels = np.repeat(np.arange(K, dtype=np.int64), block_size)
        perm = rng.permutation(n)
        return labels[perm]
    # Otherwise: multinomial
    return rng.choice(K, size=n, p=pi).astype(np.int64)


def sample_adjacency(P: np.ndarray, seed: int) -> np.ndarray:
    """Sample symmetric adjacency A from Bernoulli(P), with A_ii = 0.

    Only upper-triangular pairs (i<j) are sampled; lower mirrored.
    """
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=np.float64)
    rows, cols = np.triu_indices(n, k=1)
    probs = P[rows, cols]
    edges = (rng.random(len(probs)) < probs).astype(np.float64)
    A[rows, cols] = edges
    A[cols, rows] = edges
    return A


def sample_mask(n: int, tau: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample mask. Returns (train_mask, held_mask), both length-N bool arrays.

    Deterministic-size random subset: |H| = round(tau * N) is fixed; H is
    drawn uniformly at random among size-|H| subsets of {(i,j) : i<j}. This
    matches the paper's convention 'Fix |H|=ceil(tau N) ... draw H uniformly
    at random among size-|H| subsets' and is required for the Hoeffding-Serfling
    sampling-without-replacement argument used in the paper.

    When tau is exactly 0 or 1, returns trivial masks without sampling.
    """
    rng = np.random.default_rng(seed)
    N = n * (n - 1) // 2
    if tau <= 0.0:
        return np.ones(N, dtype=bool), np.zeros(N, dtype=bool)
    if tau >= 1.0:
        return np.zeros(N, dtype=bool), np.ones(N, dtype=bool)
    # Deterministic size, random subset.
    n_held = int(round(tau * N))
    n_held = max(1, min(N - 1, n_held))   # guard against degenerate sizes
    held_idx = rng.choice(N, size=n_held, replace=False)
    held_mask = np.zeros(N, dtype=bool)
    held_mask[held_idx] = True
    train_mask = ~held_mask
    return train_mask, held_mask


# =============================================================================
# All-in-one: prepare data dict
# =============================================================================

def prepare_sbm_data(
    sbm_cfg: Dict[str, Any],
    mask_cfg: Dict[str, Any],
    seeds: Dict[str, int],
) -> Dict[str, Any]:
    """Prepare all data structures for one SBM realization.

    Returns dict with keys (using paper notation where possible):
        n, K, tau, rho_n            (params)
        pi                          (K,) community proportions used
        B_block                     (K, K) base block matrix (no rho_n)
        B_n                         (K, K) actual p_ab = rho_n * B_block
        labels                      (n,)  community labels
        P                           (n, n) population matrix (P_ii = 0)
        A                           (n, n) observed adjacency
        rows, cols                  upper-tri indices, length N = n(n-1)/2
        train_mask, held_mask       (N,) bool, |H|/|T| pair indicators
        P_flat                      (N,) P[rows, cols]
        A_flat                      (N,) A[rows, cols]
    """
    n = int(sbm_cfg["n"])
    K = int(sbm_cfg["K"])
    rho_n = float(sbm_cfg["rho_n"])
    tau = float(mask_cfg["tau"])

    pi = resolve_pi(K, sbm_cfg["pi"])
    B_block = build_block_matrix(K, sbm_cfg["block_spec"])
    B_n = rho_n * B_block

    labels = sample_labels(n, K, pi, seeds["graph"])

    # Population matrix P is FILLED (as defined in the paper): P[i,j] = p_{c_i c_j}
    # for ALL i,j, including i == j. The analysis uses the diagonal of P.
    # Keep it filled here. The observed adjacency A (sample_adjacency below) is the
    # one that is hollow (A_ii = 0 by SBM convention). P_flat below is built from
    # triu_indices(k=1), so the diagonal of P never enters the LP loss anyway.
    P = B_n[labels][:, labels].astype(np.float64)

    A = sample_adjacency(P, seeds["edge"])

    train_mask, held_mask = sample_mask(n, tau, seeds["mask"])

    rows, cols = np.triu_indices(n, k=1)
    P_flat = P[rows, cols]
    A_flat = A[rows, cols]

    return {
        "n":          n,
        "K":          K,
        "tau":        tau,
        "rho_n":      rho_n,
        "pi":         pi,
        "B_block":    B_block,
        "B_n":        B_n,
        "labels":     labels,
        "P":          P,
        "A":          A,
        "rows":       rows,
        "cols":       cols,
        "train_mask": train_mask,
        "held_mask":  held_mask,
        "P_flat":     P_flat,
        "A_flat":     A_flat,
    }


# =============================================================================
# Encoder input: A_enc (masked adjacency)
# =============================================================================

def build_A_enc_edge_index(
    A: np.ndarray, train_mask: np.ndarray,
    rows: np.ndarray, cols: np.ndarray,
) -> np.ndarray:
    """Build the encoder input as a bidirectional edge_index (2, E) tensor.

    Edges are pairs (i,j) with (i,j) in T AND A_ij == 1. Both directions
    are included so message-passing can aggregate from both ends.
    """
    train_idx = np.where(train_mask)[0]
    ri, ci = rows[train_idx], cols[train_idx]
    edge_present = A[ri, ci].astype(bool)
    src = ri[edge_present]
    dst = ci[edge_present]
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    return np.stack([all_src, all_dst], axis=0).astype(np.int64)
