"""Sanity tests for lp_sbm.oracle."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp_sbm.oracle import (
    safe_logit, binary_entropy, sigmoid, mu_B_at,
    compute_capacity_required, compute_Delta_s_sq, compute_sbm_oracle,
    build_oracle_prototypes,
)


def test_logit_entropy_inverse():
    """logit(sigma(x)) == x and entropy is symmetric around 0.5."""
    for x in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        p = sigmoid(x)
        assert abs(safe_logit(float(p)) - x) < 1e-6
    # Entropy at 0.5 = log 2
    assert abs(binary_entropy(0.5) - np.log(2.0)) < 1e-9
    # Entropy is symmetric
    for p in [0.1, 0.3, 0.7, 0.9]:
        assert abs(binary_entropy(p) - binary_entropy(1 - p)) < 1e-9


def test_mu_B_at_known_values():
    """mu_B(0) = 1/4, mu_B(B) -> 0 as |B| grows."""
    assert abs(mu_B_at(0.0) - 0.25) < 1e-9
    assert mu_B_at(10.0) < 1e-3
    assert mu_B_at(20.0) < 1e-7


def test_capacity_symmetric_2_block():
    """For a symmetric 2-block SBM with a = -b (i.e. p_in + p_out = 1),
    ||M_tilde||_op has a known form.

    M_star = [[a, b], [b, a]]; eigenvalues of M_star - beta 11^T are
        - a - b (on eigenvector (1,-1)/sqrt(2), beta-independent)
        - a + b - 2 beta (on (1,1)/sqrt(2))
    so ||M_tilde||_op = max(|a-b|, |a+b-2 beta|), minimized when |a+b-2 beta| <= |a-b|.
    With a = -b (i.e. p_in + p_out = 1), the optimum is the entire interval
    [-(a-b)/2, (a-b)/2]; b_star is non-unique. We only assert cap = |a-b|.
    """
    p_in, p_out = 0.7, 0.3
    a = safe_logit(p_in)
    b = safe_logit(p_out)
    M_star = np.array([[a, b], [b, a]])
    cap, b_star = compute_capacity_required(M_star)
    # Capacity is well-defined and equals |a - b|
    assert abs(cap - abs(a - b)) < 1e-6, f"cap={cap}, expected {abs(a-b)}"
    # b_star should still satisfy ||M_star - b_star 11^T||_op == cap
    M_tilde = M_star - b_star * np.ones((2, 2))
    assert abs(float(np.linalg.norm(M_tilde, ord=2)) - cap) < 1e-6


def test_capacity_K3_obstruction_example():
    """Example from the paper: hub SBM with K=3 has indefinite M_tilde."""
    # p_11 = p_22 = p_33 = p_13 = 0.1; p_12 = p_23 = 0.9
    P = np.array([
        [0.1, 0.9, 0.1],
        [0.9, 0.1, 0.9],
        [0.1, 0.9, 0.1],
    ])
    M_star = np.array([[safe_logit(p) for p in row] for row in P])
    cap, b_star = compute_capacity_required(M_star)
    # x = (1, -2, 1) is orthogonal to ones; check x^T M^* x < 0
    x = np.array([1, -2, 1], dtype=float)
    assert abs(x @ np.ones(3)) < 1e-9
    quad = float(x @ M_star @ x)
    # Paper claims this equals -16 log 9
    assert quad < 0
    assert abs(quad - (-16.0 * np.log(9.0))) < 1e-6
    # cap > 0 (M_tilde nonzero)
    assert cap > 0


def test_Delta_s_sq_symmetric_2_block():
    """For balanced 2-block SBM, Delta_s^2 = (1/2)(a-b)^2 + (1/2)(b-a)^2 = (a-b)^2.

    Delta_s^2 = min_{a!=b} sum_c pi_c (m*_{ac} - m*_{bc})^2.
    For 2-SBM with c in {0, 1}: row 0 - row 1 = (a-b, b-a).
    Sum_c pi_c (...)^2 = pi_0 (a-b)^2 + pi_1 (b-a)^2 = (a-b)^2 with uniform pi.
    """
    p_in, p_out = 0.7, 0.3
    a = safe_logit(p_in)
    b = safe_logit(p_out)
    M_star = np.array([[a, b], [b, a]])
    pi = np.array([0.5, 0.5])
    Delta_sq = compute_Delta_s_sq(M_star, pi)
    expected = (a - b) ** 2
    assert abs(Delta_sq - expected) < 1e-9, f"got {Delta_sq}, expected {expected}"


def test_oracle_prototype_block_constancy():
    """Oracle prototype mu_a in R^n satisfies (mu_a)_j = m*_{a, c_j}."""
    M_star = np.array([[1.0, -1.0], [-1.0, 1.0]])
    labels = np.array([0, 1, 0, 1, 0])
    K = 2
    proto = build_oracle_prototypes(M_star, labels, K)
    assert proto.shape == (K, len(labels))
    # mu_0 should be M_star[0, c_j] for each j
    expected_mu0 = np.array([M_star[0, c] for c in labels])
    np.testing.assert_array_almost_equal(proto[0], expected_mu0)


def test_compute_sbm_oracle_full():
    """End-to-end: compute_sbm_oracle returns a complete dict."""
    K = 3
    B = np.array([[0.7, 0.2, 0.2],
                  [0.2, 0.7, 0.2],
                  [0.2, 0.2, 0.7]])
    pi = np.ones(K) / K
    out = compute_sbm_oracle(B, pi)
    keys = ["M_star", "H_matrix", "B_star", "b_star",
            "capacity_required", "Delta_s", "Delta_s_sq", "pi"]
    for k in keys:
        assert k in out, f"Missing key {k!r}"
    assert out["M_star"].shape == (K, K)
    assert out["B_star"] > 0
    assert out["capacity_required"] > 0
    assert out["Delta_s_sq"] > 0


if __name__ == "__main__":
    test_logit_entropy_inverse()
    test_mu_B_at_known_values()
    test_capacity_symmetric_2_block()
    test_capacity_K3_obstruction_example()
    test_Delta_s_sq_symmetric_2_block()
    test_oracle_prototype_block_constancy()
    test_compute_sbm_oracle_full()
    print("All oracle tests passed.")
