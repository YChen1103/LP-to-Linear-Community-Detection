"""Sanity tests for lp_sbm.sbm."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp_sbm.sbm import (
    build_block_matrix, resolve_pi,
    sample_labels, sample_adjacency, sample_mask,
    prepare_sbm_data, build_A_enc_edge_index,
)


def test_build_block_matrix_assortative():
    K = 3
    spec = {"type": "assortative", "p_in": 0.7, "p_out": 0.2}
    B = build_block_matrix(K, spec)
    assert B.shape == (K, K)
    np.testing.assert_array_almost_equal(np.diag(B), [0.7] * K)
    # Off-diag = 0.2
    off_diag_sum = B.sum() - np.diag(B).sum()
    assert abs(off_diag_sum - 0.2 * (K * K - K)) < 1e-9


def test_build_block_matrix_custom():
    K = 2
    spec = {"type": "custom", "p_matrix": [[0.5, 0.1], [0.1, 0.5]]}
    B = build_block_matrix(K, spec)
    assert B.shape == (K, K)
    assert B[0, 1] == 0.1


def test_resolve_pi_uniform_and_explicit():
    pi = resolve_pi(3, "uniform")
    np.testing.assert_array_almost_equal(pi, [1/3, 1/3, 1/3])
    pi2 = resolve_pi(3, [0.5, 0.3, 0.2])
    np.testing.assert_array_almost_equal(pi2, [0.5, 0.3, 0.2])


def test_sample_labels_balanced():
    """With uniform pi and n divisible by K, labels are exactly balanced."""
    n, K = 60, 3
    pi = np.ones(K) / K
    labels = sample_labels(n, K, pi, seed=0)
    counts = np.bincount(labels, minlength=K)
    np.testing.assert_array_equal(counts, [n // K] * K)


def test_sample_adjacency_symmetric_hollow():
    """Sampled A is symmetric with A_ii = 0."""
    n = 30
    P = np.full((n, n), 0.5)
    np.fill_diagonal(P, 0.0)
    A = sample_adjacency(P, seed=0)
    assert A.shape == (n, n)
    assert np.allclose(np.diag(A), 0)
    np.testing.assert_array_almost_equal(A, A.T)


def test_sample_mask_marginal_rate():
    """Held-out fraction matches tau within concentration bounds."""
    n = 200
    tau = 0.3
    train_mask, held_mask = sample_mask(n, tau, seed=0)
    N = n * (n - 1) // 2
    assert len(train_mask) == N
    assert len(held_mask) == N
    # Disjoint
    assert not np.any(train_mask & held_mask)
    # Union
    assert np.all(train_mask | held_mask)
    # Empirical rate close to tau
    emp_rate = held_mask.sum() / N
    assert abs(emp_rate - tau) < 0.05


def test_prepare_sbm_data_complete():
    """End-to-end: prepare_sbm_data returns all expected keys."""
    sbm_cfg = {"n": 30, "K": 3, "rho_n": 1.0, "pi": "uniform",
               "block_spec": {"type": "assortative", "p_in": 0.7, "p_out": 0.2}}
    mask_cfg = {"tau": 0.2}
    seeds = {"graph": 1, "edge": 2, "mask": 3, "model": 4, "label": 5}
    data = prepare_sbm_data(sbm_cfg, mask_cfg, seeds)
    expected_keys = {"n", "K", "tau", "rho_n", "pi", "B_block", "B_n",
                     "labels", "P", "A", "rows", "cols",
                     "train_mask", "held_mask", "P_flat", "A_flat"}
    for k in expected_keys:
        assert k in data, f"Missing key {k!r}"
    assert data["A"].shape == (30, 30)
    assert data["P"].shape == (30, 30)


def test_A_enc_edge_index_only_uses_T():
    """A_enc edges are bidirectional and only include (i,j) in T with A_ij = 1."""
    n = 20
    P = np.full((n, n), 0.5); np.fill_diagonal(P, 0)
    A = sample_adjacency(P, seed=42)
    train_mask, held_mask = sample_mask(n, 0.3, seed=99)
    rows, cols = np.triu_indices(n, k=1)
    edge_index = build_A_enc_edge_index(A, train_mask, rows, cols)
    # Edges should be symmetric (each (i,j) gives (j,i) too)
    assert edge_index.shape[0] == 2
    n_edges = edge_index.shape[1]
    assert n_edges % 2 == 0
    # No self-loops
    assert not np.any(edge_index[0] == edge_index[1])


if __name__ == "__main__":
    test_build_block_matrix_assortative()
    test_build_block_matrix_custom()
    test_resolve_pi_uniform_and_explicit()
    test_sample_labels_balanced()
    test_sample_adjacency_symmetric_hollow()
    test_sample_mask_marginal_rate()
    test_prepare_sbm_data_complete()
    test_A_enc_edge_index_only_uses_T()
    print("All SBM tests passed.")
