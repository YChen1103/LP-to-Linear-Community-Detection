"""
Oracle classifier h^star (main theorem, Part I).

This requires the true labels and oracle prototypes mu_a, both of which are
only available on synthetic data. It serves as the "best possible classifier
buildable from these embeddings + bilinear decoder structure" -- a useful
upper-performance reference.

Construction (nearest-prototype rule, linear in z):
    h^star(z) = arg max_a (w_a^T z + b_a)
    w_a = W Z^T mu_a,  b_a = b * mu_a^T 1 - 0.5 ||mu_a||^2
"""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np


def oracle_classifier_predictions(
    z: np.ndarray,            # (n, d) embeddings
    W: np.ndarray,            # (d, d) decoder W
    b: float,                 # decoder bias
    labels: np.ndarray,       # (n,) ground truth
    M_star: np.ndarray,       # (K, K) oracle score matrix
    K: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute oracle classifier predictions and error rate.

    Returns:
        pred       (n,)   predicted labels
        info       dict with 'error_rate', 'per_class_error'
    """
    n = z.shape[0]
    # Build full score profile S_tilde[i, j] = b + z_i^T W z_j
    S_tilde = z @ W @ z.T + b  # (n, n)

    # Oracle prototypes mu_a in R^n: (mu_a)_j = M_star[a, c_j]
    mu = M_star[:, labels]  # (K, n)

    # Nearest-prototype: arg min_a ||S_tilde[i, :] - mu[a, :]||^2
    # Equivalent to max_a (mu_a^T s_tilde_i - 0.5 ||mu_a||^2)
    # We compute the squared-distance form directly:
    dists = np.zeros((n, K))
    for a in range(K):
        diff = S_tilde - mu[a][None, :]
        dists[:, a] = np.sum(diff * diff, axis=1)
    pred = np.argmin(dists, axis=1).astype(np.int64)

    error_rate = float(np.mean(pred != labels))
    per_class = {}
    for a in range(K):
        mask = labels == a
        if mask.sum() > 0:
            per_class[str(int(a))] = float(np.mean(pred[mask] != labels[mask]))
    return pred, {"error_rate": error_rate, "per_class_error": per_class}
