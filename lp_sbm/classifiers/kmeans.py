"""
K-means clustering on the learned embeddings.

Two variants are provided:

(A) `kmeans_z_euclidean`: plain Euclidean K-means on z. This is a generic
    baseline that ignores the decoder geometry, so in general it does not
    match K-means on the score profiles (variant B).

(B) `kmeans_in_G`: K-means in R^d under the data-dependent PSD quadratic form
    G = W Z^T Z W (the same G as in the Mahalanobis nearest-centroid form).
    Implementation: factor
    G = M^T M (via eigendecomposition of G; entries with eigenvalues below a
    numerical threshold are zeroed, since G can be singular when W is rank-
    deficient or null(W) overlaps null(Z)). Then run plain Euclidean K-means on
    v_i := M z_i, since
        ||v_i - v_j||^2 = (z_i - z_j)^T G (z_i - z_j).
    This is equivalent to Euclidean K-means on the (n-dimensional) score
    profiles, but avoids materializing the score profiles at runtime.

Both variants do Hungarian permutation matching to align cluster labels with
ground-truth labels.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


# =============================================================================
# Helper: Hungarian-permuted error
# =============================================================================

def _hungarian_align_and_eval(
    pred_raw: np.ndarray, labels: np.ndarray, K: int,
) -> Dict[str, object]:
    confusion = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(labels, pred_raw):
        confusion[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    perm = np.zeros(K, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        perm[c] = r
    pred_perm = perm[pred_raw]
    return {
        "error_rate":  float(np.mean(pred_perm != labels)),
        "nmi":         float(normalized_mutual_info_score(labels, pred_raw)),
        "pred":        pred_perm,
        "permutation": perm.tolist(),
    }


# =============================================================================
# Variant A: plain Euclidean K-means on z (baseline)
# =============================================================================

def kmeans_z_euclidean(
    z: np.ndarray, labels: np.ndarray, K: int,
    seed: int, n_init: int = 50,
) -> Dict[str, object]:
    """Plain Euclidean K-means on the learned embeddings z.

    This is a baseline; it does NOT correspond to score-profile K-means in
    general. Use `kmeans_in_G` for that version.
    """
    km = KMeans(n_clusters=int(K), n_init=int(n_init), random_state=int(seed))
    pred_raw = km.fit_predict(z)
    return _hungarian_align_and_eval(pred_raw, labels, K)


# =============================================================================
# Variant B: K-means under PSD form G = W Z^T Z W
# =============================================================================

def _g_form_factor(
    z: np.ndarray, W: np.ndarray, *, eig_floor: float = 1e-10,
) -> "tuple[np.ndarray, int]":
    """Compute M (d x d) such that M^T M = G = W Z^T Z W.

    Eigenvalues of G below `eig_floor * max(eigvals)` are clamped to 0; this
    handles the singular case (G is PSD but may be singular when
    W is rank-deficient or when W has null directions overlapping null(Z)).
    Returns (M, effective_rank).
    """
    G = W @ (z.T @ z) @ W
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.clip(eigvals, 0.0, None)
    if eigvals.max() > 0:
        eigvals = np.where(eigvals < eig_floor * eigvals.max(), 0.0, eigvals)
    sqrt_eig = np.sqrt(eigvals)
    M = (sqrt_eig[:, None] * eigvecs.T)        # (d, d), M^T M = G
    eff_rank = int((eigvals > 0).sum())
    return M, eff_rank


def kmeans_in_G(
    z: np.ndarray, W: np.ndarray, labels: np.ndarray, K: int,
    seed: int, n_init: int = 50,
) -> Dict[str, object]:
    """G-form K-means: Euclidean K-means on M z, where M^T M = G = W Z^T Z W.

    Equivalent to (i) K-means on z under (z_i - z_j)^T G (z_i - z_j) distance,
    or (ii) Euclidean K-means on the score profiles s_tilde_i, but
    runs in d dimensions.
    """
    M, eff_rank = _g_form_factor(z, W)
    v = z @ M.T                                 # (n, d): row i is (M z_i)^T
    km = KMeans(n_clusters=int(K), n_init=int(n_init), random_state=int(seed))
    pred_raw = km.fit_predict(v)
    out = _hungarian_align_and_eval(pred_raw, labels, K)
    out["G_effective_rank"] = eff_rank
    return out


# =============================================================================
# Convenience: run both
# =============================================================================

def kmeans_classify(
    z: np.ndarray, labels: np.ndarray, K: int,
    seed: int, n_init: int = 50,
    *, W: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run both K-means variants. Returns a dict with both sets of metrics.

    Keys:
        z_euclid_error, z_euclid_nmi    plain Euclidean on z (baseline)
        G_form_error,   G_form_nmi      K-means under G (only if W given)
        G_effective_rank                rank of G (singular check)

    When W is None only the Euclidean variant is run. The Euclidean metrics
    are also exposed under the alias keys `error_rate`, `nmi` (kept for
    compatibility).
    """
    out_z = kmeans_z_euclidean(z, labels, K, seed=seed, n_init=n_init)
    res: Dict[str, object] = {
        "z_euclid_error":  out_z["error_rate"],
        "z_euclid_nmi":    out_z["nmi"],
        "z_euclid_perm":   out_z["permutation"],
        # Alias keys (= z-Euclidean), kept for compatibility
        "error_rate":      out_z["error_rate"],
        "nmi":             out_z["nmi"],
        "pred":            out_z["pred"],
        "permutation":     out_z["permutation"],
    }
    if W is not None:
        out_g = kmeans_in_G(z, W, labels, K, seed=seed + 1, n_init=n_init)
        res["G_form_error"]     = out_g["error_rate"]
        res["G_form_nmi"]       = out_g["nmi"]
        res["G_form_perm"]      = out_g["permutation"]
        res["G_effective_rank"] = out_g["G_effective_rank"]
    return res
