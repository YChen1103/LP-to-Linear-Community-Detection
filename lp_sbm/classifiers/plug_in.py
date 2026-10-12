"""
Plug-in classifier from m labels per class (main theorem, Part II).

Construction (as in the paper):
    Given a labeled set L_a subset of C_a with |L_a| = m_a, compute empirical
    prototypes mu_hat_a = (1/m_a) sum_{i in L_a} s_tilde_i where
    s_tilde_i = b_hat * 1_n + Z W z_i. Classify by nearest empirical prototype:
        c_hat(i) = arg min_a ||s_tilde_i - mu_hat_a||^2,
    which is equivalent to a linear classifier on z_i (see the Mahalanobis
    nearest-centroid form).

Reports:
    eta_L     = max_a (1/sqrt(n)) ||mu_hat_a - mu_a||
                (prototype estimation error vs oracle prototypes)
    Delta_hat = min_{a != b} (1/sqrt(n)) ||mu_hat_a - mu_hat_b||
                (empirical separation; should be close to Delta_s)
    error_rate

The label set L is supplied externally (see classifiers.sampling) so it can
be shared with the direct-LogReg baseline for a fair comparison on the same
draw of L.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from .sampling import sample_stratified_label_sets


def _plugin_one_resample(
    S_tilde: np.ndarray,         # (n, n) full score profile
    labels: np.ndarray,          # (n,) ground truth
    K: int,
    L: Dict[int, np.ndarray],    # pre-sampled stratified label set
    oracle_prototypes: np.ndarray,  # (K, n)
) -> Dict[str, object]:
    n = S_tilde.shape[0]
    # Empirical prototypes from supplied L
    mu_hat = np.zeros((K, n))
    for a in range(K):
        idx = L.get(a, np.array([], dtype=np.int64))
        if len(idx) > 0:
            mu_hat[a] = np.mean(S_tilde[idx], axis=0)
        else:
            mu_hat[a] = 0.0

    # Predict: nearest empirical prototype
    dists = np.zeros((n, K))
    for a in range(K):
        diff = S_tilde - mu_hat[a][None, :]
        dists[:, a] = np.sum(diff * diff, axis=1)
    pred = np.argmin(dists, axis=1).astype(np.int64)
    error_rate = float(np.mean(pred != labels))

    # eta_L: prototype estimation error
    eta_L = 0.0
    for a in range(K):
        diff_norm = float(np.linalg.norm(mu_hat[a] - oracle_prototypes[a]) / np.sqrt(n))
        eta_L = max(eta_L, diff_norm)

    # Empirical separation Delta_hat
    Delta_hat = float("inf")
    for a in range(K):
        for b in range(a + 1, K):
            sep = float(np.linalg.norm(mu_hat[a] - mu_hat[b]) / np.sqrt(n))
            Delta_hat = min(Delta_hat, sep)
    if Delta_hat == float("inf"):
        Delta_hat = float("nan")

    return {
        "pred":        pred,
        "error_rate":  error_rate,
        "eta_L":       eta_L,
        "Delta_hat":   Delta_hat,
    }


def plugin_classify(
    z: np.ndarray, W: np.ndarray, b: float,
    labels: np.ndarray, M_star: np.ndarray, K: int,
    *,
    label_sets: List[Dict[int, np.ndarray]] = None,
    m_per_class: int = None,
    n_resamples: int = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Run plug-in classifier with multiple resamples of L; aggregate.

    If `label_sets` is provided, those L sets are used directly (and
    `m_per_class`, `n_resamples`, `seed` are ignored). Otherwise, label sets
    are sampled internally via `sample_stratified_label_sets`.

    Aggregates over resamples: mean and std for error_rate, eta_L, Delta_hat.
    """
    n = z.shape[0]
    S_tilde = z @ W @ z.T + b
    oracle_prototypes = M_star[:, labels]

    if label_sets is None:
        if m_per_class is None or n_resamples is None:
            raise ValueError("Must provide either `label_sets` or both "
                             "`m_per_class` and `n_resamples`.")
        label_sets = sample_stratified_label_sets(
            labels, K, m_per_class, n_resamples, seed=seed,
        )

    err_list, eta_list, Delta_hat_list = [], [], []
    for L in label_sets:
        sub = _plugin_one_resample(S_tilde, labels, K, L, oracle_prototypes)
        err_list.append(sub["error_rate"])
        eta_list.append(sub["eta_L"])
        Delta_hat_list.append(sub["Delta_hat"])

    # Determine effective m_per_class from the supplied/sampled L for logging
    m_eff = int(m_per_class) if m_per_class is not None else (
        int(min(len(L_dict.get(a, [])) for a in range(K) for L_dict in label_sets[:1]))
        if label_sets else 0
    )

    return {
        "classifier":       "plug_in_nearest_centroid_score_profile",
        "m_per_class":      m_eff,
        "n_resamples":      len(label_sets),
        "error_rate_mean":  float(np.mean(err_list)) if err_list else float("nan"),
        "error_rate_std":   float(np.std(err_list)) if err_list else float("nan"),
        "error_rate_list":  err_list,
        "eta_L_mean":       float(np.mean(eta_list)) if eta_list else float("nan"),
        "eta_L_std":        float(np.std(eta_list)) if eta_list else float("nan"),
        "eta_L_list":       eta_list,
        "Delta_hat_mean":   float(np.nanmean(Delta_hat_list)) if Delta_hat_list else float("nan"),
        "Delta_hat_std":    float(np.nanstd(Delta_hat_list)) if Delta_hat_list else float("nan"),
        "Delta_hat_list":   Delta_hat_list,
    }
