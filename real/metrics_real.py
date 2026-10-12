"""
Metrics that are well defined on a real graph.

Deliberately a strict subset of `lp_sbm.metrics.compute_all_metrics`. Anything
that needs the true P (population excess risk, mu_B, Delta_s, the oracle
classifier, eta_L) is NOT computed here, because on a real graph those would be
relative to a fitted block model rather than to truth, and reporting them would
invite exactly the over-claim the experiment is meant to avoid.

What IS computed:

    LP side (no labels used)
        R_T_emp, R_H_emp     empirical BCE on visible / held-out pairs
        auc_H, ap_H          held-out ranking quality; on sparse real graphs
                             BCE is dominated by non-edges and has almost no
                             dynamic range, so AUC/AP is the usable axis
        pos_rate_H           held-out edge density, for context

    Downstream side (proxy labels used only here)
        direct_m{m}_error_mean    LogReg on z, fit on m labels/class, evaluated
                                  on all n nodes -- identical protocol to
                                  lp_sbm/classifiers/direct.py
        plug_in_m{m}_error_mean   main theorem, Part II plug-in. Its *prediction*
                                  rule needs no oracle quantity; only the eta_L
                                  diagnostic does, and eta_L is dropped here.
        kmeans_z_error            unsupervised, Hungarian-aligned
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from lp_sbm.classifiers.direct import direct_classify
from lp_sbm.classifiers.kmeans import kmeans_z_euclidean
from lp_sbm.classifiers.plug_in import plugin_classify
from lp_sbm.classifiers.sampling import sample_stratified_label_sets


# =============================================================================
# LP side
# =============================================================================

def pair_scores(z: np.ndarray, W: np.ndarray, b: float,
                rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """s_ij = b + z_i^T W z_j for the given upper-triangular pair list.

    Computed without materialising the n x n score matrix.
    """
    ZW = z @ W
    return np.einsum("ij,ij->i", ZW[rows], z[cols]) + float(b)


def _bce_with_logits(s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Numerically stable log(1+e^s) - a*s."""
    return np.logaddexp(0.0, s) - a * s


def lp_metrics(z: np.ndarray, W: np.ndarray, b: float,
               data: Dict[str, Any]) -> Dict[str, float]:
    rows, cols = data["rows"], data["cols"]
    A_flat = np.asarray(data["A_flat"], dtype=np.float64)
    tr, he = data["train_mask"], data["held_mask"]

    s = pair_scores(z, W, b, rows, cols)
    loss = _bce_with_logits(s, A_flat)

    out: Dict[str, float] = {
        "R_T_emp": float(loss[tr].mean()) if tr.any() else float("nan"),
        "R_H_emp": float(loss[he].mean()) if he.any() else float("nan"),
        "pos_rate_H": float(A_flat[he].mean()) if he.any() else float("nan"),
    }

    y, sc = A_flat[he], s[he]
    if he.any() and 0 < y.sum() < len(y):
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["auc_H"] = float(roc_auc_score(y, sc))
        out["ap_H"] = float(average_precision_score(y, sc))
    else:
        out["auc_H"] = out["ap_H"] = float("nan")
    return out


# =============================================================================
# Downstream side
# =============================================================================

def probe_metrics(
    z: np.ndarray, W: np.ndarray, b: float,
    labels: np.ndarray, K: int, classifier_seed: int,
    *,
    m_list: List[int],
    n_resamples: int = 5,
    do_plugin: bool = True,
    do_kmeans: bool = False,
    kmeans_n_init: int = 20,
    M_hat: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Few-label probes on z, seeded exactly as in lp_sbm/metrics.py."""
    out: Dict[str, float] = {}
    for m in m_list:
        L = sample_stratified_label_sets(
            labels, K, int(m), int(n_resamples),
            seed=int(classifier_seed) + int(m) * 1009,
        )
        d = direct_classify(z, labels, K, label_sets=L,
                            seed=int(classifier_seed) + int(m) * 1009 + 1)
        out[f"direct_m{m}_error_mean"] = float(d["error_rate_mean"])
        out[f"direct_m{m}_error_std"] = float(d["error_rate_std"])

        if do_plugin:
            # M_hat only feeds the eta_L diagnostic, which we drop; the
            # prediction rule uses S_tilde = z W z^T + b and the labelled set.
            Mh = M_hat if M_hat is not None else np.zeros((K, K))
            p = plugin_classify(z, W, b, labels, Mh, K, label_sets=L)
            out[f"plug_in_m{m}_error_mean"] = float(p["error_rate_mean"])
            out[f"plug_in_m{m}_error_std"] = float(p["error_rate_std"])
            out[f"plug_in_m{m}_Delta_hat_mean"] = float(p["Delta_hat_mean"])

    if do_kmeans:
        r = kmeans_z_euclidean(z, labels, K, seed=int(classifier_seed) + 2,
                               n_init=int(kmeans_n_init))
        out["kmeans_z_error"] = float(r["error_rate"])
        out["kmeans_z_nmi"] = float(r["nmi"])
    return out


def compute_real_metrics(
    z: np.ndarray, W: np.ndarray, b: float,
    *, data: Dict[str, Any], classifier_seed: int,
    m_list: List[int], n_resamples: int = 5,
    do_plugin: bool = True, do_kmeans: bool = False,
    kmeans_n_init: int = 20, M_hat: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    out = lp_metrics(z, W, b, data)
    out.update(probe_metrics(
        z, W, b, data["labels"], int(data["K"]), classifier_seed,
        m_list=m_list, n_resamples=n_resamples,
        do_plugin=do_plugin, do_kmeans=do_kmeans,
        kmeans_n_init=kmeans_n_init, M_hat=M_hat,
    ))
    return out
