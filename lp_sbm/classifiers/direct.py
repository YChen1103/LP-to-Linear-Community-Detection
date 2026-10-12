"""
Direct classifier baseline: train sklearn LogisticRegression on z using m
labels per class, evaluate on all n.

This is the baseline alternative to the plug-in classifier. Both are built
from the SAME labeled set L (passed in via `label_sets`) so that errors are
directly comparable on the same draws of L. Theoretically:
    plug-in:   structured (nearest-prototype = linear), paper-bounded
    direct:    unstructured logreg, only VC-style bounds available
The comparison reveals whether plug-in's structure helps in practice.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from .sampling import sample_stratified_label_sets


def _direct_one_resample(
    z: np.ndarray, labels: np.ndarray, K: int,
    L: Dict[int, np.ndarray], seed_for_lr: int = 0,
) -> Dict[str, object]:
    """Train LogReg on z using indices in L, evaluate on all n."""
    train_idx = np.concatenate([L[a] for a in range(K) if len(L.get(a, [])) > 0]) \
        if any(len(L.get(a, [])) > 0 for a in range(K)) else np.array([], dtype=np.int64)

    if len(train_idx) == 0 or len(np.unique(labels[train_idx])) < 2:
        return {"error_rate": float("nan"), "pred": None, "L_size": int(len(train_idx))}

    clf = LogisticRegression(max_iter=1000, random_state=int(seed_for_lr))
    clf.fit(z[train_idx], labels[train_idx])
    pred = clf.predict(z).astype(np.int64)
    error_rate = float(np.mean(pred != labels))
    return {"error_rate": error_rate, "pred": pred, "L_size": int(len(train_idx))}


def direct_classify(
    z: np.ndarray, labels: np.ndarray, K: int,
    *,
    label_sets: List[Dict[int, np.ndarray]] = None,
    m_per_class: int = None,
    n_resamples: int = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Run direct LogReg with multiple resamples of L; aggregate.

    If `label_sets` is provided, those L sets are used directly. Otherwise,
    sampled internally. Pass the SAME `label_sets` to both this and
    `plugin_classify` for fair head-to-head comparison.
    """
    if label_sets is None:
        if m_per_class is None or n_resamples is None:
            raise ValueError("Must provide either `label_sets` or both "
                             "`m_per_class` and `n_resamples`.")
        label_sets = sample_stratified_label_sets(
            labels, K, m_per_class, n_resamples, seed=seed,
        )

    errs = []
    for r, L in enumerate(label_sets):
        out = _direct_one_resample(z, labels, K, L, seed_for_lr=int(seed) + r)
        if not np.isnan(out["error_rate"]):
            errs.append(out["error_rate"])

    m_eff = int(m_per_class) if m_per_class is not None else (
        int(min(len(L.get(a, [])) for a in range(K))) if label_sets else 0
    )
    if not errs:
        return {
            "classifier":      "direct_logreg",
            "m_per_class":     m_eff,
            "n_resamples":     len(label_sets),
            "error_rate_mean": float("nan"),
            "error_rate_std":  float("nan"),
            "error_rate_list": [],
        }
    return {
        "classifier":      "direct_logreg",
        "m_per_class":     m_eff,
        "n_resamples":     len(label_sets),
        "error_rate_mean": float(np.mean(errs)),
        "error_rate_std":  float(np.std(errs)),
        "error_rate_list": errs,
    }
