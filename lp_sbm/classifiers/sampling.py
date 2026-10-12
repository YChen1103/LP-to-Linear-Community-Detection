"""
Stratified label sampling, shared by the plug-in and direct classifiers.

Why this exists. The plug-in classifier (main theorem, Part II) and the direct
LogReg baseline are both built from a labeled set L = union_a L_a obtained by
stratified uniform-without-replacement sampling. To make their errors
*directly comparable on the same draws of L*, we generate the L_a list ONCE
here and feed the SAME list into both classifiers. Otherwise, two different
random subsets are drawn (one per classifier) and the comparison conflates
"which classifier is better" with "which random labeled set you got".

Each call returns a list of length `n_resamples`, where each element is a
dict {a: indices} mapping community label a to the chosen indices in C_a.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np


def sample_stratified_label_sets(
    labels: np.ndarray,
    K: int,
    m_per_class: int,
    n_resamples: int,
    seed: int,
) -> List[Dict[int, np.ndarray]]:
    """Draw `n_resamples` independent stratified L sets.

    Each L is represented as a dict {a: array of indices into [n]}, where for
    each community a, |L_a| = min(m_per_class, |C_a|) and indices are sampled
    uniformly without replacement from C_a.

    Returns a list of length `n_resamples`. The same RNG seed is used to
    generate the entire list, so two callers with the same seed get the same
    list of L sets.
    """
    rng = np.random.default_rng(int(seed))
    out: List[Dict[int, np.ndarray]] = []
    for _ in range(int(n_resamples)):
        L: Dict[int, np.ndarray] = {}
        for a in range(K):
            class_idx = np.where(labels == a)[0]
            if len(class_idx) == 0:
                L[a] = np.array([], dtype=np.int64)
                continue
            m_actual = min(int(m_per_class), len(class_idx))
            L[a] = rng.choice(class_idx, size=m_actual, replace=False).astype(np.int64)
        out.append(L)
    return out
