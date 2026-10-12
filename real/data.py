"""
Real-graph loading, and packaging into the exact dict shape `lp_sbm` expects.

The point of this module is that NOTHING in `lp_sbm/` has to change. The
trainer consumes only

    n, K, rows, cols, train_mask, held_mask, A_flat, A

from the data dict, and touches `sbm_oracle` only inside `compute_bias_init`
(which, under the paper's `bias_init_mode: zero`, does not use it). So a real
graph can be dropped in by producing a data dict with the same keys.

Two quantities have no counterpart on a real graph and are therefore *fitted*:

    B_hat[a,b]  empirical edge density between proxy communities a and b
    P_hat       = B_hat[c_i, c_j], the best block-constant approximation of the
                  real graph given the proxy labels

`B_hat` is fed to `lp_sbm.oracle.compute_sbm_oracle`, which makes the existing
capacity resolution (`W_max_spec: from_capacity_ratio`, `B_0_spec: from_b_star`)
work unchanged. This is a *modelling convenience for setting the capacity
budget*, not a claim that the graph is an SBM. Every quantity that depends on
P_hat (excess risk, mu_B, Delta_s, the oracle classifier) is therefore
"relative to the fitted block model" and is NOT reported as a test of the
theorem. See `real/README.md`.

Supported sources
-----------------
    demo_sbm     planted partition generated on the fly; needs no download.
                 Used by the test suite and by `sanity_check.py --source demo_sbm`.
    email_eu     SNAP email-Eu-core: `email-Eu-core.txt` +
                 `email-Eu-core-department-labels.txt`. n=1005, mean degree ~33.
    cora         LINQS Cora: `cora.cites` + `cora.content`. n=2708, mean degree ~3.9.
                 Node attributes in cora.content are IGNORED by design.
    edgelist     generic: `--edges` two-column file, `--labels` two-column
                 (node label) or one-label-per-line file.
    npz          numpy archive with arrays `edges` (E,2) and `labels` (n,).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from lp_sbm.sbm import sample_mask


# =============================================================================
# Loading
# =============================================================================

def _dense_from_edges(edges: np.ndarray, n: int) -> np.ndarray:
    """Symmetric hollow 0/1 adjacency from an (E, 2) edge array."""
    A = np.zeros((n, n), dtype=np.float32)
    if len(edges) > 0:
        e = np.asarray(edges, dtype=np.int64)
        A[e[:, 0], e[:, 1]] = 1.0
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def _largest_connected_component(A: np.ndarray) -> np.ndarray:
    """Indices of the largest connected component (plain BFS, no scipy needed)."""
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    best: np.ndarray = np.array([], dtype=np.int64)
    nbrs = [np.flatnonzero(A[i]) for i in range(n)]
    for s in range(n):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in nbrs[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        if len(comp) > len(best):
            best = np.array(sorted(comp), dtype=np.int64)
    return best


def _restrict_to_top_k_classes(labels: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    """Indices of nodes whose class is among the `top_k` largest. None -> all."""
    if top_k is None:
        return np.arange(len(labels), dtype=np.int64)
    vals, counts = np.unique(labels, return_counts=True)
    keep = set(vals[np.argsort(counts)[::-1][:int(top_k)]].tolist())
    return np.array([i for i, c in enumerate(labels) if c in keep], dtype=np.int64)


def _relabel(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    vals = np.unique(labels)
    remap = {v: i for i, v in enumerate(vals)}
    return np.array([remap[v] for v in labels], dtype=np.int64), len(vals)


def _read_two_col(path: Path) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return np.array(rows, dtype=object)


def _demo_sbm(n: int, K: int, seed: int, p_in: float, p_out: float
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Planted partition, for offline smoke tests only."""
    rng = np.random.default_rng(seed)
    labels = np.sort(rng.integers(0, K, size=n)).astype(np.int64)
    P = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    U = rng.random((n, n))
    A = ((np.triu(U, 1) < np.triu(P, 1))).astype(np.float32)
    A = A + A.T
    return A, labels


def load_graph(
    source: str,
    *,
    root: Optional[str] = None,
    edges_file: Optional[str] = None,
    labels_file: Optional[str] = None,
    top_k: Optional[int] = None,
    largest_cc: bool = True,
    demo_n: int = 1000,
    demo_K: int = 3,
    demo_seed: int = 0,
    demo_p_in: float = 0.10,
    demo_p_out: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """Return (A, labels, K, meta). A is dense symmetric hollow float32."""
    src = str(source).lower()

    if src == "demo_sbm":
        A, labels = _demo_sbm(demo_n, demo_K, demo_seed, demo_p_in, demo_p_out)

    elif src == "email_eu":
        d = Path(root or ".")
        e = _read_two_col(d / "email-Eu-core.txt").astype(np.int64)
        lab_raw = _read_two_col(d / "email-Eu-core-department-labels.txt").astype(np.int64)
        n = int(max(e.max(), lab_raw[:, 0].max())) + 1
        labels = np.full(n, -1, dtype=np.int64)
        labels[lab_raw[:, 0]] = lab_raw[:, 1]
        A = _dense_from_edges(e, n)

    elif src == "cora":
        d = Path(root or ".")
        content = [ln.split() for ln in open(d / "cora.content") if ln.strip()]
        ids = [c[0] for c in content]
        # column -1 is the class string; the feature columns are deliberately unused
        lab_str = [c[-1] for c in content]
        idx = {pid: i for i, pid in enumerate(ids)}
        n = len(ids)
        labels, _ = _relabel(np.array(lab_str, dtype=object))
        cites = _read_two_col(d / "cora.cites")
        e = np.array([[idx[a], idx[b]] for a, b in cites
                      if a in idx and b in idx], dtype=np.int64)
        A = _dense_from_edges(e, n)

    elif src == "edgelist":
        if edges_file is None or labels_file is None:
            raise ValueError("source=edgelist needs --edges-file and --labels-file")
        raw = _read_two_col(Path(edges_file))
        nodes = sorted({x for pair in raw for x in pair})
        idx = {v: i for i, v in enumerate(nodes)}
        e = np.array([[idx[a], idx[b]] for a, b in raw], dtype=np.int64)
        A = _dense_from_edges(e, len(nodes))
        txt = [ln.split() for ln in open(labels_file) if ln.strip()]
        if len(txt[0]) >= 2:
            labels = np.full(len(nodes), -1, dtype=object)
            for a, b in [(t[0], t[1]) for t in txt]:
                if a in idx:
                    labels[idx[a]] = b
        else:
            labels = np.array([t[0] for t in txt], dtype=object)
        labels, _ = _relabel(labels)

    elif src == "npz":
        z = np.load(root)
        A = _dense_from_edges(np.asarray(z["edges"]), int(np.asarray(z["labels"]).shape[0]))
        labels = np.asarray(z["labels"]).astype(np.int64)

    else:
        raise ValueError(f"unknown source: {source!r}")

    labels = np.asarray(labels).astype(np.int64)
    meta: Dict[str, Any] = {"source": src, "n_raw": int(A.shape[0]),
                            "n_edges_raw": int(A.sum() / 2)}

    keep = np.array([i for i in range(len(labels)) if labels[i] >= 0], dtype=np.int64)
    keep = np.intersect1d(keep, _restrict_to_top_k_classes(labels, top_k))
    A, labels = A[np.ix_(keep, keep)], labels[keep]

    if largest_cc and A.shape[0] > 0:
        cc = _largest_connected_component(A)
        A, labels = A[np.ix_(cc, cc)], labels[cc]

    labels, K = _relabel(labels)
    meta.update(n=int(A.shape[0]), K=int(K), n_edges=int(A.sum() / 2),
                mean_degree=float(A.sum() / max(A.shape[0], 1)),
                density=float(A.sum() / max(A.shape[0] * (A.shape[0] - 1), 1)),
                class_sizes=np.bincount(labels, minlength=K).tolist(),
                majority_class_frac=float(np.bincount(labels).max() / len(labels)),
                edge_homophily=_edge_homophily(A, labels))
    return A.astype(np.float32), labels, int(K), meta


def _edge_homophily(A: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of edges whose endpoints share a proxy community label."""
    r, c = np.triu_indices(A.shape[0], k=1)
    m = A[r, c] > 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(labels[r[m]] == labels[c[m]]))


# =============================================================================
# Block-model fit (used ONLY to set the capacity budget; see module docstring)
# =============================================================================

def fit_block_matrix(A: np.ndarray, labels: np.ndarray, K: int,
                     *, eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical block densities B_hat and community proportions pi_hat."""
    n = A.shape[0]
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    B = np.zeros((K, K), dtype=np.float64)
    for a in range(K):
        ia = np.flatnonzero(labels == a)
        for b in range(a, K):
            ib = np.flatnonzero(labels == b)
            if a == b:
                possible = len(ia) * (len(ia) - 1) / 2.0
                present = A[np.ix_(ia, ia)].sum() / 2.0
            else:
                possible = float(len(ia) * len(ib))
                present = A[np.ix_(ia, ib)].sum()
            val = present / possible if possible > 0 else 0.0
            B[a, b] = B[b, a] = val
    B = np.clip(B, eps, 1.0 - eps)
    return B, counts / max(n, 1)


# =============================================================================
# Packaging into the lp_sbm data dict
# =============================================================================

def prepare_real_data(
    A: np.ndarray, labels: np.ndarray, K: int,
    mask_cfg: Dict[str, Any], seeds: Dict[str, int],
) -> Dict[str, Any]:
    """Same keys as `lp_sbm.sbm.prepare_sbm_data`, built from a real graph.

    The mask is drawn by the repository's own `sample_mask`, so the held-out
    protocol is byte-identical to the synthetic experiments.
    """
    n = int(A.shape[0])
    tau = float(mask_cfg["tau"])

    B_hat, pi_hat = fit_block_matrix(A, labels, K)
    P_hat = B_hat[labels][:, labels].astype(np.float64)

    train_mask, held_mask = sample_mask(n, tau, seeds["mask"])
    rows, cols = np.triu_indices(n, k=1)

    return {
        "n":          n,
        "K":          int(K),
        "tau":        tau,
        "rho_n":      1.0,
        "pi":         pi_hat,
        "B_block":    B_hat,
        "B_n":        B_hat,
        "labels":     np.asarray(labels, dtype=np.int64),
        "P":          P_hat,
        "A":          np.asarray(A, dtype=np.float32),
        "rows":       rows,
        "cols":       cols,
        "train_mask": train_mask,
        "held_mask":  held_mask,
        "P_flat":     P_hat[rows, cols],
        "A_flat":     np.asarray(A, dtype=np.float32)[rows, cols],
    }
