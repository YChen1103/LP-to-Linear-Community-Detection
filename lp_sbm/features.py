"""
Initial node features X for the encoder.

All features are computed from the masked encoder input A_enc (or its edge
representation), never from the full adjacency A. This preserves held-out
independence (see the paper).

Available modes:
    'ones'           (n, 1)        constant 1
    'random_normal'  (n, dim)      i.i.d. N(0, I), seeded by feature_seed
    'degree'         (n, 1)        normalized node degree (z-scored)
    'laplacian_pe'   (n, dim)      eigenvectors of normalized Laplacian
                                   I - D^{-1/2} A_enc D^{-1/2} (skipping the
                                   trivial constant eigenvector). O(n^3) precompute.
    'spectral_pe'    (n, dim)      sqrt(n) * top-K eigenvectors of A_enc itself,
                                   as in the spectral-PE GIN approximation
                                   result. O(n^3) precompute.

The `pe_normalize` argument controls how Laplacian PE columns are post-processed:
    'raw'     keep the orthonormal eigenvectors as returned by eigh
    'std'     divide each column by its std (heuristic; standard in many GNN libs;
              breaks orthonormality but rescales each PE coord to O(1))
    'sqrt_n'  multiply each column by sqrt(n), matching the spectral-PE convention
              (entries become O(1))

Default for laplacian_pe is 'std' (common GNN-library convention); spectral_pe
always uses sqrt(n), the scaling assumed by the spectral-PE GIN approximation
result.
"""
from __future__ import annotations
from typing import Optional

import torch


def build_features(
    mode: str,
    num_nodes: int,
    edge_index: torch.Tensor,
    *,
    feature_seed: int = 0,
    dim: int = 4,
    device: Optional[torch.device] = None,
    pe_normalize: str = "std",
) -> torch.Tensor:
    """Construct node-feature matrix X.

    Returns a tensor of shape (num_nodes, input_dim) where input_dim depends
    on `mode`. Use `feature_input_dim(mode, dim)` to query the dim.

    Notes
    -----
    `feature_seed` is the seed used for `random_normal` mode. It is intentionally
    decoupled from the model/optimizer seed so that features are sampled once per
    base seed and held fixed across optimizer restarts (held-out independence:
    the analysis conditions on X).
    """
    mode = str(mode).lower()
    device = device or edge_index.device

    if mode == "ones":
        return torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

    if mode == "random_normal":
        gen = torch.Generator(device="cpu").manual_seed(int(feature_seed))
        x = torch.randn(num_nodes, int(dim), generator=gen).to(device).float()
        return x

    if mode == "degree":
        src, dst = edge_index
        deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        if dst.numel() > 0:
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = (deg - deg.mean()) / (deg.std(unbiased=False) + 1e-6)
        return deg.unsqueeze(-1)

    if mode == "laplacian_pe":
        return _laplacian_pe(num_nodes, edge_index, dim, device,
                             pe_normalize=pe_normalize)

    if mode == "spectral_pe":
        return _spectral_pe(num_nodes, edge_index, dim, device)

    raise ValueError(
        f"Unknown feature mode: {mode!r}. "
        f"Choose from: ones, random_normal, degree, laplacian_pe, spectral_pe."
    )


def feature_input_dim(mode: str, dim: int) -> int:
    """Get the dimensionality of features produced by `build_features`."""
    mode = str(mode).lower()
    if mode in ("ones", "degree"):
        return 1
    if mode in ("random_normal", "laplacian_pe", "spectral_pe"):
        return int(dim)
    raise ValueError(f"Unknown feature mode: {mode!r}")


# =============================================================================
# Helpers: PE constructions via dense eigendecomposition
# =============================================================================

def _build_dense_A_enc(num_nodes: int, edge_index: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
    """Reconstruct dense A_enc (symmetric, hollow) from the edge_index."""
    A = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)
    src, dst = edge_index
    if src.numel() > 0:
        A[src, dst] = 1.0
        A = torch.maximum(A, A.t())
    A.fill_diagonal_(0.0)
    return A


def _laplacian_pe(num_nodes: int, edge_index: torch.Tensor,
                  dim: int, device: torch.device,
                  *, pe_normalize: str = "std") -> torch.Tensor:
    """Eigenvectors of normalized Laplacian L = I - D^{-1/2} A_enc D^{-1/2}.

    Returns (n, dim), corresponding to the smallest non-trivial dim eigenvalues.
    `pe_normalize` controls column scaling -- see module docstring.
    """
    A = _build_dense_A_enc(num_nodes, edge_index, device)
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
    L = torch.eye(num_nodes, device=device) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    eigvals, eigvecs = torch.linalg.eigh(L)
    k = min(int(dim), num_nodes - 1)
    pe = eigvecs[:, 1:k + 1]  # skip trivial constant eigenvector
    norm = str(pe_normalize).lower()
    if norm == "raw":
        pass
    elif norm == "std":
        pe = pe / (pe.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
    elif norm == "sqrt_n":
        pe = pe * float(num_nodes) ** 0.5
    else:
        raise ValueError(f"Unknown pe_normalize: {pe_normalize!r}")
    return pe.float()


def _spectral_pe(num_nodes: int, edge_index: torch.Tensor,
                 dim: int, device: torch.device) -> torch.Tensor:
    """sqrt(n) * top-K (by |lambda|) eigenvectors of A_enc itself.

    This is the spectral PE used in the spectral-PE GIN approximation result.
    """
    A = _build_dense_A_enc(num_nodes, edge_index, device)
    eigvals, eigvecs = torch.linalg.eigh(A)
    # Sort by |lambda| descending
    idx = torch.argsort(eigvals.abs(), descending=True)
    k = min(int(dim), num_nodes)
    top_idx = idx[:k]
    pe = eigvecs[:, top_idx] * float(num_nodes) ** 0.5  # sqrt(n) rescaling
    return pe.float()
