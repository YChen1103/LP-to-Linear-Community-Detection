"""
Encoder architectures.

Each encoder:
  - takes an edge_index (2, E) and an initial feature matrix X (n, input_dim)
  - returns embeddings z_raw of shape (n, d).

Norm-bound projection ||z_i|| <= R is applied OUTSIDE the encoder, by the
training loop (see lp_sbm.trainer.project_to_ball). This keeps the encoders
themselves clean and makes the projection layer auditable.

All encoders accept `input_dim` so they can ingest features from
lp_sbm.features.build_features (ones / random / PE / ...).
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


# =============================================================================
# CenteredPower: spectral-like, training-stable, SBM-friendly
# =============================================================================

class CenteredPowerEncoder(nn.Module):
    """Power-iteration-style encoder.

    At each layer: aggregate sum over neighbors, then center+standardize.
    Stack L iterations, apply LayerNorm, then a 2-layer MLP head.
    """

    def __init__(
        self,
        input_dim: int,
        d: int,
        hidden_dim: int = 32,
        num_layers: int = 8,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        **_unused,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = input_dim
        self.num_layers = num_layers
        stacked_dim = num_layers * input_dim
        self.dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(stacked_dim) if use_layernorm else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(stacked_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d),
        )

    @staticmethod
    def _aggregate_sum(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        out = torch.zeros_like(x)
        if src.numel() > 0:
            out.index_add_(0, dst, x[src])
        return out

    @staticmethod
    def _center_standardize(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False) + 1e-6
        return x / std

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = torch.ones((num_nodes, self.input_dim),
                           dtype=torch.float32, device=edge_index.device)
        h = self._center_standardize(self._aggregate_sum(x, edge_index))
        feats = [h]
        for _ in range(1, self.num_layers):
            h = self._center_standardize(self._aggregate_sum(h, edge_index))
            feats.append(h)
        stacked = torch.cat(feats, dim=-1)
        stacked = self.input_norm(stacked)
        stacked = self.dropout(stacked)
        return self.head(stacked)


# =============================================================================
# GCN
# =============================================================================

class GCNEncoder(nn.Module):
    """Standard GCN: H' = sigma( D^{-1/2} (A + I) D^{-1/2} H W ).

    Diagonal self-loops added to A_enc.

    Implementation note: the normalization tensors (src_all, dst_all, norm)
    depend only on edge_index and num_nodes, both of which are fixed across
    the entire training loop. We cache them on first forward and reuse --
    saves O(E) tensor allocations + 3 small CUDA kernel launches per epoch.
    """

    def __init__(
        self,
        input_dim: int,
        d: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.0,
        **_unused,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [d]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        # Norm cache (set on first forward; valid for as long as edge_index
        # is the same tensor object). Stored as non-Parameter buffers so
        # they don't show up in state_dict.
        self._cached_edge_index = None
        self._cached_norm = None  # (src_all, dst_all, norm) tuple of tensors

    @staticmethod
    def _build_norm(edge_index: torch.Tensor, num_nodes: int):
        src, dst = edge_index
        device = edge_index.device
        loop = torch.arange(num_nodes, device=device)
        src_all = torch.cat([src, loop])
        dst_all = torch.cat([dst, loop])
        deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        ones = torch.ones_like(dst_all, dtype=torch.float32)
        deg.index_add_(0, dst_all, ones)
        d_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
        norm = d_inv_sqrt[src_all] * d_inv_sqrt[dst_all]
        return src_all, dst_all, norm

    def _get_norm(self, edge_index: torch.Tensor, num_nodes: int):
        # `is` check: cheap pointer equality. Within one train_lp call,
        # edge_index is the same tensor across all epochs and restarts
        # (built once in train_lp, passed in). A fresh tensor would trigger
        # a recompute. Safer than id() (which can be reused after GC).
        if self._cached_edge_index is not edge_index:
            self._cached_edge_index = edge_index
            self._cached_norm = self._build_norm(edge_index, num_nodes)
        return self._cached_norm

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = torch.ones((num_nodes, self.input_dim),
                           dtype=torch.float32, device=edge_index.device)
        src_all, dst_all, norm = self._get_norm(edge_index, num_nodes)
        h = x
        for i, lin in enumerate(self.linears):
            h = lin(h)
            out = torch.zeros_like(h)
            out.index_add_(0, dst_all, h[src_all] * norm.unsqueeze(-1))
            h = out
            if i < len(self.linears) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


# =============================================================================
# GIN
# =============================================================================

class GINEncoder(nn.Module):
    """GIN: h_v' = MLP( (1 + eps) h_v + sum_{u in N(v)} h_u )."""

    def __init__(
        self,
        input_dim: int,
        d: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.0,
        **_unused,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [d]
        self.mlps = nn.ModuleList()
        self.eps = nn.ParameterList()
        for i in range(num_layers):
            self.mlps.append(nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dims[i + 1]),
            ))
            self.eps.append(nn.Parameter(torch.zeros(1)))

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = torch.ones((num_nodes, self.input_dim),
                           dtype=torch.float32, device=edge_index.device)
        src, dst = edge_index
        h = x
        for i, mlp in enumerate(self.mlps):
            agg = torch.zeros_like(h)
            if src.numel() > 0:
                agg.index_add_(0, dst, h[src])
            h = (1.0 + self.eps[i]) * h + agg
            h = mlp(h)
            if i < len(self.mlps) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


# =============================================================================
# GraphSAGE (mean aggregator)
# =============================================================================

class GraphSAGEEncoder(nn.Module):
    """GraphSAGE: h_v' = sigma( W_self h_v + W_neigh * mean_{u in N(v)} h_u )."""

    def __init__(
        self,
        input_dim: int,
        d: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.0,
        **_unused,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [d]
        self.lin_self = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.lin_neigh = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    @staticmethod
    def _mean_aggregate(h: torch.Tensor, edge_index: torch.Tensor,
                        num_nodes: int) -> torch.Tensor:
        src, dst = edge_index
        agg_sum = torch.zeros_like(h)
        deg = torch.zeros(num_nodes, device=h.device, dtype=h.dtype)
        if src.numel() > 0:
            agg_sum.index_add_(0, dst, h[src])
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        return agg_sum / deg

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = torch.ones((num_nodes, self.input_dim),
                           dtype=torch.float32, device=edge_index.device)
        h = x
        for i in range(self.num_layers):
            mean_agg = self._mean_aggregate(h, edge_index, num_nodes)
            h = self.lin_self[i](h) + self.lin_neigh[i](mean_agg)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


# =============================================================================
# Pure MLP (no graph) -- useful as ablation
# =============================================================================

class MLPEncoder(nn.Module):
    """MLP applied per-node, ignoring graph structure entirely.

    Useful as a baseline / ablation showing the importance of message passing.
    Note: with `feature_mode=ones`, the MLP cannot distinguish nodes at all and
    will be degenerate. Pair with `random_normal` or `laplacian_pe` instead.
    """

    def __init__(
        self,
        input_dim: int,
        d: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.0,
        **_unused,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = input_dim
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [d]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = torch.ones((num_nodes, self.input_dim),
                           dtype=torch.float32, device=edge_index.device)
        h = x
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < len(self.linears) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


# =============================================================================
# GAT (Graph Attention Network)
# =============================================================================

class GATEncoder(nn.Module):
    """Graph Attention Network (Velickovic et al., 2018).

    Per layer:
        e_ij     = LeakyReLU( a_src . (W h_i) + a_dst . (W h_j) )
        alpha_ij = softmax_{j in N(i)}( e_ij )       (soft attention over neighbors)
        h_i'     = sum_{j in N(i)} alpha_ij * (W h_j)

    Multi-head: K parallel heads. Hidden layers concat; final layer averages
    (standard GAT convention). With `hidden_dim=64, num_heads=4`, each hidden
    head has dim 16; the last layer averages num_heads heads of dim d each.

    Notes:
      - Self-loops are NOT added explicitly (matches PyG GAT default for our
        setting where the encoder input edge_index already has both directions
        but no diagonal). If you want self-loops, augment edge_index upstream.
      - Numerical stability: per-dst softmax via index_add for normalization;
        we clamp pre-exp scores to avoid overflow when leaky-relu output is
        large. This is sufficient for our setting (n <= 2000, edges sparse).
      - Edge-index unpacking is cached the same way as GCN's norm cache.
    """

    def __init__(
        self,
        input_dim: int,
        d: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        leaky_slope: float = 0.2,
        **_unused,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim_hidden = hidden_dim // num_heads
        self.head_dim_out = d
        self.leaky_slope = float(leaky_slope)
        self.dropout = nn.Dropout(dropout)

        # Per-layer weights and attention vectors.
        # Layer i: input -> num_heads * out_d_per_head
        #   - hidden layer: out_d_per_head = head_dim_hidden, then concat -> hidden_dim
        #   - last layer:   out_d_per_head = head_dim_out (= d), then average -> d
        self.lin_W = nn.ModuleList()
        self.att_src = nn.ParameterList()
        self.att_dst = nn.ParameterList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            is_last = (i == num_layers - 1)
            out_d_per_head = self.head_dim_out if is_last else self.head_dim_hidden
            self.lin_W.append(nn.Linear(in_d, num_heads * out_d_per_head, bias=False))
            # Small init for attention vectors (avoid early sharp attention).
            self.att_src.append(
                nn.Parameter(0.1 * torch.randn(1, num_heads, out_d_per_head))
            )
            self.att_dst.append(
                nn.Parameter(0.1 * torch.randn(1, num_heads, out_d_per_head))
            )

        # Edge-index cache (same pattern as GCN).
        self._cached_edge_index = None
        self._cached_src_dst = None  # (src, dst)

    def _get_src_dst(self, edge_index: torch.Tensor):
        if self._cached_edge_index is not edge_index:
            self._cached_edge_index = edge_index
            self._cached_src_dst = (edge_index[0], edge_index[1])
        return self._cached_src_dst

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = torch.ones((num_nodes, self.input_dim),
                           dtype=torch.float32, device=edge_index.device)
        src, dst = self._get_src_dst(edge_index)
        H = self.num_heads
        h = x
        for i in range(self.num_layers):
            is_last = (i == self.num_layers - 1)
            d_h = self.head_dim_out if is_last else self.head_dim_hidden

            # Linear transform (no bias) -> per-head representation
            Wh = self.lin_W[i](h).view(num_nodes, H, d_h)  # (n, H, d_h)

            # Per-node attention "left/right" scores.
            # att_src has shape (1, H, d_h); broadcast over n.
            a_src_score = (Wh * self.att_src[i]).sum(dim=-1)  # (n, H)
            a_dst_score = (Wh * self.att_dst[i]).sum(dim=-1)  # (n, H)

            if src.numel() > 0:
                # Edge-level un-normalized attention logit
                e = a_src_score[src] + a_dst_score[dst]            # (E, H)
                e = F.leaky_relu(e, self.leaky_slope)
                # Numerical safety: clamp before exp. Range is bounded by
                # |a_src| + |a_dst| <= O(R||a||) per pair; clamp to +/-20.
                e = e.clamp(-20.0, 20.0)
                e_exp = e.exp()                                    # (E, H)

                # Per-dst-node sum (softmax denominator)
                denom = torch.zeros((num_nodes, H), device=h.device, dtype=h.dtype)
                denom.index_add_(0, dst, e_exp)
                denom = denom.clamp(min=1e-12)
                alpha = e_exp / denom[dst]                         # (E, H)

                # Aggregate weighted Wh[src]
                msg = Wh[src] * alpha.unsqueeze(-1)                # (E, H, d_h)
                out = torch.zeros((num_nodes, H, d_h),
                                  device=h.device, dtype=h.dtype)
                out.index_add_(0, dst, msg)
            else:
                out = torch.zeros((num_nodes, H, d_h),
                                  device=h.device, dtype=h.dtype)

            # Combine heads
            if is_last:
                # Final layer: average heads -> (n, d_h) where d_h == d
                h = out.mean(dim=1)
            else:
                # Hidden layer: concat heads -> (n, H * d_h) == (n, hidden_dim)
                h = out.reshape(num_nodes, H * d_h)
                h = F.elu(h)
                h = self.dropout(h)
        return h



# =============================================================================
# Factory
# =============================================================================

_ENCODER_REGISTRY = {
    "centered_power": CenteredPowerEncoder,
    "gcn":            GCNEncoder,
    "gin":            GINEncoder,
    "gat":            GATEncoder,
    "graphsage":      GraphSAGEEncoder,
    "mlp":            MLPEncoder,
}


def build_encoder(encoder_cfg: Dict, input_dim: int) -> nn.Module:
    """Build encoder from the resolved encoder config block.

    `input_dim` is the dim of the initial features (computed from features.mode/dim).
    """
    family = str(encoder_cfg["family"]).lower()
    if family not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder family: {family!r}. "
            f"Available: {list(_ENCODER_REGISTRY.keys())}"
        )
    cls = _ENCODER_REGISTRY[family]
    kwargs = dict(
        input_dim=int(input_dim),
        d=int(encoder_cfg["d"]),
        hidden_dim=int(encoder_cfg.get("hidden_dim", 32)),
        num_layers=int(encoder_cfg.get("num_layers", 3)),
        dropout=float(encoder_cfg.get("dropout", 0.0)),
        use_layernorm=bool(encoder_cfg.get("use_layernorm", True)),
    )
    # GAT-specific knobs (other encoders absorb unknown kwargs via **_unused).
    if family == "gat":
        kwargs["num_heads"] = int(encoder_cfg.get("num_heads", 4))
        kwargs["leaky_slope"] = float(encoder_cfg.get("leaky_slope", 0.2))
    return cls(**kwargs)


def list_encoders() -> list:
    """Return list of available encoder family names."""
    return list(_ENCODER_REGISTRY.keys())
