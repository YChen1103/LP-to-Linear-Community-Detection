"""
Decoder modules with built-in enforcement of the bounded encoder-decoder class
constraints.

Two decoders:
    BilinearDecoder         s_ij = b + z_i^T W z_j, with W symmetric, ||W||_op <= W_max
    InnerProductDecoder     s_ij = b + z_i^T z_j (special case W = I)

Norm constraints:
    encoder side:  ||z_i|| <= R is applied OUTSIDE these modules (in trainer.py
                   via project_to_ball), so the decoder only sees the projected z.
    decoder side:  W is enforced to be symmetric AND have ||W||_op <= W_max via
                   spectral projection (eigendecomposition + eigenvalue clamp).
                   Bias |b| <= B_0 is enforced via tanh parameterization.

Both projections are done in the forward pass, so gradients flow through them
correctly. (For W this requires torch.linalg.eigh, which is differentiable.)
"""
from __future__ import annotations
from typing import Dict

import torch
from torch import nn


class BilinearDecoder(nn.Module):
    """Bilinear decoder with symmetric, spectrally-bounded W and bounded bias.

    Parameters
    ----------
    d        : int    embedding dimension
    W_max    : float  bound on ||W||_op
    B_0      : float  bound on |b|
    init_b   : float  initial bias value
    symmetric: bool   if False, allows W to be arbitrary (still bounded). Default True.
    """

    def __init__(
        self,
        d: int,
        W_max: float,
        B_0: float,
        init_b: float = 0.0,
        symmetric: bool = True,
    ):
        super().__init__()
        self.d = int(d)
        self.W_max = float(W_max)
        self.B_0 = float(B_0)
        self.symmetric = bool(symmetric)
        # Raw W param; symmetrized + projected in forward
        scale = 0.8 * min(self.W_max, 1.0)
        self.M = nn.Parameter(scale * torch.eye(d) + 0.01 * torch.randn(d, d))
        # Raw bias param: actual b = B_0 * tanh(b_raw)
        # Initialize tanh^{-1}(init_b / B_0) so initial b ~= init_b
        if abs(init_b) >= B_0:
            init_b_raw_val = float(torch.atanh(torch.tensor(0.999 * (init_b / abs(init_b)))))
        else:
            init_b_raw_val = float(torch.atanh(torch.tensor(init_b / B_0)))
        self.b_raw = nn.Parameter(torch.tensor(init_b_raw_val, dtype=torch.float32))

    def W(self) -> torch.Tensor:
        """Return the effective W matrix: symmetric (if requested) and spectrally projected.

        Note on cost: this does an `eigh` (symmetric case) or `svd` (general case)
        on every forward pass to enforce ||W||_op <= W_max with full gradient flow.
        Cost is O(d^3) per call; for d <= 32 this is negligible, but for d ~ 64+
        it becomes a non-trivial fraction of training time.
        """
        M = self.M
        if self.symmetric:
            M = 0.5 * (M + M.T)
        # Eigendecompose, clamp eigenvalues to [-W_max, W_max], reconstruct
        if self.symmetric:
            eigvals, eigvecs = torch.linalg.eigh(M)
            eigvals_clamped = torch.clamp(eigvals, -self.W_max, self.W_max)
            return eigvecs @ torch.diag(eigvals_clamped) @ eigvecs.T
        # Non-symmetric: use SVD to bound spectral norm
        U, sing, Vh = torch.linalg.svd(M, full_matrices=False)
        sing_clamped = torch.clamp(sing, max=self.W_max)
        return U @ torch.diag(sing_clamped) @ Vh

    def b(self) -> torch.Tensor:
        """Return the effective bias: B_0 * tanh(b_raw)."""
        return self.B_0 * torch.tanh(self.b_raw)

    def score_pairs(
        self, z: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute s_ij = b + z_i^T W z_j for given (i, j) pair indices.

        i, j : (P,) long tensors of node indices.
        Returns (P,) tensor of scores.
        """
        W = self.W()
        b = self.b()
        # (P, d) @ (d, d) -> (P, d), then row-wise dot with z[j]
        Wzj = z[j] @ W.T  # (P, d); since W symmetric this equals z[j] @ W
        return (z[i] * Wzj).sum(dim=-1) + b

    def score_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Full (n, n) score matrix S = b 1 1^T + Z W Z^T."""
        W = self.W()
        b = self.b()
        return z @ W @ z.T + b


class InnerProductDecoder(nn.Module):
    """Inner-product decoder: s_ij = b + z_i^T z_j (i.e. W = I).

    Has only one trainable parameter (the bias). |b| <= B_0 enforced via tanh.
    The effective ||W||_op = ||I||_op = 1 identically. We accept a `W_max_declared`
    argument purely for sanity-checking against the surrounding pipeline's
    declared bound (must be >= 1 for the boundedness assumption to be
    satisfiable).
    """

    def __init__(self, B_0: float, init_b: float = 0.0,
                 W_max_declared: float = 1.0, **_unused):
        super().__init__()
        self.B_0 = float(B_0)
        self.W_max_declared = float(W_max_declared)
        if self.W_max_declared < 1.0 - 1e-9:
            raise ValueError(
                f"InnerProductDecoder requires W_max_declared >= 1 "
                f"(since ||W||_op = 1); got {self.W_max_declared}."
            )
        if abs(init_b) >= B_0:
            init_b_raw_val = float(torch.atanh(torch.tensor(0.999 * (init_b / abs(init_b)))))
        else:
            init_b_raw_val = float(torch.atanh(torch.tensor(init_b / B_0)))
        self.b_raw = nn.Parameter(torch.tensor(init_b_raw_val, dtype=torch.float32))

    def W(self) -> torch.Tensor:
        # Returned for API symmetry; not actually used in forward
        # Note: we don't know d here; caller must use score_pairs/score_matrix
        raise NotImplementedError("InnerProductDecoder.W() returns identity implicitly; "
                                  "use score_pairs/score_matrix.")

    def b(self) -> torch.Tensor:
        return self.B_0 * torch.tanh(self.b_raw)

    def score_pairs(
        self, z: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
    ) -> torch.Tensor:
        return (z[i] * z[j]).sum(dim=-1) + self.b()

    def score_matrix(self, z: torch.Tensor) -> torch.Tensor:
        return z @ z.T + self.b()


# =============================================================================
# Factory
# =============================================================================

def build_decoder(decoder_cfg: Dict, init_b: float = 0.0) -> nn.Module:
    """Build decoder from resolved decoder config block.

    `init_b` is the initial bias value (computed by trainer based on
    bias_init_mode + data).
    """
    family = str(decoder_cfg["family"]).lower()
    B_0 = float(decoder_cfg["B_0"])
    if family == "bilinear":
        return BilinearDecoder(
            d=int(decoder_cfg["d_pass_through"]),  # injected by trainer
            W_max=float(decoder_cfg["W_max"]),
            B_0=B_0,
            init_b=float(init_b),
            symmetric=bool(decoder_cfg.get("symmetric", True)),
        )
    if family == "inner_product":
        return InnerProductDecoder(
            B_0=B_0, init_b=float(init_b),
            W_max_declared=float(decoder_cfg.get("W_max", 1.0)),
        )
    raise ValueError(f"Unknown decoder family: {family!r}")


# =============================================================================
# Verification utility
# =============================================================================

@torch.no_grad()
def verify_decoder_constraints(decoder: nn.Module, atol: float = 1e-5) -> Dict[str, float]:
    """Verify that the decoder's effective W and b satisfy the boundedness-assumption bounds.

    Returns dict with measured values, and asserts they satisfy the bounds.
    """
    out = {}
    if isinstance(decoder, BilinearDecoder):
        W = decoder.W()
        out["W_op_norm"] = float(torch.linalg.svdvals(W).max().item())
        out["W_max_target"] = float(decoder.W_max)
        assert out["W_op_norm"] <= decoder.W_max + atol, (
            f"||W||_op = {out['W_op_norm']} exceeds W_max = {decoder.W_max}"
        )
        if decoder.symmetric:
            asym = float((W - W.T).abs().max().item())
            out["W_asymmetry"] = asym
            assert asym <= atol, f"W is not symmetric: max(|W - W^T|) = {asym}"
    elif isinstance(decoder, InnerProductDecoder):
        # ||W||_op = ||I||_op = 1 identically. We just record that, and verify
        # that W_max declared by the surrounding pipeline (if any) is >= 1.
        out["W_op_norm"] = 1.0
        out["W_max_target"] = float(getattr(decoder, "W_max_declared", 1.0))
        if out["W_max_target"] < 1.0 - atol:
            raise AssertionError(
                f"InnerProductDecoder has effective ||W||_op = 1, but the declared "
                f"W_max bound is {out['W_max_target']} < 1; boundedness assumption violated."
            )
    out["b"] = float(decoder.b().item())
    out["B_0_target"] = float(decoder.B_0)
    assert abs(out["b"]) <= decoder.B_0 + atol, (
        f"|b| = {abs(out['b'])} exceeds B_0 = {decoder.B_0}"
    )
    return out
