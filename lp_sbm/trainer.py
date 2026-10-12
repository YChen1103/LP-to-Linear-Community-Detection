"""
Training loop.

Responsibilities:
    1. Build encoder and decoder from resolved config
    2. Construct edge_index from A_enc, build initial features
    3. Train encoder + decoder on visible pairs T (BCE with target = A)
    4. Project z to ball ||z|| <= R after each forward pass (boundedness assumption)
    5. Periodically evaluate (every `eval_every` epochs), recording all metrics
    6. Final eval after training, dump final embeddings/decoder

The trainer stays pure: it does not compute any bound envelopes. All
"derived" quantities are computed elsewhere.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import build_features
from .models.encoders import build_encoder
from .models.decoders import build_decoder, BilinearDecoder, InnerProductDecoder
from .sbm import build_A_enc_edge_index


# =============================================================================
# Encoder-side norm projection (boundedness-assumption enforcement)
# =============================================================================

def project_to_ball(z: torch.Tensor, R: float) -> torch.Tensor:
    """Per-node projection: z_i -> R * z_i / max(||z_i||, R). Guarantees ||z_i|| <= R."""
    norm = torch.linalg.vector_norm(z, dim=1, keepdim=True)
    scale = torch.clamp(R / norm.clamp(min=1e-12), max=1.0)
    return z * scale


# =============================================================================
# Bias init (decoder bias initial value)
# =============================================================================

# Available modes:
#   "zero"               -> b0 = 0
#   "empirical_logit"    -> b0 = logit(mean(A_train)); a data-driven heuristic
#   "mean_offdiag_logit" -> b0 = mean of off-diagonal entries of M*; a population
#                            heuristic. NOT the paper's b* (operator-norm
#                            minimizer). The mode name "oracle_bstar" is kept
#                            as an alias of this for compatibility.
#   "b_star"             -> b0 = sbm_oracle["b_star"], the operator-norm
#                            minimizer used in the paper's analysis and in
#                            capacity resolution. Most "oracle" of the choices.
#   "near_b_star"        -> b0 = b_star + N(0, sigma^2). sigma read from
#                            train_cfg["near_b_star_sigma"], default 0.1.
#                            Useful for testing whether training is robust to
#                            small bias-init perturbations.
#   "random"             -> b0 ~ Uniform(-half_range, +half_range) with
#                            half_range = train_cfg["random_init_half_range"]
#                            * B_0, default 0.5. I.e. random in [-B_0/2, B_0/2].

def compute_bias_init(
    bias_init_mode: str,
    A_train: np.ndarray,
    sbm_oracle: Dict[str, Any],
    rng_seed: int,
    B_0: float,
    *,
    near_b_star_sigma: float = 0.1,
    random_init_half_range: float = 0.5,
) -> float:
    """Compute initial value for decoder bias param, clipped to [-B_0, B_0]."""
    mode = str(bias_init_mode).lower()
    M_star = sbm_oracle["M_star"]
    b_star = float(sbm_oracle["b_star"])

    if mode == "zero":
        b0 = 0.0
    elif mode == "empirical_logit":
        rate = float(np.clip(np.mean(A_train), 1e-6, 1 - 1e-6))
        b0 = float(np.log(rate / (1 - rate)))
    elif mode in ("mean_offdiag_logit", "oracle_bstar"):
        # `oracle_bstar` is a deprecated alias kept for backward compatibility.
        K = M_star.shape[0]
        offdiag = [M_star[a, b] for a in range(K) for b in range(K) if a != b]
        b0 = float(np.mean(offdiag)) if offdiag else 0.0
    elif mode == "b_star":
        b0 = b_star
    elif mode == "near_b_star":
        rng = np.random.default_rng(rng_seed)
        b0 = b_star + float(rng.normal(0.0, float(near_b_star_sigma)))
    elif mode == "random":
        rng = np.random.default_rng(rng_seed)
        half = float(random_init_half_range) * float(B_0)
        b0 = float(rng.uniform(-half, half))
    else:
        raise ValueError(
            f"Unknown bias_init_mode: {bias_init_mode!r}. "
            f"Choose from: zero, empirical_logit, mean_offdiag_logit, "
            f"b_star, near_b_star, random."
        )
    # Clip to [-B_0, B_0] so initial decoder bias respects bound (with 5% slack
    # so the tanh parameterization can still represent it).
    return float(np.clip(b0, -B_0 * 0.95, B_0 * 0.95))


# =============================================================================
# Build full forward function: encoder -> projection -> decoder
# =============================================================================

class LPModel(nn.Module):
    """Encoder + projection + decoder, packaged for clean forward."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, R: float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.R = float(R)

    def embed(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        z_raw = self.encoder(edge_index, num_nodes, x=x)
        z = project_to_ball(z_raw, self.R)
        return z

    def score_pairs(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        x: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
    ) -> torch.Tensor:
        z = self.embed(edge_index, num_nodes, x)
        return self.decoder.score_pairs(z, i, j)


# =============================================================================
# Per-snapshot extraction: produce numpy z, W, b for metrics
# =============================================================================

@torch.no_grad()
def extract_snapshot(
    model: LPModel,
    edge_index: torch.Tensor,
    num_nodes: int,
    x: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute (z, W, b) numpy arrays for the current model state."""
    model.eval()
    z = model.embed(edge_index, num_nodes, x)
    z_np = z.detach().cpu().numpy().astype(np.float32)
    if isinstance(model.decoder, BilinearDecoder):
        W = model.decoder.W().detach().cpu().numpy().astype(np.float64)
    elif isinstance(model.decoder, InnerProductDecoder):
        d = z_np.shape[1]
        W = np.eye(d, dtype=np.float64)
    else:
        raise TypeError(f"Unknown decoder type: {type(model.decoder)}")
    b = float(model.decoder.b().detach().cpu().item())
    return z_np, W, b


# =============================================================================
# Train one restart
# =============================================================================

def train_one_restart(
    *,
    data: Dict[str, Any],
    sbm_oracle: Dict[str, Any],
    resolved_cfg: Dict[str, Any],
    seeds: Dict[str, int],
    device: torch.device,
    edge_index: torch.Tensor,
    x: torch.Tensor,
    eval_callback=None,
) -> Dict[str, Any]:
    """Train one restart. Returns final state + per-epoch losses + last snapshot.

    `edge_index` and `x` are passed in from `train_lp` (built once per base seed,
    independent of restart index -- see held-out independence in the paper;
    the analysis conditions on X).
    `eval_callback`, if provided, is called at every periodic eval point as
        eval_callback(epoch, z, W, b, metrics_extra={...})
    """
    enc_cfg = resolved_cfg["encoder"]
    dec_cfg = resolved_cfg["decoder"]
    train_cfg = resolved_cfg["training"]
    R = float(enc_cfg["R"])
    B_0 = float(dec_cfg["B_0"])
    W_max = float(dec_cfg["W_max"])

    n = data["n"]
    rows = data["rows"]
    cols = data["cols"]
    train_mask = data["train_mask"]
    held_mask = data["held_mask"]
    A_flat = data["A_flat"]

    # ------- Reproducible seeds for this restart -------
    torch.manual_seed(int(seeds["model"]))
    np.random.seed(int(seeds["model"]))

    # ------- Encoder + decoder -------
    in_dim = int(x.shape[1])
    encoder = build_encoder(enc_cfg, input_dim=in_dim).to(device)

    A_train_for_init = A_flat[train_mask]
    # Resolve bias_init_mode. Canonical location is training.bias_init_mode.
    # Backward compat: if it is missing in `training:` but present in `decoder:`,
    # we honor it and emit a warning. Configs that put it in BOTH places will
    # cause an explicit error (so the user notices an inconsistent intent).
    train_mode = train_cfg.get("bias_init_mode", None)
    decoder_mode = dec_cfg.get("bias_init_mode", None)
    if train_mode is not None and decoder_mode is not None and train_mode != decoder_mode:
        raise ValueError(
            f"Conflicting bias_init_mode: training={train_mode!r}, "
            f"decoder={decoder_mode!r}. Set it in training section only."
        )
    if train_mode is None and decoder_mode is not None:
        import warnings
        warnings.warn(
            "bias_init_mode found under `decoder:` (legacy location); "
            "please move it to `training:`. Honoring decoder value for now.",
            DeprecationWarning, stacklevel=2,
        )
        bias_mode_eff = decoder_mode
    else:
        bias_mode_eff = train_mode if train_mode is not None else "b_star"

    init_b = compute_bias_init(
        bias_mode_eff,
        A_train_for_init, sbm_oracle,
        rng_seed=int(seeds["model"]) + 2024,
        B_0=B_0,
        near_b_star_sigma=float(train_cfg.get("near_b_star_sigma", 0.1)),
        random_init_half_range=float(train_cfg.get("random_init_half_range", 0.5)),
    )
    dec_cfg_with_d = dict(dec_cfg)
    dec_cfg_with_d["d_pass_through"] = int(enc_cfg["d"])
    decoder = build_decoder(dec_cfg_with_d, init_b=init_b).to(device)

    model = LPModel(encoder, decoder, R=R).to(device)

    # ------- Train / val split inside H -------
    val_split = float(train_cfg.get("val_split_of_H", 0.5))
    held_idx = np.where(held_mask)[0]
    rng_split = np.random.default_rng(int(seeds["model"]) + 1000)
    rng_split.shuffle(held_idx)
    n_val = int(len(held_idx) * val_split)
    val_idx_in_H = held_idx[:n_val]
    test_idx_in_H = held_idx[n_val:]

    train_pair_idx = np.where(train_mask)[0]
    train_i = torch.from_numpy(rows[train_pair_idx]).long().to(device)
    train_j = torch.from_numpy(cols[train_pair_idx]).long().to(device)
    train_tgt = torch.from_numpy(A_flat[train_pair_idx].astype(np.float32)).to(device)

    val_i = torch.from_numpy(rows[val_idx_in_H]).long().to(device)
    val_j = torch.from_numpy(cols[val_idx_in_H]).long().to(device)
    val_tgt = torch.from_numpy(A_flat[val_idx_in_H].astype(np.float32)).to(device)

    # ------- Optimizer -------
    optim_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("lr", 1e-2))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    if optim_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name!r}")

    max_epochs = int(train_cfg.get("max_epochs", 400))
    patience = int(train_cfg.get("patience", 80))
    clip_norm = float(train_cfg.get("clip_grad_norm", 5.0))
    eval_every = int(train_cfg.get("eval_every", 0))
    early_stop = bool(train_cfg.get("early_stop", True))

    # Batch mode. "full_batch" (default) computes loss over all train pairs each
    # step (full-batch GD on R_T_emp); "mini_batch" samples `batch_size` train
    # pairs per step (stochastic gradient on R_T_emp; theory unaffected since
    # the SGD updates are on the same objective). When mini_batch, an "epoch" is
    # one pass over the train set in shuffled mini-batches (so #steps per epoch
    # = ceil(|T| / batch_size)).
    batch_mode = str(train_cfg.get("batch_mode", "full_batch")).lower()
    if batch_mode not in ("full_batch", "mini_batch"):
        raise ValueError(f"Unknown batch_mode={batch_mode!r}; use full_batch or mini_batch.")
    batch_size = int(train_cfg.get("batch_size", 4096))

    train_curve: List[float] = []
    val_curve: List[float] = []

    # Best-by-val tracking. We always track this for diagnostics, but we only
    # USE it to (a) stop training early and (b) select the final state when
    # early_stop=True. When early_stop=False, the final state is the LAST
    # epoch's state and selection across restarts is by best train loss.
    best_val = float("inf")
    best_val_epoch = 0
    best_val_state: Dict[str, torch.Tensor] | None = None
    no_improve = 0

    last_train_loss = float("inf")

    n_train_pairs = int(train_i.numel())
    rng_minibatch = np.random.default_rng(int(seeds["model"]) + 7777)

    for epoch in range(1, max_epochs + 1):
        model.train()
        if batch_mode == "full_batch":
            # Merged train + val forward:
            # Run encoder once -> z, then decode pairs for both train and val
            # from the SAME z. The expensive part (encoder MPNN aggregation)
            # is then evaluated once per epoch instead of twice.
            #
            # Semantic note: this evaluates val_loss at the SAME state at which
            # train_loss is computed (i.e. the pre-step state). For
            # `early_stop=False` (the main sweep mode) val is purely an
            # observable. For `early_stop=True`, we save best_val_state BEFORE
            # opt.step() so that the saved state corresponds to the val_loss
            # used to select it.
            opt.zero_grad()
            z_full = model.embed(edge_index, n, x)
            train_scores = model.decoder.score_pairs(z_full, train_i, train_j)
            loss = F.binary_cross_entropy_with_logits(train_scores, train_tgt)
            with torch.no_grad():
                val_scores = model.decoder.score_pairs(z_full.detach(), val_i, val_j)
                val_loss = F.binary_cross_entropy_with_logits(val_scores, val_tgt)
            epoch_train_loss = float(loss.item())

            # Save best-val checkpoint BEFORE the optimizer step so saved-state
            # is consistent with the val_loss that selected it.
            if val_loss.item() < best_val - 1e-12:
                best_val = float(val_loss.item())
                best_val_epoch = epoch
                best_val_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()
        else:
            # mini_batch: shuffle train pair indices, take batch_size at a time.
            perm = rng_minibatch.permutation(n_train_pairs)
            losses = []
            for start in range(0, n_train_pairs, batch_size):
                sel = perm[start : start + batch_size]
                bi = train_i[sel]
                bj = train_j[sel]
                btgt = train_tgt[sel]
                opt.zero_grad()
                scores = model.score_pairs(edge_index, n, x, bi, bj)
                loss_b = F.binary_cross_entropy_with_logits(scores, btgt)
                loss_b.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                opt.step()
                losses.append(float(loss_b.item()) * len(sel))
            epoch_train_loss = float(sum(losses) / max(n_train_pairs, 1))
            # final-of-epoch loss object + post-step val (mini_batch uses separate
            # forwards for train and val; the merged forward is full_batch only).
            with torch.no_grad():
                full_scores = model.score_pairs(edge_index, n, x, train_i, train_j)
                loss = F.binary_cross_entropy_with_logits(full_scores, train_tgt)

            model.eval()
            with torch.no_grad():
                val_scores = model.score_pairs(edge_index, n, x, val_i, val_j)
                val_loss = F.binary_cross_entropy_with_logits(val_scores, val_tgt)

            if val_loss.item() < best_val - 1e-12:
                best_val = float(val_loss.item())
                best_val_epoch = epoch
                best_val_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

        train_curve.append(epoch_train_loss)
        val_curve.append(float(val_loss.item()))
        last_train_loss = epoch_train_loss

        # Periodic eval
        if eval_every > 0 and epoch % eval_every == 0 and eval_callback is not None:
            z_e, W_e, b_e = extract_snapshot(model, edge_index, n, x)
            eval_callback(epoch=epoch, z=z_e, W=W_e, b=b_e,
                          extra={"train_loss": epoch_train_loss,
                                 "val_loss": float(val_loss.item()),
                                 "early_stop": early_stop,
                                 "batch_mode": batch_mode})

        if early_stop and no_improve >= patience:
            break

    # ------- Pick final state -------
    if early_stop:
        # Use best-by-val checkpoint
        if best_val_state is None:
            raise RuntimeError("Training produced no valid checkpoint")
        model.load_state_dict(best_val_state)
        final_epoch = best_val_epoch
        # selection_metric is best val loss (used by train_lp for restart selection)
        selection_metric = best_val
        selection_basis = "val_loss"
    else:
        # Use the last-epoch state as-is; never look at val
        final_epoch = len(train_curve)
        selection_metric = last_train_loss
        selection_basis = "train_loss"
    model.eval()

    # ------- Final snapshot -------
    z_final, W_final, b_final = extract_snapshot(model, edge_index, n, x)

    return {
        "z":                z_final,
        "W":                W_final,
        "b":                b_final,
        "best_epoch":       final_epoch,        # epoch of the returned state
        "best_val_loss":    best_val,           # always logged, even if not used
        "selection_metric": float(selection_metric),
        "selection_basis":  selection_basis,    # "val_loss" or "train_loss"
        "early_stop":       early_stop,
        "train_curve":      train_curve,
        "val_curve":        val_curve,
        "val_idx_in_H":     val_idx_in_H,
        "test_idx_in_H":    test_idx_in_H,
        "feature_dim":      in_dim,
        "encoder_family":   enc_cfg["family"],
        "decoder_family":   dec_cfg["family"],
        "bias_init_mode":   bias_mode_eff,
        "bias_init_value":  float(init_b),
    }


# =============================================================================
# Train with multi-restart
# =============================================================================

def train_lp(
    *,
    data: Dict[str, Any],
    sbm_oracle: Dict[str, Any],
    resolved_cfg: Dict[str, Any],
    seeds: Dict[str, int],
    device: torch.device,
    eval_callback=None,
) -> Dict[str, Any]:
    """Train with multi-restart; returns the best-by-val-loss restart's result.

    The encoder input edge_index and the initial node features X are constructed
    ONCE here (they only depend on the base seed: edge_index from data seeds via
    A_enc; X from the dedicated `feature` sub-seed). They are then reused across
    all optimizer restarts. This implements the paper's held-out independence
    convention for random features:
    random feature realizations are sampled once per base seed and held fixed;
    multi-restart only perturbs the optimizer initialization.

    `eval_callback` (if provided) receives periodic-eval calls only from the
    *current* restart. After all restarts finish, it is also called once with
    the best-restart's final state (so post-hoc analysis sees that final).
    """
    enc_cfg = resolved_cfg["encoder"]
    n = data["n"]
    rows = data["rows"]
    cols = data["cols"]
    train_mask = data["train_mask"]

    # ---- Edge index for encoder (built once from A_enc) ----
    edge_index_np = build_A_enc_edge_index(data["A"], train_mask, rows, cols)
    edge_index = torch.from_numpy(edge_index_np).long().to(device)

    # ---- Initial features X (built once, conditioned on A_enc + feature seed) ----
    feat_cfg = enc_cfg["features"]
    feat_mode = str(feat_cfg["mode"])
    feat_dim_arg = int(feat_cfg.get("dim", 4))
    pe_normalize = str(feat_cfg.get("pe_normalize", "std"))
    # Use the dedicated `feature` sub-seed (independent of `model`).
    # If the subseeds dict lacks `feature`, derive it deterministically from
    # `model` using a large prime offset.
    feature_seed = int(seeds.get("feature", int(seeds["model"]) + 600013))
    x = build_features(
        feat_mode, num_nodes=n, edge_index=edge_index,
        feature_seed=feature_seed, dim=feat_dim_arg, device=device,
        pe_normalize=pe_normalize,
    ).detach()

    n_restarts = int(resolved_cfg["training"].get("restarts", 1))
    base_model_seed = seeds["model"]
    best = None
    restart_vals = []

    for r in range(n_restarts):
        local_seeds = dict(seeds)
        local_seeds["model"] = int(base_model_seed) + r * 17

        def cb_wrapper(*, epoch, z, W, b, extra, _r=r):
            extra2 = dict(extra)
            extra2["restart"] = int(_r)
            if eval_callback is not None:
                eval_callback(epoch=epoch, z=z, W=W, b=b, extra=extra2)

        result = train_one_restart(
            data=data, sbm_oracle=sbm_oracle, resolved_cfg=resolved_cfg,
            seeds=local_seeds, device=device,
            edge_index=edge_index, x=x,
            eval_callback=cb_wrapper if eval_callback is not None else None,
        )
        restart_vals.append(result["selection_metric"])
        if best is None or result["selection_metric"] < best["selection_metric"]:
            best = result
            best["restart_idx"] = int(r)

    best["restart_vals"] = restart_vals
    best["restart_selection_basis"] = best["selection_basis"]
    best["feature_mode"] = feat_mode
    best["pe_normalize"] = pe_normalize
    return best
