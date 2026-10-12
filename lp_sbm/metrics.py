"""
Metrics: all paper-defined observables, computed from a (z, W, b) snapshot.

Every quantity here is a *raw observable* -- no bound envelopes, no
universal-constant assumptions, no derived rates. Bound computation lives
in analysis/bounds.py and is post-hoc.

Variable naming follows paper notation:
    R_H_emp       = R_hat_H(S_hat)            empirical held-out BCE
    R_H_pop       = R^P_H(S_hat)              population held-out BCE (synthetic only)
    R_H_star      = R^P_H(S^star_H)           Bayes floor on H
    excess_emp_H  = R_H_emp - R_H_star        empirical excess (half-empirical corollary driver)
    excess_pop_H  = R_H_pop - R_H_star        population excess (main theorem driver)
    gamma_H_at_trained = |R_H_emp - R_H_pop|  Gamma_H proxy at trained model

    score_mse_H   = (1/|H|) sum_H (s_ij - m*_{c_i c_j})^2
    score_mse_off = (1/N) sum_{i<j} ...        off-diagonal full
    score_mse_full = (1/n^2) sum_{i,j} ...     including diagonal

    tilde_e_sq    = (1/n) sum_j (s_ij - m*_{c_i c_j})^2  per node

    For each m in classifiers.m_per_class:
        plug_in_m{m}_error_mean / std
        plug_in_m{m}_eta_L_mean / std
        plug_in_m{m}_Delta_hat_mean / std
        direct_m{m}_error_mean / std

    oracle_classifier_error
    kmeans_z_euclid_error, kmeans_z_euclid_nmi    (plain Euclidean K-means on z)
    kmeans_G_form_error,   kmeans_G_form_nmi      (K-means under G; bilinear-aware)

    Verification of the boundedness assumption (keys prefixed `a6_`):
        z_norm_max, z_norm_p99, z_norm_mean
        W_op_norm, b_value
        z_constraint_hit_rate, W_constraint_hit_rate, b_constraint_hit_rate
        assumption6_holds
"""
from __future__ import annotations
from typing import Any, Dict

import numpy as np


# =============================================================================
# Numerical helpers
# =============================================================================

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def bce_per_pair(s, p):
    """Numerically-stable BCE: ell(s; p) = softplus(-s) + (1-p)*s = max(s,0) - s*p + log1p(exp(-|s|))."""
    return np.maximum(s, 0.0) - s * p + np.log1p(np.exp(-np.abs(s)))


# =============================================================================
# Score computation given (z, W, b)
# =============================================================================

def compute_pair_scores(
    z: np.ndarray, W: np.ndarray, b: float,
    rows: np.ndarray, cols: np.ndarray, mask: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute s_ij = b + z_i^T W z_j for pairs in the mask.

    Returns (s_values, mask_indices).
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([]), idx
    ri, ci = rows[idx], cols[idx]
    Wzj = z[ci] @ W.T  # symmetric W => same as z[ci] @ W
    s = np.sum(z[ri] * Wzj, axis=1) + b
    return s, idx


def compute_full_score_matrix(
    z: np.ndarray, W: np.ndarray, b: float,
) -> np.ndarray:
    """Full (n, n) score matrix S_tilde = b 1 1^T + Z W Z^T."""
    return z @ W @ z.T + b


# =============================================================================
# LP losses
# =============================================================================

def compute_R_H_emp(
    z: np.ndarray, W: np.ndarray, b: float,
    rows: np.ndarray, cols: np.ndarray,
    held_mask: np.ndarray, A_flat: np.ndarray,
) -> float:
    """R_hat_H(S_hat) = (1/|H|) sum_{(i,j) in H} ell(s_ij; A_ij)."""
    s, idx = compute_pair_scores(z, W, b, rows, cols, held_mask)
    if len(idx) == 0:
        return 0.0
    return float(np.mean(bce_per_pair(s, A_flat[idx])))


def compute_R_H_pop(
    z: np.ndarray, W: np.ndarray, b: float,
    rows: np.ndarray, cols: np.ndarray,
    held_mask: np.ndarray, P_flat: np.ndarray,
) -> float:
    """R^P_H(S_hat) = (1/|H|) sum_{(i,j) in H} ell(s_ij; P_ij)."""
    s, idx = compute_pair_scores(z, W, b, rows, cols, held_mask)
    if len(idx) == 0:
        return 0.0
    return float(np.mean(bce_per_pair(s, P_flat[idx])))


def compute_R_T_emp(
    z: np.ndarray, W: np.ndarray, b: float,
    rows: np.ndarray, cols: np.ndarray,
    train_mask: np.ndarray, A_flat: np.ndarray,
) -> float:
    """R_hat_T(S_hat) = (1/|T|) sum_{(i,j) in T} ell(s_ij; A_ij).

    This is the empirical training loss, the quantity SGD directly minimises.
    The train-side corollary bridges epsilon_opt to (R_T_emp - R_T_emp_dagger) where
    R_T_emp_dagger is the in-class infimum (not directly observable). In
    practice we report R_T_emp - R_T_star as a proxy for the residual: under
    architecturally-adequate training, R_T_emp_dagger ~ R_T_star, so the proxy
    is a useful diagnostic of training quality.
    """
    s, idx = compute_pair_scores(z, W, b, rows, cols, train_mask)
    if len(idx) == 0:
        return 0.0
    return float(np.mean(bce_per_pair(s, A_flat[idx])))


def compute_R_T_pop(
    z: np.ndarray, W: np.ndarray, b: float,
    rows: np.ndarray, cols: np.ndarray,
    train_mask: np.ndarray, P_flat: np.ndarray,
) -> float:
    """R^P_T(S_hat) = (1/|T|) sum_{(i,j) in T} ell(s_ij; P_ij).

    Population train-side BCE; synthetic-only. Used as the cross-side
    quantity in the Gamma_TH = sup |R^P_T - R^P_H| bridge of the train-side
    corollary.
    """
    s, idx = compute_pair_scores(z, W, b, rows, cols, train_mask)
    if len(idx) == 0:
        return 0.0
    return float(np.mean(bce_per_pair(s, P_flat[idx])))


# =============================================================================
# Score MSE family
# =============================================================================

def compute_score_mse_H(
    z: np.ndarray, W: np.ndarray, b: float,
    rows: np.ndarray, cols: np.ndarray, held_mask: np.ndarray,
    labels: np.ndarray, M_star: np.ndarray,
) -> float:
    """ScoreMSE_H = (1/|H|) sum_{(i,j) in H} (s_ij - m*_{c_i c_j})^2."""
    s, idx = compute_pair_scores(z, W, b, rows, cols, held_mask)
    if len(idx) == 0:
        return 0.0
    m = M_star[labels[rows[idx]], labels[cols[idx]]]
    return float(np.mean((s - m) ** 2))


def compute_score_mse_off(
    z: np.ndarray, W: np.ndarray, b: float,
    rows: np.ndarray, cols: np.ndarray,
    labels: np.ndarray, M_star: np.ndarray,
) -> float:
    """ScoreMSE_off = (1/N) sum_{i<j} (s_ij - m*_{c_i c_j})^2."""
    Wzj = z[cols] @ W.T
    s = np.sum(z[rows] * Wzj, axis=1) + b
    m = M_star[labels[rows], labels[cols]]
    return float(np.mean((s - m) ** 2))


def compute_score_mse_full(
    z: np.ndarray, W: np.ndarray, b: float,
    labels: np.ndarray, M_star: np.ndarray,
) -> float:
    """ScoreMSE_full = (1/n^2) sum_{i,j} (s_ij - m*)^2."""
    S = z @ W @ z.T + b
    M_full = M_star[labels][:, labels]
    return float(np.mean((S - M_full) ** 2))


# =============================================================================
# Per-node error
# =============================================================================

def compute_tilde_e_sq(
    z: np.ndarray, W: np.ndarray, b: float,
    labels: np.ndarray, M_star: np.ndarray,
) -> np.ndarray:
    """tilde_e_i^2 = (1/n) sum_j (s_ij - m*_{c_i c_j})^2.

    Returns (n,) array. Note: (1/n) sum_i tilde_e_i^2 = ScoreMSE_full.
    """
    S = z @ W @ z.T + b
    M_full = M_star[labels][:, labels]
    diff = S - M_full
    return np.mean(diff * diff, axis=1)


def compute_tilde_e_moments(tilde_e_sq: np.ndarray, ks=(1, 2, 3, 4)) -> Dict[str, float]:
    """Higher-order moments of tilde_e_i.

    M_{2k} = mean_i tilde_e_i^{2k}
    Ratio_{2k} = M_{2k} / M_2^k  (1 if all equal; >> 1 if heavy-tailed)
    """
    out: Dict[str, float] = {}
    M2 = float(np.mean(tilde_e_sq))
    out["M_2"] = M2
    for k in ks:
        Mk = float(np.mean(tilde_e_sq ** k))
        out[f"M_{2*k}"] = Mk
        if k > 1:
            out[f"ratio_M{2*k}_over_M2_k{k}"] = Mk / (M2 ** k) if M2 > 1e-12 else float("nan")
    out["tilde_e_sq_max"]  = float(np.max(tilde_e_sq))
    out["tilde_e_sq_p99"]  = float(np.percentile(tilde_e_sq, 99))
    out["tilde_e_sq_p95"]  = float(np.percentile(tilde_e_sq, 95))
    out["tilde_e_sq_p50"]  = float(np.percentile(tilde_e_sq, 50))
    return out


# =============================================================================
# Decoder/encoder verification (boundedness assumption + capacity)
# =============================================================================

def compute_assumption6_diagnostics(
    z: np.ndarray, W: np.ndarray, b: float,
    R: float, W_max: float, B_0: float,
) -> Dict[str, float]:
    """Verify the boundedness assumption holds and report constraint hit-rates.

    ("assumption6" / "a6" in key names refers to this boundedness assumption.)

    Returns:
        z_norm_max, z_norm_p99, z_norm_mean
        W_op_norm
        b_value
        z_constraint_hit_rate  (fraction of nodes within 1% of R)
        W_constraint_hit_rate  (||W||_op / W_max, in [0, 1])
        b_constraint_hit_rate  (|b| / B_0)
        assumption6_holds      (all bounds satisfied within 1e-5)
    """
    z_norms = np.linalg.norm(z, axis=1)
    z_norm_max  = float(np.max(z_norms))
    z_norm_mean = float(np.mean(z_norms))
    z_norm_p99  = float(np.percentile(z_norms, 99))

    sigma_W = np.linalg.svd(W, compute_uv=False)
    W_op_norm = float(sigma_W.max())

    z_hit = float(np.mean(z_norms > 0.99 * R))
    W_hit = float(W_op_norm / W_max) if W_max > 0 else float("nan")
    b_hit = float(abs(b) / B_0) if B_0 > 0 else float("nan")

    a6_ok = (
        z_norm_max <= R + 1e-5
        and W_op_norm <= W_max + 1e-5
        and abs(b) <= B_0 + 1e-5
    )

    return {
        "z_norm_max":             z_norm_max,
        "z_norm_p99":             z_norm_p99,
        "z_norm_mean":            z_norm_mean,
        "W_op_norm":              W_op_norm,
        "b_value":                float(b),
        "z_constraint_hit_rate":  z_hit,
        "W_constraint_hit_rate":  W_hit,
        "b_constraint_hit_rate":  b_hit,
        "assumption6_holds":      bool(a6_ok),
    }


# =============================================================================
# Top-level: compute all metrics for one (z, W, b) snapshot
# =============================================================================

def compute_all_metrics(
    z: np.ndarray,
    W: np.ndarray,
    b: float,
    *,
    data: Dict[str, Any],
    sbm_oracle: Dict[str, Any],
    R: float,
    W_max: float,
    B_0: float,
    classifiers_cfg: Dict[str, Any],
    classifier_seed: int,
) -> Dict[str, Any]:
    """Compute every paper observable from one snapshot.

    Returns a flat dict (no nested structure) so it serializes cleanly to JSON
    and unpacks cleanly into a pandas row in post-hoc analysis.
    """
    # Lazy imports to avoid pulling sklearn unless needed
    from .classifiers.oracle import oracle_classifier_predictions
    from .classifiers.plug_in import plugin_classify
    from .classifiers.direct import direct_classify
    from .classifiers.kmeans import kmeans_classify
    from .oracle import compute_R_H_star, compute_R_T_star

    n = data["n"]
    K = data["K"]
    rows, cols = data["rows"], data["cols"]
    train_mask = data["train_mask"]
    held_mask  = data["held_mask"]
    P_flat     = data["P_flat"]
    A_flat     = data["A_flat"]
    labels     = data["labels"]
    M_star     = sbm_oracle["M_star"]
    H_matrix   = sbm_oracle["H_matrix"]

    out: Dict[str, Any] = {}

    # --- LP losses ---
    out["R_H_emp"]       = compute_R_H_emp(z, W, b, rows, cols, held_mask, A_flat)
    out["R_H_pop"]       = compute_R_H_pop(z, W, b, rows, cols, held_mask, P_flat)
    out["R_H_star"]      = compute_R_H_star(labels, rows, cols, held_mask, H_matrix)
    out["R_T_emp"]       = compute_R_T_emp(z, W, b, rows, cols, train_mask, A_flat)
    out["R_T_pop"]       = compute_R_T_pop(z, W, b, rows, cols, train_mask, P_flat)
    out["R_T_star"]      = compute_R_T_star(labels, rows, cols, train_mask, H_matrix)
    out["excess_emp_H"]  = out["R_H_emp"] - out["R_H_star"]
    out["excess_pop_H"]  = out["R_H_pop"] - out["R_H_star"]

    # Train-side residual proxy: R_T_emp - R_T_star. Strictly an upper bound on
    # rho_hat = R_T_emp - R_T_emp_dagger (the in-class empirical infimum is
    # generally below the entropy floor); under adequate training and adequate
    # architecture, the two are close. Track separately so plots can show the
    # squeeze of R_T_emp toward R_T_star.
    out["train_residual_proxy"] = out["R_T_emp"] - out["R_T_star"]

    # NB: this is the empirical-vs-population gap at the SINGLE TRAINED MODEL,
    # NOT the uniform supremum Gamma_H = sup_{S in S_H} |R_hat_H(S) - R^P_H(S)|
    # defined in the paper. The latter is the closed-form envelope computed
    # post-hoc in analysis/bounds.py:gamma_H_envelope. Do NOT call this Gamma_H
    # in figures/text. The key `gamma_H_at_trained` is kept as an alias for
    # compatibility; `single_model_gap_H` is the clearer name.
    out["single_model_gap_H"]  = abs(out["R_H_emp"] - out["R_H_pop"])
    out["single_model_gap_T"]  = abs(out["R_T_emp"] - out["R_T_pop"])
    out["gamma_H_at_trained"]  = out["single_model_gap_H"]  # deprecated alias

    # --- Score MSE family ---
    out["score_mse_H"]    = compute_score_mse_H(z, W, b, rows, cols, held_mask,
                                                labels, M_star)
    out["score_mse_off"]  = compute_score_mse_off(z, W, b, rows, cols,
                                                  labels, M_star)
    out["score_mse_full"] = compute_score_mse_full(z, W, b, labels, M_star)

    # --- Per-node errors and moments ---
    tilde_e_sq = compute_tilde_e_sq(z, W, b, labels, M_star)
    moments = compute_tilde_e_moments(tilde_e_sq)
    for k, v in moments.items():
        out[f"node_err_{k}"] = v

    # Sanity: (1/n) sum tilde_e_i^2 should equal ScoreMSE_full
    out["score_mse_full_via_node_errors"] = float(np.mean(tilde_e_sq))

    # --- Boundedness-assumption verification (keys prefixed a6_) ---
    a6 = compute_assumption6_diagnostics(z, W, b, R, W_max, B_0)
    out.update({f"a6_{k}": v for k, v in a6.items()})

    # --- Classifiers ---
    if classifiers_cfg.get("oracle", {}).get("enabled", True):
        _, ocl = oracle_classifier_predictions(z, W, b, labels, M_star, K)
        out["oracle_classifier_error"] = ocl["error_rate"]
        out["oracle_classifier_name"]  = "oracle_nearest_prototype_score_profile"
        for cls_label, err in ocl["per_class_error"].items():
            out[f"oracle_classifier_per_class_{cls_label}"] = err

    # --- Plug-in and direct: share the SAME stratified L draws ---
    plugin_enabled = classifiers_cfg.get("plug_in", {}).get("enabled", True)
    direct_enabled = classifiers_cfg.get("direct", {}).get("enabled", True)

    if plugin_enabled or direct_enabled:
        from .classifiers.sampling import sample_stratified_label_sets

        # Decide which m_per_class list and n_resamples to use. We require
        # both classifiers to use the same list so the comparison is on the
        # same L draws. If both enabled and lists differ, error out.
        plug_cfg = classifiers_cfg.get("plug_in", {})
        dir_cfg  = classifiers_cfg.get("direct", {})
        if plugin_enabled and direct_enabled:
            plug_list = list(plug_cfg.get("m_per_class", [10]))
            dir_list  = list(dir_cfg.get("m_per_class", [10]))
            if plug_list != dir_list:
                raise ValueError(
                    f"plug_in.m_per_class={plug_list} != direct.m_per_class={dir_list}; "
                    f"set them equal so the two classifiers share the same labeled sets."
                )
            n_plug = int(plug_cfg.get("n_label_resamples", 5))
            n_dir  = int(dir_cfg.get("n_label_resamples", 5))
            if n_plug != n_dir:
                raise ValueError(
                    f"plug_in.n_label_resamples={n_plug} != direct.n_label_resamples={n_dir}; "
                    f"set them equal for shared resamples."
                )
            m_list = plug_list
            n_resamples = n_plug
        elif plugin_enabled:
            m_list = list(plug_cfg.get("m_per_class", [10]))
            n_resamples = int(plug_cfg.get("n_label_resamples", 5))
        else:
            m_list = list(dir_cfg.get("m_per_class", [10]))
            n_resamples = int(dir_cfg.get("n_label_resamples", 5))

        for m in m_list:
            # ONE shared sampling step, used by both classifiers.
            shared_sets = sample_stratified_label_sets(
                labels, K, int(m), int(n_resamples),
                seed=int(classifier_seed) + int(m) * 1009,  # decouple by m
            )
            if plugin_enabled:
                pres = plugin_classify(
                    z, W, b, labels, M_star, K,
                    label_sets=shared_sets,
                )
                out[f"plug_in_m{m}_error_mean"]    = pres["error_rate_mean"]
                out[f"plug_in_m{m}_error_std"]     = pres["error_rate_std"]
                out[f"plug_in_m{m}_eta_L_mean"]    = pres["eta_L_mean"]
                out[f"plug_in_m{m}_eta_L_std"]     = pres["eta_L_std"]
                out[f"plug_in_m{m}_Delta_hat_mean"] = pres["Delta_hat_mean"]
                out[f"plug_in_m{m}_Delta_hat_std"]  = pres["Delta_hat_std"]
                out[f"plug_in_m{m}_classifier"]    = pres["classifier"]
            if direct_enabled:
                dres = direct_classify(
                    z, labels, K,
                    label_sets=shared_sets,
                    seed=int(classifier_seed) + int(m) * 1009 + 1,
                )
                out[f"direct_m{m}_error_mean"]  = dres["error_rate_mean"]
                out[f"direct_m{m}_error_std"]   = dres["error_rate_std"]
                out[f"direct_m{m}_classifier"]  = dres["classifier"]

    if classifiers_cfg.get("kmeans", {}).get("enabled", True):
        n_init = int(classifiers_cfg["kmeans"].get("n_init", 50))
        kres = kmeans_classify(z, labels, K, seed=classifier_seed + 2,
                               n_init=n_init, W=W)
        # Explicit, classifier-tagged names.
        out["kmeans_z_euclid_error"]    = kres["z_euclid_error"]
        out["kmeans_z_euclid_nmi"]      = kres["z_euclid_nmi"]
        out["kmeans_z_euclid_classifier"] = "kmeans_euclidean_on_z_baseline"
        if "G_form_error" in kres:
            out["kmeans_G_form_error"]      = kres["G_form_error"]
            out["kmeans_G_form_nmi"]        = kres["G_form_nmi"]
            out["kmeans_G_effective_rank"]  = kres["G_effective_rank"]
            out["kmeans_G_form_classifier"] = "kmeans_in_G_form_cor38"
        # NB: there is no unqualified `kmeans_error` / `kmeans_nmi` key. Use the
        # explicit `kmeans_z_euclid_*` or `kmeans_G_form_*` keys; analysis code
        # must NOT plot a column called "kmeans_error" without specifying which
        # variant.

    # --- Sanity quantities and per-node arrays returned separately ---
    out["_tilde_e_sq_array"] = tilde_e_sq  # consumed by run-script for npz dump

    return out
