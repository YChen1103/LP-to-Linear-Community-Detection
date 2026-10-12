"""
Post-hoc bound envelope computation.

These bounds are NOT computed at training time -- they are derived from the
raw observables logged during training, with universal-constant choices made
configurable here at analysis time.

What we compute:

    Gamma_H_envelope     ~ C_stat * B * sqrt(d * log(n*B*R*W_max) / (tau*n))
                           held-out uniform deviation envelope.

    Gamma_T_envelope     ~ C_stat * B * sqrt(d * log(...) / ((1-tau)*n))
                           train-side analogue (train-side corollary).

    Gamma_TH_envelope    ~ (C_1 B^2 / sqrt(tau(1-tau))) * sqrt(d log(...) / n)
                           cross-side (R^P_T - R^P_H) envelope (train-side corollary).

    delta_mask_n         ~ C_1 * B^2 / tau * sqrt(d log(.) / n) + 4 B^2 / n
                           (mask-concentration term used by half-empirical bound).

    V_n_stat             = Gamma_H_envelope + delta_mask_n
                           (raw sum, for comparison with single_model_gap).

    V_n_stat_theorem     = (C_stat B / mu_B) sqrt(d log/(tau n))
                           + (C_1 B^2 / tau) sqrt(d log / n) + 4 B^2 / n
                           (V_n^{stat} as it appears in the main theorem).

    bound_half_empirical_oracle    = (8 / (mu_B Delta_s^2)) (R_H_emp - R_H_star + Gamma_H_env)
                                   + 4 delta_mask / Delta_s^2
                           (held-out side, half-empirical corollary; fully computable).

    bound_half_empirical_plugin    = analogous with eff sep Delta_s - 2 eta_L.

    bound_train_side_oracle = (8 / (mu_B Delta_s^2))
                                (R_T_emp - R_T_star + 2(Gamma_T + Gamma_TH))
                            + 4 V_n_stat_theorem / Delta_s^2
                           (train-side corollary; uses train_residual_proxy =
                            R_T_emp - R_T_star, an upper bound on the true
                            in-class residual rho_hat).

    bound_score_mse_H_via_strong_convexity
                          = (2 / mu_B) * (R_H_pop - R_H_star)
                           (Step 1, BCE strong convexity; tight, no const).

    bound_markov_oracle / bound_markov_plugin
                          = 4 ScoreMSE_full / Delta_s^2 [eff^2]
                           (Markov-only, uses directly-measured ScoreMSE_full).

The names `cor42_*` are kept as aliases for `half_empirical_*` for
backward compatibility.

We deliberately do NOT include a main-theorem envelope using excess_pop_H as a
proxy for B_app + R_n: that conflates eps_pop_opt with eps_emp_opt and is
neither the main theorem nor any corollary. If you need a population-side
comparator, use `bound_score_mse_H_via_strong_convexity` (a tight quadratic
relation, not a bound on classification error) or `bound_markov_oracle`
(uses the directly-measured ScoreMSE_full).

All envelopes are pandas-compatible: they take a row of observables and
return a scalar (or a column when `compute_bounds` is mapped over a frame).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Universal-constant container
# =============================================================================

@dataclass
class BoundConstants:
    """Universal constants in the main-theorem / half-empirical-corollary envelopes.

    These are the constants whose explicit values are not tracked in the
    paper's proofs. Default = 1 (a-la "rate-only" envelope). Increase to
    inflate the envelope.
    """
    C_stat:       float = 1.0    # Gamma_H envelope multiplier (Step 0)
    C_1:          float = 1.0    # delta_mask_n leading multiplier (Step 2)
    log_floor:    float = 1.0    # min argument to log(.) (avoid log(0/negative))


_DEFAULT_CONSTS = BoundConstants()


# =============================================================================
# Closed-form bound terms
# =============================================================================

def gamma_H_envelope(
    n, B, R_norm, W_max_norm, d, tau,
    consts: BoundConstants = _DEFAULT_CONSTS,
) -> float | np.ndarray:
    """Step-0 envelope Gamma_H <= C_stat * B * sqrt(d log(...) / (tau n))."""
    arg = np.maximum(n * B * np.maximum(R_norm, 1.0) * np.maximum(W_max_norm, 1.0),
                     consts.log_floor)
    return consts.C_stat * B * np.sqrt(d * np.log(arg) / (tau * n))


def delta_mask_n(
    n, B, R_norm, W_max_norm, d, tau,
    consts: BoundConstants = _DEFAULT_CONSTS,
) -> float | np.ndarray:
    """Step-2 mask-concentration term."""
    arg = np.maximum(n * B * np.maximum(R_norm, 1.0) * np.maximum(W_max_norm, 1.0),
                     consts.log_floor)
    main = consts.C_1 * (B ** 2) / tau * np.sqrt(d * np.log(arg) / n)
    return main + 4.0 * (B ** 2) / n


def V_n_stat_raw(
    n, B, R_norm, W_max_norm, d, tau,
    consts: BoundConstants = _DEFAULT_CONSTS,
) -> float | np.ndarray:
    """Sum of the two statistical envelope pieces, BEFORE strong-convexity scaling.

        V_n_stat_raw  =  Gamma_H_envelope  +  delta_mask_n

    This is the right thing to compare against measured single-model gaps
    (|R_hat_H - R^P_H| = single_model_gap_H) and ScoreMSE residuals; it is
    NOT what appears as V_n^stat in the main theorem, which carries an
    extra B/mu_B factor on the Gamma piece. Use `V_n_stat_theorem` for that.
    """
    return (gamma_H_envelope(n, B, R_norm, W_max_norm, d, tau, consts)
            + delta_mask_n(n, B, R_norm, W_max_norm, d, tau, consts))


def V_n_stat_theorem(
    n, B, R_norm, W_max_norm, d, tau, mu_B,
    consts: BoundConstants = _DEFAULT_CONSTS,
) -> float | np.ndarray:
    """V_n^{stat} as defined in the main theorem:

        V_n_stat_theorem
          =  (C_stat * B / mu_B) * sqrt(d log(...) / (tau n))      <- from Gamma_H
           +  (C_1 * B^2 / tau)   * sqrt(d log(...) /  n)           <- mask
           +   4 B^2 / n                                            <- diagonal correction

    This is the right object for comparing against the bound on classification
    error: 4 (B_app + R_n + V_n_stat_theorem) / Delta_s^2 (main theorem).
    """
    arg = np.maximum(n * B * np.maximum(R_norm, 1.0) * np.maximum(W_max_norm, 1.0),
                     consts.log_floor)
    log_term = np.log(arg)
    mu_safe = np.maximum(mu_B, 1e-12)
    gamma_part = consts.C_stat * B / mu_safe * np.sqrt(d * log_term / (tau * n))
    mask_part  = consts.C_1 * (B ** 2) / tau * np.sqrt(d * log_term / n)
    diag_part  = 4.0 * (B ** 2) / n
    return gamma_part + mask_part + diag_part


# Backward-compat alias. Prefer V_n_stat_raw or V_n_stat_theorem.
def V_n_stat(*args, **kwargs):
    """Deprecated alias for V_n_stat_raw. Use V_n_stat_raw or V_n_stat_theorem
    explicitly to avoid ambiguity (the raw sum and the theorem's V_n^stat
    differ by a B/mu_B factor on the Gamma piece).
    """
    import warnings
    warnings.warn(
        "V_n_stat is deprecated; use V_n_stat_raw (raw sum) or "
        "V_n_stat_theorem (with strong-convexity scaling).",
        DeprecationWarning, stacklevel=2,
    )
    return V_n_stat_raw(*args, **kwargs)


# =============================================================================
# Score-MSE-side bound (tight)
# =============================================================================

def bound_score_mse_H_via_strong_convexity(
    excess_pop_H, mu_B,
) -> float | np.ndarray:
    """ScoreMSE_H <= (2 / mu_B) * (R_H_pop - R_H_star).

    From Step 1 (strong convexity of BCE in [-B, B]). Tight up to mu_B.
    Requires excess_pop_H, which is observable on synthetic SBMs.
    """
    return 2.0 / np.maximum(mu_B, 1e-12) * np.maximum(excess_pop_H, 0.0)


# =============================================================================
# Half-empirical (held-out side) oracle / plug-in bounds.
# These are the bounds of the half-empirical corollary;
# the alias names "bound.cor42_oracle" / "bound.cor42_plugin" map to these.
# =============================================================================

def bound_half_empirical_oracle(
    excess_emp_H, mu_B, Delta_s, gamma_env, delta_mask,
) -> float | np.ndarray:
    """Half-empirical bound on oracle classifier h^star error.

        err <= (8 / (mu_B Delta_s^2)) (R_H_emp - R_H_star + Gamma_H_env)
             + 4 delta_mask / Delta_s^2

    Half-empirical: empirical excess on H + statistical envelopes; fully
    computable on any held-out set.
    """
    inner = np.maximum(excess_emp_H, 0.0) + gamma_env
    Delta_sq = np.maximum(Delta_s ** 2, 1e-12)
    mu_safe = np.maximum(mu_B, 1e-12)
    return (8.0 / (mu_safe * Delta_sq)) * inner + 4.0 * delta_mask / Delta_sq


def bound_half_empirical_plugin(
    excess_emp_H, mu_B, Delta_s, eta_L, gamma_env, delta_mask,
) -> float | np.ndarray:
    """Plug-in version of the half-empirical bound (effective separation
    Delta_s - 2 eta_L). Vacuous when eta_L >= Delta_s / 2."""
    eff = np.maximum(Delta_s - 2.0 * eta_L, 1e-12)
    eff_sq = eff ** 2
    inner = np.maximum(excess_emp_H, 0.0) + gamma_env
    mu_safe = np.maximum(mu_B, 1e-12)
    return (8.0 / (mu_safe * eff_sq)) * inner + 4.0 * delta_mask / eff_sq


# =============================================================================
# Train-side envelopes (train-side corollary).
# =============================================================================

def gamma_T_envelope(
    n, B, R_norm, W_max_norm, d, tau,
    consts: BoundConstants = _DEFAULT_CONSTS,
) -> float | np.ndarray:
    """Train-side envelope:
        Gamma_T <= C_stat * B * sqrt(d log(...) / ((1-tau) n))
    Mirrors Gamma_H with tau <-> 1-tau (T is the complement of H).
    """
    arg = np.maximum(n * B * np.maximum(R_norm, 1.0) * np.maximum(W_max_norm, 1.0),
                     consts.log_floor)
    one_minus_tau = np.maximum(1.0 - tau, 1e-12)
    return consts.C_stat * B * np.sqrt(d * np.log(arg) / (one_minus_tau * n))


def gamma_TH_envelope(
    n, B, R_norm, W_max_norm, d, tau,
    consts: BoundConstants = _DEFAULT_CONSTS,
) -> float | np.ndarray:
    """Cross-side envelope (train-side corollary):
        Gamma_TH <= (C_1 B^2 / sqrt(tau (1-tau))) * sqrt(d log(...) / n)
    Cross-side deviation between population train- and held-out BCE.
    """
    arg = np.maximum(n * B * np.maximum(R_norm, 1.0) * np.maximum(W_max_norm, 1.0),
                     consts.log_floor)
    tau_geom = np.maximum(tau * (1.0 - tau), 1e-12)
    return (consts.C_1 * (B ** 2) / np.sqrt(tau_geom)) * np.sqrt(d * np.log(arg) / n)


def bound_train_side(
    train_residual_proxy, mu_B, Delta_s, gamma_T, gamma_TH, eapp_proxy=0.0,
    V_n_stat_thm=0.0,
) -> float | np.ndarray:
    """Train-side bound on oracle classifier error (train-side corollary):

        err <= (8 / (mu_B Delta_s^2))
                * (rho_hat + eapp + 2(Gamma_T + Gamma_TH))
             + 4 V_n_stat / Delta_s^2

    `train_residual_proxy` = R_T_emp - R_T_star (proxy for rho_hat; equals
    rho_hat when in-class infimum equals entropy floor).
    `eapp_proxy` defaults to 0 (unobservable; pass a positive value to inflate).
    `V_n_stat_thm` is V_n_stat_theorem (already includes the strong-convexity
    scaling on Gamma_H from the held-out chain). For a pure train-side reading,
    pass V_n_stat_thm=0.

    Vacuous as a number when train_residual_proxy < 0 (empirical risk below
    entropy floor by chance); we max-clip to 0.
    """
    inner = np.maximum(train_residual_proxy, 0.0) + eapp_proxy + 2.0 * (gamma_T + gamma_TH)
    Delta_sq = np.maximum(Delta_s ** 2, 1e-12)
    mu_safe = np.maximum(mu_B, 1e-12)
    return (8.0 / (mu_safe * Delta_sq)) * inner + 4.0 * V_n_stat_thm / Delta_sq


# =============================================================================
# Backward-compat aliases (bound.cor42_*).
# =============================================================================

def bound_cor42_oracle(*args, **kwargs):
    """Deprecated alias for `bound_half_empirical_oracle`."""
    return bound_half_empirical_oracle(*args, **kwargs)


def bound_cor42_plugin(*args, **kwargs):
    """Deprecated alias for `bound_half_empirical_plugin`."""
    return bound_half_empirical_plugin(*args, **kwargs)


# =============================================================================
# Markov-only classification bound
# =============================================================================

def bound_markov_oracle(score_mse_full, Delta_s):
    """Oracle err <= 4 ScoreMSE_full / Delta_s^2 (Step 3, Markov)."""
    Delta_sq = np.maximum(Delta_s ** 2, 1e-12)
    return 4.0 * score_mse_full / Delta_sq


def bound_markov_plugin(score_mse_full, Delta_s, eta_L):
    """Plug-in err <= 4 ScoreMSE_full / (Delta_s - 2 eta_L)^2 (Step 3, plug-in form)."""
    eff_sq = np.maximum((Delta_s - 2.0 * eta_L) ** 2, 1e-12)
    return 4.0 * score_mse_full / eff_sq


# =============================================================================
# DataFrame-level: add bound columns en masse
# =============================================================================

def compute_bounds(
    df: pd.DataFrame,
    consts: BoundConstants = _DEFAULT_CONSTS,
    *,
    R_col: str = "cfg.encoder.R",
    W_max_col: str = "cfg.decoder.W_max",
    d_col: str = "cfg.encoder.d",
    tau_col: str = "cfg.mask.tau",
    n_col: str = "cfg.sbm.n",
    B_col: str = "derived.B",
    mu_B_col: str = "derived.mu_B_at_B",
    Delta_s_col: str = "derived.Delta_s",
) -> pd.DataFrame:
    """Add bound columns to a DataFrame. Returns a new DataFrame with the
    original columns + bound columns prefixed `bound.`.

    The column name args allow flexibility for different DataFrame shapes,
    but the defaults match what `load.load_experiment` produces.
    """
    df = df.copy()

    n   = df[n_col].astype(float).values
    B   = df[B_col].astype(float).values
    R   = df[R_col].astype(float).values
    Wm  = df[W_max_col].astype(float).values
    d   = df[d_col].astype(float).values
    tau = df[tau_col].astype(float).values
    muB = df[mu_B_col].astype(float).values
    Ds  = df[Delta_s_col].astype(float).values

    g  = gamma_H_envelope(n, B, R, Wm, d, tau, consts)
    dm = delta_mask_n(n, B, R, Wm, d, tau, consts)
    V_raw = g + dm
    V_thm = V_n_stat_theorem(n, B, R, Wm, d, tau, muB, consts)

    df["bound.gamma_H_envelope"]  = g
    df["bound.delta_mask_n"]      = dm
    df["bound.V_n_stat_raw"]      = V_raw   # raw sum (compare to single_model_gap)
    df["bound.V_n_stat_theorem"]  = V_thm   # theorem form (compare to err bound)
    # Legacy alias:
    df["bound.V_n_stat"]          = V_raw

    # Train-side envelopes (train-side corollary).
    g_T  = gamma_T_envelope(n, B, R, Wm, d, tau, consts)
    g_TH = gamma_TH_envelope(n, B, R, Wm, d, tau, consts)
    df["bound.gamma_T_envelope"]  = g_T
    df["bound.gamma_TH_envelope"] = g_TH

    if "excess_pop_H" in df.columns:
        df["bound.score_mse_H_via_strong_convexity"] = (
            bound_score_mse_H_via_strong_convexity(df["excess_pop_H"].values, muB)
        )

    if "excess_emp_H" in df.columns:
        df["bound.half_empirical_oracle"] = bound_half_empirical_oracle(
            df["excess_emp_H"].values, muB, Ds, g, dm
        )
        # Backward-compat alias.
        df["bound.cor42_oracle"] = df["bound.half_empirical_oracle"]

    if "train_residual_proxy" in df.columns:
        df["bound.train_side_oracle"] = bound_train_side(
            df["train_residual_proxy"].values, muB, Ds, g_T, g_TH,
            eapp_proxy=0.0, V_n_stat_thm=0.0,
        )

    if "score_mse_full" in df.columns:
        df["bound.markov_oracle"] = bound_markov_oracle(
            df["score_mse_full"].values, Ds
        )

    # For each plug_in_m{m} column, add corresponding half-empirical + markov bounds
    for col in df.columns:
        if col.startswith("plug_in_m") and col.endswith("_eta_L_mean"):
            prefix = col[: -len("_eta_L_mean")]   # plug_in_mNN
            eta = df[col].values
            if "excess_emp_H" in df.columns:
                df[f"bound.half_empirical_{prefix}"] = bound_half_empirical_plugin(
                    df["excess_emp_H"].values, muB, Ds, eta, g, dm
                )
                df[f"bound.cor42_{prefix}"] = df[f"bound.half_empirical_{prefix}"]
            if "score_mse_full" in df.columns:
                df[f"bound.markov_{prefix}"] = bound_markov_plugin(
                    df["score_mse_full"].values, Ds, eta
                )

    return df
