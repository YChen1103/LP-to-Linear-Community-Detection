"""
Generic plot helpers used by the per-experiment analysis scripts.

We deliberately keep these small and composable -- each helper takes a
DataFrame and produces one Axes / Figure. The per-experiment scripts orchestrate.
"""
from __future__ import annotations
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Aggregator: per-sweep mean + IQR
# =============================================================================

def aggregate_by_sweep(
    df: pd.DataFrame, *, x_col: str, y_col: str, seed_col: str = "seed",
) -> pd.DataFrame:
    """Group by x_col, compute mean / std / median / 25,75 percentiles of y."""
    g = df.groupby(x_col)[y_col]
    out = pd.DataFrame({
        "mean":   g.mean(),
        "std":    g.std(),
        "median": g.median(),
        "q25":    g.quantile(0.25),
        "q75":    g.quantile(0.75),
        "n":      g.count(),
    }).reset_index()
    return out.sort_values(x_col).reset_index(drop=True)


# =============================================================================
# Single curve with error band (median + IQR shading)
# =============================================================================

def plot_curve_iqr(
    ax, df: pd.DataFrame, *, x_col: str, y_col: str, label: Optional[str] = None,
    seed_col: str = "seed", color=None, marker="o", linestyle="-",
):
    """Plot median curve with IQR shading."""
    agg = aggregate_by_sweep(df, x_col=x_col, y_col=y_col, seed_col=seed_col)
    line, = ax.plot(agg[x_col], agg["median"],
                    marker=marker, linestyle=linestyle,
                    color=color, label=label)
    ax.fill_between(agg[x_col], agg["q25"], agg["q75"],
                    color=line.get_color(), alpha=0.2)
    return line


# =============================================================================
# Multiple curves (one per group) with IQR
# =============================================================================

def plot_grouped_curves(
    ax, df: pd.DataFrame, *, x_col: str, y_col: str,
    group_col: str, seed_col: str = "seed",
):
    """Plot one IQR curve per unique value of group_col."""
    for g_val, sub in df.groupby(group_col):
        plot_curve_iqr(ax, sub, x_col=x_col, y_col=y_col,
                       label=f"{group_col}={g_val}", seed_col=seed_col)
    ax.legend()


# =============================================================================
# Envelope-vs-measured comparison
# =============================================================================

def plot_envelope_vs_measured(
    ax, df: pd.DataFrame, *, x_col: str,
    measured_col: str, envelope_cols: Sequence[str],
    seed_col: str = "seed",
    log_x: bool = True, log_y: bool = True,
):
    """Plot the measured quantity and one or more envelopes on the same axes.

    Use case: x_col = sbm.n; measured_col = score_mse_H;
              envelope_cols = ["bound.gamma_H_envelope", "bound.cor42_oracle"]
    """
    # Measured
    plot_curve_iqr(ax, df, x_col=x_col, y_col=measured_col,
                   label=f"measured {measured_col}", seed_col=seed_col,
                   marker="o", linestyle="-")
    # Each envelope
    for env_col in envelope_cols:
        if env_col not in df.columns:
            continue
        plot_curve_iqr(ax, df, x_col=x_col, y_col=env_col,
                       label=f"envelope {env_col}", seed_col=seed_col,
                       marker="x", linestyle="--")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_col)
    ax.legend(fontsize=8)


# =============================================================================
# Ratio plot: measured / theoretical-rate
# =============================================================================

def plot_ratio_to_rate(
    ax, df: pd.DataFrame, *, x_col: str, y_col: str,
    rate_fn,                                # callable: x -> theoretical rate
    seed_col: str = "seed", label: Optional[str] = None,
):
    """Plot y / rate_fn(x) vs x.

    If the bound is tight, this ratio should be roughly constant.
    Decreasing ratio => bound is loose; increasing ratio => bound under-predicts.
    """
    df = df.copy()
    df["__ratio__"] = df[y_col] / np.maximum(rate_fn(df[x_col].astype(float)), 1e-30)
    plot_curve_iqr(ax, df, x_col=x_col, y_col="__ratio__",
                   label=label or f"{y_col} / rate", seed_col=seed_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel("ratio")


# =============================================================================
# Constraint hit-rate diagnostic plot (for norm-ablation sanity)
# =============================================================================

def plot_constraint_hit_rates(
    ax, df: pd.DataFrame, *, x_col: str, seed_col: str = "seed",
):
    """Plot z / W / b constraint hit rates vs sweep axis."""
    for col, label in [
        ("a6_z_constraint_hit_rate", "z hit rate (frac at boundary)"),
        ("a6_W_constraint_hit_rate", "W hit rate (||W||_op / W_max)"),
        ("a6_b_constraint_hit_rate", "b hit rate (|b| / B_0)"),
    ]:
        if col in df.columns:
            plot_curve_iqr(ax, df, x_col=x_col, y_col=col, label=label,
                           seed_col=seed_col)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel("hit rate")
    ax.legend(fontsize=8)
