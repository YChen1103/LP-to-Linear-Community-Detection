"""
Analyze SBM-parameter sweeps: sparsity (rho_n) or separation (p_in / Delta_s).

Both share an analysis structure:
    - Aggregated metrics vs sweep axis
    - For each setting, derived quantities (B, mu_B, Delta_s) plotted alongside
    - Bound vs measured comparison: when bound becomes vacuous (>= 1 - 1/K)
      versus when measured stays well below

Each row in the resulting summary table includes the derived oracle values
to make it easy to plot e.g. score_mse_H vs Delta_s instead of vs the
nominal sweep axis.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load import load_final
from .bounds import compute_bounds, BoundConstants
from .plots import plot_curve_iqr, plot_envelope_vs_measured


_BASE_METRICS = [
    "score_mse_H", "score_mse_full",
    "excess_emp_H", "excess_pop_H",
    "oracle_classifier_error", "kmeans_z_euclid_error", "kmeans_G_form_error",
]


def analyze_sbm_sweep(
    exp_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    consts: BoundConstants = BoundConstants(),
    secondary_axis: str | None = None,
) -> dict:
    """Analyze a sweep_sbm_* experiment.

    `secondary_axis` (optional): a column name in the loaded DF to plot
    against, in addition to sweep_value. E.g. "derived.Delta_s" for the
    separation experiment, or "derived.B_star" for the sparsity experiment.

    If None, we auto-detect:
      - sweep on sbm.rho_n        -> secondary "derived.B_star"
      - sweep on sbm.block_spec.* -> secondary "derived.Delta_s"
    """
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir) if out_dir is not None else exp_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_final(exp_dir)
    if df.empty:
        print(f"No final-eval rows in {exp_dir}.")
        return {}
    df = compute_bounds(df, consts=consts)

    sweep_axis = df["sweep_axis"].iloc[0]
    if secondary_axis is None:
        if "rho_n" in (sweep_axis or ""):
            secondary_axis = "derived.B_star"
        else:
            secondary_axis = "derived.Delta_s"

    print(f"sweep_sbm: axis = {sweep_axis}; secondary = {secondary_axis}")
    print(f"Sweep values: {sorted(df['sweep_value'].unique())}")

    metrics = list(_BASE_METRICS)
    metrics += [c for c in df.columns
                if c.startswith("plug_in_m") and c.endswith("_error_mean")]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for sw, sub in df.groupby("sweep_value"):
        r = {sweep_axis: sw, "n_seeds": len(sub)}
        for col in [secondary_axis, "derived.B", "derived.mu_B_at_B",
                    "derived.Delta_s", "derived.capacity_required"]:
            if col in sub.columns:
                r[col] = float(sub[col].median())
        for m in metrics:
            r[f"{m}.median"] = float(sub[m].median())
            r[f"{m}.q25"] = float(sub[m].quantile(0.25))
            r[f"{m}.q75"] = float(sub[m].quantile(0.75))
        rows.append(r)
    summary = pd.DataFrame(rows).sort_values(sweep_axis)
    summary.to_csv(out_dir / "table_summary.csv", index=False)
    print(f"Saved {out_dir / 'table_summary.csv'}")

    # ---- Metrics vs sweep axis ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in ["score_mse_H", "score_mse_full"]:
        if m in df.columns:
            plot_curve_iqr(axes[0], df, x_col="sweep_value", y_col=m, label=m)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(sweep_axis)
    axes[0].set_ylabel("score MSE")
    axes[0].legend()
    axes[0].set_title(f"Score MSE vs {sweep_axis}")

    for m in ["oracle_classifier_error", "kmeans_z_euclid_error", "kmeans_G_form_error"]:
        if m in df.columns:
            plot_curve_iqr(axes[1], df, x_col="sweep_value", y_col=m, label=m)
    for col in df.columns:
        if col.startswith("plug_in_m") and col.endswith("_error_mean"):
            label = col.replace("_error_mean", "")
            plot_curve_iqr(axes[1], df, x_col="sweep_value", y_col=col, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(sweep_axis)
    axes[1].set_ylabel("classifier error")
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"Classifier errors vs {sweep_axis}")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_metrics_vs_axis.png", dpi=150)
    plt.close(fig)

    # ---- Derived quantities along axis (for context) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    derived_cols = ["derived.B", "derived.B_star", "derived.mu_B_at_B",
                    "derived.Delta_s", "derived.capacity_required"]
    derived_cols = [c for c in derived_cols if c in df.columns]
    for c in derived_cols:
        plot_curve_iqr(ax, df, x_col="sweep_value", y_col=c, label=c)
    ax.set_xlabel(sweep_axis)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Derived oracle quantities vs sweep axis")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_derived_vs_axis.png", dpi=150)
    plt.close(fig)

    # ---- Bound vs measured (with K-dependent random baseline) ----
    if "bound.cor42_oracle" in df.columns and "oracle_classifier_error" in df.columns:
        K = int(df["cfg.sbm.K"].iloc[0]) if "cfg.sbm.K" in df.columns else 2
        random_baseline = 1.0 - 1.0 / max(K, 2)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_envelope_vs_measured(
            ax, df, x_col="sweep_value",
            measured_col="oracle_classifier_error",
            envelope_cols=["bound.cor42_oracle", "bound.markov_oracle"],
            log_x=False, log_y=True,
        )
        ax.axhline(random_baseline, color="r", linestyle=":",
                   label=f"random baseline = 1 - 1/K = {random_baseline:.3f}")
        ax.legend(fontsize=8)
        ax.set_title(f"Bound vs measured (half-empirical corollary)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_envelope_vs_measured.png", dpi=150)
        plt.close(fig)

    return {"summary": summary, "df": df}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--secondary-axis", default=None)
    args = p.parse_args()
    analyze_sbm_sweep(args.exp_dir, args.out_dir,
                      secondary_axis=args.secondary_axis)
