"""
Analyze the norm-constraint ablation experiment(s).

Handles all three sub-experiments (capacity_ratio, R-W invariance, B_0)
through a single function. Detects which sweep axis was used and renders
the appropriate plots.

Key plots produced:
    fig_capacity_transition.png    measured score_mse_H / classifier err
                                   vs capacity_ratio (the phase-transition
                                   figure for ablation_norm_capacity)
    fig_constraint_hit_rates.png   z/W/b hit rates as diagnostic
    fig_envelope_vs_measured.png   measured vs half-empirical envelope along the axis
    table_summary.csv              tidy aggregated table
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load import load_final
from .bounds import compute_bounds, BoundConstants
from .plots import (
    plot_curve_iqr, plot_constraint_hit_rates, plot_envelope_vs_measured,
)


_DEFAULT_METRICS = [
    "score_mse_H",
    "score_mse_full",
    "excess_emp_H",
    "excess_pop_H",
    "oracle_classifier_error",
    "kmeans_z_euclid_error",
]


def analyze_norm_ablation(
    exp_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    consts: BoundConstants = BoundConstants(),
) -> dict:
    """Run analysis for a norm-ablation experiment.

    Parameters
    ----------
    exp_dir : path to outputs/<experiment_name>/
    out_dir : where to save figures and tables (default: exp_dir / "analysis")
    consts  : universal constants for bound computation
    """
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir) if out_dir is not None else exp_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_final(exp_dir)
    if df.empty:
        print(f"No final-eval rows found in {exp_dir}.")
        return {}

    df = compute_bounds(df, consts=consts)

    sweep_axis = df["sweep_axis"].iloc[0]
    x_col = "sweep_value"
    print(f"Norm-ablation analysis: axis={sweep_axis}")
    print(f"Settings: {sorted(df[x_col].unique())}")
    print(f"Seeds per setting: ~{df.groupby(x_col).size().median():.0f}")

    # ---- Aggregate summary table ----
    agg_metrics = _DEFAULT_METRICS + [c for c in df.columns
                                       if c.startswith("plug_in_m")
                                       and c.endswith("_error_mean")]
    agg_metrics = [m for m in agg_metrics if m in df.columns]

    summary_rows = []
    for sweep_value, sub in df.groupby(x_col):
        row = {"sweep_axis": sweep_axis, "sweep_value": sweep_value,
               "n_seeds": len(sub)}
        for m in agg_metrics:
            row[f"{m}.median"] = float(sub[m].median())
            row[f"{m}.q25"] = float(sub[m].quantile(0.25))
            row[f"{m}.q75"] = float(sub[m].quantile(0.75))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("sweep_value")
    summary.to_csv(out_dir / "table_summary.csv", index=False)
    print(f"Saved {out_dir / 'table_summary.csv'}")

    # ---- Capacity transition figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    plot_curve_iqr(axes[0], df, x_col=x_col, y_col="score_mse_H",
                   label="score_mse_H")
    if "score_mse_full" in df.columns:
        plot_curve_iqr(axes[0], df, x_col=x_col, y_col="score_mse_full",
                       label="score_mse_full", marker="s")
    axes[0].set_ylabel("score MSE")
    axes[0].set_xlabel(sweep_axis)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Score MSE")

    if "oracle_classifier_error" in df.columns:
        plot_curve_iqr(axes[1], df, x_col=x_col, y_col="oracle_classifier_error",
                       label="oracle err")
    if "kmeans_z_euclid_error" in df.columns:
        plot_curve_iqr(axes[1], df, x_col=x_col, y_col="kmeans_z_euclid_error",
                       label="k-means err", marker="s")
    for col in df.columns:
        if col.startswith("plug_in_m") and col.endswith("_error_mean"):
            m = col.split("plug_in_m")[1].split("_")[0]
            plot_curve_iqr(axes[1], df, x_col=x_col, y_col=col,
                           label=f"plug-in m={m}", marker="^")
    axes[1].set_ylabel("classification error")
    axes[1].set_xlabel(sweep_axis)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Classifier errors")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_capacity_transition.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'fig_capacity_transition.png'}")

    # ---- Constraint hit rates ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_constraint_hit_rates(ax, df, x_col=x_col)
    ax.set_title(f"Bounded-class constraint hit rates vs {sweep_axis}")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_constraint_hit_rates.png", dpi=150)
    plt.close(fig)

    # ---- Envelope vs measured ----
    if "bound.cor42_oracle" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_envelope_vs_measured(
            ax, df, x_col=x_col,
            measured_col="oracle_classifier_error",
            envelope_cols=["bound.cor42_oracle", "bound.markov_oracle"],
            log_x=False, log_y=True,
        )
        ax.set_title(f"Oracle err vs half-empirical / Markov envelope")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_envelope_vs_measured.png", dpi=150)
        plt.close(fig)

    return {"summary": summary, "df": df}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    analyze_norm_ablation(args.exp_dir, args.out_dir)
