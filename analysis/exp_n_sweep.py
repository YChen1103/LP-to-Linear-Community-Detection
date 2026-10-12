"""
Analyze the finite-sample (n) sweep experiment.

Key plots / tables:
    fig_metric_vs_n.png            measured metrics vs n on log-log
    fig_envelope_vs_measured.png   measured + envelopes
    fig_ratio_to_rate.png          metric / sqrt(log n / n)
    table_loglog_slopes.csv        slope +/- CI for each metric
    table_summary.csv              tidy aggregate
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load import load_final
from .bounds import compute_bounds, BoundConstants
from .fits import fit_many_metrics
from .plots import plot_curve_iqr, plot_envelope_vs_measured, plot_ratio_to_rate


_RATE_FNS = {
    "sqrt_logn_over_n":  lambda n: np.sqrt(np.log(np.maximum(n, 2)) / np.maximum(n, 1)),
    "logn_over_n":       lambda n: np.log(np.maximum(n, 2)) / np.maximum(n, 1),
    "1_over_sqrt_n":     lambda n: 1.0 / np.sqrt(np.maximum(n, 1)),
}


def analyze_n_sweep(
    exp_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    consts: BoundConstants = BoundConstants(),
    drop_n_below: float | None = None,
) -> dict:
    """Analyze a sweep_n experiment.

    `drop_n_below` excludes small-n points from log-log fits (since the bound
    has a self-consistency requirement of "n sufficiently large"). Set to
    e.g. 500 to drop n in {100, 200}.
    """
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir) if out_dir is not None else exp_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_final(exp_dir)
    if df.empty:
        print(f"No final-eval rows in {exp_dir}.")
        return {}
    df = compute_bounds(df, consts=consts)
    x_col = "sweep_value"   # = n in this sweep

    print(f"sweep_n: n values = {sorted(df[x_col].unique())}")
    print(f"Seeds per n: ~{df.groupby(x_col).size().median():.0f}")

    # ---- Summary table ----
    metrics = [
        "score_mse_H", "score_mse_full",
        "excess_emp_H", "excess_pop_H",
        "gamma_H_at_trained",
        "oracle_classifier_error",
        "kmeans_z_euclid_error", "kmeans_G_form_error",
    ]
    metrics += [c for c in df.columns
                if c.startswith("plug_in_m") and c.endswith("_error_mean")]
    metrics += [c for c in df.columns
                if c.startswith("direct_m") and c.endswith("_error_mean")]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for n_val, sub in df.groupby(x_col):
        r = {"n": n_val, "n_seeds": len(sub)}
        for m in metrics:
            r[f"{m}.median"] = float(sub[m].median())
            r[f"{m}.q25"] = float(sub[m].quantile(0.25))
            r[f"{m}.q75"] = float(sub[m].quantile(0.75))
        rows.append(r)
    summary = pd.DataFrame(rows).sort_values("n")
    summary.to_csv(out_dir / "table_summary.csv", index=False)

    # ---- Log-log fits ----
    fit_metrics = [
        "score_mse_H", "score_mse_full", "excess_emp_H", "excess_pop_H",
        "gamma_H_at_trained",
    ]
    fit_metrics = [m for m in fit_metrics if m in df.columns]
    fits = fit_many_metrics(df, x_col=x_col, y_cols=fit_metrics, seed_col="seed",
                            n_bootstrap=1000, drop_x_below=drop_n_below)
    fits.to_csv(out_dir / "table_loglog_slopes.csv", index=False)
    print("Log-log slopes vs n:")
    print(fits.to_string(index=False))

    # ---- Metric vs n ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for m in ["score_mse_H", "score_mse_full"]:
        if m in df.columns:
            plot_curve_iqr(axes[0], df, x_col=x_col, y_col=m, label=m)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("ScoreMSE")
    axes[0].legend()
    axes[0].set_title("Score MSE vs n")

    for m in ["excess_emp_H", "excess_pop_H", "gamma_H_at_trained"]:
        if m in df.columns:
            plot_curve_iqr(axes[1], df, x_col=x_col, y_col=m, label=m)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("excess / Gamma_H proxy")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Excess and Gamma_H proxy")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_metric_vs_n.png", dpi=150)
    plt.close(fig)

    # ---- Classifier errors vs n ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if "oracle_classifier_error" in df.columns:
        plot_curve_iqr(ax, df, x_col=x_col, y_col="oracle_classifier_error",
                       label="oracle")
    if "kmeans_z_euclid_error" in df.columns:
        plot_curve_iqr(ax, df, x_col=x_col, y_col="kmeans_z_euclid_error",
                       label="k-means (Euclidean on z, baseline)", marker="s")
    if "kmeans_G_form_error" in df.columns:
        plot_curve_iqr(ax, df, x_col=x_col, y_col="kmeans_G_form_error",
                       label="k-means (G-form, Mahalanobis nearest-centroid form)", marker="D")
    for col in df.columns:
        if col.startswith("plug_in_m") and col.endswith("_error_mean"):
            m_label = col.split("plug_in_m")[1].split("_")[0]
            plot_curve_iqr(ax, df, x_col=x_col, y_col=col,
                           label=f"plug-in m={m_label}", marker="^")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("classification error")
    ax.legend(fontsize=8)
    ax.set_title("Classifier errors vs n")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_classifier_vs_n.png", dpi=150)
    plt.close(fig)

    # ---- Envelope vs measured ----
    if "bound.cor42_oracle" in df.columns and "oracle_classifier_error" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_envelope_vs_measured(
            ax, df, x_col=x_col,
            measured_col="oracle_classifier_error",
            envelope_cols=["bound.cor42_oracle", "bound.markov_oracle"],
        )
        ax.set_title("Oracle err vs envelope")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_envelope_vs_measured.png", dpi=150)
        plt.close(fig)

    # ---- Ratio plot ----
    if "score_mse_H" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_ratio_to_rate(
            ax, df, x_col=x_col, y_col="score_mse_H",
            rate_fn=_RATE_FNS["sqrt_logn_over_n"],
            label="score_mse_H / sqrt(log n / n)",
        )
        ax.set_xscale("log")
        ax.set_title("Ratio of measured to predicted rate")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_ratio_to_rate.png", dpi=150)
        plt.close(fig)

    return {"summary": summary, "fits": fits, "df": df}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--drop-n-below", type=float, default=None)
    args = p.parse_args()
    analyze_n_sweep(args.exp_dir, args.out_dir, drop_n_below=args.drop_n_below)
