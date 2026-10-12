"""
Analyze architecture sweep (encoder family) or feature mode sweep.

Architecture is a categorical axis, so plots are bar charts / categorical
points rather than line charts.

Key plots:
    fig_arch_metrics.png       bar plot of key metrics across architectures
    fig_arch_classifier.png    classifier errors across architectures
    table_summary.csv          tidy table

The same code handles both encoder.family sweeps and encoder.features.mode
sweeps -- the only difference is the categorical labels.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load import load_final
from .bounds import compute_bounds, BoundConstants


_BASE_METRICS = [
    "score_mse_H", "score_mse_full",
    "excess_emp_H", "excess_pop_H",
    "oracle_classifier_error", "kmeans_z_euclid_error", "kmeans_G_form_error",
]


def analyze_arch_sweep(
    exp_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    consts: BoundConstants = BoundConstants(),
) -> dict:
    """Analyze a sweep_arch or sweep_features experiment."""
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir) if out_dir is not None else exp_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_final(exp_dir)
    if df.empty:
        print(f"No final-eval rows in {exp_dir}.")
        return {}
    df = compute_bounds(df, consts=consts)

    sweep_axis = df["sweep_axis"].iloc[0]
    print(f"sweep_arch: axis = {sweep_axis}")
    arch_values = sorted(df["sweep_value"].unique(), key=str)
    print(f"Architectures / features: {arch_values}")

    metrics = list(_BASE_METRICS)
    metrics += [c for c in df.columns
                if c.startswith("plug_in_m") and c.endswith("_error_mean")]
    metrics += [c for c in df.columns
                if c.startswith("direct_m") and c.endswith("_error_mean")]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for arch, sub in df.groupby("sweep_value"):
        r = {"arch": arch, "n_seeds": len(sub)}
        for m in metrics:
            r[f"{m}.median"] = float(sub[m].median())
            r[f"{m}.q25"] = float(sub[m].quantile(0.25))
            r[f"{m}.q75"] = float(sub[m].quantile(0.75))
        rows.append(r)
    summary = pd.DataFrame(rows).sort_values("arch")
    summary.to_csv(out_dir / "table_summary.csv", index=False)
    print(f"Saved {out_dir / 'table_summary.csv'}")

    # ---- Bar plot: ScoreMSE_H + classifier errors per architecture ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    _bar_with_iqr(axes[0], df, "sweep_value", "score_mse_H",
                  arch_values, label="score_mse_H")
    if "score_mse_full" in df.columns:
        _bar_with_iqr(axes[0], df, "sweep_value", "score_mse_full",
                      arch_values, offset=0.35,
                      color="C1", label="score_mse_full")
    axes[0].set_xticks(np.arange(len(arch_values)))
    axes[0].set_xticklabels([str(a) for a in arch_values], rotation=20)
    axes[0].set_ylabel("score MSE (log)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Score MSE per {sweep_axis}")

    classifier_metrics = [
        ("oracle_classifier_error", "oracle"),
        ("kmeans_z_euclid_error", "k-means (Euclidean on z, baseline)"),
    ]
    classifier_metrics += [
        (c, c.replace("_error_mean", ""))
        for c in df.columns
        if c.startswith("plug_in_m") and c.endswith("_error_mean")
    ]
    n_groups = len(classifier_metrics)
    width = 0.8 / max(n_groups, 1)
    cmap = plt.get_cmap("tab10")
    for i, (col, lab) in enumerate(classifier_metrics):
        if col not in df.columns:
            continue
        offset = (i - n_groups / 2) * width + width / 2
        _bar_with_iqr(axes[1], df, "sweep_value", col,
                      arch_values, offset=offset, width=width,
                      color=cmap(i % 10), label=lab)
    axes[1].set_xticks(np.arange(len(arch_values)))
    axes[1].set_xticklabels([str(a) for a in arch_values], rotation=20)
    axes[1].set_ylabel("classifier error")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_title(f"Classifier errors per {sweep_axis}")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_arch_metrics.png", dpi=150)
    plt.close(fig)

    return {"summary": summary, "df": df}


# =============================================================================
# Helper: categorical bar with IQR error bars
# =============================================================================

def _bar_with_iqr(ax, df, group_col, y_col, group_order,
                  offset=0.0, width=0.35, color=None, label=None):
    xs = []
    medians = []
    lows = []
    highs = []
    for arch in group_order:
        sub = df[df[group_col] == arch][y_col]
        if len(sub) == 0:
            xs.append(np.nan); medians.append(np.nan)
            lows.append(np.nan); highs.append(np.nan)
            continue
        med = float(sub.median())
        q25 = float(sub.quantile(0.25))
        q75 = float(sub.quantile(0.75))
        medians.append(med); lows.append(med - q25); highs.append(q75 - med)
    positions = np.arange(len(group_order)) + offset
    medians_arr = np.array(medians, dtype=float)
    err = np.array([lows, highs], dtype=float)
    ax.bar(positions, medians_arr, yerr=err,
           width=width, color=color, label=label,
           alpha=0.85, capsize=3)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    analyze_arch_sweep(args.exp_dir, args.out_dir)
