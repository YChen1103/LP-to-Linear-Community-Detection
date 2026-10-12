"""
Log-log fits with bootstrap confidence intervals.

Goal: given a DataFrame with one (sweep_value, seed, metric) row per run,
fit a power-law metric ~ sweep_value^slope and return slope +/- CI.

Key design choice: bootstrap is over seeds (entire seeds resampled), not
over points (individual rows). This honors the dependency structure
between metrics computed on the same seed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    slope: float
    intercept: float
    slope_ci_low: float
    slope_ci_high: float
    n_seeds: int
    n_points: int

    def __repr__(self) -> str:
        return (f"FitResult(slope={self.slope:.3f} "
                f"[{self.slope_ci_low:.3f}, {self.slope_ci_high:.3f}], "
                f"intercept={self.intercept:.3f}, "
                f"n_seeds={self.n_seeds}, n_points={self.n_points})")


# =============================================================================
# Single fit
# =============================================================================

def loglog_linfit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Plain log-log linear regression: log y = a + slope * log x.

    Returns (slope, intercept) where intercept = a (so y = exp(a) * x^slope).
    Drops non-positive points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return float("nan"), float("nan")
    lx = np.log(x)
    ly = np.log(y)
    # Standard least squares
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    slope, intercept = float(sol[0]), float(sol[1])
    return slope, intercept


# =============================================================================
# Bootstrap-over-seeds fit
# =============================================================================

def loglog_fit_with_bootstrap(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    seed_col: str = "seed",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    rng_seed: int = 0,
    drop_x_below: Optional[float] = None,
) -> FitResult:
    """Bootstrap over seeds for slope CI.

    Procedure:
      1. Find unique seed values.
      2. For each bootstrap iter, sample seeds with replacement; collect all
         (x, y) pairs from those seeds; do a log-log fit.
      3. Slope CI = quantiles of bootstrap-slope distribution.

    Parameters
    ----------
    drop_x_below : if set, drop rows with x_col < this value before fitting
                   (useful to skip pre-asymptotic small-n points).
    """
    sub = df[[x_col, y_col, seed_col]].dropna()
    if drop_x_below is not None:
        sub = sub[sub[x_col].astype(float) >= drop_x_below]
    sub = sub[(sub[x_col].astype(float) > 0) & (sub[y_col].astype(float) > 0)]
    if len(sub) < 2:
        return FitResult(float("nan"), float("nan"), float("nan"), float("nan"),
                         0, 0)

    seeds_unique = sub[seed_col].unique()

    # Point estimate using all data
    slope_pt, intercept_pt = loglog_linfit(sub[x_col].values, sub[y_col].values)

    # Bootstrap
    rng = np.random.default_rng(rng_seed)
    by_seed = {s: sub[sub[seed_col] == s] for s in seeds_unique}
    slopes = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(seeds_unique, size=len(seeds_unique), replace=True)
        chunks = [by_seed[s] for s in sampled]
        cat = pd.concat(chunks, ignore_index=True)
        s, _ = loglog_linfit(cat[x_col].values, cat[y_col].values)
        if np.isfinite(s):
            slopes.append(s)
    if len(slopes) < 2:
        return FitResult(slope_pt, intercept_pt, float("nan"), float("nan"),
                         len(seeds_unique), len(sub))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(slopes, alpha))
    hi = float(np.quantile(slopes, 1.0 - alpha))
    return FitResult(slope_pt, intercept_pt, lo, hi,
                     len(seeds_unique), len(sub))


# =============================================================================
# Convenience: fit + collect over multiple metrics
# =============================================================================

def fit_many_metrics(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_cols: list,
    seed_col: str = "seed",
    n_bootstrap: int = 1000,
    drop_x_below: Optional[float] = None,
) -> pd.DataFrame:
    """Run loglog_fit_with_bootstrap for each metric in y_cols.

    Returns a DataFrame with one row per metric, columns:
        metric, slope, slope_ci_low, slope_ci_high, intercept, n_seeds, n_points
    """
    rows = []
    for y_col in y_cols:
        if y_col not in df.columns:
            rows.append({"metric": y_col, "slope": float("nan"),
                         "slope_ci_low": float("nan"),
                         "slope_ci_high": float("nan"),
                         "intercept": float("nan"),
                         "n_seeds": 0, "n_points": 0,
                         "note": "column missing"})
            continue
        fit = loglog_fit_with_bootstrap(
            df, x_col=x_col, y_col=y_col, seed_col=seed_col,
            n_bootstrap=n_bootstrap, drop_x_below=drop_x_below,
        )
        rows.append({
            "metric":         y_col,
            "slope":          fit.slope,
            "slope_ci_low":   fit.slope_ci_low,
            "slope_ci_high":  fit.slope_ci_high,
            "intercept":      fit.intercept,
            "n_seeds":        fit.n_seeds,
            "n_points":       fit.n_points,
        })
    return pd.DataFrame(rows)
