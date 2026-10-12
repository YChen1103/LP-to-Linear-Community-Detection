#!/usr/bin/env python3
"""
Summarize real-graph trajectories as tables and association statistics.

Everything here is text/tables only -- no figures.

Produces
--------
 1. Checkpoint table: one row per selected epoch, median over seeds, with both
    LP-side and downstream-side columns.

 2. Association: Spearman rho between LP quality and few-label probe error,
    pooled over all trajectory points.

 3. Held-out vs train: the same rho computed against the TRAIN-side loss. The
    theorem's antecedent is a *held-out* quantity (the half-empirical
    corollary), with the train-side version as a separate statement (the
    train-side corollary). If the run overfits, these two
    come apart, and which one keeps tracking the probe error is a real test
    rather than a restatement of "training helps".

 4. Elasticity: relative dynamic range of the probe error divided by that of
    the LP-side quantity. On the synthetic sweep (K=3) this ratio is ~6x at
    n=1000 and ~90x at n=2000 (reference values printed below), i.e. the
    association is monotone but extremely steep. Reporting the same ratio on
    the real graph says precisely *how* predictive the synthetic validation is.

Usage
-----
    python real/analyze_trajectory.py --runs outputs/real_trajectory \\
        --arch gin_pe --probe direct_m10_error_mean --out results/real_traj
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr


def load_runs(root: Path, arch: str | None) -> List[List[Dict]]:
    runs = []
    for p in sorted(root.rglob("trajectory.jsonl")):
        rows = [json.loads(ln) for ln in open(p) if ln.strip()]
        rows = [r for r in rows if not r.get("is_final", False)]
        if not rows:
            continue
        if arch and rows[0].get("arch") != arch:
            continue
        runs.append(sorted(rows, key=lambda r: r["epoch"]))
    return runs


def pick_epochs(runs: List[List[Dict]], k: int = 7) -> List[int]:
    """A geometric-ish subset of the epochs present in every run."""
    common = set(r["epoch"] for r in runs[0])
    for run in runs[1:]:
        common &= set(r["epoch"] for r in run)
    eps = np.array(sorted(common))
    if len(eps) <= k:
        return eps.tolist()
    targets = np.geomspace(eps.min(), eps.max(), k)
    out = sorted({int(eps[np.argmin(np.abs(eps - t))]) for t in targets})
    return out


def median_at(runs: List[List[Dict]], epoch: int, key: str) -> float:
    v = [r[key] for run in runs for r in run
         if r["epoch"] == epoch and key in r and r[key] == r[key]]
    return float(np.median(v)) if v else float("nan")


def fold(x: np.ndarray, floor: float = 1e-9) -> float:
    """max/min fold-change with the minimum floored.

    Same definition as analysis/synthetic_reference.py, so the two numbers are
    directly comparable. For error rates the floor is 1/n: an error rate cannot
    be resolved below one node.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(x.max() / max(x.min(), floor))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="outputs/real_trajectory")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--probe", default="direct_m10_error_mean")
    ap.add_argument("--lp-key", default="R_H_emp",
                    help="R_H_emp (held-out BCE) or auc_H")
    ap.add_argument("--n-rows", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = load_runs(Path(args.runs), args.arch)
    if not runs:
        raise SystemExit(f"no trajectories found under {args.runs}")
    meta = runs[0][0]
    n_pts = sum(len(r) for r in runs)
    print(f"{len(runs)} run(s), {n_pts} trajectory points, "
          f"graph={meta.get('source')} n={meta.get('n')} K={meta.get('K')} "
          f"mean_degree={meta.get('mean_degree'):.2f} arch={meta.get('arch')}")

    # ---------------------------------------------------------------- table
    eps = pick_epochs(runs, args.n_rows)
    cols = [("train BCE", "R_T_emp", 4), ("held-out BCE", "R_H_emp", 4),
            ("held-out AUC", "auc_H", 3),
            ("err m=1", "direct_m1_error_mean", 3),
            ("err m=10", "direct_m10_error_mean", 3),
            ("plug-in m=10", "plug_in_m10_error_mean", 3)]
    lines = [f"Real graph: {meta.get('source')}, n={meta.get('n')}, K={meta.get('K')}, "
             f"mean degree {meta.get('mean_degree'):.1f}, tau={meta.get('tau')}. "
             f"Median over {len(runs)} seeds. Error is over all n nodes.", "",
             "| epoch | " + " | ".join(c[0] for c in cols) + " |",
             "|" + "---|" * (1 + len(cols))]
    for e in eps:
        cells = [f"{median_at(runs, e, k):.{d}f}" for _, k, d in cols]
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    # ---------------------------------------------------- association / test
    def pooled(key):
        return np.array([r.get(key, np.nan) for run in runs for r in run], float)

    y = pooled(args.probe)
    lp_h, lp_t, auc = pooled("R_H_emp"), pooled("R_T_emp"), pooled("auc_H")
    ok = np.isfinite(y) & np.isfinite(lp_h) & np.isfinite(lp_t)

    print("\nAssociation over all trajectory points "
          f"(probe = {args.probe}, n = {int(ok.sum())}):")
    r_h = spearmanr(lp_h[ok], y[ok]).statistic
    r_t = spearmanr(lp_t[ok], y[ok]).statistic
    print(f"  Spearman(held-out BCE, probe error) = {r_h:+.3f}")
    print(f"  Spearman(train    BCE, probe error) = {r_t:+.3f}")
    if np.isfinite(auc).sum() > 3:
        ok2 = ok & np.isfinite(auc)
        print(f"  Spearman(held-out AUC, probe error) = "
              f"{spearmanr(auc[ok2], y[ok2]).statistic:+.3f}   (expected negative)")

    # overfitting segment: after the held-out loss reaches its minimum
    seg_h, seg_t, seg_y = [], [], []
    for run in runs:
        h = np.array([r.get("R_H_emp", np.nan) for r in run], float)
        if not np.isfinite(h).any():
            continue
        k = int(np.nanargmin(h))
        if k < len(run) - 3:
            seg_h += [r.get("R_H_emp", np.nan) for r in run[k:]]
            seg_t += [r.get("R_T_emp", np.nan) for r in run[k:]]
            seg_y += [r.get(args.probe, np.nan) for r in run[k:]]
    seg_h, seg_t, seg_y = map(lambda v: np.array(v, float), (seg_h, seg_t, seg_y))
    m = np.isfinite(seg_h) & np.isfinite(seg_t) & np.isfinite(seg_y)
    print("\nPost-minimum segment (held-out loss rising, train loss still falling):")
    if m.sum() >= 8:
        print(f"  {int(m.sum())} points")
        print(f"  Spearman(held-out BCE, probe error) = "
              f"{spearmanr(seg_h[m], seg_y[m]).statistic:+.3f}")
        print(f"  Spearman(train    BCE, probe error) = "
              f"{spearmanr(seg_t[m], seg_y[m]).statistic:+.3f}")
        print("  -> if the held-out column keeps tracking the probe error and the "
              "train column does not,\n     the association is not merely "
              "'more training helps'.")
    else:
        print("  not enough points past the held-out minimum; "
              "the runs did not overfit within the budget.")

    # ------------------------------------------------------------ elasticity
    n_nodes = int(meta.get("n", 1) or 1)
    f_bce = fold(lp_h)
    f_probe = fold(y, floor=1.0 / n_nodes)
    print("\nDynamic range over trajectory points (max/min fold-change):")
    print(f"  held-out BCE                {f_bce:.3f}x")
    print(f"  {args.probe:<26}{f_probe:.1f}x")
    e = f_probe / f_bce if np.isfinite(f_bce) and f_bce > 0 else float("nan")
    print(f"  elasticity  probe / held-out BCE : {e:.0f}x")
    print("  synthetic reference (analysis/synthetic_reference.py, K=3):")
    print("      n=1000  BCE 1.038x  probe 5.8x   ->   6x")
    print("      n=2000  BCE 1.026x  probe 90.3x  ->  88x")

    if args.out:
        d = Path(args.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "table.md").write_text(table + "\n")
        json.dump({"n_runs": len(runs), "n_points": int(ok.sum()),
                   "spearman_heldout": float(r_h), "spearman_train": float(r_t),
                   "fold_heldout_bce": f_bce, "fold_probe": f_probe,
                   "elasticity": float(e),
                   "epochs": eps, "graph": {k: meta.get(k) for k in
                                            ("source", "n", "K", "mean_degree", "tau")}},
                  open(d / "summary.json", "w"), indent=1)
        print(f"\nwritten to {d}/")


if __name__ == "__main__":
    main()
