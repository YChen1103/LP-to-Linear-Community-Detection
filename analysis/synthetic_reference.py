#!/usr/bin/env python3
"""
Reproduce the synthetic reference numbers used alongside the real-graph
experiment. Reads only `summary_sweep_n.json` -- no reruns.

Sections
--------
 A. Architecture separability. Per-arch median [IQR] of held-out BCE and of the
    few-label probe error. This is the evidence that a cross-architecture
    design does NOT work on a fixed graph: the three PE architectures overlap
    inside the seed-to-seed IQR, so only 2 of 5 points are actually resolved.

 B. Where the association lives. Spearman(held-out BCE, probe error)
      - pooled over every run in the sweep
      - within a setting, i.e. seed noise only
      - at fixed (K, n), across architectures x seeds
    The middle one is ~0, which is why the trajectory design is used instead.

 C. Rank agreement (Kendall tau) between the ordering induced by held-out LP
    loss and by probe error, per condition; and transfer of the architecture
    ordering between conditions.

 D. Elasticity: relative dynamic range of the probe error divided by that of
    held-out BCE. This is the headline calibration number for the real-graph
    comparison.

Usage
-----
    python analysis/synthetic_reference.py --summary path/to/summary_sweep_n.json
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
from scipy.stats import kendalltau, spearmanr

ARCHS = ["gin_pe", "gcn_pe", "gat_pe", "gin_random", "cp_random"]


def load(summary: str):
    d = json.load(open(summary))
    per_run, per_setting = [], {}
    for s in d["settings"]:
        lab = s["sweep_label"]
        try:
            K, n, arch = lab.replace("K=", "").replace("n=", "").split("__")
            K, n = int(K), int(n)
        except ValueError:
            continue
        per_setting[(K, n, arch)] = s
        for sd in s["seeds"]:
            m = sd["metrics"]
            if "R_H_emp" in m:
                per_run.append(dict(K=K, n=n, arch=arch, label=lab, **m))
    return per_run, per_setting


def med_iqr(s, key):
    v = np.array([sd["metrics"][key] for sd in s["seeds"] if key in sd["metrics"]])
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)))


def fold(x, floor: float = 1e-9) -> float:
    """max/min fold-change, with min floored.

    For error rates the floor is the resolution 1/n: an error cannot be
    resolved below one node, so without the floor the fold-change of a column
    containing an exact zero is meaningless.
    """
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 2:
        return float("nan")
    return float(x.max() / max(x.min(), floor))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--probe", default="direct_m10_error_mean")
    args = ap.parse_args()
    runs, setting = load(args.summary)
    print(f"loaded {len(runs)} runs across {len(setting)} settings\n")

    # ---------------------------------------------------------------- A
    print("=" * 78)
    print("A. ARCHITECTURE SEPARABILITY  (median [IQR] over seeds)")
    print("=" * 78)
    for K, n in [(3, 1000), (3, 2000)]:
        print(f"\n  K={K}, n={n}")
        print(f"    {'arch':<12}{'held-out BCE':>28}{'probe err @ m=10':>30}")
        for a in ARCHS:
            s = setting.get((K, n, a))
            if s is None:
                continue
            b = med_iqr(s, "R_H_emp")
            e = med_iqr(s, args.probe)
            print(f"    {a:<12}{b[0]:>10.4f} [{b[1]:.4f}, {b[2]:.4f}]"
                  f"{e[0]:>12.4f} [{e[1]:.4f}, {e[2]:.4f}]")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 78)
    print("B. WHERE THE ASSOCIATION LIVES  (Spearman rho)")
    print("=" * 78)
    sub_all = [r for r in runs if args.probe in r and "excess_emp_H" in r]
    xr = np.array([r["R_H_emp"] for r in sub_all])
    xe = np.array([r["excess_emp_H"] for r in sub_all])
    y = np.array([r[args.probe] for r in sub_all])
    print(f"  pooled, x = held-out EXCESS risk ({len(xe)} runs)   "
          f"rho = {spearmanr(xe, y).statistic:+.3f}")
    print(f"  pooled, x = raw held-out BCE                     "
          f"rho = {spearmanr(xr, y).statistic:+.3f}   <-- meaningless: the")
    print("      irreducible entropy term differs across K and n, so raw BCE is only")
    print("      comparable WITHIN one graph. On a single real graph it is fixed, so")
    print("      raw held-out BCE is the right x-axis there.")

    within = []
    for lab in {r["label"] for r in runs}:
        sub = [r for r in runs if r["label"] == lab and args.probe in r]
        if len(sub) >= 8:
            xs = np.array([r["R_H_emp"] for r in sub])
            ys = np.array([r[args.probe] for r in sub])
            if len(np.unique(xs)) > 3 and len(np.unique(ys)) > 3:
                rr = spearmanr(xs, ys).statistic
                if np.isfinite(rr):
                    within.append(rr)
    print(f"  within a setting, seed noise only (median of {len(within)})   "
          f"rho = {np.median(within):+.3f}   <-- essentially zero")

    print("\n  at fixed (K, n), across architectures x seeds:")
    for K in (2, 3, 4):
        for n in (500, 1000, 2000):
            sub = [r for r in runs if r["K"] == K and r["n"] == n and args.probe in r]
            if len(sub) < 20:
                continue
            xs = np.array([r["R_H_emp"] for r in sub])
            ys = np.array([r[args.probe] for r in sub])
            print(f"    K={K}, n={n:<5} ({len(sub)} pts)   "
                  f"rho = {spearmanr(xs, ys).statistic:+.3f}")

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 78)
    print("C. RANK AGREEMENT  (Kendall tau over the 5 architectures)")
    print("=" * 78)
    conds = [(K, n) for K in (2, 3, 4) for n in (1000, 2000)]
    order: Dict = {}
    print("\n  internal: held-out BCE ordering vs probe-error ordering")
    for K, n in conds:
        b = [med_iqr(setting[(K, n, a)], "R_H_emp")[0] for a in ARCHS
             if (K, n, a) in setting]
        e = [med_iqr(setting[(K, n, a)], args.probe)[0] for a in ARCHS
             if (K, n, a) in setting]
        if len(b) < 3:
            continue
        order[(K, n)] = e
        print(f"    K={K}, n={n:<5}  tau = {kendalltau(b, e).statistic:+.2f}")
    print("\n  transfer: probe-error architecture ordering between conditions")
    ks = sorted(order)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            t = kendalltau(order[ks[i]], order[ks[j]]).statistic
            print(f"    K={ks[i][0]},n={ks[i][1]} -> K={ks[j][0]},n={ks[j][1]}"
                  f"   tau = {t:+.2f}")

    # ---------------------------------------------------------------- D
    print("\n" + "=" * 78)
    print("D. ELASTICITY  (relative dynamic range across the 5 architectures)")
    print("=" * 78)
    print(f"    {'condition':<14}{'BCE min':>10}{'BCE max':>10}{'fold':>8}"
          f"{'probe min':>12}{'probe max':>11}{'fold':>9}{'ratio':>9}")
    for K, n in conds:
        b = [med_iqr(setting[(K, n, a)], "R_H_emp")[0] for a in ARCHS
             if (K, n, a) in setting]
        e = [med_iqr(setting[(K, n, a)], args.probe)[0] for a in ARCHS
             if (K, n, a) in setting]
        if len(b) < 3:
            continue
        fb, fe = fold(b), fold(e, floor=1.0 / n)
        print(f"    K={K}, n={n:<7}{min(b):>10.4f}{max(b):>10.4f}{fb:>8.3f}"
              f"{max(min(e), 1.0 / n):>12.4f}{max(e):>11.4f}{fe:>9.1f}"
              f"{fe / fb:>8.0f}x")
    print("\n  Read as: a few-percent change in held-out LP loss corresponds to a\n"
          "  several-hundred-percent change in few-label probe error. The\n"
          "  association is monotone but very steep, so held-out LP loss is a\n"
          "  usable qualitative signal and a poor quantitative predictor.")


if __name__ == "__main__":
    main()
