#!/usr/bin/env python3
"""
Single entry point for post-hoc analysis.

Usage:
    python scripts/analyze.py --kind n_sweep --exp-dir outputs/main_sweep_n
    python scripts/analyze.py --kind norm_ablation --exp-dir outputs/ablation_W_capratio
    python scripts/analyze.py --kind sbm_sweep --exp-dir outputs/ablation_rho_sparsity
    python scripts/analyze.py --kind arch_sweep --exp-dir outputs/main_sweep_n

The `--kind` flag dispatches to the appropriate analyze_*() function.

For a quick auto-detection (try to infer kind from sweep axis), use --kind auto.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import (
    analyze_norm_ablation, analyze_n_sweep,
    analyze_sbm_sweep, analyze_arch_sweep,
)
from analysis.bounds import BoundConstants
from analysis.load import load_final


_KIND_DISPATCH = {
    "norm_ablation": analyze_norm_ablation,
    "n_sweep":       analyze_n_sweep,
    "sbm_sweep":     analyze_sbm_sweep,
    "arch_sweep":    analyze_arch_sweep,
}


def _infer_kind(exp_dir: Path) -> str:
    """Heuristic: look at the sweep axis to guess which analyzer to use."""
    df = load_final(exp_dir)
    if df.empty:
        raise RuntimeError(f"No final-eval rows in {exp_dir}; cannot infer kind.")
    axis = df["sweep_axis"].iloc[0] or ""
    if axis.endswith(".W_max_spec.value") or "_spec" in axis or "R_spec" in axis or "B_0_spec" in axis:
        return "norm_ablation"
    if axis == "sbm.n":
        return "n_sweep"
    if axis.startswith("sbm."):
        return "sbm_sweep"
    if axis.startswith("encoder.family") or axis.startswith("encoder.features"):
        return "arch_sweep"
    raise RuntimeError(f"Cannot infer kind from sweep axis: {axis!r}")


def main():
    parser = argparse.ArgumentParser(description="Run post-hoc analysis.")
    parser.add_argument("--kind", choices=list(_KIND_DISPATCH) + ["auto"],
                        default="auto",
                        help="Type of analysis (or 'auto' to infer).")
    parser.add_argument("--exp-dir", required=True,
                        help="Path to outputs/<experiment_name>/")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write figures/tables (default: <exp-dir>/analysis/)")
    parser.add_argument("--C-stat", type=float, default=1.0,
                        help="Universal constant C_stat for Gamma_H envelope.")
    parser.add_argument("--C-1", type=float, default=1.0,
                        help="Universal constant C_1 for delta_mask_n.")
    parser.add_argument("--drop-n-below", type=float, default=None,
                        help="(n_sweep only) Drop n < this from log-log fits.")
    parser.add_argument("--secondary-axis", default=None,
                        help="(sbm_sweep only) Secondary axis for derived plots.")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    kind = args.kind
    if kind == "auto":
        kind = _infer_kind(exp_dir)
        print(f"[auto] inferred kind = {kind!r}")

    consts = BoundConstants(C_stat=args.C_stat, C_1=args.C_1)

    fn = _KIND_DISPATCH[kind]
    if kind == "n_sweep":
        fn(exp_dir, args.out_dir, consts=consts, drop_n_below=args.drop_n_below)
    elif kind == "sbm_sweep":
        fn(exp_dir, args.out_dir, consts=consts,
           secondary_axis=args.secondary_axis)
    else:
        fn(exp_dir, args.out_dir, consts=consts)


if __name__ == "__main__":
    main()
