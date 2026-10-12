#!/usr/bin/env python3
"""
Driver: launch the W_max (decoder capacity) ablation -- cap_ratio x n 2D sweep.

This is a thin wrapper around `scripts/run.py` that:
  1. Loads `configs/experiments/ablation_W_capratio.yaml`.
  2. Optionally restricts seeds / cap_ratios / n values for quick iteration.
  3. Invokes the main runner.

Usage:
    # Full ablation (7 cap_ratio x 5 n x 10 seeds = 350 runs)
    python scripts/exp_ablation_W_capratio.py --num-workers 6 --device cuda

    # Smoke test (1 seed, max_epochs=10)
    python scripts/exp_ablation_W_capratio.py --smoke

    # Restrict cap_ratios (e.g. only test phase-boundary region)
    python scripts/exp_ablation_W_capratio.py --cap-ratios 0.5 1.0 2.0 \\
        --ns 1000 --seeds 0 1 2

    # Skip already-completed runs (resume after a failure)
    python scripts/exp_ablation_W_capratio.py --num-workers 6 \\
        --device cuda --skip-completed
"""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

import scripts.run as run_module
from lp_sbm.config import load_raw_config


CONFIG_PATH = ROOT / "configs" / "experiments" / "ablation_W_capratio.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W_max ablation driver: cap_ratio x n 2D sweep."
    )
    p.add_argument("--smoke", action="store_true",
                   help="Quick sanity check: 1 seed, 1 cap_ratio, n in {200, 500}, "
                        "max_epochs=10. Verifies path runs end-to-end.")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Override the seed list.")
    p.add_argument("--ns", type=int, nargs="*", default=None,
                   help="Override the n values to sweep.")
    p.add_argument("--cap-ratios", type=float, nargs="*", default=None,
                   help="Override the cap_ratio values to sweep.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override training.max_epochs.")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p.add_argument("--output-base", type=str, default=None,
                   help="Override output.base_dir (default: outputs/).")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Override experiment.name (output folder name).")
    p.add_argument("--max-runs", type=int, default=None,
                   help="Cap total runs (for debug).")
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip already-completed (sid, seed) runs from manifest.")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Parallel workers. 1/0 = sequential.")
    return p.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides to the loaded config in place."""
    cfg = copy.deepcopy(cfg)

    if args.smoke:
        cfg["seeds"]["list"] = [0]
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "n":
                ax["values"] = [200, 500]
            if ax.get("name") == "cap_ratio":
                ax["values"] = [1.0, 2.0]
        cfg["training"]["max_epochs"] = 10
        cfg["training"]["patience"] = 10
        cfg["training"]["early_stop"] = False
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = (
            cfg["experiment"].get("name", "ablation_W_capratio") + "_smoke"
        )
        cfg["experiment"]["description"] = (
            cfg["experiment"].get("description", "") + " [SMOKE]"
        )

    if args.seeds is not None:
        cfg["seeds"]["list"] = list(args.seeds)
    if args.ns is not None:
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "n":
                ax["values"] = list(args.ns)
    if args.cap_ratios is not None:
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "cap_ratio":
                ax["values"] = list(args.cap_ratios)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = int(args.max_epochs)
    if args.exp_name is not None:
        cfg.setdefault("experiment", {})["name"] = args.exp_name
    if args.device is not None:
        cfg.setdefault("runtime", {})["device"] = args.device

    return cfg


def main() -> None:
    args = parse_args()
    raw_cfg = load_raw_config(CONFIG_PATH)
    cfg = apply_overrides(raw_cfg, args)

    # Persist customised config to a temp file in the project, then call run.py
    # with --config pointing to it (so run.py's manifest etc. all work).
    tmp_dir = ROOT / "configs" / "_runtime"
    tmp_dir.mkdir(exist_ok=True)
    tmp_cfg_path = tmp_dir / "ablation_W_capratio_active.yaml"
    with open(tmp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    # Build argv for run.main().
    sys.argv = ["run.py", "--config", str(tmp_cfg_path)]
    if args.output_base is not None:
        sys.argv += ["--output-base", args.output_base]
    if args.max_runs is not None:
        sys.argv += ["--max-runs", str(args.max_runs)]
    if args.skip_completed:
        sys.argv += ["--skip-completed"]
    if args.device is not None:
        sys.argv += ["--device", args.device]
    if args.num_workers is not None:
        sys.argv += ["--num-workers", str(args.num_workers)]

    run_module.main()


if __name__ == "__main__":
    main()
