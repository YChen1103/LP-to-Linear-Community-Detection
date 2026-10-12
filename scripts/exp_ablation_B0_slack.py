#!/usr/bin/env python3
"""
Driver: launch the B_0 (bias cap) ablation -- B_0 x n 2D sweep.

This is a thin wrapper around `scripts/run.py` that:
  1. Loads `configs/experiments/ablation_B0_slack.yaml`.
  2. Optionally restricts seeds / B_0 / n values for quick iteration.
  3. Invokes the main runner.

Usage:
    # Full ablation (6 B_0 x 5 n x 10 seeds = 300 runs)
    python scripts/exp_ablation_B0_slack.py --num-workers 6 --device cuda

    # Smoke test (1 seed, max_epochs=10)
    python scripts/exp_ablation_B0_slack.py --smoke

    # Restrict B_0 (e.g. only test bias-binding region)
    python scripts/exp_ablation_B0_slack.py --b0s 2.0 2.6 3.4 \\
        --ns 1000 --seeds 0 1 2

    # Skip already-completed runs (resume after a failure)
    python scripts/exp_ablation_B0_slack.py --num-workers 6 \\
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


CONFIG_PATH = ROOT / "configs" / "experiments" / "ablation_B0_slack.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B_0 ablation driver: B_0 x n 2D sweep."
    )
    p.add_argument("--smoke", action="store_true",
                   help="Quick sanity check: 1 seed, B_0 in {2.0, 3.4}, "
                        "n in {200, 500}, max_epochs=10. Verifies path runs "
                        "end-to-end.")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Override the seed list.")
    p.add_argument("--ns", type=int, nargs="*", default=None,
                   help="Override the n values to sweep.")
    p.add_argument("--b0s", type=float, nargs="*", default=None,
                   help="Override the B_0 values to sweep.")
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
            if ax.get("name") == "B_0":
                ax["values"] = [2.0, 3.4]
        cfg["training"]["max_epochs"] = 10
        cfg["training"]["patience"] = 10
        cfg["training"]["early_stop"] = False
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = (
            cfg["experiment"].get("name", "ablation_B0_slack") + "_smoke"
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
    if args.b0s is not None:
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "B_0":
                ax["values"] = list(args.b0s)
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
    tmp_cfg_path = tmp_dir / "ablation_B0_slack_active.yaml"
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
