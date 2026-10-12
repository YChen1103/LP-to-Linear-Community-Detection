#!/usr/bin/env python3
"""
Driver: launch the main sweep-n experiment.

This is a thin wrapper around `scripts/run.py` that:
  1. Loads `configs/experiments/main_sweep_n.yaml`.
  2. Optionally restricts seeds / archs / n values for quick iteration.
  3. Invokes the main runner.

Usage:
    # Full sweep
    python scripts/exp_main_sweep_n.py

    # Quick smoke test (1 seed, n in [200, 500])
    python scripts/exp_main_sweep_n.py --smoke

    # Override n values
    python scripts/exp_main_sweep_n.py --ns 500 1000 --seeds 0 1 2

    # Override archs (matches `label:` field)
    python scripts/exp_main_sweep_n.py --archs gin_pe gcn_pe

    # Use a custom output dir (so different sweeps don't clobber each other)
    python scripts/exp_main_sweep_n.py --output-base outputs_smoke
"""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

# Re-use run.py's main pipeline -- we don't reimplement the loop.
import scripts.run as run_module
from lp_sbm.config import load_raw_config


CONFIG_PATH = ROOT / "configs" / "experiments" / "main_sweep_n.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Main sweep-n experiment driver.")
    p.add_argument("--smoke", action="store_true",
                   help="Quick sanity check: 1 seed, only n in {200, 500}, "
                        "max_epochs=10. Verifies the path runs end-to-end.")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Override the seed list.")
    p.add_argument("--ns", type=int, nargs="*", default=None,
                   help="Override the n values to sweep.")
    p.add_argument("--archs", type=str, nargs="*", default=None,
                   help="Restrict to these arch labels (subset of "
                        "{gcn_pe, gin_pe, cp_random, gin_random, gat_pe}).")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override training.max_epochs.")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p.add_argument("--output-base", type=str, default=None,
                   help="Override output.base_dir (default: outputs/).")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Override experiment.name (folder name).")
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

    # Smoke shortcut.
    if args.smoke:
        cfg["seeds"]["list"] = [0]
        # Find the n axis and shrink its values.
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "n":
                ax["values"] = [200, 500]
        cfg["training"]["max_epochs"] = 10
        cfg["training"]["patience"] = 10
        cfg["training"]["early_stop"] = False
        if "experiment" not in cfg:
            cfg["experiment"] = {}
        cfg["experiment"]["name"] = (cfg["experiment"].get("name", "main_sweep_n")
                                      + "_smoke")
        cfg["experiment"]["description"] = (cfg["experiment"].get("description", "")
                                             + " [SMOKE]")

    # Explicit overrides take precedence.
    if args.seeds is not None:
        cfg["seeds"]["list"] = list(args.seeds)
    if args.ns is not None:
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "n":
                ax["values"] = list(args.ns)
    if args.archs is not None:
        wanted = set(args.archs)
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "arch":
                ax["values"] = [v for v in ax["values"]
                                if v.get("label") in wanted]
                if not ax["values"]:
                    raise ValueError(
                        f"--archs {args.archs} matched no labels in config; "
                        f"available: gcn_pe gin_pe cp_random gin_random gat_pe"
                    )
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

    # Persist the customised config to a temp file inside the project, then call
    # run.py with --config pointing to it (so run.py's manifest etc. all work).
    tmp_dir = ROOT / "configs" / "_runtime"
    tmp_dir.mkdir(exist_ok=True)
    tmp_cfg_path = tmp_dir / "main_sweep_n_active.yaml"
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
