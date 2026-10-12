#!/usr/bin/env python3
"""
Driver: launch the Delta_s sweep experiment.

Sweeps p_in (with p_out fixed) -> sweeps the score-profile separation
Delta_s.  Single architecture (GIN+PE by default), at multiple n.

Usage:
    # Full sweep
    python scripts/exp_sweep_delta.py

    # Quick smoke test (1 seed, 10 epochs, 2 p_in values)
    python scripts/exp_sweep_delta.py --smoke

    # Override p_in or n values
    python scripts/exp_sweep_delta.py --pins 0.10 0.20 0.30 --ns 1000

    # Different architecture (must be a label in the YAML)
    python scripts/exp_sweep_delta.py --archs gin_pe
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


CONFIG_PATH = ROOT / "configs" / "experiments" / "sweep_delta.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delta_s sweep experiment driver.")
    p.add_argument("--smoke", action="store_true",
                   help="Quick sanity check: 1 seed, 2 p_in values, max_epochs=10.")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Override the seed list.")
    p.add_argument("--ns", type=int, nargs="*", default=None,
                   help="Override the n axis values.")
    p.add_argument("--pins", type=float, nargs="*", default=None,
                   help="Override the p_in axis values.")
    p.add_argument("--p-out", type=float, default=None,
                   help="Override the (fixed) p_out value.")
    p.add_argument("--archs", type=str, nargs="*", default=None,
                   help="Restrict to these arch labels.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override training.max_epochs.")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p.add_argument("--output-base", type=str, default=None,
                   help="Override output.base_dir.")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Override experiment.name (folder name).")
    p.add_argument("--max-runs", type=int, default=None,
                   help="Cap total runs (for debug).")
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip already-completed (sid, seed) runs.")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Parallel workers. 1/0 = sequential.")
    return p.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)

    if args.smoke:
        cfg["seeds"]["list"] = [0]
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "p_in":
                ax["values"] = [0.10, 0.30]
            if ax.get("name") == "n":
                ax["values"] = [500]
        cfg["training"]["max_epochs"] = 10
        cfg["training"]["patience"] = 10
        cfg["training"]["early_stop"] = False
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = (
            cfg["experiment"].get("name", "sweep_delta") + "_smoke"
        )

    if args.seeds is not None:
        cfg["seeds"]["list"] = list(args.seeds)
    if args.ns is not None:
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "n":
                ax["values"] = list(args.ns)
    if args.pins is not None:
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "p_in":
                ax["values"] = list(args.pins)
    if args.p_out is not None:
        cfg["sbm"]["block_spec"]["p_out"] = float(args.p_out)
    if args.archs is not None:
        wanted = set(args.archs)
        for ax in cfg["sweep"]["axes"]:
            if ax.get("name") == "arch":
                ax["values"] = [v for v in ax["values"]
                                if v.get("label") in wanted]
                if not ax["values"]:
                    raise ValueError(
                        f"--archs {args.archs} matched no labels in config."
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

    tmp_dir = ROOT / "configs" / "_runtime"
    tmp_dir.mkdir(exist_ok=True)
    tmp_cfg_path = tmp_dir / "sweep_delta_active.yaml"
    with open(tmp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

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