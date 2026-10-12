#!/usr/bin/env python3
"""
Control: UNTRAINED GIN+PE at initialization.

Complements the raw-PE baselines in scripts/pe_baselines.py by separating the
effect of the LP objective from the GNN's random nonlinear mixing: it evaluates
the *same* architecture, on the *same* SBM realizations, with the *same*
spectral PE input and the *same* labeled sets, at initialization -- i.e. after
zero gradient steps.

Implementation note: this is not a reimplementation. It loads
configs/experiments/main_sweep_n.yaml, restricts the grid to (K=3, gin_pe, the
requested n values), sets training.max_epochs = 0, and calls scripts/run.py.
With max_epochs = 0 the epoch loop is skipped, and since the main sweep uses
early_stop = false the returned state is the initialization itself
(lp_sbm/trainer.py: `final_epoch = len(train_curve)`), which then flows through
the ordinary compute_all_metrics path. Every reported quantity -- oracle error,
plug-in error, direct LogReg, k-means, ScoreMSE -- is therefore produced by
exactly the same code as the trained runs.

Usage
-----
    python scripts/untrained_control.py --ns 500 1000 2000 \
        --seeds 0 1 2 3 4 5 6 7 8 9 --device cuda

    # then, as usual:
    python scripts/extract_summary.py outputs_untrained/untrained_control \
        --out summary_untrained.json

    # and finally the side-by-side against the trained runs:
    python scripts/untrained_control.py --compare \
        --untrained-summary summary_untrained.json \
        --trained-summary  summary_sweep_n.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run as run_module                      # noqa: E402
from lp_sbm.config import load_raw_config             # noqa: E402

CONFIG_PATH = ROOT / "configs" / "experiments" / "main_sweep_n.yaml"

# Metrics worth reporting for an untrained model: the theorem chain plus the
# downstream classifiers.
REPORT_KEYS = [
    "excess_pop_H",
    "score_mse_full",
    "oracle_classifier_error",
    "plug_in_m1_error_mean",
    "plug_in_m10_error_mean",
    "direct_m1_error_mean",
    "direct_m10_error_mean",
    "kmeans_z_euclid_error",
]


def build_config(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(load_raw_config(CONFIG_PATH))

    cfg["seeds"]["list"] = list(args.seeds)
    for ax in cfg["sweep"]["axes"]:
        if ax.get("name") == "K":
            ax["values"] = [args.K]
        elif ax.get("name") == "n":
            ax["values"] = list(args.ns)
        elif ax.get("name") == "arch":
            ax["values"] = [v for v in ax["values"] if v.get("label") == args.arch]
            if not ax["values"]:
                raise ValueError(f"--arch {args.arch!r} matched no label in the config")
            # An arch entry may carry its own training.max_epochs override
            # (cp_random does); strip it so the zero-epoch setting is not undone.
            for v in ax["values"]:
                v.get("overrides", {}).pop("training.max_epochs", None)

    cfg["training"]["max_epochs"] = 0
    cfg["training"]["early_stop"] = False
    cfg["training"]["restarts"] = 1
    cfg.setdefault("experiment", {})["name"] = args.exp_name
    cfg["experiment"]["description"] = (
        "Untrained control: identical pipeline, zero gradient steps. "
        "Isolates the contribution of the LP objective from the architecture "
        "and from the spectral PE."
    )
    if args.device is not None:
        cfg.setdefault("runtime", {})["device"] = args.device
    if args.num_workers is not None:
        cfg.setdefault("runtime", {})["num_workers"] = int(args.num_workers)
    return cfg


def launch(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    tmp_dir = ROOT / "configs" / "_runtime"
    tmp_dir.mkdir(exist_ok=True)
    tmp_cfg_path = tmp_dir / "untrained_control_active.yaml"
    with open(tmp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    sys.argv = ["run.py", "--config", str(tmp_cfg_path),
                "--output-base", args.output_base]
    print(f"[untrained control] K={args.K} arch={args.arch} "
          f"n={args.ns} seeds={args.seeds} max_epochs=0")
    print(f"[untrained control] config -> {tmp_cfg_path}")
    run_module.main()
    print(f"\nNext:\n  python scripts/extract_summary.py "
          f"{args.output_base}/{args.exp_name} --out summary_untrained.json")


# =============================================================================
# Side-by-side reporting
# =============================================================================

def _medians(summary_path: str, K: int, arch: str) -> Dict[int, Dict[str, float]]:
    d = json.load(open(summary_path))
    out: Dict[int, Dict[str, float]] = {}
    for s in d["settings"]:
        label = s["sweep_label"]
        if f"__{arch}" not in label or f"K={K}__" not in label:
            continue
        n = int(label.split("__n=")[1].split("__")[0])
        cell = {}
        for k in REPORT_KEYS:
            v = [sd["metrics"][k] for sd in s["seeds"] if k in sd["metrics"]]
            cell[k] = float(np.median(v)) if v else float("nan")
        out[n] = cell
    return out


def compare(args: argparse.Namespace) -> None:
    un = _medians(args.untrained_summary, args.K, args.arch)
    tr = _medians(args.trained_summary, args.K, args.arch)
    ns = sorted(set(un) & set(tr))
    if not ns:
        raise SystemExit("no overlapping n between the two summaries")

    width = max(len(k) for k in REPORT_KEYS) + 2
    header = "metric".ljust(width) + "".join(f"{'n=' + str(n):>22}" for n in ns)
    print(header)
    print("-" * len(header))
    print("".ljust(width) + "".join(f"{'untrained':>11}{'trained':>11}" for _ in ns))
    for k in REPORT_KEYS:
        row = k.ljust(width)
        for n in ns:
            row += f"{un[n][k]:>11.4f}{tr[n][k]:>11.4f}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--arch", type=str, default="gin_pe")
    ap.add_argument("--ns", type=int, nargs="*", default=[500, 1000, 2000])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--output-base", type=str, default="outputs_untrained")
    ap.add_argument("--exp-name", type=str, default="untrained_control")
    ap.add_argument("--compare", action="store_true",
                    help="skip the run; print untrained-vs-trained medians")
    ap.add_argument("--untrained-summary", type=str, default=None)
    ap.add_argument("--trained-summary", type=str, default=None)
    args = ap.parse_args()

    if args.compare:
        if not (args.untrained_summary and args.trained_summary):
            ap.error("--compare requires --untrained-summary and --trained-summary")
        compare(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
