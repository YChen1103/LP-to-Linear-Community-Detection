#!/usr/bin/env python3
"""
Single entry point for running an experiment.

Usage:
    python scripts/run.py --config configs/experiments/main_sweep_n.yaml

This:
    1. Loads the YAML config (with extends: chain).
    2. Expands the sweep into a list of (setting_id, raw_config) pairs.
    3. For each (setting, seed) combination:
         a. Sample SBM (using sub-seeds derived from the seed).
         b. Compute SBM oracle.
         c. Resolve config (capacity-aware *_spec resolution).
         d. Train (with optional periodic eval).
         e. Compute all metrics from final state.
         f. Save raw + resolved configs, logs, final embeddings.
    4. Updates a manifest tracking all runs.
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make the package importable when running from any directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lp_sbm.config import (
    load_raw_config, expand_sweep, resolve_config,
    save_configs, derive_subseeds, get_sweep_summary, get_dotted,
)
from lp_sbm.sbm import prepare_sbm_data
from lp_sbm.oracle import compute_sbm_oracle
from lp_sbm.trainer import train_lp
from lp_sbm.metrics import compute_all_metrics
from lp_sbm.logging_io import (
    JSONLWriter, Manifest, ensure_dir, seed_dir, setting_dir, dump_final_npz,
)


def _experiment_dir(top_cfg: dict, base_dir: str) -> Path:
    """Determine the experiment output directory."""
    name = top_cfg.get("experiment", {}).get("name", "unnamed")
    return Path(base_dir) / name


def run_one(
    *,
    sid: int,
    seed: int,
    raw_cfg_for_setting: dict,
    exp_dir: Path,
    device: torch.device,
    print_progress: bool = True,
    sweep_label: Optional[str] = None,
) -> dict:
    """Run one (setting, seed). Returns a status dict."""
    out_dir = ensure_dir(seed_dir(exp_dir, sid, seed, label=sweep_label))
    sub_seeds = derive_subseeds(seed)

    # 1) Prepare data (need it before resolving capacity-relative *_spec)
    sbm_cfg = raw_cfg_for_setting["sbm"]
    mask_cfg = raw_cfg_for_setting["mask"]
    data = prepare_sbm_data(sbm_cfg, mask_cfg, sub_seeds)

    # 2) Compute SBM oracle (B_star, b_star, capacity_required, Delta_s)
    sbm_oracle = compute_sbm_oracle(data["B_n"], data["pi"])

    # 3) Resolve all *_spec dicts using oracle-derived quantities
    resolved_cfg = resolve_config(raw_cfg_for_setting, sbm_oracle)

    # 4) Save configs (both raw and resolved) up-front for traceability
    save_configs(raw_cfg_for_setting, resolved_cfg, out_dir)

    # 5) Train (with periodic eval if configured)
    eval_writer = JSONLWriter(out_dir / "eval_log.jsonl")
    train_writer = JSONLWriter(out_dir / "train_log.jsonl")  # currently unused for per-epoch

    classifiers_cfg = resolved_cfg["classifiers"]
    classifier_seed = sub_seeds["label"]

    R = float(resolved_cfg["encoder"]["R"])
    W_max = float(resolved_cfg["decoder"]["W_max"])
    B_0 = float(resolved_cfg["decoder"]["B_0"])

    # Eval callback: compute and log metrics at every periodic eval point
    def eval_cb(*, epoch, z, W, b, extra):
        m = compute_all_metrics(
            z, W, b, data=data, sbm_oracle=sbm_oracle,
            R=R, W_max=W_max, B_0=B_0,
            classifiers_cfg=classifiers_cfg,
            classifier_seed=int(classifier_seed),
        )
        m.pop("_tilde_e_sq_array", None)  # don't dump per-node into jsonl
        m["epoch"] = int(epoch)
        m["is_final"] = False
        m.update(extra)
        eval_writer.write(m)

    t0 = time.time()
    try:
        result = train_lp(
            data=data, sbm_oracle=sbm_oracle, resolved_cfg=resolved_cfg,
            seeds=sub_seeds, device=device,
            eval_callback=eval_cb,
        )
    except Exception:
        eval_writer.close()
        train_writer.close()
        traceback.print_exc()
        return {"status": "failed", "elapsed_s": time.time() - t0}

    train_time = time.time() - t0

    # 6) Final metrics
    z, W, b = result["z"], result["W"], result["b"]
    final_metrics = compute_all_metrics(
        z, W, b, data=data, sbm_oracle=sbm_oracle,
        R=R, W_max=W_max, B_0=B_0,
        classifiers_cfg=classifiers_cfg, classifier_seed=int(classifier_seed),
    )
    tilde_e_sq = final_metrics.pop("_tilde_e_sq_array")
    final_metrics["epoch"] = int(result["best_epoch"])
    final_metrics["is_final"] = True
    final_metrics["best_val_loss"] = float(result["best_val_loss"])
    final_metrics["restart_idx"] = int(result.get("restart_idx", 0))
    final_metrics["restart_vals"] = result.get("restart_vals", [])
    # Training-side observables that downstream analysis might filter on:
    for k in ["early_stop", "selection_basis", "selection_metric",
              "bias_init_mode", "bias_init_value", "feature_mode",
              "pe_normalize", "encoder_family", "decoder_family",
              "restart_selection_basis"]:
        if k in result:
            final_metrics[k] = result[k]
    eval_writer.write(final_metrics)

    # 7) Dump derived oracle quantities + train curve to convenience files
    derived = resolved_cfg["_derived"]
    summary_metrics = {
        "sid":           sid,
        "seed":          seed,
        "elapsed_s":     train_time,
        "best_epoch":    int(result["best_epoch"]),
        "best_val_loss": float(result["best_val_loss"]),
    }
    summary_metrics.update({k: v for k, v in derived.items()
                            if isinstance(v, (int, float, str, bool))})
    summary_metrics.update({k: v for k, v in final_metrics.items()
                            if isinstance(v, (int, float, bool))})

    if resolved_cfg["output"].get("save_train_curve", True):
        # Dump train/val curve as JSONL (one entry per epoch)
        with JSONLWriter(out_dir / "train_curve.jsonl") as tcw:
            for ep, (tl, vl) in enumerate(zip(result["train_curve"],
                                               result["val_curve"]), start=1):
                tcw.write({"epoch": ep, "train_loss": tl, "val_loss": vl})

    # 8) Dump final.npz
    if resolved_cfg["output"].get("save_embeddings", True):
        extras = {
            "labels":          data["labels"],
            "P_diag":          data["P"].diagonal(),  # P as defined in the paper
            "M_star":          sbm_oracle["M_star"],
            "B_n":             data["B_n"],
            "val_idx_in_H":    np.asarray(result["val_idx_in_H"]),
            "test_idx_in_H":   np.asarray(result["test_idx_in_H"]),
        }
        per_node = tilde_e_sq if resolved_cfg["output"].get(
            "save_per_node_diagnostics", True) else None
        dump_final_npz(out_dir, z, W, b,
                       tilde_e_sq=per_node, extras=extras)

    eval_writer.close()
    train_writer.close()

    if print_progress:
        print(f"  [sid={sid:03d} seed={seed:03d}]  "
              f"elapsed={train_time:.1f}s  "
              f"R_H_emp={final_metrics['R_H_emp']:.4f}  "
              f"R_H_star={final_metrics['R_H_star']:.4f}  "
              f"score_mse_H={final_metrics['score_mse_H']:.4f}  "
              f"oracle_err={final_metrics.get('oracle_classifier_error', float('nan')):.4f}")

    return {"status": "completed", "elapsed_s": train_time, **summary_metrics}


# =============================================================================
# Worker functions for the multi-process pool.
# Defined at module level so the spawn-context Pool can pickle them.
# =============================================================================

def _worker_init():
    """Per-worker setup. Runs once when the subprocess starts.

    Sets torch/numpy thread caps to 1 so workers don't fight for CPU; in our
    setting compute is GPU-bound and CPU-side ops are tiny, so 1 thread per
    worker is correct.
    """
    import torch
    torch.set_num_threads(1)
    try:
        import numpy as _np
        _np.seterr(all="ignore")  # suppress noisy warnings inside workers
    except Exception:
        pass


def _worker_main(task_tuple):
    """Run a single (sid, seed) task inside a worker process.

    Returns (sid, seed, sweep_label, result_dict).
    """
    from pathlib import Path as _Path
    import torch as _torch
    sid, seed, raw_cfg, exp_dir_str, device_str, sweep_label = task_tuple
    device = _torch.device(device_str)
    exp_dir = _Path(exp_dir_str)
    res = run_one(
        sid=sid, seed=seed,
        raw_cfg_for_setting=raw_cfg,
        exp_dir=exp_dir, device=device,
        print_progress=False,            # avoid interleaved stdout from many workers
        sweep_label=sweep_label,
    )
    return sid, seed, sweep_label, res


def main():
    parser = argparse.ArgumentParser(description="Run an LP-SBM experiment.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override runtime.device.")
    parser.add_argument("--output-base", type=str, default=None,
                        help="Override output.base_dir.")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Cap total (setting, seed) runs (for debugging).")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip runs already in manifest with status=completed.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel worker processes "
                             "(overrides runtime.num_workers). "
                             "1 or 0 = sequential.")
    args = parser.parse_args()

    # Load top-level raw config
    top_cfg = load_raw_config(args.config)

    # Determine output directory
    base_dir = args.output_base or top_cfg.get("output", {}).get("base_dir", "outputs")
    exp_dir = _experiment_dir(top_cfg, base_dir)
    ensure_dir(exp_dir)

    # Save the top-level config (post-extends, pre-sweep-expansion)
    import yaml
    with open(exp_dir / "config_full.yaml", "w") as f:
        yaml.safe_dump(top_cfg, f, sort_keys=False, default_flow_style=False)

    # Resolve device
    device_str = args.device or top_cfg.get("runtime", {}).get("device", "cpu")
    device = torch.device(device_str)

    # Expand sweep
    settings = expand_sweep(top_cfg)
    seeds = top_cfg.get("seeds", {}).get("list", [0])
    sweep_summary = get_sweep_summary(top_cfg)

    # Concurrency: 0 (default) or 1 -> sequential single-process loop;
    # > 1 -> multi-process worker pool. CLI flag overrides config.
    n_workers_cfg = int(top_cfg.get("runtime", {}).get("num_workers", 0))
    n_workers = int(getattr(args, "num_workers", None) or n_workers_cfg or 0)

    print(f"\n{'='*70}")
    print(f"Experiment: {top_cfg.get('experiment', {}).get('name', 'unnamed')}")
    print(f"  Output dir:    {exp_dir}")
    print(f"  Sweep mode:    {sweep_summary['mode']}")
    for ax in sweep_summary["axes"]:
        print(f"    axis {ax['name']!r} ({ax['kind']}): {ax['values_or_labels']}")
    print(f"  Settings:      {len(settings)}")
    print(f"  Seeds:         {len(seeds)}")
    print(f"  Total runs:    {len(settings) * len(seeds)}")
    print(f"  Device:        {device}")
    print(f"  Workers:       {n_workers if n_workers > 1 else 'sequential'}")
    print(f"{'='*70}\n")

    # Manifest
    manifest = Manifest(exp_dir, top_cfg.get("experiment", {}).get("name", "unnamed"))
    manifest.data["sweep_summary"] = sweep_summary
    manifest._save()

    # Track completed
    completed_keys = set()
    if args.skip_completed:
        for r in manifest.data["runs"]:
            if r["status"] == "completed":
                completed_keys.add((int(r["sid"]), int(r["seed"])))

    # Helper: extract a "primary" sweep_value used for plot labels.
    def _primary_value(raw_cfg_for_setting, sweep_label):
        primary_axis = sweep_summary["axes"][0] if sweep_summary["axes"] else None
        if primary_axis is not None and primary_axis["kind"] == "scalar":
            primary_path = top_cfg.get("sweep", {}).get("axis")
            if primary_path is None and "axes" in top_cfg.get("sweep", {}):
                primary_path = top_cfg["sweep"]["axes"][0].get("axis")
            if primary_path is not None:
                try:
                    return get_dotted(raw_cfg_for_setting, primary_path)
                except Exception:
                    return sweep_label
            return sweep_label
        return sweep_label

    # ------------------------------------------------------------------
    # Sequential path (n_workers <= 1)
    # ------------------------------------------------------------------
    if n_workers <= 1:
        runs_done = 0
        for sid, raw_cfg_for_setting in settings:
            sweep_label = raw_cfg_for_setting.get("_sweep_label", f"sid_{sid:04d}")
            sweep_value = _primary_value(raw_cfg_for_setting, sweep_label)
            manifest.add_setting_full(sid, sweep_value, sweep_label,
                                       raw_cfg_for_setting)
            for seed in seeds:
                if (sid, seed) in completed_keys:
                    continue
                if args.max_runs is not None and runs_done >= args.max_runs:
                    print(f"Reached max_runs={args.max_runs}; stopping.")
                    return
                print(f"Run sid={sid:03d} seed={seed:03d}  ({sweep_label})")
                res = run_one(
                    sid=sid, seed=seed,
                    raw_cfg_for_setting=raw_cfg_for_setting,
                    exp_dir=exp_dir, device=device,
                    sweep_label=sweep_label,
                )
                manifest.add_run(sid, seed, res["status"],
                                 seed_dir(exp_dir, sid, seed, label=sweep_label),
                                 meta={k: v for k, v in res.items() if k != "status"})
                runs_done += 1
    else:
        # ------------------------------------------------------------------
        # Multi-process path: ProcessPoolExecutor with `spawn`
        # context. Each worker re-imports torch, gets its own CUDA context,
        # and runs `run_one` for one (sid, seed) at a time. Manifest writes
        # are serialized through the main process, so no file-lock needed.
        # ------------------------------------------------------------------
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Pre-register all settings (so the manifest contains them even if
        # some seeds fail or get cancelled by --max-runs).
        all_tasks = []
        for sid, raw_cfg_for_setting in settings:
            sweep_label = raw_cfg_for_setting.get("_sweep_label", f"sid_{sid:04d}")
            sweep_value = _primary_value(raw_cfg_for_setting, sweep_label)
            manifest.add_setting_full(sid, sweep_value, sweep_label,
                                       raw_cfg_for_setting)
            for seed in seeds:
                if (sid, seed) in completed_keys:
                    continue
                all_tasks.append((sid, seed, raw_cfg_for_setting,
                                  str(exp_dir), str(device), sweep_label))

        if args.max_runs is not None:
            all_tasks = all_tasks[: int(args.max_runs)]

        if not all_tasks:
            print("No pending tasks (all completed or skipped).")
        else:
            print(f"Dispatching {len(all_tasks)} runs to {n_workers} workers "
                  f"(spawn context)...")
            ctx = mp.get_context("spawn")
            t0 = time.time()
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_worker_init,
            ) as pool:
                futures = {pool.submit(_worker_main, t): t for t in all_tasks}
                done = 0
                for fut in as_completed(futures):
                    task = futures[fut]
                    sid, seed, _raw, _exp, _dev, sweep_label = task
                    try:
                        sid_back, seed_back, sw_back, res = fut.result()
                    except Exception as e:
                        print(f"  [WORKER FAILED] sid={sid:03d} seed={seed:03d} "
                              f"({sweep_label}): {type(e).__name__}: {e}")
                        res = {"status": "failed", "elapsed_s": 0.0,
                               "error": f"{type(e).__name__}: {e}"}
                    manifest.add_run(sid, seed, res["status"],
                                     seed_dir(exp_dir, sid, seed, label=sweep_label),
                                     meta={k: v for k, v in res.items() if k != "status"})
                    done += 1
                    elapsed = time.time() - t0
                    print(f"  [{done}/{len(all_tasks)}] sid={sid:03d} seed={seed:03d} "
                          f"({sweep_label}) status={res['status']} "
                          f"run={res.get('elapsed_s', 0):.1f}s "
                          f"wall={elapsed:.0f}s")

    # Final summary
    print(f"\n{'='*70}")
    print(f"Done. Manifest: {manifest.path}")
    print(f"  {manifest.summary()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
