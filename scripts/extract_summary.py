"""
extract_summary.py  --  compact-output extractor for sweep experiments.

Purpose: produce a single JSON file (~100-300 KB) capturing the information
needed to analyze results without the full outputs/ tree (tens of MB of
train_curves + final.npz checkpoints).

What it includes
----------------
- Manifest-resolved (sid -> sweep_label) mapping, so duplicate / stale
  setting dirs that are not in the manifest are skipped.
- Per-seed final eval metrics (every numeric field from eval_log,
  is_final entry only -- periodic checkpoints dropped).
- Compact config snapshot per setting (only the ~10 fields that matter
  for diagnosis: epochs, lr, bias init, decoder family, W_max, etc.).
- Per-(setting) aggregate: median / mean / std / min / max for each metric.

What it skips
-------------
- train_curve.*.jsonl  (epoch-level training loss -- large, rarely needed)
- train_log.*.jsonl    (epoch-level training log)
- final.npz            (model weights; meaningless without code)
- Periodic eval entries from eval_log (only final entry kept).

Usage
-----
    python scripts/extract_summary.py outputs/main_sweep_n
    python scripts/extract_summary.py outputs/main_sweep_n --out summary.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Config fields worth keeping in the summary. Add more if needed.
CONFIG_KEYS_OF_INTEREST = [
    "training.max_epochs",
    "training.lr",
    "training.weight_decay",
    "training.bias_init_mode",
    "training.early_stop",
    "training.val_split_of_H",
    "training.batch_mode",
    "decoder.family",
    "decoder.symmetric",
    "decoder.W_max",
    "decoder.B_0",
    "encoder.family",
    "encoder.d",
    "encoder.hidden_dim",
    "encoder.num_layers",
    "encoder.features.mode",
    "encoder.features.dim",
    "encoder.features.pe_normalize",
    "encoder.R",
    "sbm.n",
    "sbm.K",
    "sbm.rho_n",
    "sbm.block_spec.p_in",
    "sbm.block_spec.p_out",
    "mask.tau",
]


def _read_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _read_jsonl_final(seed_dir: Path) -> Optional[dict]:
    """Read all eval_log shards, return the last is_final=True entry, or None."""
    final = None
    for f in sorted(seed_dir.glob("eval_log.*.jsonl")):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("is_final", False):
                    final = entry
    return final


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict using dot notation."""
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
    return out


def _agg(values: List[float]) -> Dict[str, float]:
    """Robust aggregates over per-seed values."""
    if not values:
        return {}
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = var ** 0.5
    median = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    q25 = vals[max(0, int(0.25 * n))]
    q75 = vals[min(n - 1, int(0.75 * n))]
    return {
        "median": round(median, 6),
        "mean":   round(mean, 6),
        "std":    round(std, 6),
        "min":    round(vals[0], 6),
        "max":    round(vals[-1], 6),
        "q25":    round(q25, 6),
        "q75":    round(q75, 6),
        "n":      n,
    }


def extract(exp_dir: Path) -> Dict[str, Any]:
    manifest_path = exp_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {exp_dir}")
    manifest = _read_json(manifest_path)

    # Build sid -> expected_label mapping (manifest = ground truth)
    expected = {int(s["sid"]): s["sweep_label"] for s in manifest["settings"]}
    sweep_axis = manifest.get("sweep_summary", {}).get("axis", "n")

    settings_out: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for sid, expected_label in sorted(expected.items()):
        # Find the dir whose name is exactly setting_NNNN_<expected_label>
        matches = list(exp_dir.glob(f"setting_{sid:04d}_{expected_label}"))
        if not matches:
            # Tolerate dirs without the label suffix (kept for compatibility)
            matches = list(exp_dir.glob(f"setting_{sid:04d}"))
        if not matches:
            skipped.append(f"sid={sid} ({expected_label}): no matching dir")
            continue
        if len(matches) > 1:
            skipped.append(f"sid={sid}: multiple matching dirs, taking first")
        setting_dir = matches[0]

        seed_records = []
        config_snapshot: Dict[str, Any] = {}

        for seed_dir in sorted(setting_dir.glob("seed_*")):
            cfg_path = seed_dir / "config_resolved.json"
            if not cfg_path.exists():
                continue
            seed = int(seed_dir.name.split("_")[-1])
            try:
                cfg = _read_json(cfg_path)
                cfg_flat = _flatten(cfg.get("cfg", cfg))
            except Exception:
                cfg_flat = {}

            # Snapshot config from the first seed
            if not config_snapshot:
                config_snapshot = {
                    k: cfg_flat.get(k)
                    for k in CONFIG_KEYS_OF_INTEREST
                    if k in cfg_flat
                }

            final_entry = _read_jsonl_final(seed_dir)
            if final_entry is None:
                continue

            # Keep only numeric / bool fields. Skip:
            #  - strings (classifier descriptions, modes -- already in config)
            #  - arrays / dicts
            # Round floats to 6 decimals for compactness.
            metrics: Dict[str, Any] = {}
            for k, v in final_entry.items():
                if k in ("is_final", "epoch"):
                    continue
                # Drop redundant per-seed strings that duplicate the config block:
                if k.endswith("_classifier") or k.endswith("_classifier_name"):
                    continue
                if k in ("selection_basis", "selection_metric",
                         "restart_selection_basis", "bias_init_mode",
                         "feature_mode", "pe_normalize",
                         "encoder_family", "decoder_family", "early_stop"):
                    continue
                if isinstance(v, bool):
                    metrics[k] = v
                elif isinstance(v, int):
                    metrics[k] = v
                elif isinstance(v, float):
                    metrics[k] = round(v, 6)
                elif v is None:
                    metrics[k] = None
                # other types (str, list, dict) silently dropped

            seed_records.append({
                "seed": seed,
                "epoch": final_entry.get("epoch"),
                "metrics": metrics,
            })

        # Aggregate per-metric across seeds
        all_keys = set()
        for r in seed_records:
            all_keys.update(k for k, v in r["metrics"].items()
                            if isinstance(v, (int, float)))
        agg = {}
        for k in sorted(all_keys):
            vals = [r["metrics"][k] for r in seed_records
                    if isinstance(r["metrics"].get(k), (int, float))]
            agg[k] = _agg(vals)

        settings_out.append({
            "sid": sid,
            "sweep_value": next(s["sweep_value"] for s in manifest["settings"]
                                if int(s["sid"]) == sid),
            "sweep_label": expected_label,
            "config": config_snapshot,
            "seeds": seed_records,
            "aggregate": agg,
        })

    summary = {
        "experiment_name": manifest.get("experiment_name", exp_dir.name),
        "manifest_created": manifest.get("created"),
        "sweep_axis": sweep_axis,
        "n_settings": len(settings_out),
        "n_skipped": len(skipped),
        "skipped_notes": skipped,
        "settings": settings_out,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=str, help="Path to outputs/<experiment_name>/")
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSON path (default: <exp_dir>_summary.json)")
    ap.add_argument("--no-per-seed", action="store_true",
                    help="Drop per-seed records, keep only aggregates (smallest)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_dir():
        raise SystemExit(f"Not a directory: {exp_dir}")

    summary = extract(exp_dir)

    if args.no_per_seed:
        for s in summary["settings"]:
            s["seeds"] = [{"seed": r["seed"]} for r in s["seeds"]]  # keep IDs only

    out = Path(args.out) if args.out else exp_dir.parent / f"{exp_dir.name}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({size_kb:.1f} KB,  {summary['n_settings']} settings,  "
          f"{sum(len(s['seeds']) for s in summary['settings'])} seed records)")
    if summary["n_skipped"]:
        print(f"Skipped {summary['n_skipped']} settings:")
        for note in summary["skipped_notes"]:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
