"""
Load experiment runs into a pandas DataFrame.

Each row of the returned DataFrame represents one (sid, seed, eval-point)
record from an `eval_log.jsonl` file. Configuration fields (sbm.n,
encoder.family, etc.) and derived oracle quantities (capacity_required,
b_star, B_star, Delta_s, ...) are joined onto each row so that downstream
analysis only deals with one flat table.

Typical usage:
    df = load_experiment("outputs/sweep_n")
    final = df[df.is_final].copy()
    grouped = final.groupby(["sid", "sweep_value"])["score_mse_H"].agg(
        ["mean", "std", "count"]
    )
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


# =============================================================================
# Low-level loaders
# =============================================================================

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_jsonl_shards(directory: Path, basename: str) -> List[Dict[str, Any]]:
    """Read all `<basename>*.jsonl` files in `directory`, concatenated in
    filename-sort order. Returns [] if none found."""
    if not directory.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(directory.glob(f"{basename}*.jsonl")):
        out.extend(_read_jsonl(p))
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict to dotted-path keys, skipping leading-underscore keys."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full))
        elif isinstance(v, (list, tuple)):
            # Keep lists as a string repr or skip; for sweep analysis
            # we usually only care about scalars.
            try:
                out[full] = list(v)
            except Exception:
                out[full] = str(v)
        else:
            out[full] = v
    return out


# =============================================================================
# Experiment-level loader
# =============================================================================

def load_experiment(
    exp_dir: str | Path,
    *,
    only_final: bool = False,
    include_periodic: bool = True,
) -> pd.DataFrame:
    """Walk an experiment directory and return a long-form DataFrame.

    `exp_dir` is e.g. `outputs/sweep_n`. Expected structure:

        exp_dir/
            manifest.json
            setting_NNNN/
                seed_KKKK/
                    config_resolved.json
                    eval_log.jsonl
                    ...

    Each row of the returned DataFrame is one eval-log entry, augmented with:
      - sid, seed (ints)
      - is_final (bool)
      - sweep_axis, sweep_value (from manifest)
      - all (flattened) fields of config_resolved.json with `cfg.` prefix
      - all derived oracle quantities (already in resolved cfg under _derived)

    If `only_final=True`, returns only the rows where is_final==True (one row
    per (sid, seed)).
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    manifest_path = exp_dir / "manifest.json"
    sweep_axis: Optional[str] = None
    settings_meta: Dict[int, Dict[str, Any]] = {}
    sweep_summary: Optional[Dict[str, Any]] = None
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        sweep_summary = manifest.get("sweep_summary")
        # Pick up sweep axis from full config (saved at top level)
        full_cfg_path = exp_dir / "config_full.yaml"
        if full_cfg_path.exists():
            full_cfg = _read_yaml(full_cfg_path)
            # Multi-axis: take first axis as primary; otherwise legacy single axis.
            sw = full_cfg.get("sweep", {}) or {}
            if "axes" in sw and sw["axes"]:
                first_ax = sw["axes"][0]
                sweep_axis = first_ax.get("name") or first_ax.get("axis")
            else:
                sweep_axis = sw.get("axis")
        for s in manifest.get("settings", []):
            settings_meta[int(s["sid"])] = {
                "sweep_value": s.get("sweep_value"),
                "sweep_label": s.get("sweep_label"),
            }

    rows: List[Dict[str, Any]] = []

    # Setting dirs: setting_NNNN or setting_NNNN_<label>
    for setting_path in sorted(exp_dir.glob("setting_*")):
        if not setting_path.is_dir():
            continue
        # Parse the leading "setting_NNNN" to get sid; ignore the optional suffix.
        name_parts = setting_path.name.split("_")
        try:
            sid = int(name_parts[1])
        except (IndexError, ValueError):
            continue
        for seed_path in sorted(setting_path.glob("seed_*")):
            if not seed_path.is_dir():
                continue
            seed = int(seed_path.name.split("_")[-1])

            cfg_path = seed_path / "config_resolved.json"
            evals = _read_jsonl_shards(seed_path, "eval_log")
            if not evals or not cfg_path.exists():
                continue

            cfg = _read_json(cfg_path)
            cfg_flat = _flatten_dict(cfg, prefix="cfg")
            derived = cfg.get("_derived", {})
            for k, v in derived.items():
                if isinstance(v, (int, float, bool, str)):
                    cfg_flat[f"derived.{k}"] = v

            sm = settings_meta.get(sid, {})
            sweep_value = sm.get("sweep_value")
            sweep_label = sm.get("sweep_label")

            for entry in evals:
                if only_final and not entry.get("is_final", False):
                    continue
                if not include_periodic and not entry.get("is_final", False):
                    continue
                row: Dict[str, Any] = {
                    "sid":         sid,
                    "seed":        seed,
                    "sweep_axis":  sweep_axis,
                    "sweep_value": sweep_value,
                    "sweep_label": sweep_label,
                }
                row.update(cfg_flat)
                row.update(entry)
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_final(exp_dir: str | Path) -> pd.DataFrame:
    """Convenience: only the final-eval row per (sid, seed)."""
    return load_experiment(exp_dir, only_final=True)


# =============================================================================
# Discover available experiments
# =============================================================================

def list_experiments(base_dir: str | Path = "outputs") -> List[Path]:
    """List all experiment directories under base_dir that have manifests."""
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir()
                   if p.is_dir() and (p / "manifest.json").exists()])
