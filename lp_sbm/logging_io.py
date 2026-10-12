"""
Logging / IO helpers.

Per-run output structure:
    outputs/<experiment_name>/
        manifest.json              # All settings/seeds and their statuses
        config_full.yaml           # Top-level experiment config (post-extends)
        setting_<sid>/
            seed_<k>/
                config_raw.yaml          # User-intent config for this setting
                config_resolved.json     # After capacity resolution
                eval_log.jsonl           # One line per eval (final + periodic).
                                          # When multiple processes write the same
                                          # (sid, seed) concurrently, each process
                                          # writes to eval_log.<pid>.jsonl; the
                                          # analysis loader merges them.
                train_log.jsonl          # One line per epoch (light metrics)
                final.npz                # Embeddings, W, b, per-node arrays

JSONL format: one JSON object per line. Easy to append, easy to load (pandas).
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# =============================================================================
# JSONL writer
# =============================================================================

class JSONLWriter:
    """Append-only JSONL writer, line-buffered.

    To make this safe under concurrent writers (e.g. parallel sweeps that target
    the same output dir), the writer redirects the path

        foo/bar.jsonl   ->   foo/bar.<pid>.jsonl

    when `shard_by_pid=True` (the default). The analysis-side loader merges all
    `bar*.jsonl` files in the directory. Set `shard_by_pid=False` to write to
    the literal path (useful when concurrency is impossible by construction).
    """

    def __init__(self, path: Path | str, *, shard_by_pid: bool = True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if shard_by_pid:
            stem, suffix = path.stem, path.suffix
            path = path.with_name(f"{stem}.{os.getpid()}{suffix}")
        self.path = path
        self._fh = open(self.path, "a", buffering=1)  # line-buffered

    def write(self, obj: Dict[str, Any]) -> None:
        clean = _to_jsonable(obj)
        self._fh.write(json.dumps(clean, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    """Read JSONL into a list of dicts (one per line)."""
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def read_jsonl_shards(directory: Path | str, basename: str) -> List[Dict[str, Any]]:
    """Read all PID-sharded JSONL files matching `<basename>*.jsonl` under directory.

    Reads in lexicographic order of filename, which preserves intra-process
    ordering. Returns a single concatenated list of dicts.

    Examples
    --------
    >>> read_jsonl_shards("seed_0001", "eval_log")
    # reads "eval_log.jsonl" + "eval_log.<pid>.jsonl" if any
    """
    directory = Path(directory)
    out: List[Dict[str, Any]] = []
    if not directory.exists():
        return out
    candidates = sorted(directory.glob(f"{basename}*.jsonl"))
    for p in candidates:
        out.extend(read_jsonl(p))
    return out


# =============================================================================
# Run directory structure
# =============================================================================

def setting_dir(base: Path, setting_id: int, label: Optional[str] = None) -> Path:
    """Setting directory.

    If `label` is given, dir is `setting_<sid:04d>_<label>`; otherwise just
    `setting_<sid:04d>`. The label is sanitized (slashes / spaces -> '_').
    """
    if label is None:
        return Path(base) / f"setting_{setting_id:04d}"
    safe = str(label).replace("/", "_").replace(" ", "_").replace(",", "")
    # Cap length to keep paths manageable.
    if len(safe) > 80:
        safe = safe[:80]
    return Path(base) / f"setting_{setting_id:04d}_{safe}"


def seed_dir(base: Path, setting_id: int, seed: int,
             label: Optional[str] = None) -> Path:
    return setting_dir(base, setting_id, label) / f"seed_{seed:04d}"


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Final-state dump
# =============================================================================

def dump_final_npz(
    out_dir: Path | str,
    z: np.ndarray, W: np.ndarray, b: float,
    *, tilde_e_sq: Optional[np.ndarray] = None,
    extras: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save final embeddings, W, b, and optional per-node diagnostics to npz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"z": np.asarray(z, dtype=np.float32),
                                "W": np.asarray(W, dtype=np.float64),
                                "b": float(b)}
    if tilde_e_sq is not None:
        payload["tilde_e_sq"] = np.asarray(tilde_e_sq, dtype=np.float64)
    if extras:
        for k, v in extras.items():
            payload[k] = v
    np.savez_compressed(out_dir / "final.npz", **payload)


# =============================================================================
# Manifest
# =============================================================================

class Manifest:
    """Track all (setting, seed) runs and their statuses for an experiment.

    Saved as `manifest.json` under the experiment dir. Updated incrementally
    after each run.
    """

    def __init__(self, exp_dir: Path | str, experiment_name: str):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.exp_dir / "manifest.json"
        self.data: Dict[str, Any] = {
            "experiment_name":    experiment_name,
            "created":            datetime.now().isoformat(),
            "settings":           [],   # list of {sid, sweep_value, n_seeds, ...}
            "runs":               [],   # list of {sid, seed, status, path, ...}
        }
        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)

    def add_setting(self, sid: int, sweep_value: Any, raw_config: Dict) -> None:
        """Single-axis setting record. Prefer add_setting_full for multi-axis."""
        existing = next((s for s in self.data["settings"] if s["sid"] == sid), None)
        if existing is None:
            self.data["settings"].append({
                "sid":         sid,
                "sweep_value": _to_jsonable(sweep_value),
                "sweep_label": str(sweep_value),
            })
            self._save()

    def add_setting_full(self, sid: int, sweep_value: Any, sweep_label: str,
                          raw_config: Dict) -> None:
        """Multi-axis-aware setting record. `sweep_label` is the joined per-axis
        labels (used for setting_dir naming); `sweep_value` is the primary axis
        value (e.g. n for sweep_n) used for plotting."""
        existing = next((s for s in self.data["settings"] if s["sid"] == sid), None)
        if existing is None:
            self.data["settings"].append({
                "sid":         sid,
                "sweep_value": _to_jsonable(sweep_value),
                "sweep_label": str(sweep_label),
            })
            self._save()

    def add_run(self, sid: int, seed: int, status: str, path: Path,
                meta: Optional[Dict] = None) -> None:
        run = {
            "sid":      sid,
            "seed":     seed,
            "status":   status,
            "path":     str(path),
            "updated":  datetime.now().isoformat(),
        }
        if meta:
            run.update(_to_jsonable(meta))
        # Replace if existing
        self.data["runs"] = [r for r in self.data["runs"]
                             if not (r["sid"] == sid and r["seed"] == seed)]
        self.data["runs"].append(run)
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def summary(self) -> Dict[str, int]:
        out = {"total": len(self.data["runs"]),
               "completed": 0, "failed": 0, "running": 0}
        for r in self.data["runs"]:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out


# =============================================================================
# JSON-friendly conversion
# =============================================================================

def _to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays + paths to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
