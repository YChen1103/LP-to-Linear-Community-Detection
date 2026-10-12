"""
Configuration loading and resolution.

There are TWO stages of config processing:

  1. *Loading*. Parse YAML, follow `extends:` chain, apply `overrides`.
     Result: a "raw config" -- the user's specified intent, possibly with
     `*_spec` dicts that need derivation (e.g. capacity-relative `W_max`).

  2. *Resolution*. Compute SBM oracle quantities (||M_tilde||_op, b*, B*),
     then resolve each `*_spec` into a concrete numerical value.
     Also resolve coupled sweep expressions like
        decoder.W_max_spec.value: "2.0 * $capacity_required / $encoder.R_spec.value**2"

Both raw and resolved configs are saved to disk per run for reproducibility.
"""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


# =============================================================================
# Loading: YAML + extends + overrides
# =============================================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file. Handles extends: chain (relative to file's parent dir)."""
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if "extends" in cfg:
        parent_path = path.parent / cfg.pop("extends")
        parent = _load_yaml(parent_path)
        cfg = _deep_merge(parent, cfg)

    return cfg


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep-merge `override` into `base`. `override` wins at every leaf."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_raw_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file, following `extends:` chain.

    The result is the "raw" config: intent expressed in `*_spec` form,
    not yet resolved to numerical values.
    """
    return _load_yaml(Path(path))


# =============================================================================
# Dotted-path access for sweep axes and overrides
# =============================================================================

def get_dotted(d: Dict, path: str) -> Any:
    """Get d[a][b][c] from path 'a.b.c'."""
    keys = path.split(".")
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def set_dotted(d: Dict, path: str, value: Any) -> None:
    """Set d[a][b][c] = value from path 'a.b.c'. Modifies d in-place."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


# =============================================================================
# Sweep expansion: produce list of (setting_id, raw_config) pairs
# =============================================================================
# Sweep schemas
# -------------
# (a) Single-axis:
#       sweep:
#         axis: sbm.n
#         values: [200, 500, 1000]
#
# (b) Multi-axis:
#       sweep:
#         mode: grid                  # "grid" (cartesian) | "zip" (parallel)
#         axes:
#           - name: n
#             axis: sbm.n
#             values: [200, 500, 1000]
#           - name: arch
#             values:
#               - label: gin_pe
#                 overrides:
#                   encoder.family: gin
#                   encoder.features.mode: spectral_pe
#               - label: gin_random
#                 overrides:
#                   encoder.family: gin
#                   encoder.features.mode: random_normal
#
# An axis can either:
#   - have `axis` (dotted path) and `values` (list of scalars) -- like (a)
#   - have `values` as a list of {label, overrides} dicts -- applies a bundle
#
# `mode: grid` -> cartesian product (default); `mode: zip` -> parallel pairing
# (lengths must agree).
#
# When multi-axis, settings have integer ids and a human-readable label
# (joined with "_") that the run script will use for setting_dir naming.
# =============================================================================


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply a flat dotted-path -> value overrides dict to cfg in place."""
    for path, value in overrides.items():
        set_dotted(cfg, path, value)


def _expand_axis_to_choices(axis_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize one axis spec into a list of "choice" dicts.

    Each choice has:
      - label: str (used in setting_dir naming)
      - overrides: dict {dotted_path: value}
      - sweep_value: the canonical sweep value for this choice (for manifest)

    Two axis-spec forms supported:

      Form A (scalar): {axis: "sbm.n", values: [200, 500]}
        -> choices: [{label:"n=200", overrides:{"sbm.n":200}, sweep_value:200}, ...]

      Form B (bundle): {values: [{label, overrides}, ...]}
        -> choices: [{label, overrides, sweep_value=label}, ...]
    """
    name = axis_spec.get("name")
    values = axis_spec.get("values", [])
    axis_path = axis_spec.get("axis")

    out: List[Dict[str, Any]] = []
    if axis_path is not None:
        # Form A: scalar values along a single dotted path.
        for v in values:
            label = f"{name}={v}" if name else f"{axis_path}={v}"
            out.append({
                "label":      str(label),
                "overrides":  {axis_path: v},
                "sweep_value": v,
            })
    else:
        # Form B: each value is a {label, overrides} bundle.
        for v in values:
            if not isinstance(v, dict) or "label" not in v or "overrides" not in v:
                raise ValueError(
                    f"Bundle-axis value must be {{label, overrides}}; got {v!r}"
                )
            out.append({
                "label":      str(v["label"]),
                "overrides":  dict(v["overrides"]),
                "sweep_value": v["label"],
            })
    return out


def expand_sweep(raw_config: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    """Expand a sweep into a list of (setting_id, single_setting_raw_config).

    Each setting has the sweep axis set to one of the sweep values, plus any
    coupled fields evaluated. Coupled values can reference *raw* config fields
    (e.g. $encoder.R_spec.value) but NOT derived quantities (which are not
    known until resolve_config is called).

    For coupled fields that depend on derived quantities (e.g. $capacity_required),
    use the `coupled_after_derive` block in the sweep spec instead -- these are
    evaluated during resolve_config.

    Each returned setting also has a "_sweep_label" field injected at top level
    (joined from per-axis labels, e.g. "n=200__arch=gin_pe") that downstream
    code can use for setting_dir naming.
    """
    sweep = raw_config.get("sweep", {})
    axes_spec = sweep.get("axes")
    axis = sweep.get("axis")
    values = sweep.get("values", [])
    coupled = sweep.get("coupled", {})

    # Case A: single-axis (axis + values).
    if axes_spec is None:
        if axis is None or not values:
            # No sweep -- single setting.
            cfg = copy.deepcopy(raw_config)
            cfg["_sweep_label"] = "default"
            return [(0, cfg)]
        out: List[Tuple[int, Dict[str, Any]]] = []
        for sid, v in enumerate(values):
            cfg = copy.deepcopy(raw_config)
            set_dotted(cfg, axis, v)
            for coupled_path, expr in coupled.items():
                resolved = _eval_expression(expr, _flatten(cfg, prefix=""))
                set_dotted(cfg, coupled_path, resolved)
            cfg["_sweep_label"] = f"{axis.replace('.', '_')}={v}"
            out.append((sid, cfg))
        return out

    # Case B: multi-axis (axes list).
    mode = str(sweep.get("mode", "grid")).lower()
    if mode not in ("grid", "zip"):
        raise ValueError(f"sweep.mode must be 'grid' or 'zip'; got {mode!r}")

    axis_choices = [_expand_axis_to_choices(ax) for ax in axes_spec]

    if mode == "grid":
        import itertools
        combos = list(itertools.product(*axis_choices))
    else:  # zip
        lens = {len(c) for c in axis_choices}
        if len(lens) != 1:
            raise ValueError(
                f"sweep.mode=zip requires all axes same length; got {[len(c) for c in axis_choices]}"
            )
        combos = list(zip(*axis_choices))

    out: List[Tuple[int, Dict[str, Any]]] = []
    for sid, combo in enumerate(combos):
        cfg = copy.deepcopy(raw_config)
        for choice in combo:
            _apply_overrides(cfg, choice["overrides"])
        for coupled_path, expr in coupled.items():
            resolved = _eval_expression(expr, _flatten(cfg, prefix=""))
            set_dotted(cfg, coupled_path, resolved)
        # Setting label for filesystem naming.
        cfg["_sweep_label"] = "__".join(c["label"] for c in combo)
        out.append((sid, cfg))
    return out


def get_sweep_summary(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small summary of sweep structure for the manifest.

    Keys:
        mode: "single" | "grid" | "zip"
        axes: list of {name, kind: "scalar"|"bundle", values_or_labels}
    """
    sweep = raw_config.get("sweep", {})
    axes_spec = sweep.get("axes")
    if axes_spec is None:
        if sweep.get("axis") is None:
            return {"mode": "single", "axes": []}
        return {
            "mode": "single",
            "axes": [{
                "name": sweep["axis"],
                "kind": "scalar",
                "values_or_labels": list(sweep.get("values", [])),
            }],
        }
    mode = str(sweep.get("mode", "grid")).lower()
    out_axes = []
    for ax in axes_spec:
        if ax.get("axis") is not None:
            out_axes.append({
                "name": ax.get("name") or ax["axis"],
                "kind": "scalar",
                "values_or_labels": list(ax.get("values", [])),
            })
        else:
            out_axes.append({
                "name": ax.get("name") or "bundle",
                "kind": "bundle",
                "values_or_labels": [v["label"] for v in ax.get("values", [])],
            })
    return {"mode": mode, "axes": out_axes}


# =============================================================================
# Capacity-aware resolution (for *_spec dicts)
# =============================================================================

def resolve_config(
    raw_config: Dict[str, Any],
    sbm_oracle: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve all `*_spec` dicts and coupled-after-derive expressions.

    `sbm_oracle` must already contain keys: B_star, b_star, capacity_required,
    Delta_s, Delta_s_sq, mu_B (these are computed by lp_sbm.oracle).

    Returns a deep copy of `raw_config` with:
      - encoder.R       (resolved float)
      - decoder.W_max   (resolved float)
      - decoder.B_0     (resolved float)
      - _derived        (dict of all derived quantities for logging)

    Raw `*_spec` dicts are left in place for traceability.
    """
    cfg = copy.deepcopy(raw_config)
    derived = _build_derived_context(cfg, sbm_oracle)

    # Resolve R (encoder norm bound) -- only depends on SBM oracle quantities
    R = _resolve_R(cfg["encoder"]["R_spec"], derived)
    cfg["encoder"]["R"] = float(R)
    derived["R"] = float(R)

    # Apply coupled_after_derive *before* resolving W_max / B_0, so that any
    # coupled expression that targets W_max_spec.value or B_0_spec.value (and
    # references derived quantities like $capacity_required or already-resolved
    # $R) takes effect before those *_spec dicts are read.
    for coupled_path, expr in cfg.get("sweep", {}).get("coupled_after_derive", {}).items():
        flat_for_eval = _flatten(cfg, prefix="")
        flat_for_eval.update(derived)
        value = _eval_expression(expr, flat_for_eval)
        set_dotted(cfg, coupled_path, value)

    # Resolve W_max (decoder operator norm bound) -- may use the updated spec
    W_max_spec_resolved = _resolve_W_max(cfg["decoder"]["W_max_spec"], derived)
    cfg["decoder"]["W_max_spec_value"] = float(W_max_spec_resolved)

    # For inner-product decoder, ||W||_op is identically 1 (W = I). The user's
    # configured W_max_spec is then used only as a sanity check (we require
    # W_max_spec >= 1 so the inner-product instance lies within the declared
    # bounded class), and the *effective* W_max for downstream computations
    # (B_model, V_n_stat, gamma envelope, ...) is 1.
    decoder_family = str(cfg["decoder"].get("family", "bilinear")).lower()
    if decoder_family == "inner_product":
        if W_max_spec_resolved < 1.0 - 1e-9:
            raise ValueError(
                f"Inner-product decoder requires W_max_spec >= 1 (since ||W||_op = 1 "
                f"identically), got configured {W_max_spec_resolved:.4f}. "
                f"Set decoder.W_max_spec to a value >= 1 or use bilinear."
            )
        W_max = 1.0
    else:
        W_max = float(W_max_spec_resolved)
    cfg["decoder"]["W_max"] = float(W_max)
    derived["W_max"] = float(W_max)
    derived["W_max_spec_resolved"] = float(W_max_spec_resolved)
    derived["W_max_effective_is_one"] = bool(decoder_family == "inner_product")

    # Resolve B_0 (decoder bias bound)
    B_0 = _resolve_B_0(cfg["decoder"]["B_0_spec"], derived)
    cfg["decoder"]["B_0"] = float(B_0)
    derived["B_0"] = float(B_0)

    # Compute and store further derived bound quantities. For inner-product,
    # B_model = B_0 + R^2 (since W = I). For bilinear, B_model = B_0 + W_max R^2.
    # Both are captured by the unified formula B_model = B_0 + W_max_eff * R^2
    # where W_max_eff = 1 for inner-product and W_max for bilinear.
    B_model = float(B_0 + W_max * R**2)
    # Definition of B from the paper: B = max(B_model, B_star, log 2).
    # The log 2 floor is what makes the per-pair BCE loss l(s; p) lie in [0, 2B]
    # on |s| <= B (used in the proof of the train-side corollary). Numerically
    # it is rarely the active term
    # (B_model >> log 2 typically), but keeping it ensures the Train-side
    # uniform-deviation rates are exact rather than "up to a B >= log 2"
    # convention.
    B = float(max(B_model, derived["B_star"], np.log(2.0)))
    sigma_B = 1.0 / (1.0 + np.exp(-B))
    mu_B_at_B = float(sigma_B * (1.0 - sigma_B))

    derived["B_model"] = B_model
    derived["B"] = B
    derived["mu_B_at_B"] = mu_B_at_B
    derived["capacity_available"] = float(W_max * R**2)
    derived["capacity_ratio_actual"] = (
        float((W_max * R**2) / derived["capacity_required"])
        if derived["capacity_required"] > 1e-12 else float("inf")
    )

    cfg["_derived"] = derived
    return cfg


def _build_derived_context(raw_cfg: Dict, sbm_oracle: Dict) -> Dict[str, Any]:
    """Build the derived-context dict consumed by resolution + coupled exprs."""
    return {
        "n":                  int(raw_cfg["sbm"]["n"]),
        "K":                  int(raw_cfg["sbm"]["K"]),
        "rho_n":              float(raw_cfg["sbm"]["rho_n"]),
        "tau":                float(raw_cfg["mask"]["tau"]),
        "d":                  int(raw_cfg["encoder"]["d"]),
        "B_star":             float(sbm_oracle["B_star"]),
        "b_star":             float(sbm_oracle["b_star"]),
        "capacity_required":  float(sbm_oracle["capacity_required"]),
        "Delta_s":            float(sbm_oracle["Delta_s"]),
        "Delta_s_sq":         float(sbm_oracle["Delta_s_sq"]),
    }


def _resolve_R(R_spec: Dict, ctx: Dict) -> float:
    mode = R_spec.get("mode", "fixed")
    if mode == "fixed":
        return float(R_spec["value"])
    if mode == "from_capacity_ratio":
        # R^2 = capacity_ratio * capacity_required (with W_max=1 implicit)
        return float(np.sqrt(R_spec["capacity_ratio"] * ctx["capacity_required"]))
    raise ValueError(f"Unknown R_spec.mode: {mode!r}")


def _resolve_W_max(W_max_spec: Dict, ctx: Dict) -> float:
    mode = W_max_spec.get("mode", "fixed")
    if mode == "fixed":
        return float(W_max_spec["value"])
    if mode == "from_capacity_ratio":
        # W_max R^2 = capacity_ratio * ||M_tilde||_op
        R = ctx["R"]
        cap = ctx["capacity_required"]
        return float(W_max_spec["capacity_ratio"] * cap / max(R**2, 1e-12))
    raise ValueError(f"Unknown W_max_spec.mode: {mode!r}")


def _resolve_B_0(B_0_spec: Dict, ctx: Dict) -> float:
    mode = B_0_spec.get("mode", "fixed")
    if mode == "fixed":
        return float(B_0_spec["value"])
    if mode == "from_b_star":
        return float(abs(ctx["b_star"]) + B_0_spec.get("slack", 0.0))
    if mode == "multiple_of_b_star":
        return float(B_0_spec["factor"] * abs(ctx["b_star"]))
    raise ValueError(f"Unknown B_0_spec.mode: {mode!r}")


# =============================================================================
# Expression evaluation for coupled sweeps
# =============================================================================
# Syntax: "2.0 * $capacity_required / $encoder.R_spec.value**2"
# $-prefixed names are looked up in the flat context dict.
# Only basic arithmetic (+, -, *, /, **, parentheses, math functions) allowed.

_ALLOWED_NAMES = {
    "abs": abs, "min": min, "max": max,
    "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
    "pi": np.pi, "e": np.e,
}


def _eval_expression(expr: str, context: Dict[str, Any]) -> Any:
    """Safely evaluate a $-prefixed expression."""
    # Replace each $name with context['name'] (longest names first)
    expr_resolved = expr
    for name in sorted(context.keys(), key=len, reverse=True):
        # Match $name[.subname]* word-boundary-ish
        pattern = re.escape("$" + name) + r"(?![a-zA-Z0-9_.])"
        if re.search(pattern, expr_resolved):
            value = context[name]
            expr_resolved = re.sub(pattern, repr(value), expr_resolved)
    # Now any remaining $name was unrecognized -- error early
    leftover = re.findall(r"\$[a-zA-Z_][a-zA-Z0-9_.]*", expr_resolved)
    if leftover:
        raise ValueError(
            f"Unresolved variables in expression {expr!r}: {leftover}. "
            f"Available: {sorted(context.keys())}"
        )
    return eval(expr_resolved, {"__builtins__": {}}, _ALLOWED_NAMES)


def _flatten(d: Dict, prefix: str) -> Dict[str, Any]:
    """Flatten nested dict to dotted-path keys."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full))
        else:
            out[full] = v
    return out


# =============================================================================
# Serialization: save raw + resolved configs to disk
# =============================================================================

def save_configs(
    raw_cfg: Dict, resolved_cfg: Dict, out_dir: Path
) -> None:
    """Save raw and resolved configs side-by-side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_raw.yaml", "w") as f:
        yaml.safe_dump(raw_cfg, f, sort_keys=False, default_flow_style=False)
    with open(out_dir / "config_resolved.json", "w") as f:
        json.dump(_to_jsonable(resolved_cfg), f, indent=2, default=str)


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy types etc. to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# =============================================================================
# Sub-seed derivation (deterministic from a single base seed)
# =============================================================================

def derive_subseeds(base_seed: int) -> Dict[str, int]:
    """Derive sub-seeds from one base seed.

    The 'feature' sub-seed is intentionally independent of 'model'. Per the
    paper's held-out independence convention, random initial features X are
    sampled once per base seed
    and then held fixed; in particular, multi-restart training (which only
    perturbs the optimizer init) must NOT resample X across restarts.
    """
    return {
        "graph":   int(base_seed) + 100003,
        "edge":    int(base_seed) + 200003,
        "mask":    int(base_seed) + 300007,
        "model":   int(base_seed) + 400009,
        "label":   int(base_seed) + 500011,
        "feature": int(base_seed) + 600013,
    }
