"""Sanity tests for lp_sbm.config (loading, sweep expansion, resolution)."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp_sbm.config import (
    load_raw_config, expand_sweep, resolve_config, derive_subseeds,
    get_dotted, set_dotted, _eval_expression, _flatten,
)


# A minimal raw config that doesn't extend anything (so we can test in isolation)
def _minimal_raw():
    return {
        "experiment": {"name": "t"},
        "seeds": {"list": [0]},
        "sbm": {"n": 100, "K": 3, "rho_n": 1.0, "pi": "uniform",
                "block_spec": {"type": "assortative", "p_in": 0.7, "p_out": 0.2}},
        "mask": {"tau": 0.3},
        "encoder": {
            "family": "centered_power", "d": 8, "hidden_dim": 16,
            "num_layers": 2, "use_layernorm": True,
            "features": {"mode": "ones", "dim": 4},
            "R_spec": {"mode": "fixed", "value": 1.0},
        },
        "decoder": {
            "family": "bilinear", "symmetric": True,
            "W_max_spec": {"mode": "from_capacity_ratio", "capacity_ratio": 2.0},
            "B_0_spec":   {"mode": "from_b_star", "slack": 0.5},
        },
        "training": {"lr": 0.01, "max_epochs": 10, "patience": 5,
                     "restarts": 1, "eval_every": 0,
                     "bias_init_mode": "b_star"},
        "classifiers": {
            "oracle":  {"enabled": True},
            "plug_in": {"enabled": False, "m_per_class": [5], "n_label_resamples": 2},
            "direct":  {"enabled": False, "m_per_class": [5], "n_label_resamples": 2},
            "kmeans":  {"enabled": False, "n_init": 5},
        },
        "output": {"base_dir": "outputs"},
        "runtime": {"device": "cpu"},
        "sweep": {"axis": None, "values": []},
    }


def _fake_oracle():
    """A fake SBM oracle dict for resolve_config testing."""
    return {
        "B_star":            1.5,
        "b_star":            -0.8,
        "capacity_required": 2.0,
        "Delta_s":           1.5,
        "Delta_s_sq":        2.25,
    }


# =============================================================================
# Dotted path access
# =============================================================================

def test_get_set_dotted():
    d = {"a": {"b": {"c": 42}}}
    assert get_dotted(d, "a.b.c") == 42
    set_dotted(d, "a.b.c", 99)
    assert d["a"]["b"]["c"] == 99
    set_dotted(d, "a.x.y", "hello")
    assert d["a"]["x"]["y"] == "hello"


# =============================================================================
# Sweep expansion
# =============================================================================

def test_expand_no_sweep():
    raw = _minimal_raw()
    out = expand_sweep(raw)
    assert len(out) == 1
    assert out[0][0] == 0


def test_expand_simple_sweep():
    raw = _minimal_raw()
    raw["sweep"] = {"axis": "sbm.n", "values": [100, 200, 500]}
    out = expand_sweep(raw)
    assert len(out) == 3
    assert out[0][1]["sbm"]["n"] == 100
    assert out[2][1]["sbm"]["n"] == 500


def test_expand_with_coupled_raw():
    """coupled (raw) is evaluated at sweep expansion time."""
    raw = _minimal_raw()
    raw["sweep"] = {"axis": "sbm.n",
                    "values": [100, 200],
                    "coupled": {"sbm.K": "$sbm.n // 50"}}
    out = expand_sweep(raw)
    assert out[0][1]["sbm"]["K"] == 2
    assert out[1][1]["sbm"]["K"] == 4


# =============================================================================
# Resolution
# =============================================================================

def test_resolve_fixed_R():
    raw = _minimal_raw()
    resolved = resolve_config(raw, _fake_oracle())
    assert resolved["encoder"]["R"] == 1.0


def test_resolve_W_from_capacity_ratio():
    raw = _minimal_raw()
    resolved = resolve_config(raw, _fake_oracle())
    # W_max R^2 = capacity_ratio * capacity_required
    # = 2.0 * 2.0 = 4.0; with R=1.0, W_max = 4.0
    assert abs(resolved["decoder"]["W_max"] - 4.0) < 1e-6
    assert abs(resolved["_derived"]["capacity_ratio_actual"] - 2.0) < 1e-6


def test_resolve_B_0_from_b_star():
    raw = _minimal_raw()
    resolved = resolve_config(raw, _fake_oracle())
    # B_0 = |b_star| + slack = 0.8 + 0.5 = 1.3
    assert abs(resolved["decoder"]["B_0"] - 1.3) < 1e-6


def test_resolve_derived_quantities():
    raw = _minimal_raw()
    resolved = resolve_config(raw, _fake_oracle())
    d = resolved["_derived"]
    expected_B_model = 1.3 + 4.0 * 1.0 ** 2  # B_0 + W_max R^2
    assert abs(d["B_model"] - expected_B_model) < 1e-6
    # B = max(B_model, B_star) = max(5.3, 1.5) = 5.3
    assert abs(d["B"] - expected_B_model) < 1e-6
    # mu_B = sigma(B)(1-sigma(B))
    s = 1.0 / (1.0 + np.exp(-expected_B_model))
    assert abs(d["mu_B_at_B"] - s * (1 - s)) < 1e-9


def test_coupled_after_derive_resolves_W():
    """coupled_after_derive on W_max_spec.value takes effect before W_max resolution."""
    raw = _minimal_raw()
    raw["encoder"]["R_spec"] = {"mode": "fixed", "value": 2.0}
    raw["decoder"]["W_max_spec"] = {"mode": "fixed", "value": 999.0}  # should be overridden
    raw["sweep"] = {
        "axis": None, "values": [],
        "coupled_after_derive": {
            "decoder.W_max_spec.value": "2.0 * $capacity_required / $encoder.R_spec.value**2"
        }
    }
    resolved = resolve_config(raw, _fake_oracle())
    # 2 * 2.0 / 4.0 = 1.0
    assert abs(resolved["decoder"]["W_max"] - 1.0) < 1e-6


def test_eval_expression_basic():
    ctx = {"a": 2.0, "b": 3.0, "encoder.R": 0.5}
    assert _eval_expression("$a + $b", ctx) == 5.0
    assert abs(_eval_expression("sqrt($a * $b)", ctx) - np.sqrt(6.0)) < 1e-9
    assert _eval_expression("$encoder.R * 4", ctx) == 2.0


def test_eval_expression_unresolved_raises():
    try:
        _eval_expression("$missing + 1", {"a": 1.0})
        raised = False
    except ValueError:
        raised = True
    assert raised


# =============================================================================
# YAML loading + extends:
# =============================================================================

def test_extends_chain(tmp_path=None):
    """`extends:` resolves relative to the file's parent dir."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "parent.yaml").write_text(yaml.safe_dump({
        "experiment": {"name": "p"},
        "sbm": {"n": 50, "K": 2},
    }))
    (tmp / "child.yaml").write_text(yaml.safe_dump({
        "extends": "parent.yaml",
        "experiment": {"name": "c"},
        "sbm": {"n": 100},  # override n; keep K from parent
    }))
    cfg = load_raw_config(tmp / "child.yaml")
    assert cfg["experiment"]["name"] == "c"   # child wins
    assert cfg["sbm"]["n"] == 100             # child wins
    assert cfg["sbm"]["K"] == 2               # inherited from parent


# =============================================================================
# Sub-seeds
# =============================================================================

def test_subseeds_distinct():
    s = derive_subseeds(0)
    keys = ["graph", "edge", "mask", "model", "label"]
    for k in keys:
        assert k in s
    vals = [s[k] for k in keys]
    assert len(set(vals)) == len(vals), "Sub-seeds should be distinct"


if __name__ == "__main__":
    test_get_set_dotted()
    test_expand_no_sweep()
    test_expand_simple_sweep()
    test_expand_with_coupled_raw()
    test_resolve_fixed_R()
    test_resolve_W_from_capacity_ratio()
    test_resolve_B_0_from_b_star()
    test_resolve_derived_quantities()
    test_coupled_after_derive_resolves_W()
    test_eval_expression_basic()
    test_eval_expression_unresolved_raises()
    test_extends_chain()
    test_subseeds_distinct()
    print("All config tests passed.")
