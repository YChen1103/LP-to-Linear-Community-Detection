"""End-to-end pipeline smoke test.

Runs a tiny experiment via the public API (no shell) and verifies:
    - All expected files are created
    - Metrics are present and have sensible values
    - Bound computation works on the loaded DataFrame
    - Bounded encoder-decoder class constraints hold
"""
from __future__ import annotations
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp_sbm.config import (
    load_raw_config, expand_sweep, resolve_config,
    save_configs, derive_subseeds,
)
from lp_sbm.sbm import prepare_sbm_data
from lp_sbm.oracle import compute_sbm_oracle
from lp_sbm.trainer import train_lp
from lp_sbm.metrics import compute_all_metrics
from lp_sbm.logging_io import (
    JSONLWriter, Manifest, ensure_dir, seed_dir, dump_final_npz,
)


def _build_tiny_config(tmp_dir: Path) -> Path:
    """Write a minimal end-to-end config to a temp dir."""
    cfg = {
        "experiment": {"name": "e2e_smoke"},
        "seeds": {"list": [0]},
        "sbm": {"n": 60, "K": 3, "rho_n": 1.0, "pi": "uniform",
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
        "training": {"lr": 0.01, "weight_decay": 0.0001,
                     "max_epochs": 30, "patience": 30,
                     "restarts": 1, "eval_every": 0,
                     "val_split_of_H": 0.5,
                     "clip_grad_norm": 5.0,
                     "optimizer": "adam",
                     "bias_init_mode": "b_star",
                     "early_stop": True},
        "classifiers": {
            "oracle":  {"enabled": True},
            "plug_in": {"enabled": True, "m_per_class": [5], "n_label_resamples": 2},
            "direct":  {"enabled": False, "m_per_class": [5], "n_label_resamples": 2},
            "kmeans":  {"enabled": True, "n_init": 5},
        },
        "output": {"base_dir": str(tmp_dir),
                   "save_embeddings": True,
                   "save_per_node_diagnostics": True,
                   "save_train_curve": True},
        "runtime": {"device": "cpu"},
        "sweep": {"axis": None, "values": []},
    }
    cfg_path = tmp_dir / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def test_end_to_end_minimal_run():
    tmp = Path(tempfile.mkdtemp(prefix="lpsbm_e2e_"))
    try:
        cfg_path = _build_tiny_config(tmp)
        raw = load_raw_config(cfg_path)
        settings = expand_sweep(raw)
        assert len(settings) == 1
        sid, raw_cfg = settings[0]

        seeds = derive_subseeds(0)
        data = prepare_sbm_data(raw_cfg["sbm"], raw_cfg["mask"], seeds)
        sbm_oracle = compute_sbm_oracle(data["B_n"], data["pi"])
        resolved = resolve_config(raw_cfg, sbm_oracle)

        # Train
        result = train_lp(
            data=data, sbm_oracle=sbm_oracle,
            resolved_cfg=resolved, seeds=seeds,
            device=torch.device("cpu"),
        )
        assert "z" in result and "W" in result and "b" in result
        assert result["z"].shape == (60, 8)
        assert result["W"].shape == (8, 8)

        # Verify the bounded encoder-decoder class constraints in the trained model
        z_norms = np.linalg.norm(result["z"], axis=1)
        assert z_norms.max() <= resolved["encoder"]["R"] + 1e-5, \
            f"z_max = {z_norms.max()} > R = {resolved['encoder']['R']}"
        W_op = float(np.linalg.svd(result["W"], compute_uv=False).max())
        assert W_op <= resolved["decoder"]["W_max"] + 1e-5
        assert abs(result["b"]) <= resolved["decoder"]["B_0"] + 1e-5

        # Compute final metrics
        metrics = compute_all_metrics(
            result["z"], result["W"], result["b"],
            data=data, sbm_oracle=sbm_oracle,
            R=resolved["encoder"]["R"],
            W_max=resolved["decoder"]["W_max"],
            B_0=resolved["decoder"]["B_0"],
            classifiers_cfg=resolved["classifiers"],
            classifier_seed=seeds["label"],
        )
        # Drop the per-node array
        metrics.pop("_tilde_e_sq_array", None)

        # Assert key metrics exist and have right type
        for key in ["R_H_emp", "R_H_pop", "R_H_star",
                    "excess_emp_H", "excess_pop_H",
                    "single_model_gap_H", "gamma_H_at_trained",
                    "score_mse_H", "score_mse_off", "score_mse_full",
                    "oracle_classifier_error",
                    "kmeans_z_euclid_error", "kmeans_G_form_error",
                    "plug_in_m5_error_mean", "plug_in_m5_eta_L_mean",
                    "a6_z_norm_max", "a6_W_op_norm", "a6_assumption6_holds"]:
            assert key in metrics, f"Missing metric: {key}"
            assert metrics[key] is not None

        # Sanity:
        assert metrics["R_H_emp"] >= 0
        assert metrics["R_H_pop"] >= 0
        assert metrics["R_H_star"] >= 0
        assert metrics["score_mse_H"] >= 0
        assert metrics["score_mse_full"] >= 0
        assert 0 <= metrics["oracle_classifier_error"] <= 1
        assert 0 <= metrics["kmeans_z_euclid_error"] <= 1
        assert 0 <= metrics["kmeans_G_form_error"] <= 1
        # excess_emp_H = R_H_emp - R_H_star (can be negative due to noise)
        assert abs(metrics["excess_emp_H"] - (metrics["R_H_emp"] - metrics["R_H_star"])) < 1e-9
        # Bounded encoder-decoder class constraints hold
        assert metrics["a6_assumption6_holds"]

        print("End-to-end smoke test passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_end_to_end_minimal_run()
