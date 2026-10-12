#!/usr/bin/env python3
"""
Torch-free tests for the real-graph path (real/).

Covers everything up to and including the call into `train_lp`: graph loading,
the data dict, the fitted block model, capacity resolution through the
repository's own `resolve_config`, and the real-graph metrics. Training itself
needs torch and is exercised by `real/run_trajectory.py --device cpu` on the
demo graph.

    python tests/test_real_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lp_sbm.config import (derive_subseeds, expand_sweep, load_raw_config,
                           resolve_config, set_dotted)
from lp_sbm.oracle import compute_sbm_oracle
from lp_sbm.sbm import build_A_enc_edge_index
from real.data import fit_block_matrix, load_graph, prepare_real_data
from real.metrics_real import compute_real_metrics, lp_metrics, pair_scores

CONFIG = ROOT / "configs" / "experiments" / "real_trajectory.yaml"
SBM_KEYS = {"n", "K", "tau", "rho_n", "pi", "B_block", "B_n", "labels", "P",
            "A", "rows", "cols", "train_mask", "held_mask", "P_flat", "A_flat"}


def test_load_graph():
    A, labels, K, meta = load_graph("demo_sbm", demo_n=300, demo_K=3)
    assert A.shape == (len(labels), len(labels))
    assert np.allclose(A, A.T) and np.allclose(np.diag(A), 0)
    assert set(np.unique(labels)) == set(range(K))
    assert 0.0 < meta["density"] < 1.0 and meta["mean_degree"] > 0
    assert 0.0 <= meta["edge_homophily"] <= 1.0
    print("  load_graph                 ok  "
          f"(n={meta['n']}, K={K}, deg={meta['mean_degree']:.1f})")
    return A, labels, K, meta


def test_top_k_filter():
    A, labels, K, _ = load_graph("demo_sbm", demo_n=300, demo_K=4, top_k=2)
    assert K == 2, f"top_k=2 should leave 2 classes, got {K}"
    print("  top-k class filter         ok")


def test_data_dict(A, labels, K):
    data = prepare_real_data(A, labels, K, {"tau": 0.2}, derive_subseeds(0))
    assert SBM_KEYS <= set(data), f"missing keys: {SBM_KEYS - set(data)}"
    n, N = data["n"], len(data["rows"])
    assert N == n * (n - 1) // 2
    assert data["train_mask"].sum() + data["held_mask"].sum() == N
    assert not (data["train_mask"] & data["held_mask"]).any()
    assert np.allclose(data["A_flat"], data["A"][data["rows"], data["cols"]])
    ei = build_A_enc_edge_index(data["A"], data["train_mask"],
                                data["rows"], data["cols"])
    assert ei.shape[0] == 2 and ei.shape[1] > 0
    # every encoder-visible edge must be a real edge on a visible pair
    assert data["A"][ei[0], ei[1]].min() == 1.0
    print(f"  data dict / mask / A_enc   ok  (|H|={int(data['held_mask'].sum())}, "
          f"E_enc={ei.shape[1] // 2})")
    return data


def test_block_fit(A, labels, K):
    B, pi = fit_block_matrix(A, labels, K)
    assert B.shape == (K, K) and np.allclose(B, B.T)
    assert (B > 0).all() and (B < 1).all()
    assert abs(pi.sum() - 1.0) < 1e-9
    # planted partition: within-block density should exceed between-block
    assert np.diag(B).mean() > (B.sum() - np.trace(B)) / max(K * K - K, 1)
    print(f"  block fit                  ok  (diag={np.diag(B).mean():.4f}, "
          f"offdiag={(B.sum() - np.trace(B)) / max(K * K - K, 1):.4f})")
    return B, pi


def test_capacity_resolution(data):
    raw = load_raw_config(CONFIG)
    settings = {c.get("_sweep_label"): c for _sid, c in expand_sweep(raw)}
    assert {"gin_pe", "gcn_pe", "gat_pe", "gin_random", "cp_random"} <= set(settings)
    cfg = json.loads(json.dumps(settings["gin_pe"]))
    set_dotted(cfg, "sbm.n", data["n"])
    set_dotted(cfg, "sbm.K", data["K"])
    assert cfg["encoder"]["features"]["mode"] == "spectral_pe"
    assert cfg["encoder"]["features"]["dim"] == 8
    assert cfg["training"]["eval_every"] > 0
    assert cfg["training"]["bias_init_mode"] == "zero"

    oracle = compute_sbm_oracle(data["B_n"], data["pi"])
    for k in ("M_star", "b_star", "B_star", "capacity_required",
              "Delta_s", "Delta_s_sq", "H_matrix"):
        assert k in oracle, k
    res = resolve_config(cfg, oracle)
    R = float(res["encoder"]["R"])
    W = float(res["decoder"]["W_max"])
    B0 = float(res["decoder"]["B_0"])
    assert all(np.isfinite([R, W, B0])) and R > 0 and W > 0 and B0 > 0
    print(f"  capacity resolution        ok  (R={R:.3f}, W_max={W:.3f}, B_0={B0:.3f})")
    return res, oracle


def test_metrics(data, oracle, res):
    n, d = data["n"], int(res["encoder"]["d"])
    rng = np.random.default_rng(0)
    # a representation that is informative but not perfect
    z = np.eye(data["K"])[data["labels"]] @ rng.normal(size=(data["K"], d))
    z = z + 0.7 * rng.normal(size=(n, d))
    W = np.eye(d) * 0.3
    b = -2.0

    s = pair_scores(z, W, b, data["rows"], data["cols"])
    S = z @ W @ z.T + b
    assert np.allclose(s, S[data["rows"], data["cols"]], atol=1e-8), \
        "pair_scores must match the dense score matrix"

    lp = lp_metrics(z, W, b, data)
    assert np.isfinite(lp["R_T_emp"]) and np.isfinite(lp["R_H_emp"])
    assert 0.0 <= lp["auc_H"] <= 1.0
    assert lp["auc_H"] > 0.5, "informative z should beat chance ranking"

    m = compute_real_metrics(z, W, b, data=data, classifier_seed=7,
                             m_list=[1, 10], n_resamples=3,
                             do_plugin=True, do_kmeans=True, kmeans_n_init=5,
                             M_hat=oracle["M_star"])
    for k in ("direct_m1_error_mean", "direct_m10_error_mean",
              "plug_in_m10_error_mean", "kmeans_z_error", "auc_H", "R_H_emp"):
        assert k in m and np.isfinite(m[k]), k
    assert 0.0 <= m["direct_m10_error_mean"] <= 1.0
    assert m["direct_m10_error_mean"] < m["direct_m1_error_mean"] + 0.2
    # no oracle-dependent quantity should leak into the real-graph metrics
    forbidden = {"excess_pop_H", "R_H_star", "mu_B", "Delta_s",
                 "oracle_classifier_error", "eta_L_mean"}
    assert not (forbidden & set(m)), forbidden & set(m)
    print(f"  real metrics               ok  (AUC={m['auc_H']:.3f}, "
          f"err@m=10 {m['direct_m10_error_mean']:.3f}, "
          f"kmeans {m['kmeans_z_error']:.3f})")


def main() -> None:
    print("real-graph pipeline tests")
    A, labels, K, _meta = test_load_graph()
    test_top_k_filter()
    data = test_data_dict(A, labels, K)
    test_block_fit(A, labels, K)
    res, oracle = test_capacity_resolution(data)
    test_metrics(data, oracle, res)
    print("All real-pipeline tests passed.")


if __name__ == "__main__":
    main()
