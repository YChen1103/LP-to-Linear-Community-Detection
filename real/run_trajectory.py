#!/usr/bin/env python3
"""
Real-graph trajectory runs (external-validity check).

One training run per (arch, seed). At every `eval_every` epochs the current
(z, W, b) is scored on both sides of the theorem:

    LP side         held-out BCE, held-out AUC/AP   (no labels)
    downstream side LogReg / plug-in / K-means probes at m labels per class

so that a single run yields ~100 paired observations spanning the whole range
from random initialisation to convergence. This is the spread the experiment
needs: on a fixed graph, architecture-to-architecture differences in held-out
LP loss are smaller than the seed-to-seed IQR (see
analysis/synthetic_reference.py), whereas the training trajectory spans it
by orders of magnitude.

Nothing in `lp_sbm/` is modified. `train_lp` consumes only
{n, K, rows, cols, train_mask, held_mask, A_flat, A} from the data dict, and
`compute_bias_init` is called with bias_init_mode="zero", which does not read
the oracle. The block matrix fitted in real/data.py is used ONLY to resolve
the capacity budget (W_max, B_0) through the repository's existing
`resolve_config`, so the decoder capacity is set on the same footing as in the
synthetic runs.

Usage
-----
    # offline smoke test: tiny graph, 60 epochs, CPU
    python real/run_trajectory.py --source demo_sbm --demo-n 400 --demo-K 3 \\
        --archs gin_pe --seeds 0 --max-epochs 60 --eval-every 20 --device cpu

    # the real thing
    python real/run_trajectory.py --source email_eu --root data/email-Eu-core \\
        --top-k 5 --archs gin_pe --seeds 0,1,2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lp_sbm.config import (                                              # noqa: E402
    derive_subseeds, expand_sweep, load_raw_config, resolve_config,
    save_configs, set_dotted,
)
from lp_sbm.logging_io import JSONLWriter, ensure_dir                    # noqa: E402
from lp_sbm.oracle import compute_sbm_oracle                             # noqa: E402
from lp_sbm.trainer import train_lp                                      # noqa: E402
from real.data import load_graph, prepare_real_data                      # noqa: E402
from real.metrics_real import compute_real_metrics                       # noqa: E402

CONFIG_PATH = ROOT / "configs" / "experiments" / "real_trajectory.yaml"


def parse_seeds(spec: str) -> List[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def settings_by_label(raw_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for _sid, cfg in expand_sweep(raw_cfg):
        out[str(cfg.get("_sweep_label", "default"))] = cfg
    return out


def run_one(*, arch: str, seed: int, cfg: Dict[str, Any],
            A: np.ndarray, labels: np.ndarray, K: int, meta: Dict[str, Any],
            out_root: Path, device: torch.device, args) -> Dict[str, Any]:
    sub = derive_subseeds(seed)
    data = prepare_real_data(A, labels, K, cfg["mask"], sub)

    # Fitted block model -> capacity budget only. See real/data.py docstring.
    oracle = compute_sbm_oracle(data["B_n"], data["pi"])
    resolved = resolve_config(cfg, oracle)

    out_dir = ensure_dir(out_root / f"{meta['source']}__{arch}" / f"seed_{seed:04d}")
    save_configs(cfg, resolved, out_dir)
    json.dump({k: v for k, v in meta.items()},
              open(out_dir / "graph_meta.json", "w"), indent=1)

    R = float(resolved["encoder"]["R"])
    W_max = float(resolved["decoder"]["W_max"])
    B_0 = float(resolved["decoder"]["B_0"])
    cls = resolved["classifiers"]
    m_list = list(cls["direct"]["m_per_class"])
    n_res = int(cls["direct"]["n_label_resamples"])
    cs = int(sub["label"])

    traj = JSONLWriter(out_dir / "trajectory.jsonl")
    t_eval = [0.0]

    def record(epoch: int, z, W, b, extra: Dict[str, Any], is_final: bool) -> None:
        t0 = time.time()
        m = compute_real_metrics(
            np.asarray(z, dtype=np.float64), np.asarray(W, dtype=np.float64),
            float(b), data=data, classifier_seed=cs,
            m_list=m_list, n_resamples=n_res,
            do_plugin=bool(cls["plug_in"]["enabled"]),
            do_kmeans=bool(cls["kmeans"]["enabled"]) and is_final,
            kmeans_n_init=int(cls["kmeans"].get("n_init", 20)),
            M_hat=oracle["M_star"],
        )
        m.update(epoch=int(epoch), is_final=bool(is_final), arch=arch, seed=int(seed),
                 source=meta["source"], n=int(meta["n"]), K=int(meta["K"]),
                 mean_degree=float(meta["mean_degree"]), tau=float(data["tau"]),
                 R=R, W_max=W_max, B_0=B_0)
        m.update({k: v for k, v in extra.items()
                  if isinstance(v, (int, float, bool, str))})
        traj.write(m)
        t_eval[0] += time.time() - t0

    # The left end of the trajectory is the first periodic eval (epoch
    # eval_every), which is still essentially the untrained network. The
    # untrained *representation* anchor -- raw spectral PE with the same probe,
    # no GNN and no LP training -- comes from real/sanity_check.py, which uses
    # the same raw positional-encoding baselines as scripts/pe_baselines.py.
    t0 = time.time()
    try:
        result = train_lp(data=data, sbm_oracle=oracle, resolved_cfg=resolved,
                          seeds=sub, device=device,
                          eval_callback=lambda *, epoch, z, W, b, extra:
                              record(epoch, z, W, b, extra, False))
    except Exception:
        traceback.print_exc()
        traj.close()
        return {"status": "failed", "arch": arch, "seed": seed}
    elapsed = time.time() - t0

    record(int(result["best_epoch"]), result["z"], result["W"], float(result["b"]),
           extra={"best_val_loss": float(result["best_val_loss"])}, is_final=True)
    traj.close()

    np.savez_compressed(out_dir / "final.npz", z=result["z"], W=result["W"],
                        b=np.array([result["b"]]), labels=data["labels"])
    print(f"  [{arch} seed={seed}] train={elapsed - t_eval[0]:.0f}s "
          f"eval={t_eval[0]:.0f}s  -> {out_dir}", flush=True)
    return {"status": "completed", "arch": arch, "seed": seed,
            "elapsed_s": elapsed, "out_dir": str(out_dir)}



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="demo_sbm",
                    choices=["demo_sbm", "email_eu", "cora", "edgelist", "npz"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--edges-file", default=None)
    ap.add_argument("--labels-file", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--no-largest-cc", action="store_true")
    ap.add_argument("--demo-n", type=int, default=1000)
    ap.add_argument("--demo-K", type=int, default=3)
    ap.add_argument("--archs", type=str, default="gin_pe")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/real_trajectory")
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    seeds = parse_seeds(args.seeds)

    A, labels, K, meta = load_graph(
        args.source, root=args.root, edges_file=args.edges_file,
        labels_file=args.labels_file, top_k=args.top_k,
        largest_cc=not args.no_largest_cc,
        demo_n=args.demo_n, demo_K=args.demo_K)
    print(f"graph: {meta['source']}  n={meta['n']}  K={meta['K']}  "
          f"mean_degree={meta['mean_degree']:.2f}  "
          f"homophily={meta['edge_homophily']:.3f}")

    raw = load_raw_config(CONFIG_PATH)
    by_label = settings_by_label(raw)
    unknown = [a for a in archs if a not in by_label]
    if unknown:
        raise SystemExit(f"unknown arch(s) {unknown}; available: {sorted(by_label)}")

    dev_str = args.device or raw.get("runtime", {}).get("device", "cpu")
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("  cuda requested but unavailable; falling back to cpu")
        dev_str = "cpu"
    device = torch.device(dev_str)

    out_root = Path(args.out)
    results = []
    for arch in archs:
        for seed in seeds:
            cfg = json.loads(json.dumps(by_label[arch]))  # deep copy
            set_dotted(cfg, "sbm.n", int(meta["n"]))
            set_dotted(cfg, "sbm.K", int(meta["K"]))
            if args.tau is not None:
                set_dotted(cfg, "mask.tau", float(args.tau))
            if args.eval_every is not None:
                set_dotted(cfg, "training.eval_every", int(args.eval_every))
            if args.max_epochs is not None:
                set_dotted(cfg, "training.max_epochs", int(args.max_epochs))
            set_dotted(cfg, "training.restarts", 1)
            results.append(run_one(arch=arch, seed=seed, cfg=cfg, A=A, labels=labels,
                                   K=K, meta=meta, out_root=out_root,
                                   device=device, args=args))

    ensure_dir(out_root)
    json.dump(results, open(out_root / "runs.json", "w"), indent=1)
    ok = sum(r["status"] == "completed" for r in results)
    print(f"\n{ok}/{len(results)} runs completed -> {out_root}/")


if __name__ == "__main__":
    main()
