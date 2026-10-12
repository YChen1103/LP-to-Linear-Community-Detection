#!/usr/bin/env python3
"""
Pre-flight check for the real-graph trajectory experiment. NO TRAINING.

Run this BEFORE spending GPU time. It answers one question: does this graph,
under this masking rate, contain community signal that a *linear* probe can
read out of the encoder's own input? If the untrained baseline is already at
the trivial-classifier level, then no LP-training trajectory can be
informative on this graph, and the right move is to pick a denser one --- not
to train for an afternoon and report a flat line.

It reuses exactly the representations and probes from
`scripts/pe_baselines.py`, so the numbers here are directly
comparable to the raw positional-encoding baselines computed there.

Usage
-----
    # offline smoke test, no download needed
    python real/sanity_check.py --source demo_sbm --seeds 0-2

    # SNAP email-Eu-core (n=1005, mean degree ~33)
    python real/sanity_check.py --source email_eu --root data/email-Eu-core \\
        --top-k 5 --seeds 0-2

    # LINQS Cora (n=2708, mean degree ~3.9); node attributes are ignored
    python real/sanity_check.py --source cora --root data/cora --seeds 0-2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lp_sbm.config import derive_subseeds                                # noqa: E402
from lp_sbm.classifiers.sampling import sample_stratified_label_sets     # noqa: E402
from real.data import load_graph, prepare_real_data                      # noqa: E402
from scripts.pe_baselines import (                              # noqa: E402
    adjacency_spectral_embedding, dense_A_enc, laplacian_pe,
    probe_kmeans, probe_logreg, spectral_pe,
)


def parse_seeds(spec: str) -> List[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def run(args) -> Dict[str, object]:
    A, labels, K, meta = load_graph(
        args.source, root=args.root, edges_file=args.edges_file,
        labels_file=args.labels_file, top_k=args.top_k,
        largest_cc=not args.no_largest_cc,
        demo_n=args.demo_n, demo_K=args.demo_K,
    )

    print("=" * 68)
    print("GRAPH")
    print("=" * 68)
    for k in ("source", "n", "K", "n_edges", "mean_degree", "density",
              "majority_class_frac", "edge_homophily"):
        v = meta[k]
        print(f"  {k:<22}{v:.4f}" if isinstance(v, float) else f"  {k:<22}{v}")
    print(f"  {'class_sizes':<22}{meta['class_sizes']}")

    trivial_majority = 1.0 - float(meta["majority_class_frac"])
    trivial_uniform = 1.0 - 1.0 / K
    print(f"\n  trivial error, predict majority class : {trivial_majority:.3f}")
    print(f"  trivial error, uniform random guess   : {trivial_uniform:.3f}")

    per_seed: Dict[str, Dict[int, List[float]]] = {}
    gaps: List[float] = []

    for seed in args.seeds:
        sub = derive_subseeds(seed)
        data = prepare_real_data(A, labels, K, {"tau": args.tau}, sub)
        A_enc = dense_A_enc(data)
        cs = int(sub["label"])

        w = np.linalg.eigvalsh(A_enc)
        aw = np.sort(np.abs(w))[::-1]
        if len(aw) > K:
            gaps.append(float((aw[K - 1] - aw[K]) / max(aw[K - 1], 1e-12)))

        reps = {
            f"specPE_d{args.pe_dim}": spectral_pe(A_enc, args.pe_dim),
            "specPE_dK": spectral_pe(A_enc, K),
            "ASE_K": adjacency_spectral_embedding(A_enc, K),
            f"lapPE_d{args.pe_dim}": laplacian_pe(A_enc, args.pe_dim),
        }
        for m in args.m_list:
            Ls = sample_stratified_label_sets(labels, K, int(m), args.n_resamples,
                                              seed=cs + int(m) * 1009)
            for name, X in reps.items():
                e = probe_logreg(X, labels, K, Ls, cs + int(m) * 1009 + 1)
                per_seed.setdefault(name, {}).setdefault(m, []).append(e)
        for name in ("specPE_dK", "ASE_K"):
            err, _ = probe_kmeans(reps[name], labels, K, cs + 2,
                                  n_init=args.kmeans_n_init)
            per_seed.setdefault(name, {}).setdefault(0, []).append(err)
        print(f"  ... seed {seed} done", flush=True)

    def med(name, m):
        v = per_seed.get(name, {}).get(m, [])
        return float(np.median(v)) if v else float("nan")

    print("\n" + "=" * 68)
    print(f"UNTRAINED BASELINES  (median over {len(args.seeds)} seeds, "
          f"tau={args.tau}, error over all n nodes)")
    print("=" * 68)
    head = "  " + f"{'representation':<24}" + "".join(f"m={m:<8}" for m in args.m_list)
    print(head)
    for name in per_seed:
        cells = "".join(f"{med(name, m):<10.3f}" for m in args.m_list)
        print(f"  {name:<24}{cells}")
    print(f"\n  spectral clustering (no labels)   : {med('specPE_dK', 0):.3f}")
    print(f"  K-means on ASE      (no labels)   : {med('ASE_K', 0):.3f}")
    if gaps:
        print(f"  spectral gap (|l_K|-|l_K+1|)/|l_K|: {np.median(gaps):.3f}")

    best10 = min(med(nm, 10) for nm in per_seed if 10 in per_seed[nm]) \
        if any(10 in v for v in per_seed.values()) else float("nan")
    best_unsup = min(med("specPE_dK", 0), med("ASE_K", 0))

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    ratio = best10 / trivial_majority if trivial_majority > 0 else float("nan")
    print(f"  best untrained probe error @ m=10 : {best10:.3f}")
    print(f"  as a fraction of trivial error    : {ratio:.2f}")
    print(f"  best unsupervised error           : {best_unsup:.3f}")
    if ratio < 0.6:
        verdict = "PASS -- clear linearly readable community signal"
    elif ratio < 0.85:
        verdict = "MARGINAL -- signal exists but is weak; expect a noisy trajectory"
    else:
        verdict = ("FAIL -- untrained probe is near the trivial classifier. "
                   "Pick a denser graph or fewer/larger classes (--top-k).")
    print(f"\n  {verdict}\n")

    out = {"meta": meta, "tau": args.tau, "seeds": args.seeds,
           "trivial_majority": trivial_majority, "trivial_uniform": trivial_uniform,
           "median_spectral_gap": float(np.median(gaps)) if gaps else None,
           "baselines": {nm: {str(m): med(nm, m) for m in per_seed[nm]}
                         for nm in per_seed},
           "best_probe_m10": best10, "ratio_to_trivial": ratio,
           "verdict": verdict}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"  written to {args.out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="demo_sbm",
                    choices=["demo_sbm", "email_eu", "cora", "edgelist", "npz"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--edges-file", default=None)
    ap.add_argument("--labels-file", default=None)
    ap.add_argument("--top-k", type=int, default=None,
                    help="keep only the top-k largest classes")
    ap.add_argument("--no-largest-cc", action="store_true")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--pe-dim", type=int, default=8)
    ap.add_argument("--m-list", type=str, default="1,5,10,20")
    ap.add_argument("--seeds", type=str, default="0-2")
    ap.add_argument("--n-resamples", type=int, default=5)
    ap.add_argument("--kmeans-n-init", type=int, default=20)
    ap.add_argument("--demo-n", type=int, default=1000)
    ap.add_argument("--demo-K", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    args.m_list = [int(x) for x in args.m_list.split(",")]
    args.seeds = parse_seeds(args.seeds)
    run(args)


if __name__ == "__main__":
    main()
