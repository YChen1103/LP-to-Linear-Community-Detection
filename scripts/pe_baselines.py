#!/usr/bin/env python3
"""
Baselines on the raw positional encodings.

Question: does LP pretraining add anything beyond the spectral positional
encoding that is already fed to the encoder?

This script evaluates *untrained* representations derived from exactly the same
A_enc that the encoder saw, under exactly the same labeled sets L that the
paper's plug-in / direct classifiers used, and (optionally) re-probes the
LP-trained embeddings z from final.npz with the same probes.

No GNN training is performed here. The SBM realization, the mask, and the
labeled sets are all reproduced deterministically from the base seed via the
repository's own sampling code, so every number is paired seed-by-seed with the
numbers already in summary_sweep_n.json.

Representations
---------------
    specPE_d{D}   sqrt(n) * top-D eigenvectors of A_enc by |lambda|
                  D = encoder.features.dim (8 in the paper's runs)  -> the
                  encoder's ACTUAL input
    specPE_dK     the same construction truncated at D = K           -> the
                  rank of P; the classical spectral-recovery choice and the
                  object analyzed in the spectral-PE GIN approximation result
    ASE_K         U_K |Lambda_K|^{1/2}                               -> adjacency
                  spectral embedding (Sussman et al., 2012)
    lapPE_d{D}    normalized-Laplacian PE, std-scaled (what GCN+PE / GAT+PE
                  actually received)                     [--include-lap]
    LP_z          LP-trained embeddings from final.npz   [--npz-root]

Probes (all evaluated on all n nodes, matching the main theorem's (1/n) sum_i 1{.})
------
    logreg        multinomial logistic regression, protocol-identical to
                  lp_sbm/classifiers/direct.py
    centroid      Euclidean nearest centroid on the same L
    kmeans        unsupervised, Hungarian-aligned; reported separately since it
                  has no label budget

Usage
-----
    python scripts/pe_baselines.py \
        --K 3 --n-list 500,1000,2000 --seeds 0-9 \
        --summary /path/to/summary_sweep_n.json \
        --npz-root /path/to/outputs/main_sweep_n \
        --out results/pe_baselines

    # protocol self-check: recompute direct-LogReg on the trained z and compare
    # against the logged direct_m*_error_mean (requires --npz-root)
    python scripts/pe_baselines.py --verify --npz-root ... --summary ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lp_sbm.sbm import prepare_sbm_data, build_A_enc_edge_index          # noqa: E402
from lp_sbm.config import derive_subseeds                                 # noqa: E402
from lp_sbm.classifiers.sampling import sample_stratified_label_sets      # noqa: E402
from lp_sbm.classifiers.kmeans import _hungarian_align_and_eval           # noqa: E402

from sklearn.linear_model import LogisticRegression                       # noqa: E402
from sklearn.cluster import KMeans                                        # noqa: E402


# =============================================================================
# Data reproduction
# =============================================================================

def rebuild_realization(n: int, K: int, seed: int, *, p_in: float, p_out: float,
                        rho_n: float, tau: float) -> Dict[str, object]:
    """Reproduce one (graph, edges, mask) realization exactly as scripts/run.py does."""
    sub = derive_subseeds(seed)
    sbm_cfg = {
        "n": n, "K": K, "rho_n": rho_n, "pi": "uniform",
        "block_spec": {"type": "assortative", "p_in": p_in, "p_out": p_out},
    }
    data = prepare_sbm_data(sbm_cfg, {"tau": tau}, sub)
    data["sub_seeds"] = sub
    return data


def dense_A_enc(data: Dict[str, object]) -> np.ndarray:
    """Dense symmetric hollow A_enc, built from the same edge_index the encoder got.

    Mirrors lp_sbm/features.py::_build_dense_A_enc so that the PE computed here
    is bit-identical to the PE the encoder consumed.
    """
    ei = build_A_enc_edge_index(data["A"], data["train_mask"],
                                data["rows"], data["cols"])
    n = int(data["n"])
    A = np.zeros((n, n), dtype=np.float32)
    if ei.shape[1] > 0:
        A[ei[0], ei[1]] = 1.0
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


# =============================================================================
# Representations
# =============================================================================

def spectral_pe(A_enc: np.ndarray, dim: int) -> np.ndarray:
    """sqrt(n) * top-`dim` eigenvectors of A_enc by |lambda| (as in features.py)."""
    n = A_enc.shape[0]
    w, v = np.linalg.eigh(A_enc)
    idx = np.argsort(np.abs(w))[::-1][:min(int(dim), n)]
    return (v[:, idx] * np.sqrt(n)).astype(np.float64)


def adjacency_spectral_embedding(A_enc: np.ndarray, K: int) -> np.ndarray:
    """U_K |Lambda_K|^{1/2}: textbook adjacency spectral embedding (ASE)."""
    w, v = np.linalg.eigh(A_enc)
    idx = np.argsort(np.abs(w))[::-1][:K]
    return (v[:, idx] * np.sqrt(np.abs(w[idx]))[None, :]).astype(np.float64)


def laplacian_pe(A_enc: np.ndarray, dim: int) -> np.ndarray:
    """Normalized-Laplacian PE with 'std' column scaling (features.py default)."""
    n = A_enc.shape[0]
    deg = A_enc.sum(axis=1)
    d_inv_sqrt = np.power(np.clip(deg, 1e-6, None), -0.5)
    L = np.eye(n) - d_inv_sqrt[:, None] * A_enc * d_inv_sqrt[None, :]
    _, v = np.linalg.eigh(L)
    k = min(int(dim), n - 1)
    pe = v[:, 1:k + 1]
    pe = pe / (pe.std(axis=0, keepdims=True) + 1e-6)
    return pe.astype(np.float64)


def load_lp_embedding(npz_root: Optional[str], sid_label: str, seed: int) -> Optional[np.ndarray]:
    """Load trained z from a run's final.npz. Returns None when unavailable.

    Layout produced by lp_sbm/logging_io.py:
        <npz_root>/setting_<sid:04d>_<sanitized label>/seed_<seed:04d>/final.npz
    The label is sanitized by dropping commas and replacing '/' and ' ' with '_',
    and truncated to 80 chars, so we match on the sanitized prefix.
    """
    if npz_root is None:
        return None
    root = Path(npz_root)
    safe = sid_label.replace("/", "_").replace(" ", "_").replace(",", "")[:80]
    hits = sorted(root.glob(f"setting_*_{safe}/seed_{seed:04d}/final.npz"))
    if not hits:  # tolerate truncated or renamed setting dirs
        hits = [p for p in root.rglob(f"seed_{seed:04d}/final.npz")
                if safe in p.parent.parent.name]
    if not hits:
        return None
    with np.load(hits[0]) as f:
        return np.asarray(f["z"], dtype=np.float64)


# =============================================================================
# Probes
# =============================================================================

def probe_logreg(X: np.ndarray, labels: np.ndarray, K: int,
                 label_sets: List[Dict[int, np.ndarray]], base_seed: int) -> float:
    """Mean error over label resamples. Protocol-identical to direct.py.

    direct_classify passes seed_for_lr = base_seed + r for resample r, fits on
    the labeled subset and predicts on all n nodes.
    """
    errs = []
    for r, L in enumerate(label_sets):
        tr = np.concatenate([L[a] for a in range(K) if len(L.get(a, [])) > 0]) \
            if any(len(L.get(a, [])) > 0 for a in range(K)) else np.array([], dtype=np.int64)
        if len(tr) == 0 or len(np.unique(labels[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, random_state=int(base_seed) + r)
        clf.fit(X[tr], labels[tr])
        errs.append(float(np.mean(clf.predict(X) != labels)))
    return float(np.mean(errs)) if errs else float("nan")


def probe_centroid(X: np.ndarray, labels: np.ndarray, K: int,
                   label_sets: List[Dict[int, np.ndarray]]) -> float:
    """Euclidean nearest-centroid on the same L. Mean error over resamples."""
    errs = []
    for L in label_sets:
        if any(len(L.get(a, [])) == 0 for a in range(K)):
            continue
        cents = np.stack([X[L[a]].mean(axis=0) for a in range(K)])
        d2 = ((X ** 2).sum(1)[:, None] - 2 * X @ cents.T + (cents ** 2).sum(1)[None, :])
        errs.append(float(np.mean(d2.argmin(axis=1) != labels)))
    return float(np.mean(errs)) if errs else float("nan")


def probe_kmeans(X: np.ndarray, labels: np.ndarray, K: int, seed: int,
                 n_init: int = 50) -> Tuple[float, float]:
    """Unsupervised K-means + Hungarian alignment, as in classifiers/kmeans.py."""
    pred = KMeans(n_clusters=K, n_init=n_init, random_state=int(seed)).fit_predict(X)
    res = _hungarian_align_and_eval(pred, labels, K)
    return float(res["error_rate"]), float(res["nmi"])


# =============================================================================
# Driver
# =============================================================================

def run_cell(n: int, K: int, seed: int, args) -> List[Dict[str, object]]:
    data = rebuild_realization(n, K, seed, p_in=args.p_in, p_out=args.p_out,
                               rho_n=args.rho_n, tau=args.tau)
    labels = data["labels"]
    A_enc = dense_A_enc(data)
    classifier_seed = int(data["sub_seeds"]["label"])

    reps: Dict[str, np.ndarray] = {
        f"specPE_d{args.pe_dim}": spectral_pe(A_enc, args.pe_dim),
        "specPE_dK": spectral_pe(A_enc, K),
        "ASE_K": adjacency_spectral_embedding(A_enc, K),
    }
    if args.include_lap:
        reps[f"lapPE_d{args.pe_dim}"] = laplacian_pe(A_enc, args.pe_dim)

    z = load_lp_embedding(args.npz_root, f"K={K}__n={n}__{args.lp_arch}", seed)
    if z is not None:
        reps["LP_z"] = z

    out: List[Dict[str, object]] = []
    for m in args.m_list:
        # Identical to metrics.py: one shared draw of L per m, decoupled by m*1009.
        label_sets = sample_stratified_label_sets(
            labels, K, int(m), int(args.n_resamples),
            seed=classifier_seed + int(m) * 1009,
        )
        lr_seed = classifier_seed + int(m) * 1009 + 1   # direct.py convention
        for name, X in reps.items():
            out.append(dict(n=n, K=K, seed=seed, rep=name, probe="logreg", m=m,
                            error=probe_logreg(X, labels, K, label_sets, lr_seed)))
            out.append(dict(n=n, K=K, seed=seed, rep=name, probe="centroid", m=m,
                            error=probe_centroid(X, labels, K, label_sets)))

    for name in ("specPE_dK", "ASE_K"):
        err, nmi = probe_kmeans(reps[name], labels, K, classifier_seed + 2,
                                n_init=args.kmeans_n_init)
        out.append(dict(n=n, K=K, seed=seed, rep=name, probe="kmeans_unsup",
                        m=0, error=err, nmi=nmi))
    return out


# =============================================================================
# Comparison rows pulled from the existing summary
# =============================================================================

LP_KEYS = {
    "oracle": "oracle_classifier_error",
    "plug_in": "plug_in_m{m}_error_mean",
    "direct": "direct_m{m}_error_mean",
    "kmeans_z": "kmeans_z_euclid_error",
}


def lp_reference(summary_path: str, K: int, n_list: List[int], arch: str,
                 m_list: List[int]) -> Dict[Tuple[int, str, int], float]:
    """Median-over-seeds LP-trained numbers already reported in the paper."""
    d = json.load(open(summary_path))
    ref: Dict[Tuple[int, str, int], float] = {}
    for s in d["settings"]:
        for n in n_list:
            if s["sweep_label"] != f"K={K}__n={n}__{arch}":
                continue
            def med(key):
                v = [sd["metrics"][key] for sd in s["seeds"] if key in sd["metrics"]]
                return float(np.median(v)) if v else float("nan")
            ref[(n, "oracle", 0)] = med(LP_KEYS["oracle"])
            ref[(n, "kmeans_z", 0)] = med(LP_KEYS["kmeans_z"])
            for m in m_list:
                ref[(n, "plug_in", m)] = med(LP_KEYS["plug_in"].format(m=m))
                ref[(n, "direct", m)] = med(LP_KEYS["direct"].format(m=m))
    return ref


# =============================================================================
# Reporting
# =============================================================================

def aggregate(records: List[Dict[str, object]]) -> Dict[Tuple, float]:
    """Median over seeds of the per-seed mean-over-resamples error."""
    buckets: Dict[Tuple, List[float]] = {}
    for r in records:
        if r["error"] == r["error"]:  # not NaN
            buckets.setdefault((r["n"], r["rep"], r["probe"], r["m"]), []).append(r["error"])
    return {k: float(np.median(v)) for k, v in buckets.items()}


def markdown_table(agg, ref, args) -> str:
    ns = args.n_list
    ms = args.m_list
    lines = []
    lines.append(f"Median over {len(args.seeds)} seeds, K={args.K}, "
                 f"p_in={args.p_in}, p_out={args.p_out}, tau={args.tau}. "
                 f"Error is over all n nodes. Probe: logistic regression.")
    lines.append("")
    head = "| representation | " + " | ".join(
        f"n={n}, m={m}" for n in ns for m in ms) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (1 + len(ns) * len(ms)))

    rep_order = [f"specPE_d{args.pe_dim}", "specPE_dK", "ASE_K"]
    if args.include_lap:
        rep_order.append(f"lapPE_d{args.pe_dim}")
    if any(k[1] == "LP_z" for k in agg):
        rep_order.append("LP_z")

    label = {
        f"specPE_d{args.pe_dim}": f"raw spectral PE (d={args.pe_dim}, encoder input)",
        "specPE_dK": f"raw spectral PE (d=K={args.K})",
        "ASE_K": "ASE  U_K|L_K|^(1/2)",
        f"lapPE_d{args.pe_dim}": f"raw Laplacian PE (d={args.pe_dim})",
        "LP_z": "LP-trained z (same probe)",
    }
    for rep in rep_order:
        cells = [f"{agg.get((n, rep, 'logreg', m), float('nan')):.3f}"
                 for n in ns for m in ms]
        lines.append(f"| {label.get(rep, rep)} | " + " | ".join(cells) + " |")

    if ref:
        for nm, disp in (("plug_in", "LP-trained, plug-in (main theorem, Part II)"),
                         ("direct", "LP-trained z, LogReg (paper)")):
            cells = [f"{ref.get((n, nm, m), float('nan')):.3f}" for n in ns for m in ms]
            lines.append(f"| {disp} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Unsupervised reference (no label budget, Hungarian-aligned):")
    lines.append("")
    lines.append("| method | " + " | ".join(f"n={n}" for n in ns) + " |")
    lines.append("|" + "---|" * (1 + len(ns)))
    for rep, disp in (("specPE_dK", "spectral clustering on top-K eigenvectors"),
                      ("ASE_K", "K-means on ASE")):
        cells = [f"{agg.get((n, rep, 'kmeans_unsup', 0), float('nan')):.3f}" for n in ns]
        lines.append(f"| {disp} | " + " | ".join(cells) + " |")
    if ref:
        lines.append("| K-means on LP-trained z (paper) | " +
                     " | ".join(f"{ref.get((n, 'kmeans_z', 0), float('nan')):.3f}"
                                for n in ns) + " |")
        lines.append("| LP-trained oracle classifier (paper) | " +
                     " | ".join(f"{ref.get((n, 'oracle', 0), float('nan')):.3f}"
                                for n in ns) + " |")
    return "\n".join(lines)


def verify(args) -> None:
    """Protocol self-check: recompute direct-LogReg on trained z, compare to log."""
    ref = lp_reference(args.summary, args.K, args.n_list, args.lp_arch, args.m_list)
    print("Recomputed vs logged direct-LogReg on the trained embeddings")
    print(f"{'n':>6} {'m':>4} {'recomputed':>12} {'logged':>10} {'|diff|':>10}")
    for n in args.n_list:
        per_seed: Dict[int, List[Tuple[int, float]]] = {}
        for seed in args.seeds:
            z = load_lp_embedding(args.npz_root, f"K={args.K}__n={n}__{args.lp_arch}", seed)
            if z is None:
                print(f"  n={n} seed={seed}: final.npz not found, skipping")
                continue
            data = rebuild_realization(n, args.K, seed, p_in=args.p_in, p_out=args.p_out,
                                       rho_n=args.rho_n, tau=args.tau)
            cs = int(data["sub_seeds"]["label"])
            for m in args.m_list:
                Ls = sample_stratified_label_sets(data["labels"], args.K, m,
                                                  args.n_resamples, seed=cs + m * 1009)
                e = probe_logreg(z, data["labels"], args.K, Ls, cs + m * 1009 + 1)
                per_seed.setdefault(m, []).append((seed, e))
        for m in args.m_list:
            if m not in per_seed:
                continue
            got = float(np.median([e for _, e in per_seed[m]]))
            want = ref.get((n, "direct", m), float("nan"))
            print(f"{n:>6} {m:>4} {got:>12.6f} {want:>10.6f} {abs(got - want):>10.2e}")


def parse_seeds(spec: str) -> List[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--n-list", type=str, default="500,1000,2000")
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--m-list", type=str, default="1,5,10,20")
    ap.add_argument("--pe-dim", type=int, default=8,
                    help="encoder.features.dim used in the paper's runs")
    ap.add_argument("--p-in", type=float, default=0.10)
    ap.add_argument("--p-out", type=float, default=0.04)
    ap.add_argument("--rho-n", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--n-resamples", type=int, default=5)
    ap.add_argument("--kmeans-n-init", type=int, default=50)
    ap.add_argument("--include-lap", action="store_true",
                    help="also evaluate the Laplacian PE that GCN+PE / GAT+PE received")
    ap.add_argument("--lp-arch", type=str, default="gin_pe")
    ap.add_argument("--npz-root", type=str, default=None,
                    help="experiment output dir containing per-run final.npz")
    ap.add_argument("--summary", type=str, default=None,
                    help="summary_sweep_n.json, for the LP-trained comparison rows")
    ap.add_argument("--out", type=str, default="results/pe_baselines")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    args.n_list = [int(x) for x in args.n_list.split(",")]
    args.m_list = [int(x) for x in args.m_list.split(",")]
    args.seeds = parse_seeds(args.seeds)

    if args.verify:
        if not (args.npz_root and args.summary):
            ap.error("--verify requires both --npz-root and --summary")
        verify(args)
        return

    records: List[Dict[str, object]] = []
    for n in args.n_list:
        for seed in args.seeds:
            records.extend(run_cell(n, args.K, seed, args))
            print(f"  done n={n} seed={seed}", flush=True)

    agg = aggregate(records)
    ref = lp_reference(args.summary, args.K, args.n_list, args.lp_arch,
                       args.m_list) if args.summary else {}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(out_dir / "records.json", "w"), indent=1)
    json.dump({"|".join(map(str, k)): v for k, v in agg.items()},
              open(out_dir / "aggregate.json", "w"), indent=1)
    table = markdown_table(agg, ref, args)
    (out_dir / "table.md").write_text(table)
    print("\n" + table)
    print(f"\nwritten to {out_dir}/")


if __name__ == "__main__":
    main()
