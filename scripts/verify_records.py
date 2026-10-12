#!/usr/bin/env python3
"""
Compare a freshly produced PE-baseline records.json (scripts/pe_baselines.py)
against a reference one.

Usage
-----
    python verify_records.py NEW/records.json REF/records.json
    python verify_records.py NEW/records.json REF/records.json --tol 1e-12

Checks
------
 1. the two files cover exactly the same (n, seed, rep, probe, m) cells
 2. per-cell errors agree (exact by default; --tol allows a tolerance)
 3. the median-over-seeds aggregate agrees at 3 decimals, i.e. the published
    table.md numbers are reproduced

Exit code 0 iff everything matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

KEY = ("n", "seed", "rep", "probe", "m")


def load(path):
    recs = json.load(open(path))
    return {tuple(r[k] for k in KEY): r for r in recs}


def aggregate(d):
    """Median over seeds -> {(n, rep, probe, m): median error}."""
    b = defaultdict(list)
    for (n, seed, rep, probe, m), r in d.items():
        e = r["error"]
        if e == e:  # skip NaN
            b[(n, rep, probe, m)].append(e)
    return {k: float(np.median(v)) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("ref")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="allowed |new-ref| per cell (default 0 = bit-exact)")
    ap.add_argument("--allow-subset", action="store_true",
                    help="new may cover only part of ref (partial / smoke run)")
    args = ap.parse_args()

    new, ref = load(args.new), load(args.ref)
    ok = True

    only_new = sorted(set(new) - set(ref))
    only_ref = sorted(set(ref) - set(new))
    print(f"records: new={len(new)}  ref={len(ref)}")
    if only_new or (only_ref and not args.allow_subset):
        ok = False
    if only_new or only_ref:
        print(f"  !! cells only in new: {len(only_new)} e.g. {only_new[:3]}")
        print(f"  !! cells only in ref: {len(only_ref)} e.g. {only_ref[:3]}")
    shared = sorted(set(new) & set(ref))
    if not shared:
        print("no overlapping cells - nothing to compare")
        return 1

    diffs = [(k, ref[k]["error"], new[k]["error"]) for k in shared
             if abs(ref[k]["error"] - new[k]["error"]) > args.tol]
    worst = max(abs(ref[k]["error"] - new[k]["error"]) for k in shared)
    print(f"per-cell: compared {len(shared)}  mismatches={len(diffs)}  "
          f"max|diff|={worst:.3e}  (tol={args.tol:g})")
    for k, a, b in diffs[:15]:
        print(f"   {dict(zip(KEY, k))}  ref={a:.6f}  new={b:.6f}  d={b - a:+.2e}")
    if diffs:
        ok = False

    # Medians are computed on the SHARED cells only, so that a partial run
    # (e.g. 3 of the 10 seeds) is still compared against the same seed subset.
    an = aggregate({k: new[k] for k in shared})
    ar = aggregate({k: ref[k] for k in shared})
    shared_agg = sorted(set(an) & set(ar))
    bad = [(k, ar[k], an[k]) for k in shared_agg
           if f"{ar[k]:.3f}" != f"{an[k]:.3f}"]
    print(f"table cells (median over seeds, 3 decimals): compared "
          f"{len(shared_agg)}  mismatches={len(bad)}")
    for k, a, b in bad[:15]:
        print(f"   n={k[0]} rep={k[1]} probe={k[2]} m={k[3]}  "
              f"ref={a:.3f}  new={b:.3f}")
    if bad:
        ok = False

    print("\nRESULT:", "MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
