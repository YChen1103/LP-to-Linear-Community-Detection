# Real-graph experiment

This folder runs the masked link-prediction (LP) pipeline on a real graph. The
training code in `lp_sbm/` is used unchanged: `real/data.py` builds the same
data dictionary that the trainer reads for synthetic graphs.

## What is measured

Every `eval_every` epochs during training, the current model is scored on two
sides:

| Side | Quantity | Labels used |
|---|---|---|
| LP | held-out BCE (`R_H_emp`), held-out AUC and AP | none |
| Downstream | logistic-regression and plug-in probes with `m` labels per class, error over all nodes | `m` per class |

A real graph has no ground-truth edge probabilities. Population excess risk,
`mu_B`, `Delta_s`, the oracle classifier and `eta_L` are therefore not
computed, and `tests/test_real_pipeline.py` checks that they never appear in
the output.

`real/data.py` estimates an empirical block density matrix from the proxy
labels. It is used only to set the decoder norm budgets (`W_max`, `B_0`) with
the same rule as in the synthetic runs, and it does not enter the training
objective.

## Why a training trajectory

Across encoders, the final held-out BCE differs by a few percent, often within
seed-to-seed variation, so a comparison across encoders has little resolution.
A single training run moves held-out LP quality over a much wider range and
gives about 100 paired observations. Held-out and train BCE separate once a run
starts to overfit, so `analyze_trajectory.py` also reports the association on
the segment after the held-out minimum, separately for held-out and train BCE.

## Data

Only the adjacency and the labels are read. Node attributes are ignored.

- **email-Eu-core** (SNAP, <https://snap.stanford.edu/data/email-Eu-core.html>):
  `email-Eu-core.txt` and `email-Eu-core-department-labels.txt`, 1,005 nodes and
  42 departments. Use `--top-k` to keep the largest departments.
- **Cora** (LINQS): `cora.cites` and `cora.content`.
- **Other graphs**: `--source edgelist --edges-file E.txt --labels-file L.txt`,
  or `--source npz` with arrays `edges` of shape (E, 2) and `labels` of shape (n,).
- **No download**: `--source demo_sbm` generates a planted-partition graph.

By default the loader keeps the largest connected component
(`--no-largest-cc` disables this).

## Running

1. Check the pipeline on a small synthetic graph:

   ```bash
   python tests/test_real_pipeline.py
   python real/run_trajectory.py --source demo_sbm --demo-n 400 --demo-K 3 \
       --archs gin_pe --seeds 0 --max-epochs 60 --eval-every 20 --device cpu
   ```

2. Check the graph before training. `sanity_check.py` prints graph statistics,
   the spectral gap, and the errors of untrained baselines (raw spectral PE,
   adjacency spectral embedding and Laplacian PE with the same probe, plus
   spectral clustering) next to the error of the trivial classifier. It ends
   with `PASS`, `MARGINAL` or `FAIL`. `FAIL` means the untrained probe is close
   to the trivial classifier; use a denser graph or fewer classes.

   ```bash
   python real/sanity_check.py --source email_eu --root data/email-Eu-core \
       --top-k 5 --seeds 0-2 --out results/real/sanity_email_eu.json
   ```

3. Train and log the trajectories:

   ```bash
   python real/run_trajectory.py --source email_eu --root data/email-Eu-core \
       --top-k 5 --archs gin_pe --seeds 0,1,2 --device cuda \
       --out outputs/real_trajectory
   ```

   Each run writes `trajectory.jsonl` (one row per evaluation), `final.npz`,
   the raw and resolved configurations, and `graph_meta.json`.

4. Summarize:

   ```bash
   python real/analyze_trajectory.py --runs outputs/real_trajectory \
       --arch gin_pe --probe direct_m10_error_mean --out results/real/traj
   python analysis/synthetic_reference.py \
       --summary results/summary/summary_sweep_n.json
   ```

   `analyze_trajectory.py` prints a checkpoint table and the Spearman
   correlation between LP quality and probe error. `synthetic_reference.py`
   computes the same statistics on the synthetic sample-size sweep for
   comparison.

On sparse graphs the held-out BCE is dominated by non-edges and varies little;
held-out AUC (`auc_H`) is logged alongside for that case.
