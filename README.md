# From Link Prediction to Linear Community Detection

Code for the NeurIPS 2026 paper *From Link Prediction to Linear Community
Detection: A Finite-Sample Guarantee for Graph Pretraining*.

The code trains graph neural network (GNN) encoders with a symmetric bilinear
decoder by masked link prediction (LP) on stochastic block models (SBMs). For
each trained model it records the held-out LP risk, the score error with
respect to the SBM oracle logits, and the community-classification error of
linear classifiers built on the learned node embeddings (oracle, plug-in,
logistic regression and k-means). The `real/` folder runs the same pipeline on
a real graph.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10 or later and PyTorch 2.0 or later. The experiments were
run with CUDA 12.x; training also runs on CPU (`--device cpu`).

## Repository layout

```
lp_sbm/            SBM sampling, masking, encoders, decoder, training, metrics
  classifiers/     oracle, plug-in, logistic-regression and k-means classifiers
configs/
  base.yaml        default configuration, with every option documented
  experiments/     one file per experiment
scripts/           experiment drivers, baselines, summary extraction
analysis/          bound envelopes, slope fits and plots computed from logs
real/              real-graph experiment (see real/README.md)
tests/             unit and end-to-end tests
```

## Synthetic experiments

Each sweep has one configuration file in `configs/experiments/` and one driver
in `scripts/`. All sweeps use 10 seeds per setting.

| Sweep | What varies | Config | Driver |
|---|---|---|---|
| Sample size | `n` from 100 to 2000, `K` in {2, 3, 4}, five encoders | `main_sweep_n.yaml` | `exp_main_sweep_n.py` |
| Separation | `Delta_s`, through `p_in`, at `K = 3` | `sweep_delta.yaml` | `exp_sweep_delta.py` |
| Decoder capacity | `W_max`, through the capacity ratio | `ablation_W_capratio.yaml` | `exp_ablation_W_capratio.py` |
| Bias cap | `B_0` | `ablation_B0_slack.yaml` | `exp_ablation_B0_slack.py` |
| Edge density | `rho_n` | `ablation_rho_sparsity.yaml` | `exp_ablation_rho_sparsity.py` |

```bash
python scripts/exp_main_sweep_n.py --device cuda   # full sweep
python scripts/exp_main_sweep_n.py --smoke         # one seed, two values of n
```

Each run writes its logs to `outputs/<experiment_name>/setting_<id>/seed_<k>/`.
Run any driver with `--help` to see the options for restricting seeds, sizes
or encoders.

### Encoders and input features

| Label | Encoder | Input features |
|---|---|---|
| `gin_pe` | GIN, 2 layers | spectral PE: the 8 eigenvectors of the masked adjacency `A_enc` with the largest absolute eigenvalues, scaled by `sqrt(n)` |
| `gcn_pe` | GCN, 2 layers | Laplacian PE: the 8 eigenvectors of the normalized Laplacian of `A_enc` with the smallest nontrivial eigenvalues, each column standardized |
| `gat_pe` | GAT, 2 layers, 4 heads | Laplacian PE, as for `gcn_pe` |
| `gin_random` | GIN, 2 layers | i.i.d. Gaussian features, 16 dimensions |
| `cp_random` | power-iteration encoder, 10 layers | i.i.d. Gaussian features, 16 dimensions |

All graph-derived features are computed from `A_enc`, which excludes the
held-out pairs. The embedding dimension is 16. The norm constraints are
enforced in the forward pass: embeddings are projected onto the ball of radius
`R = 1.5`, the eigenvalues of the symmetric decoder matrix `W` are clipped to
`[-W_max, W_max]`, and the bias is parameterized as `B_0 * tanh(b_raw)`.

### Summaries

`scripts/extract_summary.py` compresses the output folder of one sweep into a
single JSON file with the final metrics of every seed and per-setting medians,
quartiles and other aggregates:

```bash
python scripts/extract_summary.py outputs/main_sweep_n --out summary_sweep_n.json
```

The summary files for the runs reported in the paper are in
`results/summary/`.

## Baselines without LP training

`scripts/pe_baselines.py` evaluates representations that involve no LP
training: the raw 8-dimensional spectral PE, the same PE truncated to `K`
dimensions, the adjacency spectral embedding, and the Laplacian PE. It uses the
same graphs, masks, labeled sets and probes as the trained models, and also
reports spectral clustering and k-means. `scripts/untrained_control.py`
evaluates the GIN+PE encoder at initialization, before any gradient step.

```bash
python scripts/pe_baselines.py --K 3 --n-list 500,1000,2000 --seeds 0-9 \
    --summary results/summary/summary_sweep_n.json --include-lap
python scripts/untrained_control.py --ns 500 1000 2000 --device cuda
```

## Real-graph experiment

`real/` runs masked LP pretraining on the SNAP email-Eu-core network, with
department labels as proxy communities. It logs held-out LP quality and
few-label probe error along the training trajectory. Quantities that require
the true edge probabilities are not computed. See `real/README.md`.

## Tests

```bash
python tests/run_all.py
```

## Compute

The synthetic sweeps took about 250 GPU-hours on two NVIDIA L40S GPUs. One seed
of the largest setting (`n = 2000`, `K = 4`) takes 5 to 10 minutes on one L40S.
