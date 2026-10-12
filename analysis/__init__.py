"""Post-hoc analysis package: load runs, compute bounds, fit rates, plot."""

from .load import load_experiment, load_final, list_experiments
from .bounds import (
    BoundConstants, compute_bounds,
    gamma_H_envelope, delta_mask_n,
    V_n_stat, V_n_stat_raw, V_n_stat_theorem,
    bound_score_mse_H_via_strong_convexity,
    bound_cor42_oracle, bound_cor42_plugin,
    bound_markov_oracle, bound_markov_plugin,
)
from .fits import FitResult, loglog_linfit, loglog_fit_with_bootstrap, fit_many_metrics
from .plots import (
    aggregate_by_sweep, plot_curve_iqr, plot_grouped_curves,
    plot_envelope_vs_measured, plot_ratio_to_rate, plot_constraint_hit_rates,
)
from .exp_norm_ablation import analyze_norm_ablation
from .exp_n_sweep import analyze_n_sweep
from .exp_sbm_sweep import analyze_sbm_sweep
from .exp_arch_sweep import analyze_arch_sweep
