"""Experiments: the run grid, its execution, and the results tables.

A study is built as data first (:mod:`ecg.experiments.plan`) and executed
second (:mod:`ecg.experiments.runner`), so the whole grid can be printed and
checked before any GPU time is spent, and so every run in it provably derives
from one base configuration.
"""

from __future__ import annotations

from ecg.experiments.plan import (
    DEFAULT_SSL_BUDGETS,
    RunSpec,
    ablation_plan,
    describe_plan,
    experiment_plan,
    budget_name,
    pretrain_name,
    seed_name,
    ssl_budget_plan,
    supervised_name,
    variant_name,
)
from ecg.experiments.report import (
    RunPredictions,
    collect_predictions,
    comparison_table,
    detailed_metrics,
    label_distribution,
    per_class_table,
    plot_label_distribution,
    plot_pr_curves,
    plot_training_curves,
    select_seed,
    style_table,
    training_history,
)
from ecg.experiments.runner import (
    RunOutcome,
    Workspace,
    aggregate_runs,
    embedder_benefit,
    format_mean_sd,
    label_efficiency_report,
    label_efficiency_table,
    load_results,
    paired_contrast,
    results_frame,
    run_one,
    run_plan,
    ssl_benefit,
)

__all__ = [
    "DEFAULT_SSL_BUDGETS",
    "RunOutcome",
    "RunPredictions",
    "RunSpec",
    "Workspace",
    "ablation_plan",
    "aggregate_runs",
    "budget_name",
    "collect_predictions",
    "comparison_table",
    "describe_plan",
    "detailed_metrics",
    "embedder_benefit",
    "experiment_plan",
    "format_mean_sd",
    "label_distribution",
    "label_efficiency_report",
    "label_efficiency_table",
    "load_results",
    "paired_contrast",
    "per_class_table",
    "plot_label_distribution",
    "plot_pr_curves",
    "plot_training_curves",
    "pretrain_name",
    "results_frame",
    "run_one",
    "run_plan",
    "seed_name",
    "select_seed",
    "ssl_benefit",
    "ssl_budget_plan",
    "style_table",
    "supervised_name",
    "training_history",
    "variant_name",
]
