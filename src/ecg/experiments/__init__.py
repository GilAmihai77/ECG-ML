"""Experiments: the run grid, its execution, and the results tables.

A study is built as data first (:mod:`ecg.experiments.plan`) and executed
second (:mod:`ecg.experiments.runner`), so the whole grid can be printed and
checked before any GPU time is spent, and so every run in it provably derives
from one base configuration.
"""

from __future__ import annotations

from ecg.experiments.plan import (
    RunSpec,
    ablation_plan,
    describe_plan,
    experiment_plan,
    pretrain_name,
    supervised_name,
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
    style_table,
    training_history,
)
from ecg.experiments.runner import (
    RunOutcome,
    Workspace,
    label_efficiency_table,
    results_frame,
    run_one,
    run_plan,
    ssl_benefit,
)

__all__ = [
    "RunOutcome",
    "RunPredictions",
    "RunSpec",
    "Workspace",
    "ablation_plan",
    "collect_predictions",
    "comparison_table",
    "describe_plan",
    "detailed_metrics",
    "experiment_plan",
    "label_distribution",
    "label_efficiency_table",
    "per_class_table",
    "plot_label_distribution",
    "plot_pr_curves",
    "plot_training_curves",
    "pretrain_name",
    "results_frame",
    "run_one",
    "run_plan",
    "ssl_benefit",
    "style_table",
    "supervised_name",
    "training_history",
]
