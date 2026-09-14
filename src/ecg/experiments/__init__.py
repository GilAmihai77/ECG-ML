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
    "RunSpec",
    "Workspace",
    "ablation_plan",
    "describe_plan",
    "experiment_plan",
    "label_efficiency_table",
    "pretrain_name",
    "results_frame",
    "run_one",
    "run_plan",
    "ssl_benefit",
    "supervised_name",
]
