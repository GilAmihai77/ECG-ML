"""Run configuration: everything needed to reproduce an experiment.

Integrity rules 4, 5 and 6 in one place. A :class:`RunConfig` names the data,
the model, the SSL settings, the optimisation budget and every seed; it round
-trips through YAML, and a copy is written into each checkpoint. Nothing about
a run should live only in a notebook cell or a shell history.

The optimisation settings sit in :class:`TrainConfig` and are deliberately
shared across arms. Arms A and B must differ only in ``model.embedder``; arms C
and D must differ only in that and in having a pretrained encoder. If the
learning rate or the epoch budget moved between arms, the comparison would be
about tuning rather than about the embedder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ecg.models.config import ModelConfig, SslConfig

#: Variant name for the original architecture. Runs in the default variant are
#: named exactly as they always were, so a study that already finished still
#: resumes instead of re-running under a new name.
DEFAULT_VARIANT: str = "base"


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation budget, identical across the arms being compared.

    Attributes:
        epochs: Passes over the training cohort.
        batch_size: Records per step.
        lr: Peak AdamW learning rate.
        weight_decay: AdamW decoupled weight decay.
        grad_clip: Global gradient-norm clip; ``0`` disables it.
        min_lr_ratio: Cosine floor as a fraction of ``lr``.
        seed: Seed for weights, shuffling and masks.
        amp: Use bfloat16 autocast on CUDA. Ignored on CPU. bf16 needs no
            gradient scaler, unlike fp16, so there is no scaler to misconfigure.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        eval_every: Epochs between validation passes. Supervised training only;
            pretraining evaluates its holdout every epoch.
        patience: Stop after this many evaluations without improvement; ``0``
            disables early stopping. Applies to both loops -- supervised
            training watches val macro AUROC, pretraining watches the held-out
            reconstruction loss -- so a run of either kind can end before
            ``epochs``. Because ``eval_every`` is supervised-only, this counts
            evaluations there and epochs in pretraining; at the default
            ``eval_every=1`` those are the same thing.

            It is a safety net, not a substitute for choosing ``epochs``. The
            cosine schedule is spread across ``epochs``, so a large budget
            stretches the decay rather than merely capping the run, and a run
            that stops at a fifth of its budget never reaches its low-rate
            phase. Both loops warn when that happens.
    """

    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.01
    seed: int = 0
    amp: bool = True
    device: str = "auto"
    eval_every: int = 1
    patience: int = 0

    def __post_init__(self) -> None:
        """Validate the budget.

        Raises:
            ValueError: If epochs, batch size or evaluation interval is not
                positive, or the learning rate is not positive.
        """
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be positive, got {self.eval_every}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")


@dataclass(frozen=True)
class RunConfig:
    """A complete, reproducible experiment.

    Attributes:
        name: Run name, used for MLflow and the checkpoint directory.
        variant: Which version of the architecture this run belongs to, e.g.
            ``"deep6"``. It prefixes every run name, so two variants never
            share an output directory and never collide in MLflow. Pick a slug
            that says what changed; a version ladder stops meaning anything
            after the fourth change. The label alone does not prove two runs
            ran the same code -- see
            :func:`ecg.training.tracking.git_provenance` for the part that
            does.
        model: Architecture, including which embedder this arm uses.
        train: Optimisation budget.
        ssl: Masking settings; only read by pretraining runs.
        store_path: Waveform store directory.
        metadata_path: PTB-XL root, for the metadata frame.
        label_fraction: Fraction of the labelled training cohort to use, for
            the label-efficiency curve.
        subset_seed: Seed for the nested-subset permutation. Held fixed across
            arms so every arm sees the *same* records at a given fraction.
        ssl_holdout: Fraction of the SSL pool held out to watch reconstruction
            loss. Taken from the pool itself, not from val, so pretraining
            never touches a cohort used for model selection.
        pretrained_from: Checkpoint to initialise the encoder from. ``None``
            for arms A and B.
        experiment: MLflow experiment name.
        tracking_uri: MLflow tracking database. Local by default and copied to
            durable storage at each checkpoint -- see
            :func:`ecg.training.checkpoints.sync_tracking`. A ``.db`` path
            selects the SQLite backend, which is one file rather than the file
            store's thousands.
        output_dir: Where checkpoints are written. On Colab this is the Drive
            path; it is the only Drive write in the loop.
    """

    name: str = "run"
    variant: str = DEFAULT_VARIANT
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ssl: SslConfig = field(default_factory=SslConfig)
    store_path: str = "data/store_100hz"
    metadata_path: str = "data/ptbxl"
    label_fraction: float = 1.0
    subset_seed: int = 0
    ssl_holdout: float = 0.05
    pretrained_from: str | None = None
    experiment: str = "ecg-ssl"
    tracking_uri: str = "mlruns/mlflow.db"
    output_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        """Validate the run.

        Raises:
            ValueError: If ``label_fraction`` is outside ``(0, 1]``,
                ``ssl_holdout`` is outside ``[0, 1)``, or ``variant`` is not
                usable as a directory name.
        """
        # The variant becomes part of a run name, and a run name becomes a
        # directory on Drive. Rejecting separators here turns a confusing
        # mid-study path error into an immediate one.
        if not self.variant or any(bad in self.variant for bad in "/\\ "):
            raise ValueError(
                f"variant must be a non-empty slug without spaces or path "
                f"separators, got {self.variant!r}"
            )
        if not 0.0 < self.label_fraction <= 1.0:
            raise ValueError(
                f"label_fraction must be in (0, 1], got {self.label_fraction}"
            )
        if not 0.0 <= self.ssl_holdout < 1.0:
            raise ValueError(
                f"ssl_holdout must be in [0, 1), got {self.ssl_holdout}"
            )

    @property
    def arm(self) -> str:
        """Short arm label, e.g. ``"conv+ssl"``, for logs and run names."""
        suffix = "+ssl" if self.pretrained_from else ""
        return f"{self.model.embedder}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        """Render as plain nested dictionaries and lists.

        Returns:
            A structure ``yaml.safe_dump`` accepts.
        """
        return asdict(self)

    def to_yaml(self, path: str | Path) -> Path:
        """Write the configuration to a YAML file.

        Args:
            path: Destination.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
        return destination

    def with_(self, **changes: Any) -> RunConfig:
        """Return a copy with fields replaced.

        Args:
            **changes: Fields to override.

        Returns:
            A new configuration.
        """
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Rebuild a configuration from plain data.

        YAML has no tuple type, so the nested sequences come back as lists and
        are coerced here. Without that, a loaded config would compare unequal
        to the one that was saved and the checkpoint-config equality check in
        :func:`ecg.models.ssl.transfer_encoder` would reject valid transfers.

        Args:
            data: Mapping as produced by :meth:`to_dict`.

        Returns:
            The configuration.
        """
        payload = dict(data)
        model = dict(payload.pop("model", {}))
        if model.get("conv_hidden") is not None:
            model["conv_hidden"] = tuple(tuple(layer) for layer in model["conv_hidden"])
        if model.get("conv_final") is not None:
            model["conv_final"] = tuple(model["conv_final"])
        return cls(
            model=ModelConfig(**model),
            train=TrainConfig(**payload.pop("train", {})),
            ssl=SslConfig(**payload.pop("ssl", {})),
            **payload,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load a configuration from a YAML file.

        Args:
            path: Source file.

        Returns:
            The configuration.
        """
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def mlflow_params(self) -> dict[str, Any]:
        """Flatten to scalars for MLflow parameter logging.

        Returns:
            A flat mapping; nested configs are prefixed with ``model.``,
            ``train.`` and ``ssl.``.
        """
        params: dict[str, Any] = {"arm": self.arm}
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                for inner, inner_value in value.items():
                    params[f"{key}.{inner}"] = inner_value
            else:
                params[key] = value
        return params
