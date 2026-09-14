"""Models: the two patch embedders and the Transformer encoder they share.

Every arm of the experiment is one :class:`~ecg.models.config.ModelConfig` with
a different ``embedder`` field, and arms C and D add a
:class:`~ecg.models.config.SslConfig` that is identical between them. Building
models through this package is how integrity rule 3 is kept.
"""

from __future__ import annotations

from ecg.models.config import ModelConfig, SslConfig
from ecg.models.embeddings import (
    ConvPatchEmbedding,
    LinearPatchEmbedding,
    build_embedder,
    patchify,
)
from ecg.models.encoder import (
    EcgClassifier,
    EcgEncoder,
    build_classifier,
    count_parameters,
    embedder_parity,
    parameter_report,
)
from ecg.models.masking import (
    apply_mask,
    contaminated_fraction,
    mask_summary,
    reach_of,
    sample_block_mask,
)
from ecg.models.ssl import (
    MaskedReconstruction,
    SslStep,
    baseline_loss,
    build_pretrainer,
    transfer_encoder,
)

__all__ = [
    "ConvPatchEmbedding",
    "EcgClassifier",
    "EcgEncoder",
    "LinearPatchEmbedding",
    "MaskedReconstruction",
    "ModelConfig",
    "SslConfig",
    "SslStep",
    "apply_mask",
    "baseline_loss",
    "build_classifier",
    "build_embedder",
    "build_pretrainer",
    "contaminated_fraction",
    "count_parameters",
    "embedder_parity",
    "mask_summary",
    "parameter_report",
    "patchify",
    "reach_of",
    "sample_block_mask",
    "transfer_encoder",
]
