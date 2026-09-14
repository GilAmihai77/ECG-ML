"""Models: the two patch embedders and the Transformer encoder they share.

Every arm of the experiment is one :class:`~ecg.models.config.ModelConfig` with
a different ``embedder`` field. Nothing else about the model may vary between
arms, so building them through this package is the way integrity rule 3 is kept.
"""

from __future__ import annotations

from ecg.models.config import ModelConfig
from ecg.models.embeddings import (
    ConvPatchEmbedding,
    LinearPatchEmbedding,
    build_embedder,
)
from ecg.models.encoder import (
    EcgClassifier,
    EcgEncoder,
    build_classifier,
    count_parameters,
    embedder_parity,
    parameter_report,
)

__all__ = [
    "ConvPatchEmbedding",
    "EcgClassifier",
    "EcgEncoder",
    "LinearPatchEmbedding",
    "ModelConfig",
    "build_classifier",
    "build_embedder",
    "count_parameters",
    "embedder_parity",
    "parameter_report",
]
