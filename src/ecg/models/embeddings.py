"""The two patch embedders under comparison.

Both take raw signal shaped ``(batch, n_leads, n_samples)`` and return tokens
shaped ``(batch, n_tokens, d_model)``. They are interchangeable by construction:
same input, same output, same token count, parameter counts matched to within a
few percent. The only thing that differs is how a patch becomes a vector.

* :class:`LinearPatchEmbedding` flattens a patch across all leads and applies a
  single ``nn.Linear``. A token sees exactly its own 500 ms and nothing else.
* :class:`ConvPatchEmbedding` runs a strided convolutional stem whose kernels
  overlap, so a token sees 1.27 s centred on its patch, through two
  nonlinearities.

The trap this module exists to avoid: ``Conv1d(kernel_size=patch,
stride=patch)`` is *mathematically identical* to the linear projection -- same
function, differently spelled -- so an arm built that way would compare a model
against itself. :attr:`ConvPatchEmbedding.receptive_field` makes the difference
measurable, and the test suite asserts it exceeds one patch.
"""

from __future__ import annotations

import torch
from torch import nn

from ecg.models.config import CONV_NORM_GROUPS, ModelConfig


class LinearPatchEmbedding(nn.Module):
    """Flatten each patch across all leads and project it once.

    The patch is laid out lead-major -- all ``patch_samples`` of lead I, then
    all of lead II, and so on -- so a single weight row spans one lead's whole
    window. Which layout is used does not change what the layer can represent,
    only which weights are adjacent, but it is fixed here so checkpoints stay
    interpretable.

    Args:
        config: Model configuration. Only the patch geometry and ``d_model``
            are read; the ``embedder`` field is not consulted, so this class
            can be instantiated from either arm's config in tests.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.patch_features, config.d_model)

    @property
    def n_tokens(self) -> int:
        """Tokens produced per record."""
        return self.config.n_tokens

    @property
    def receptive_field(self) -> int:
        """Samples one token depends on: exactly its own patch."""
        return self.config.patch_samples

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Embed a batch of raw signal.

        Args:
            signal: ``(batch, n_leads, n_samples)`` float tensor.

        Returns:
            ``(batch, n_tokens, d_model)`` token embeddings.

        Raises:
            ValueError: If the input shape does not match the configuration.
        """
        _check_input(signal, self.config)
        batch, leads, _ = signal.shape
        patches = signal.reshape(
            batch, leads, self.config.n_tokens, self.config.patch_samples
        )
        # (batch, token, lead, sample) -> flatten lead-major within the patch.
        patches = patches.permute(0, 2, 1, 3).reshape(
            batch, self.config.n_tokens, self.config.patch_features
        )
        return self.project(patches)


class ConvPatchEmbedding(nn.Module):
    """A strided convolutional stem with overlapping kernels.

    Three layers, GELU between them and GroupNorm on the hidden ones. The
    strides multiply to ``patch_samples``, so the stem emits exactly as many
    tokens as the linear embedder, but the kernels are wider than the strides,
    so each token integrates its neighbours' signal as well as its own. At the
    default geometry that is a 127-sample receptive field -- 1.27 s, enough to
    span a QRS complex and the segments either side of it -- against a 50-sample
    patch.

    That overlap is the reason SSL must mask the raw signal *before* this
    module rather than masking token embeddings after it. Masking afterwards
    would leave the masked region's samples visible through neighbouring
    tokens, and the reconstruction task would be trivially solvable.

    Args:
        config: Model configuration, read for the stem spec and ``d_model``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.n_leads
        for out_channels, kernel, stride, padding in config.conv_hidden:
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding)
            )
            layers.append(nn.GroupNorm(CONV_NORM_GROUPS, out_channels))
            layers.append(nn.GELU())
            in_channels = out_channels

        kernel, stride, padding = config.conv_final
        layers.append(
            nn.Conv1d(in_channels, config.d_model, kernel, stride=stride, padding=padding)
        )
        self.stem = nn.Sequential(*layers)

    @property
    def n_tokens(self) -> int:
        """Tokens produced per record."""
        return self.config.n_tokens

    @property
    def receptive_field(self) -> int:
        """Input samples one output token depends on.

        Accumulated over the stem as ``rf += (kernel - 1) * jump``, where
        ``jump`` is the product of the strides below the current layer. A value
        equal to ``patch_samples`` would mean the stem had collapsed into the
        linear projection; the tests assert it is larger.
        """
        spec = [(k, s) for _, k, s, _ in self.config.conv_hidden]
        spec.append((self.config.conv_final[0], self.config.conv_final[1]))
        field, jump = 1, 1
        for kernel, stride in spec:
            field += (kernel - 1) * jump
            jump *= stride
        return field

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Embed a batch of raw signal.

        Args:
            signal: ``(batch, n_leads, n_samples)`` float tensor.

        Returns:
            ``(batch, n_tokens, d_model)`` token embeddings.

        Raises:
            ValueError: If the input shape does not match the configuration, or
                if the stem emitted an unexpected number of tokens -- which
                would mean the padding arithmetic had drifted from the config.
        """
        _check_input(signal, self.config)
        tokens = self.stem(signal).transpose(1, 2)
        if tokens.shape[1] != self.config.n_tokens:
            raise ValueError(
                f"stem produced {tokens.shape[1]} tokens, expected "
                f"{self.config.n_tokens}; check conv padding"
            )
        return tokens


def build_embedder(config: ModelConfig) -> nn.Module:
    """Construct the embedder named by the configuration.

    Args:
        config: Model configuration.

    Returns:
        A :class:`LinearPatchEmbedding` or :class:`ConvPatchEmbedding`.
    """
    if config.embedder == "linear":
        return LinearPatchEmbedding(config)
    return ConvPatchEmbedding(config)


def _check_input(signal: torch.Tensor, config: ModelConfig) -> None:
    """Fail loudly on a shape mismatch rather than broadcasting into nonsense."""
    if signal.dim() != 3:
        raise ValueError(
            f"expected (batch, n_leads, n_samples), got shape {tuple(signal.shape)}"
        )
    if signal.shape[1] != config.n_leads or signal.shape[2] != config.n_samples:
        raise ValueError(
            f"expected (batch, {config.n_leads}, {config.n_samples}), "
            f"got {tuple(signal.shape)}"
        )
