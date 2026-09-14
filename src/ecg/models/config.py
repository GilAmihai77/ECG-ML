"""Model configuration, shared by every arm of the experiment.

One dataclass describes the whole model, and the only field an arm may change
is :attr:`ModelConfig.embedder`. Everything else -- depth, width, heads,
dropout, token count -- is held identical, which turns integrity rule 3 into a
property of the type rather than a convention someone has to remember.

Validation is deliberately strict: a configuration that would make the two arms
emit different numbers of tokens is rejected at construction, not discovered
later in a metrics table.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

#: One convolutional stem layer: ``(out_channels, kernel, stride, padding)``.
ConvLayer = tuple[int, int, int, int]

Embedder = Literal["linear", "conv"]

#: Channel groups for the stem's normalisation. GroupNorm rather than
#: BatchNorm: it keeps no running statistics, so a stem pretrained with SSL at
#: one batch size behaves identically when fine-tuned at another.
CONV_NORM_GROUPS: int = 8

#: First stem layer, fixed. Kernel 7 against stride 5 is where the overlap --
#: the inductive bias under test -- starts.
CONV_FIRST_LAYER: ConvLayer = (48, 7, 5, 1)

#: Second stem layer's ``(kernel, stride, padding)``. Its width is not fixed:
#: it is solved for by :func:`matched_conv_hidden` to match the linear
#: embedder's parameter count at whatever ``d_model`` is in use.
CONV_SECOND_SHAPE: tuple[int, int, int] = (5, 5, 0)

#: Final stem layer as ``(kernel, stride, padding)``. Its output width is
#: always ``d_model``, so the encoder sees the same shape whichever stem ran.
DEFAULT_CONV_FINAL: tuple[int, int, int] = (5, 2, 2)

#: Largest second-layer width the parity search will consider.
_MAX_STEM_WIDTH: int = 512


def linear_embedder_parameters(patch_features: int, d_model: int) -> int:
    """Parameters in the linear patch embedder.

    Args:
        patch_features: Values in one patch across all leads.
        d_model: Token width.

    Returns:
        Weight plus bias count.
    """
    return patch_features * d_model + d_model


def conv_embedder_parameters(
    n_leads: int,
    d_model: int,
    conv_hidden: tuple[ConvLayer, ...],
    conv_final: tuple[int, int, int],
) -> int:
    """Parameters in the convolutional stem, counted without building it.

    Counting arithmetically rather than by instantiating a module lets the
    parity search run inside configuration construction, where importing the
    model layer would be circular.

    Args:
        n_leads: Input channels.
        d_model: Token width; the final layer's output channels.
        conv_hidden: Hidden layers.
        conv_final: Final layer's ``(kernel, stride, padding)``.

    Returns:
        Total parameter count, including GroupNorm affine terms.
    """
    total = 0
    in_channels = n_leads
    for out_channels, kernel, _, _ in conv_hidden:
        total += in_channels * out_channels * kernel + out_channels
        total += 2 * out_channels  # GroupNorm weight and bias
        in_channels = out_channels
    total += in_channels * d_model * conv_final[0] + d_model
    return total


def matched_conv_hidden(
    n_leads: int,
    d_model: int,
    patch_features: int,
    *,
    first: ConvLayer = CONV_FIRST_LAYER,
    second_shape: tuple[int, int, int] = CONV_SECOND_SHAPE,
    conv_final: tuple[int, int, int] = DEFAULT_CONV_FINAL,
) -> tuple[ConvLayer, ...]:
    """Solve for the stem width that matches the linear embedder's size.

    The two embedders do not scale alike: the linear one grows as
    ``patch_features * d_model``, while the stem's cost is a fixed hidden part
    plus a final layer growing as ``width * kernel * d_model``. A stem spec
    hand-tuned at one ``d_model`` therefore drifts out of parity at another --
    at ``d_model=256`` a spec tuned for 128 is 18% light, which would make a
    conv loss partly a capacity loss. So the second layer's width is searched
    rather than fixed, over multiples of :data:`CONV_NORM_GROUPS`.

    Args:
        n_leads: Input channels.
        d_model: Token width.
        patch_features: Values in one patch, the linear embedder's input.
        first: Fixed first stem layer.
        second_shape: Second layer's ``(kernel, stride, padding)``.
        conv_final: Final layer's ``(kernel, stride, padding)``.

    Returns:
        The two hidden layers, with the second's width chosen so the stem's
        parameter count is as close as possible to the linear embedder's.
    """
    target = linear_embedder_parameters(patch_features, d_model)
    kernel, stride, padding = second_shape
    best: tuple[ConvLayer, ...] = ()
    best_gap = float("inf")
    for width in range(CONV_NORM_GROUPS, _MAX_STEM_WIDTH + 1, CONV_NORM_GROUPS):
        candidate = (first, (width, kernel, stride, padding))
        gap = abs(
            conv_embedder_parameters(n_leads, d_model, candidate, conv_final) - target
        )
        if gap < best_gap:
            best, best_gap = candidate, gap
    return best


#: The resolved stem at the default configuration, kept as a named constant so
#: the geometry appears in one readable place: 12 -> 48 -> 80 -> d_model, with
#: strides 5, 5, 2 multiplying to a 50-sample patch.
DEFAULT_CONV_HIDDEN: tuple[ConvLayer, ...] = matched_conv_hidden(
    n_leads=12, d_model=128, patch_features=600
)


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to build the encoder and either embedder.

    Attributes:
        n_leads: Input channels; 12 for a standard ECG.
        n_samples: Samples per record. 1000 is 10 s at 100 Hz.
        patch_samples: Samples per token. 50 is 500 ms at 100 Hz, giving 20
            tokens, and divides evenly at 500 Hz too.
        d_model: Token width, identical across arms.
        n_layers: Transformer blocks.
        n_heads: Attention heads.
        ffn_mult: Feed-forward width as a multiple of ``d_model``.
        dropout: Dropout inside the encoder and before the classifier head.
        n_classes: Output dimension; 5 diagnostic superclasses.
        embedder: ``"linear"`` or ``"conv"`` -- the only field an arm may vary.
        conv_hidden: Hidden stem layers. Left as ``None``, it is solved for by
            :func:`matched_conv_hidden` so the arms stay parameter-matched at
            any ``d_model``. Ignored when ``embedder="linear"``.
        conv_final: Final stem layer as ``(kernel, stride, padding)``.
    """

    n_leads: int = 12
    n_samples: int = 1000
    patch_samples: int = 50
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    n_classes: int = 5
    embedder: Embedder = "linear"
    conv_hidden: tuple[ConvLayer, ...] | None = field(default=None)
    conv_final: tuple[int, int, int] = field(default=DEFAULT_CONV_FINAL)

    def __post_init__(self) -> None:
        """Resolve the stem, then reject anything that makes the arms differ.

        Raises:
            ValueError: If the embedder name is unknown, the record does not
                divide into whole patches, the heads do not divide ``d_model``,
                the stem's strides do not multiply to ``patch_samples``, or a
                stem width is not divisible by :data:`CONV_NORM_GROUPS`.
        """
        if self.embedder not in ("linear", "conv"):
            raise ValueError(f"unknown embedder {self.embedder!r}")
        if self.n_samples % self.patch_samples:
            raise ValueError(
                f"n_samples={self.n_samples} is not a whole number of "
                f"patch_samples={self.patch_samples} patches"
            )
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )

        if self.conv_hidden is None:
            object.__setattr__(
                self,
                "conv_hidden",
                matched_conv_hidden(
                    n_leads=self.n_leads,
                    d_model=self.d_model,
                    patch_features=self.patch_features,
                    conv_final=self.conv_final,
                ),
            )

        stride = self.conv_final[1]
        for _, _, layer_stride, _ in self.conv_hidden:
            stride *= layer_stride
        if stride != self.patch_samples:
            raise ValueError(
                f"convolutional strides multiply to {stride}, but one token is "
                f"{self.patch_samples} samples; the two arms would emit "
                "different numbers of tokens"
            )
        for channels, _, _, _ in self.conv_hidden:
            if channels % CONV_NORM_GROUPS:
                raise ValueError(
                    f"stem width {channels} is not divisible by "
                    f"{CONV_NORM_GROUPS} GroupNorm groups"
                )

    @property
    def n_tokens(self) -> int:
        """Tokens per record, identical for both embedders."""
        return self.n_samples // self.patch_samples

    @property
    def patch_features(self) -> int:
        """Values in one patch across all leads: the linear embedder's input."""
        return self.patch_samples * self.n_leads

    @property
    def d_ffn(self) -> int:
        """Feed-forward width inside a Transformer block."""
        return self.d_model * self.ffn_mult

    def for_arm(self, embedder: Embedder) -> ModelConfig:
        """Return this configuration with a different embedder.

        The intended way to build a paired arm: every other field is copied, so
        the two models cannot drift apart.

        Args:
            embedder: ``"linear"`` or ``"conv"``.

        Returns:
            A new configuration.
        """
        return replace(self, embedder=embedder)
