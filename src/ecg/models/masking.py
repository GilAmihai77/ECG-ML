"""Block masking of raw ECG signal for masked-reconstruction pretraining.

Two decisions are encoded here, and both are about the pretext task being hard
enough to be worth solving.

**Masking is applied to the raw signal, before the patch embedder.** The
convolutional stem's receptive field spans 127 samples against a 50-sample
stride, so a mask applied to token embeddings would leave the masked region
visible through its neighbours' inputs and the reconstruction would be
solvable by copying. Masking raw samples is the only placement that works for
both arms.

**Masks are contiguous blocks, not independent per token.** One token is
500 ms; one RR interval at 60-75 bpm is 800-1000 ms. A single-token hole is
therefore always shorter than a beat, and the model fills it by interpolating
*within* the same beat -- a local smoothness task that teaches nothing about
morphology. A two-token block spans a whole cardiac cycle, so the beat has to
be reconstructed from *other* beats, which is the representation we want.

A consequence worth measuring rather than assuming:
:func:`contaminated_fraction` reports how many visible tokens have a masked
sample inside their receptive field. It is zero for the linear embedder and
non-zero for the convolutional one, so the two arms do not face quite the same
task. The effect is partial -- a boundary token keeps its own patch intact --
but it is logged every run rather than left unexamined.
"""

from __future__ import annotations

import math

import torch

from ecg.models.config import ModelConfig, SslConfig

#: Value written into masked samples. The signal is per-record normalised, so
#: zero is the record's own mean: the least informative constant available.
MASK_VALUE: float = 0.0


def sample_block_mask(
    batch_size: int,
    n_tokens: int,
    config: SslConfig,
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Draw one block mask per record.

    Blocks are placed uniformly at random without overlap, using the standard
    stars-and-bars bijection: choosing ``k`` sorted indices from
    ``n_visible + k`` positions is equivalent to choosing the ``k + 1`` gap
    sizes between blocks. Blocks may land adjacent to one another, which merges
    them into a longer run and adds useful variety to the mask patterns.

    Every record in the batch is masked to exactly the same token count, so the
    loss is comparable across records and across steps.

    **The first and last tokens are masked less often than the interior**, and
    this is inherent rather than a bug: a block of length ``L`` covers token 0
    only if it starts there, while an interior token is covered by ``L``
    different block positions. At the defaults the marginals are exactly 1/3 at
    each end, 4/7 one token in, and about 1/2 across the interior. The only fix
    would be to wrap blocks around the record boundary, which would split them
    into two short holes and defeat the point of spanning a cardiac cycle. The
    edge patches are still masked on a third of steps, and they have one-sided
    context anyway, so the bias is accepted and recorded here.

    Args:
        batch_size: Records in the batch.
        n_tokens: Tokens per record.
        config: SSL configuration supplying the ratio and span.
        generator: Torch generator for reproducibility (integrity rule 4). Must
            be a CPU generator; the mask is moved to ``device`` afterwards.
        device: Device for the returned mask.

    Returns:
        Boolean tensor ``(batch_size, n_tokens)``, ``True`` where masked.
    """
    sizes = config.block_sizes(n_tokens)
    n_blocks = len(sizes)
    n_masked = sum(sizes)
    n_visible = n_tokens - n_masked

    # Sorted sample without replacement from range(n_visible + n_blocks).
    scores = torch.rand(batch_size, n_visible + n_blocks, generator=generator)
    chosen, _ = scores.argsort(dim=1)[:, :n_blocks].sort(dim=1)

    # chosen[i] counts the tokens and blocks below block i; subtracting i
    # removes the blocks and adding the sizes below i restores the offset.
    before = torch.tensor(
        [sum(sizes[:i]) for i in range(n_blocks)], dtype=torch.long
    )
    starts = chosen - torch.arange(n_blocks) + before

    mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool)
    for i, size in enumerate(sizes):
        positions = starts[:, i : i + 1] + torch.arange(size)
        mask.scatter_(1, positions, torch.ones_like(positions, dtype=torch.bool))
    return mask.to(device)


def expand_to_samples(token_mask: torch.Tensor, patch_samples: int) -> torch.Tensor:
    """Expand a per-token mask to a per-sample mask.

    Args:
        token_mask: ``(batch, n_tokens)`` boolean tensor.
        patch_samples: Samples per token.

    Returns:
        ``(batch, n_tokens * patch_samples)`` boolean tensor.
    """
    return token_mask.repeat_interleave(patch_samples, dim=1)


def apply_mask(
    signal: torch.Tensor, token_mask: torch.Tensor, patch_samples: int
) -> torch.Tensor:
    """Blank the masked patches of a raw signal batch.

    Args:
        signal: ``(batch, n_leads, n_samples)`` raw signal.
        token_mask: ``(batch, n_tokens)`` boolean tensor.
        patch_samples: Samples per token.

    Returns:
        A new tensor with masked samples set to :data:`MASK_VALUE`. All leads
        of a masked time patch are blanked together -- a token *is* the whole
        12-lead column, so there is no way to mask leads independently and no
        opportunity to reconstruct a lead from its neighbours.
    """
    sample_mask = expand_to_samples(token_mask, patch_samples).unsqueeze(1)
    return signal.masked_fill(sample_mask, MASK_VALUE)


def receptive_reach(receptive_field: int, patch_samples: int) -> int:
    """How many tokens either side a token's receptive field can reach into.

    Args:
        receptive_field: Input samples one token depends on.
        patch_samples: Samples per token, equal to the embedder's stride.

    Returns:
        ``0`` when a token sees only its own patch, as for the linear
        embedder; ``1`` at the default convolutional geometry, where 127
        samples against a 50-sample patch reach 38.5 samples past each edge.
    """
    overhang = (receptive_field - patch_samples) / 2
    return max(0, math.ceil(overhang / patch_samples))


def contaminated_fraction(token_mask: torch.Tensor, reach: int) -> float:
    """Fraction of visible tokens whose receptive field touches a masked sample.

    The number that quantifies how differently the two arms experience the same
    mask. Zero for the linear embedder by construction; at the default
    convolutional geometry a visible token is affected whenever either
    neighbour is masked, though it keeps its own patch intact.

    Args:
        token_mask: ``(batch, n_tokens)`` boolean tensor, ``True`` where masked.
        reach: Tokens either side, from :func:`receptive_reach`.

    Returns:
        Fraction in ``[0, 1]``, or ``0.0`` if nothing is visible.
    """
    visible = ~token_mask
    n_visible = int(visible.sum())
    if n_visible == 0 or reach == 0:
        return 0.0

    touched = torch.zeros_like(token_mask)
    for shift in range(1, reach + 1):
        touched[:, shift:] |= token_mask[:, :-shift]
        touched[:, :-shift] |= token_mask[:, shift:]
    return float((touched & visible).sum()) / n_visible


def mask_summary(token_mask: torch.Tensor, reach: int) -> dict[str, float]:
    """Summarise a batch's masks for MLflow.

    Args:
        token_mask: ``(batch, n_tokens)`` boolean tensor.
        reach: Tokens either side, from :func:`receptive_reach`.

    Returns:
        Mapping with ``"mask_ratio"``, ``"masked_tokens"`` and
        ``"contaminated_fraction"``.
    """
    batch, n_tokens = token_mask.shape
    masked = float(token_mask.sum()) / batch
    return {
        "mask_ratio": masked / n_tokens,
        "masked_tokens": masked,
        "contaminated_fraction": contaminated_fraction(token_mask, reach),
    }


def reach_of(config: ModelConfig, receptive_field: int) -> int:
    """Convenience wrapper resolving :func:`receptive_reach` from a config.

    Args:
        config: Model configuration supplying ``patch_samples``.
        receptive_field: The embedder's receptive field in samples.

    Returns:
        Tokens either side.
    """
    return receptive_reach(receptive_field, config.patch_samples)
