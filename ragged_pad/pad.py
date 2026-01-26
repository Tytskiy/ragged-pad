import math
from typing import Tuple

import torch
import triton
import triton.language as tl


def _get_autotune_configs():
    """Generate autotune configurations for pad kernels."""
    configs = []
    for block_seq in [1, 2, 4, 8, 16, 32]:
        for block_d in [64, 128, 256, 512, 1024, 2048]:
            configs.append(triton.Config({"BLOCK_SEQ": block_seq, "BLOCK_D": block_d}))
    return configs


def _bucket_seqlen(max_seqlen: int, bucket_size: int = 64) -> int:
    """Round max_seqlen up to the nearest multiple of bucket_size for autotune caching."""
    return ((max_seqlen + bucket_size - 1) // bucket_size) * bucket_size


@triton.autotune(configs=_get_autotune_configs(), key=["D", "max_seqlen_bucket"])
@triton.jit
def _pad_kernel(
    x_ptr,  # (outer_batch, varlen, D)
    out_ptr,  # (outer_batch, B, max_seqlen, D)
    cu_lengths_ptr,  # (B + 1,)
    lengths_ptr,  # (B,)
    D: tl.constexpr,
    max_seqlen: tl.constexpr,
    max_seqlen_bucket: tl.constexpr,  # used only for autotune key
    varlen_total: tl.constexpr,  # total varlen for input stride
    B: tl.constexpr,  # batch size
    BLOCK_SEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # program_id(0) = outer_batch_idx * B + batch_idx
    outer_batch_idx = tl.program_id(0) // B
    batch_idx = tl.program_id(0) % B
    seq_block_idx = tl.program_id(1)

    seq_start = tl.load(cu_lengths_ptr + batch_idx)
    seq_len = tl.load(lengths_ptr + batch_idx)

    x_base = outer_batch_idx * varlen_total * D
    out_base = outer_batch_idx * B * max_seqlen * D + batch_idx * max_seqlen * D

    for s in range(BLOCK_SEQ):
        seq_idx = seq_block_idx * BLOCK_SEQ + s

        if seq_idx < max_seqlen:
            out_offset = out_base + seq_idx * D

            if seq_idx < seq_len:
                x_offset = x_base + (seq_start + seq_idx) * D

                for d_start in range(0, D, BLOCK_D):
                    d_offsets = d_start + tl.arange(0, BLOCK_D)
                    mask = d_offsets < D
                    vals = tl.load(x_ptr + x_offset + d_offsets, mask=mask, other=0.0)
                    tl.store(out_ptr + out_offset + d_offsets, vals, mask=mask)


def _pad(
    x: torch.Tensor,
    lengths: torch.Tensor,
    cu_lengths: torch.Tensor,
    max_seqlen: int,
    pad_value: float = 0.0,
    dim: int = 0,
) -> torch.Tensor:
    assert x.is_contiguous()
    assert lengths.is_contiguous()
    assert cu_lengths.is_contiguous()

    dim = dim % x.dim()

    batch_size = lengths.shape[0]
    varlen = x.shape[dim]

    outer_batch, D, out_shape = _compute_shapes(
        x.shape, dim, is_pad=True, batch_size=batch_size, max_seqlen=max_seqlen, varlen=varlen
    )

    out = x.new_full(out_shape, fill_value=pad_value)

    def grid(meta):
        return (outer_batch * batch_size, triton.cdiv(max_seqlen, meta["BLOCK_SEQ"]))

    _pad_kernel[grid](
        x,
        out,
        cu_lengths,
        lengths,
        D=D,
        max_seqlen=max_seqlen,
        max_seqlen_bucket=_bucket_seqlen(max_seqlen),
        varlen_total=varlen,
        B=batch_size,
    )

    return out


@triton.autotune(configs=_get_autotune_configs(), key=["D", "max_seqlen_bucket"])
@triton.jit
def _unpad_kernel(
    padded_ptr,  # (outer_batch, B, max_seqlen, D)
    out_ptr,  # (outer_batch, varlen, D)
    cu_lengths_ptr,  # (B + 1,)
    lengths_ptr,  # (B,)
    D: tl.constexpr,
    max_seqlen: tl.constexpr,
    max_seqlen_bucket: tl.constexpr,  # used only for autotune key
    varlen_total: tl.constexpr,  # total varlen for output stride
    B: tl.constexpr,  # batch size
    BLOCK_SEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # program_id(0) = outer_batch_idx * B + batch_idx
    outer_batch_idx = tl.program_id(0) // B
    batch_idx = tl.program_id(0) % B
    seq_block_idx = tl.program_id(1)

    seq_start = tl.load(cu_lengths_ptr + batch_idx)
    seq_len = tl.load(lengths_ptr + batch_idx)

    padded_base = outer_batch_idx * B * max_seqlen * D + batch_idx * max_seqlen * D
    out_base = outer_batch_idx * varlen_total * D

    for s in range(BLOCK_SEQ):
        seq_idx = seq_block_idx * BLOCK_SEQ + s

        if seq_idx < seq_len:
            padded_offset = padded_base + seq_idx * D
            out_offset = out_base + (seq_start + seq_idx) * D

            for d_start in range(0, D, BLOCK_D):
                d_offsets = d_start + tl.arange(0, BLOCK_D)
                mask = d_offsets < D
                vals = tl.load(padded_ptr + padded_offset + d_offsets, mask=mask, other=0.0)
                tl.store(out_ptr + out_offset + d_offsets, vals, mask=mask)


def _unpad(
    padded: torch.Tensor,
    lengths: torch.Tensor,
    cu_lengths: torch.Tensor,
    varlen: int,
    dim: int = 0,
) -> torch.Tensor:
    assert padded.is_contiguous()
    assert lengths.is_contiguous()
    assert cu_lengths.is_contiguous()

    dim = dim % padded.dim()

    batch_size = lengths.shape[0]
    max_seqlen = padded.shape[dim + 1]

    outer_batch, D, out_shape = _compute_shapes(
        padded.shape, dim, is_pad=False, batch_size=batch_size, max_seqlen=max_seqlen, varlen=varlen
    )

    out = torch.empty(out_shape, device=padded.device, dtype=padded.dtype)

    def grid(meta):
        return (outer_batch * batch_size, triton.cdiv(max_seqlen, meta["BLOCK_SEQ"]))

    _unpad_kernel[grid](
        padded,
        out,
        cu_lengths,
        lengths,
        D=D,
        max_seqlen=max_seqlen,
        max_seqlen_bucket=_bucket_seqlen(max_seqlen),
        varlen_total=varlen,
        B=batch_size,
    )

    return out


def _compute_shapes(
    shape: Tuple[int, ...], dim: int, is_pad: bool, batch_size: int, max_seqlen: int, varlen: int
) -> Tuple[int, int, Tuple[int, ...]]:
    """
    Compute outer_batch, D, and output shape for pad/unpad operations.

    For pad:  input shape (..., varlen, ...) -> output shape (..., batch_size, max_seqlen, ...)
    For unpad: input shape (..., batch_size, max_seqlen, ...) -> output shape (..., varlen, ...)
    """
    dim = dim % len(shape)

    if is_pad:
        # Input: (..., varlen, ...)
        prefix_shape = shape[:dim]
        suffix_shape = shape[dim + 1 :]
        outer_batch = math.prod(prefix_shape) if prefix_shape else 1
        D = math.prod(suffix_shape) if suffix_shape else 1
        out_shape = prefix_shape + (batch_size, max_seqlen) + suffix_shape
    else:
        # Input: (..., batch_size, max_seqlen, ...)
        prefix_shape = shape[:dim]
        suffix_shape = shape[dim + 2 :]
        outer_batch = math.prod(prefix_shape) if prefix_shape else 1
        D = math.prod(suffix_shape) if suffix_shape else 1
        out_shape = prefix_shape + (varlen,) + suffix_shape

    return outer_batch, D, out_shape


class PadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lengths: torch.Tensor,
        cu_lengths: torch.Tensor,
        max_seqlen: int,
        pad_value: float,
        dim: int,
    ) -> torch.Tensor:
        out = _pad(x, lengths, cu_lengths, max_seqlen, pad_value, dim)

        ctx.save_for_backward(lengths, cu_lengths)
        ctx.varlen = x.shape[dim]
        ctx.dim = dim

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        lengths, cu_lengths = ctx.saved_tensors
        varlen = ctx.varlen
        dim = ctx.dim

        grad_x = _unpad(grad_out, lengths, cu_lengths, varlen, dim)

        return grad_x, None, None, None, None, None


class UnpadFunction(torch.autograd.Function):
    """
    Unpad is the inverse of pad.
    Forward: padded (..., B, max_seqlen, ...) -> packed (..., varlen, ...)
    Backward: packed (..., varlen, ...) -> padded (..., B, max_seqlen, ...)  (i.e., pad!)
    """

    @staticmethod
    def forward(
        ctx, padded: torch.Tensor, lengths: torch.Tensor, cu_lengths: torch.Tensor, varlen: int, dim: int
    ) -> torch.Tensor:
        out = _unpad(padded, lengths, cu_lengths, varlen, dim)

        ctx.save_for_backward(lengths, cu_lengths)

        dim = dim % padded.dim()
        ctx.max_seqlen = padded.shape[dim + 1]
        ctx.dim = dim

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        lengths, cu_lengths = ctx.saved_tensors
        max_seqlen = ctx.max_seqlen
        dim = ctx.dim
        grad_padded = pad(grad_out, lengths, cu_lengths, max_seqlen, 0.0, dim)

        return grad_padded, None, None, None, None


def pad(
    x: torch.Tensor,
    lengths: torch.Tensor,
    cu_lengths: torch.Tensor,
    max_seqlen: int,
    pad_value: float = 0.0,
    dim: int = 0,
) -> torch.Tensor:
    """
    Pads the input tensor x with pad_value to the length of the longest sequence in lengths.

    Args:
        x: Input tensor of shape (..., varlen, ...). x.shape[dim] is varlen
        lengths: Tensor of shape (batch_size,) with sequence lengths.
        cu_lengths: Tensor of shape (batch_size + 1,) with cumulative sequence lengths.
        max_seqlen: Maximum sequence length.
        pad_value: Value to pad with.
        dim: Dimension of varlen.

    Returns:
        Padded tensor of shape (..., batch_size, max_seqlen, ...). x.shape[dim] is batch_size
    """
    return PadFunction.apply(x, lengths, cu_lengths, max_seqlen, pad_value, dim)  # type: ignore


def unpad(x: torch.Tensor, lengths: torch.Tensor, cu_lengths: torch.Tensor, varlen: int, dim: int = 0) -> torch.Tensor:
    """
    Unpads a padded tensor back to packed/ragged format. Differentiable.

    Args:
        x: Padded tensor of shape (..., batch_size, max_seqlen, ...). x.shape[dim] is batch_size
        lengths: Tensor of shape (batch_size,) with sequence lengths.
        cu_lengths: Tensor of shape (batch_size + 1,) with cumulative sequence lengths.
        varlen: Total number of elements in the packed tensor.
        dim: Dimension of varlen.

    Returns:
        Packed tensor of shape (..., varlen, ...). x.shape[dim] is varlen
    """
    return UnpadFunction.apply(x, lengths, cu_lengths, varlen, dim)  # type: ignore
