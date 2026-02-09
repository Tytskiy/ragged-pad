import math

import pytest
import torch

from ragged_pad import pad, unpad


def get_mask(lengths, maxlen=None):
    """[2, 3, 1] -> [[True,  True,  False],
    [True,  True,  True ],
    [True,  False, False]])"""
    if maxlen is None:
        maxlen = lengths.max().item()
    return torch.arange(maxlen, device=lengths.device)[None, :] < lengths[:, None]


def naive_pad(
    x: torch.Tensor, lengths: torch.Tensor, max_seqlen: int, pad_value: float = 0.0, dim: int = 0
) -> torch.Tensor:
    """Naive padding that works for arbitrary dimension."""

    dim = dim % x.dim()
    batch_size = lengths.shape[0]

    # Get prefix and suffix shapes
    prefix_shape = x.shape[:dim]
    suffix_shape = x.shape[dim + 1 :]

    # Output shape: (..., batch_size, max_seqlen, ...)
    out_shape = prefix_shape + (batch_size, max_seqlen) + suffix_shape

    outer_batch = math.prod(prefix_shape)
    D = math.prod(suffix_shape)

    x_flat = x.reshape(outer_batch, -1, D)
    padded_flat = x.new_full((outer_batch, batch_size, max_seqlen, D), fill_value=pad_value)

    mask = get_mask(lengths, maxlen=max_seqlen)
    for ob in range(outer_batch):
        padded_flat[ob][mask] = x_flat[ob]

    return padded_flat.reshape(out_shape)


def naive_unpad(x: torch.Tensor, lengths: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Naive unpadding that works for arbitrary dimension."""
    dim = dim % x.dim()
    batch_size = lengths.shape[0]
    max_seqlen = x.shape[dim + 1]

    # Get prefix and suffix shapes
    prefix_shape = x.shape[:dim]
    suffix_shape = x.shape[dim + 2 :]

    varlen = int(lengths.sum().item())

    # Output shape: (..., varlen, ...)
    out_shape = prefix_shape + (varlen,) + suffix_shape

    outer_batch = math.prod(prefix_shape)
    D = math.prod(suffix_shape)

    x_flat = x.reshape(outer_batch, batch_size, max_seqlen, D)
    out_flat = torch.empty(outer_batch, varlen, D, device=x.device, dtype=x.dtype)

    mask = get_mask(lengths, maxlen=max_seqlen)
    for ob in range(outer_batch):
        out_flat[ob] = x_flat[ob][mask]

    return out_flat.reshape(out_shape)

@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires at least 1 gpu")
def test_simple_forward():
    torch.manual_seed(42)

    batch_size = 16
    max_seqlen = 128
    fill_value = 2.0
    D = 64

    lengths = torch.randint(1, max_seqlen + 1, (batch_size,), device="cuda", dtype=torch.int32)
    cu_lengths = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    cu_lengths[1:] = torch.cumsum(lengths, dim=0)
    varlen = int(cu_lengths[-1].item())

    # Test basic (varlen, D) -> (batch_size, max_seqlen, D)
    x = torch.randn(varlen, D, device="cuda", dtype=torch.bfloat16)
    triton_out = pad(x, lengths, cu_lengths, max_seqlen, pad_value=fill_value)
    naive_out = naive_pad(x, lengths, max_seqlen, pad_value=fill_value)
    assert torch.allclose(triton_out, naive_out), "Padding mismatch!"

    padded = torch.randn(batch_size, max_seqlen, D, device="cuda", dtype=torch.bfloat16)
    triton_out = unpad(padded, lengths, cu_lengths, varlen)
    naive_out = naive_unpad(padded, lengths)
    assert torch.allclose(triton_out, naive_out), "Unpadding mismatch!"

@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires at least 1 gpu")
def test_backward_simple():
    torch.manual_seed(42)

    batch_size = 16
    max_seqlen = 128
    D = 64

    lengths = torch.randint(1, max_seqlen + 1, (batch_size,), device="cuda", dtype=torch.int32)
    cu_lengths = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), torch.cumsum(lengths, dim=0)])
    varlen = int(lengths.sum().item())

    x_triton = torch.randn(varlen, D, device="cuda", dtype=torch.float32, requires_grad=True)
    x_naive = x_triton.detach().clone().requires_grad_(True)

    triton_out = pad(x_triton, lengths, cu_lengths, max_seqlen)
    naive_out = naive_pad(x_naive, lengths, max_seqlen)

    grad_out = torch.randn_like(triton_out)
    triton_out.backward(grad_out)
    naive_out.backward(grad_out)

    assert x_triton.grad is not None and x_naive.grad is not None
    assert torch.allclose(x_triton.grad, x_naive.grad, atol=1e-5), (
        f"Padding backward mismatch! Max diff: {(x_triton.grad - x_naive.grad).abs().max()}"
    )

    padded_triton = torch.randn(batch_size, max_seqlen, D, device="cuda", dtype=torch.float32, requires_grad=True)
    padded_naive = padded_triton.detach().clone().requires_grad_(True)

    triton_out = unpad(padded_triton, lengths, cu_lengths, varlen)
    naive_out = naive_unpad(padded_naive, lengths)

    grad_out = torch.randn_like(triton_out)
    triton_out.backward(grad_out)
    naive_out.backward(grad_out)

    assert padded_triton.grad is not None and padded_naive.grad is not None
    assert torch.allclose(padded_triton.grad, padded_naive.grad, atol=1e-5), (
        f"Unpadding backward mismatch! Max diff: {(padded_triton.grad - padded_naive.grad).abs().max()}"
    )


# Test configurations: (prefix_shape, suffix_shape, dim)
# varlen will be inserted at position `dim`
MULTIDIM_CONFIGS = [
    pytest.param((4,), (32,), 1, id="prefix_only"),  # (outer, varlen, D)
    pytest.param((), (8, 16), 0, id="suffix_only"),  # (varlen, D1, D2)
    pytest.param((4,), (8, 16), 1, id="prefix_suffix"),  # (outer, varlen, D1, D2)
    pytest.param((2, 3), (32,), 2, id="multi_prefix"),  # (p1, p2, varlen, D)
]


def _make_ragged_data(batch_size=8, max_seqlen=256):
    lengths = torch.randint(1, max_seqlen + 1, (batch_size,), device="cuda", dtype=torch.int32)
    cu_lengths = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), torch.cumsum(lengths, dim=0)])
    varlen = int(lengths.sum().item())

    return {
        "batch_size": batch_size,
        "max_seqlen": max_seqlen,
        "lengths": lengths,
        "cu_lengths": cu_lengths,
        "varlen": varlen,
    }


@pytest.fixture
def ragged_fixture():
    """Pytest fixture wrapper for ragged test data."""
    return _make_ragged_data()


@pytest.mark.parametrize("prefix_shape,suffix_shape,dim", MULTIDIM_CONFIGS)
@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires at least 1 gpu")
def test_pad_forward_multidim(ragged_fixture, prefix_shape, suffix_shape, dim):
    """Test padding forward with various dimension configurations."""
    batch_size = ragged_fixture["batch_size"]
    max_seqlen = ragged_fixture["max_seqlen"]
    lengths = ragged_fixture["lengths"]
    cu_lengths = ragged_fixture["cu_lengths"]
    varlen = ragged_fixture["varlen"]
    fill_value = -1.0

    input_shape = prefix_shape + (varlen,) + suffix_shape
    expected_shape = prefix_shape + (batch_size, max_seqlen) + suffix_shape

    x = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
    triton_out = pad(x, lengths, cu_lengths, max_seqlen, pad_value=fill_value, dim=dim)
    naive_out = naive_pad(x, lengths, max_seqlen, pad_value=fill_value, dim=dim)

    assert triton_out.shape == expected_shape, f"Shape mismatch: {triton_out.shape} != {expected_shape}"
    assert torch.allclose(triton_out, naive_out), "Padding mismatch!"


@pytest.mark.parametrize("prefix_shape,suffix_shape,dim", MULTIDIM_CONFIGS)
@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires at least 1 gpu")
def test_unpad_forward_multidim(ragged_fixture, prefix_shape, suffix_shape, dim):
    """Test unpadding forward with various dimension configurations."""
    batch_size = ragged_fixture["batch_size"]
    max_seqlen = ragged_fixture["max_seqlen"]
    lengths = ragged_fixture["lengths"]
    cu_lengths = ragged_fixture["cu_lengths"]
    varlen = ragged_fixture["varlen"]

    input_shape = prefix_shape + (batch_size, max_seqlen) + suffix_shape
    expected_shape = prefix_shape + (varlen,) + suffix_shape

    padded = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
    triton_out = unpad(padded, lengths, cu_lengths, varlen, dim=dim)
    naive_out = naive_unpad(padded, lengths, dim=dim)

    assert triton_out.shape == expected_shape, f"Shape mismatch: {triton_out.shape} != {expected_shape}"
    assert torch.allclose(triton_out, naive_out), "Unpadding mismatch!"


@pytest.mark.parametrize("prefix_shape,suffix_shape,dim", MULTIDIM_CONFIGS)
@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires at least 1 gpu")
def test_pad_backward_multidim(ragged_fixture, prefix_shape, suffix_shape, dim):
    """Test padding backward with various dimension configurations."""
    max_seqlen = ragged_fixture["max_seqlen"]
    lengths = ragged_fixture["lengths"]
    cu_lengths = ragged_fixture["cu_lengths"]
    varlen = ragged_fixture["varlen"]

    input_shape = prefix_shape + (varlen,) + suffix_shape

    x_triton = torch.randn(input_shape, device="cuda", dtype=torch.float32, requires_grad=True)
    x_naive = x_triton.detach().clone().requires_grad_(True)

    triton_out = pad(x_triton, lengths, cu_lengths, max_seqlen, dim=dim)
    naive_out = naive_pad(x_naive, lengths, max_seqlen, dim=dim)

    grad_out = torch.randn_like(triton_out)
    triton_out.backward(grad_out)
    naive_out.backward(grad_out)

    assert x_triton.grad is not None and x_naive.grad is not None
    assert torch.allclose(x_triton.grad, x_naive.grad, atol=1e-5), (
        f"Padding backward mismatch! Max diff: {(x_triton.grad - x_naive.grad).abs().max()}"
    )


@pytest.mark.parametrize("prefix_shape,suffix_shape,dim", MULTIDIM_CONFIGS)
@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires at least 1 gpu")
def test_unpad_backward_multidim(ragged_fixture, prefix_shape, suffix_shape, dim):
    """Test unpadding backward with various dimension configurations."""
    batch_size = ragged_fixture["batch_size"]
    max_seqlen = ragged_fixture["max_seqlen"]
    lengths = ragged_fixture["lengths"]
    cu_lengths = ragged_fixture["cu_lengths"]
    varlen = ragged_fixture["varlen"]

    input_shape = prefix_shape + (batch_size, max_seqlen) + suffix_shape

    padded_triton = torch.randn(input_shape, device="cuda", dtype=torch.float32, requires_grad=True)
    padded_naive = padded_triton.detach().clone().requires_grad_(True)

    triton_out = unpad(padded_triton, lengths, cu_lengths, varlen, dim=dim)
    naive_out = naive_unpad(padded_naive, lengths, dim=dim)

    grad_out = torch.randn_like(triton_out)
    triton_out.backward(grad_out)
    naive_out.backward(grad_out)

    assert padded_triton.grad is not None and padded_naive.grad is not None
    assert torch.allclose(padded_triton.grad, padded_naive.grad, atol=1e-5), (
        f"Unpadding backward mismatch! Max diff: {(padded_triton.grad - padded_naive.grad).abs().max()}"
    )


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires at least 1 gpu")
def test_non_contiguous_forward():
    torch.manual_seed(42)

    batch_size = 4
    max_seqlen = 32
    prefix = 2
    suffix = 16

    lengths = torch.randint(1, max_seqlen + 1, (batch_size,), device="cuda", dtype=torch.int32)
    cu_lengths = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), torch.cumsum(lengths, dim=0)])
    varlen = int(lengths.sum().item())

    # Test pad: (suffix, prefix, varlen) -> permute -> (prefix, varlen, suffix) non-contiguous
    x_base = torch.randn(suffix, prefix, varlen, device="cuda", dtype=torch.bfloat16)
    x_non_contig = x_base.permute(1, 2, 0)
    assert not x_non_contig.is_contiguous()

    x_contig = x_non_contig.contiguous()

    triton_out = pad(x_non_contig, lengths, cu_lengths, max_seqlen, dim=1)
    expected_out = pad(x_contig, lengths, cu_lengths, max_seqlen, dim=1)
    assert torch.allclose(triton_out, expected_out), "Non-contiguous pad mismatch!"

    # Test unpad: (suffix, prefix, batch_size, max_seqlen) -> permute -> (prefix, batch_size, max_seqlen, suffix)
    padded_base = torch.randn(suffix, prefix, batch_size, max_seqlen, device="cuda", dtype=torch.bfloat16)
    padded_non_contig = padded_base.permute(1, 2, 3, 0)
    assert not padded_non_contig.is_contiguous()

    padded_contig = padded_non_contig.contiguous()

    triton_out = unpad(padded_non_contig, lengths, cu_lengths, varlen, dim=1)
    expected_out = unpad(padded_contig, lengths, cu_lengths, varlen, dim=1)
    assert torch.allclose(triton_out, expected_out), "Non-contiguous unpad mismatch!"

@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires at least 1 gpu")
def test_non_contiguous_backward():
    torch.manual_seed(45)

    batch_size = 4
    max_seqlen = 32
    prefix = 2
    suffix = 16

    lengths = torch.randint(1, max_seqlen + 1, (batch_size,), device="cuda", dtype=torch.int32)
    cu_lengths = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), torch.cumsum(lengths, dim=0)])
    varlen = int(lengths.sum().item())

    # Test pad backward
    x_base = torch.randn(suffix, prefix, varlen, device="cuda", dtype=torch.float32)
    x_non_contig = x_base.permute(1, 2, 0).requires_grad_(True)
    x_contig = x_non_contig.detach().contiguous().requires_grad_(True)

    out_non_contig = pad(x_non_contig, lengths, cu_lengths, max_seqlen, dim=1)
    out_contig = pad(x_contig, lengths, cu_lengths, max_seqlen, dim=1)

    grad_out = torch.randn_like(out_non_contig)
    out_non_contig.backward(grad_out)
    out_contig.backward(grad_out)

    assert x_non_contig.grad is not None and x_contig.grad is not None
    assert torch.allclose(x_non_contig.grad, x_contig.grad, atol=1e-5), (
        f"Non-contiguous pad backward mismatch! Max diff: {(x_non_contig.grad - x_contig.grad).abs().max()}"
    )

    # Test unpad backward
    padded_base = torch.randn(suffix, prefix, batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    padded_non_contig = padded_base.permute(1, 2, 3, 0).requires_grad_(True)
    padded_contig = padded_non_contig.detach().contiguous().requires_grad_(True)

    out_non_contig = unpad(padded_non_contig, lengths, cu_lengths, varlen, dim=1)
    out_contig = unpad(padded_contig, lengths, cu_lengths, varlen, dim=1)

    grad_out = torch.randn_like(out_non_contig)
    out_non_contig.backward(grad_out)
    out_contig.backward(grad_out)

    assert padded_non_contig.grad is not None and padded_contig.grad is not None
    assert torch.allclose(padded_non_contig.grad, padded_contig.grad, atol=1e-5), (
        f"Non-contiguous unpad backward mismatch! Max diff: {(padded_non_contig.grad - padded_contig.grad).abs().max()}"
    )

@torch.compile()
def naive_pad_bench(x: torch.Tensor, lengths: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """
    Naive padding implementation.
    """
    padded_x = x.new_zeros((lengths.shape[0], max_seqlen, x.shape[-1]))

    mask = get_mask(lengths, maxlen=max_seqlen)
    padded_x[mask] = x

    return padded_x

@torch.compile()
def naive_unpad_bench(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Naive unpadding implementation.
    """
    return x[get_mask(lengths, maxlen=x.shape[1])]


def benchmark():
    import time

    batch_size = 64
    max_seqlen = 2048
    D = 1024

    torch.manual_seed(42)
    lengths = torch.randint(max_seqlen - 48, max_seqlen + 1, (batch_size,), device="cuda", dtype=torch.int32)
    cu_lengths = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    cu_lengths[1:] = torch.cumsum(lengths, dim=0)
    varlen = int(cu_lengths[-1].item())

    x = torch.randn(varlen, D, device="cuda", dtype=torch.bfloat16)
    padded = torch.randn(batch_size, max_seqlen, D, device="cuda", dtype=torch.bfloat16)

    n_warmup = 10
    n_iters = 100

    print(f"Config: batch_size={batch_size}, max_seqlen={max_seqlen}, D={D}, varlen={varlen}")
    print()

    # ===== Padding benchmark =====
    print("=== Padding ===")

    # Warmup
    for _ in range(n_warmup):
        _ = pad(x, lengths, cu_lengths, max_seqlen)
        _ = naive_pad_bench(x, lengths, max_seqlen)
    torch.cuda.synchronize()

    # Triton
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = pad(x, lengths, cu_lengths, max_seqlen)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iters * 1000

    # Naive
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = naive_pad_bench(x, lengths, max_seqlen)
    torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"  Triton: {triton_time:.3f} ms")
    print(f"  Naive:  {naive_time:.3f} ms")
    print(f"  Speedup: {naive_time / triton_time:.2f}x")
    print()

    # ===== Unpadding benchmark =====
    print("=== Unpadding ===")

    # Warmup
    for _ in range(n_warmup):
        _ = unpad(padded, lengths, cu_lengths, varlen)
        _ = naive_unpad_bench(padded, lengths)
    torch.cuda.synchronize()

    # Triton
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = unpad(padded, lengths, cu_lengths, varlen)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iters * 1000

    # Naive
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = naive_unpad_bench(padded, lengths)
    torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"  Triton: {triton_time:.3f} ms")
    print(f"  Naive:  {naive_time:.3f} ms")
    print(f"  Speedup: {naive_time / triton_time:.2f}x")


if __name__ == "__main__":
    test_simple_forward()
    print("Test forward passed!")
    test_backward_simple()
    print("Test backward passed!")

    benchmark()
