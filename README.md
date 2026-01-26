# ragged-pad

Fast Triton kernels for padding and unpadding ragged/variable-length tensors in PyTorch.

## Installation
```bash
git clone https://github.com/Tytskiy/ragged-pad.git
pip install .
```

## Usage

```python
import torch
from ragged_pad import pad, unpad

# Create variable-length sequences
lengths = torch.tensor([3, 5, 2], device="cuda", dtype=torch.int32)
cu_lengths = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), 
                        torch.cumsum(lengths, dim=0)])
varlen = int(lengths.sum().item())  # 10
max_seqlen = int(lengths.max().item())  # 5

# Packed tensor of shape (varlen, D)
x = torch.randn(varlen, 64, device="cuda", dtype=torch.bfloat16)

# Pad: (varlen, D) -> (batch_size, max_seqlen, D)
padded = pad(x, lengths, cu_lengths, max_seqlen, pad_value=0.0)
# Shape: (3, 5, 64)

# Unpad: (batch_size, max_seqlen, D) -> (varlen, D)
unpacked = unpad(padded, lengths, cu_lengths, varlen)
# Shape: (10, 64)
```

### Multi-dimensional support

Both `pad` and `unpad` support arbitrary dimensions via the `dim` parameter:

```python
# Pad along dimension 1: (outer, varlen, D) -> (outer, batch_size, max_seqlen, D)
x = torch.randn(4, varlen, 64, device="cuda")
padded = pad(x, lengths, cu_lengths, max_seqlen, dim=1)
```

### Differentiable

Both operations support autograd and can be used in training:

```python
x = torch.randn(varlen, 64, device="cuda", requires_grad=True)
padded = pad(x, lengths, cu_lengths, max_seqlen)
loss = padded.sum()
loss.backward()  # Gradients flow through pad/unpad
```

## API

### `pad(x, lengths, cu_lengths, max_seqlen, pad_value=0.0, dim=0)`

Pads a packed/ragged tensor to fixed-size batched format.

**Args:**
- `x`: Input tensor of shape `(..., varlen, ...)` where `varlen` is at position `dim`
- `lengths`: Tensor of shape `(batch_size,)` with sequence lengths
- `cu_lengths`: Tensor of shape `(batch_size + 1,)` with cumulative sequence lengths
- `max_seqlen`: Maximum sequence length for output padding
- `pad_value`: Value to fill padded positions (default: 0.0)
- `dim`: Dimension of `varlen` in input (default: 0)

**Returns:** Padded tensor of shape `(..., batch_size, max_seqlen, ...)`

### `unpad(x, lengths, cu_lengths, varlen, dim=0)`

Unpads a batched tensor back to packed/ragged format.

**Args:**
- `x`: Padded tensor of shape `(..., batch_size, max_seqlen, ...)`
- `lengths`: Tensor of shape `(batch_size,)` with sequence lengths
- `cu_lengths`: Tensor of shape `(batch_size + 1,)` with cumulative sequence lengths
- `varlen`: Total number of elements in the packed output
- `dim`: Dimension of `batch_size` in input (default: 0)

**Returns:** Packed tensor of shape `(..., varlen, ...)`

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- Triton >= 2.0.0

## License

MIT
