"""Compare different methods of creating constant tensors in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel1(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    """Matrix multiplication with tl.full constant matrix."""
    tid = (
        tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    x_offset = x_ptr + tid
    y_offset = y_ptr + tid
    lhs = tl.load(x_offset)
    rhs = tl.full([BLOCK_SIZE, BLOCK_SIZE], 1.5, tl.float32)
    tl.store(y_offset, tl.dot(lhs, rhs).to(tl.float32))


@triton.jit
def kernel2(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    """Matrix multiplication with explicit constant matrix creation."""
    tid = (
        tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    x_offset = x_ptr + tid
    y_offset = y_ptr + tid
    lhs = tl.load(x_offset)
    rhs = (tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32) + 1.5).to(tl.float32)
    tl.store(y_offset, tl.dot(lhs, rhs).to(tl.float32))


BLOCK_SIZE = 128
x = torch.randn((BLOCK_SIZE, BLOCK_SIZE), device="cuda", dtype=torch.float32)
y_0 = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), device="cuda", dtype=torch.float32)
y_1 = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), device="cuda", dtype=torch.float32)
for _ in range(100):
    kernel1[(1024,)](x, y_0, BLOCK_SIZE)
    kernel2[(1024,)](x, y_1, BLOCK_SIZE)
torch.allclose(y_0, y_1)
