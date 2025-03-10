import torch

import triton
import triton.language as tl


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    STRIDE: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = (
        tl.arange(0, BLOCK_SIZE)[:, None] * STRIDE + tl.arange(0, CHANNEL_SIZE)[None, :]
    )
    x_val = tl.load(x_ptr + offsets + pid * BLOCK_SIZE * CHANNEL_SIZE)
    for _ in range(10):
        x_val = tl.cumsum(x_val, axis=0)
    tl.store(y_ptr + offsets + pid * BLOCK_SIZE * CHANNEL_SIZE, x_val)


BATCH_SIZE = 1024
BLOCK_SIZE = 64
CHANNEL_SIZE = 1
x = torch.randn(
    (BATCH_SIZE, BLOCK_SIZE, CHANNEL_SIZE), device="cuda", dtype=torch.float32
)
y = torch.zeros_like(x)

kernel[(BATCH_SIZE,)](x, y, CHANNEL_SIZE, CHANNEL_SIZE, BLOCK_SIZE)
for _ in range(10):
    x = torch.cumsum(x, axis=1)

torch.testing.assert_close(x, y, rtol=1e-2, atol=1e-2)
