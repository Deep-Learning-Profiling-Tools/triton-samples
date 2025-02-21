import torch

import triton
import triton.language as tl
import triton.profiler as proton


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
    tl.store(y_ptr + offsets + pid * BLOCK_SIZE * CHANNEL_SIZE, x_val)


BATCH_SIZE = 1024
BLOCK_SIZE = 1024

for dtype in [torch.float16, torch.float32]:
    for c in [1, 2, 4, 8, 16]:
        x = torch.randn((BATCH_SIZE, BLOCK_SIZE, c), device="cuda")
        y = torch.zeros_like(x)
        with proton.scope(
            f"{dtype}_{c}", {"bytes": x.nelement() * x.element_size() * 2 * 10}
        ):
            for _ in range(10):
                _ = torch.randn((BATCH_SIZE, BLOCK_SIZE, c), device="cuda")
                kernel[(BATCH_SIZE,)](x, y, c, c, BLOCK_SIZE)

        torch.testing.assert_close(x, y, rtol=1e-2, atol=1e-2)
