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
    C0: tl.constexpr,
    C1: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = (
        tl.arange(0, BLOCK_SIZE)[:, None] * STRIDE + tl.arange(0, CHANNEL_SIZE)[None, :]
    )
    offsets = tl.max_contiguous(offsets, (C0, C1))
    x_val = tl.load(x_ptr + offsets + pid * BLOCK_SIZE * CHANNEL_SIZE)
    tl.store(y_ptr + offsets + pid * BLOCK_SIZE * CHANNEL_SIZE, x_val)


BATCH_SIZE = 512
BLOCK_SIZE = 64
CHANNEL_SIZE = 64

for dtype in [torch.float16, torch.float32]:
    for c0 in [1, 2, 4, 8]:
        for c1 in [1, 2, 4, 8]:
            x = torch.randn((BATCH_SIZE, BLOCK_SIZE, CHANNEL_SIZE), device="cuda")
            y = torch.zeros_like(x)
            with proton.scope(
                f"{dtype}_{c0}_{c1}", {"bytes": x.nelement() * x.element_size() * 2}
            ):
                kernel[(BATCH_SIZE,)](
                    x, y, CHANNEL_SIZE, CHANNEL_SIZE, BLOCK_SIZE, c0, c1
                )

            torch.testing.assert_close(x, y, rtol=1e-2, atol=1e-2)
