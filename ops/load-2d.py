import torch

import triton
import triton.language as tl
import triton.profiler as proton


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    BLOCK_SIZE: tl.constexpr,
    C: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)[:, None] * C + tl.arange(0, C)[None, :]
    x_val = tl.load(x_ptr + offsets + pid * BLOCK_SIZE * C)
    tl.store(y_ptr + offsets + pid * BLOCK_SIZE * C, x_val)


BATCH_SIZE = 512
BLOCK_SIZE = 512

for dtype in [torch.float8_e4m3fn, torch.float16]:
    for c in [1, 2, 4, 8, 16]:
        x = torch.randn((BATCH_SIZE, BLOCK_SIZE, c), device="cuda", dtype=dtype)
        y = torch.zeros_like(x)
        with proton.scope(
            f"{dtype}_{c}", {"bytes": x.nelement() * x.element_size() * 2}
        ):
            kernel[(BATCH_SIZE,)](x, y, BLOCK_SIZE, c)

        torch.testing.assert_close(x, y, rtol=1e-2, atol=1e-2)
