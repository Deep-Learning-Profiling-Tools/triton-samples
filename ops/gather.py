import torch

import triton
import triton.language as tl
import triton.profiler as proton


@triton.jit
def kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    axis: tl.constexpr,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
):
    src_offs = tl.arange(0, dim0)[:, None] * dim1 + tl.arange(0, dim1)[None, :]
    src = tl.load(src_ptr + src_offs)

    idx_offs = tl.arange(0, dim0)[:, None] * dim1 + tl.arange(0, dim1)[None, :]
    idx = tl.load(idx_ptr + idx_offs)

    for i in range(1):
        src = tl.gather(src, idx, axis)

    out = src

    out_offs = tl.arange(0, dim0)[:, None] * dim1 + tl.arange(0, dim1)[None, :]
    tl.store(out_ptr + out_offs, out)


BLOCK_SIZE = 512

for dtype in [torch.float8_e5m2, torch.float16]:
    for c in [4, 8, 16, 32, 64]:
        x = torch.randn((BLOCK_SIZE, c), device="cuda").to(dtype)
        idx = torch.randint(
            0,
            c,
            (
                BLOCK_SIZE,
                c,
            ),
            device="cuda",
        ).to(torch.int32)
        y = torch.zeros_like(x)
        with proton.scope(f"{dtype}_{c}"):
            kernel[(1,)](x, idx, y, 1, BLOCK_SIZE, c)

    y_ref = x.to(torch.float16).gather(1, idx.to(torch.int64))
    torch.testing.assert_close(y.to(torch.float16), y_ref, rtol=1e-2, atol=1e-2)
