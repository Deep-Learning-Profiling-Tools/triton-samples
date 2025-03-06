import triton
import pathlib
import torch
import triton.profiler as proton


def run(TSIZE, VSIZE, dtype, torch_dtype):
    ttgir = f"""
    #blocked = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}}>
    #test = #ttg.blocked<{{sizePerThread = [{VSIZE}], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}}>
    module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
            %c1_i32 = arith.constant 1 : i32
            %c1000000_i32 = arith.constant 100000 : i32
            %c0_i32 = arith.constant 0 : i32
            %0 = tt.make_range {{end = {TSIZE} : i32, start = 0 : i32}} : tensor<{TSIZE}xi32, #blocked>
            %7 = tt.splat %arg0 : !tt.ptr<{dtype}> -> tensor<{TSIZE}x!tt.ptr<{dtype}>, #blocked>
            %8 = tt.addptr %7, %0 : tensor<{TSIZE}x!tt.ptr<{dtype}>, #blocked>, tensor<{TSIZE}xi32, #blocked>
            %9 = tt.load %8 : tensor<{TSIZE}x!tt.ptr<{dtype}>, #blocked>

            %10 = scf.for %arg2 = %c0_i32 to %c1000000_i32 step %c1_i32 iter_args(%arg3 = %9) -> (tensor<{TSIZE}x{dtype}, #blocked>)  : i32 {{
                %x = ttg.convert_layout %arg3 : tensor<{TSIZE}x{dtype}, #blocked> -> tensor<{TSIZE}x{dtype}, #test>
                %13 = ttg.convert_layout %x : tensor<{TSIZE}x{dtype}, #test> -> tensor<{TSIZE}x{dtype}, #blocked>
                scf.yield %13 : tensor<{TSIZE}x{dtype}, #blocked>
            }}
            %11 = tt.splat %arg1 : !tt.ptr<{dtype}> -> tensor<{TSIZE}x!tt.ptr<{dtype}>, #blocked>
            %12 = tt.addptr %11, %0 : tensor<{TSIZE}x!tt.ptr<{dtype}>, #blocked>, tensor<{TSIZE}xi32, #blocked>
            tt.store %12, %10 : tensor<{TSIZE}x!tt.ptr<{dtype}>, #blocked>
            tt.return
        }}
    }}
    """

    temp_file = pathlib.Path("test.ttgir")
    temp_file.write_text(ttgir)
    kernel = triton.compile(str(temp_file))
    a = torch.randint(0, 100, (TSIZE,), device="cuda").to(dtype=torch_dtype)
    b = torch.empty_like(a)
    kernel[(1, 1, 1)](a, b)
    torch.testing.assert_close(a, b)
    with proton.scope(f"tsize_{TSIZE}_dtype_{dtype}"):
        for _ in range(5):
            kernel[(1, 1, 1)](a, b)


for tsize in [32, 64, 128, 256]:
    for vsize in [2]:
        for dtype, torch_dtype in [
            ("i8", torch.int8),
            ("i16", torch.int16),
            ("i32", torch.int32),
        ]:
            run(tsize, vsize, dtype, torch_dtype)
