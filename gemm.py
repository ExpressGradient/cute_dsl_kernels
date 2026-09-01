import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import time
import statistics


@cute.kernel
def gemm_kernel(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, tiled_mma: cute.TiledMma
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    cta_tiler = (128, 128, 32)
    cta_coord = (bidx, bidy, None)

    # Create (M, K, Rest_K) and (N, K, Rest_K) tiles of A and B
    gA = cute.local_tile(mA, tiler=cta_tiler, coord=cta_coord, proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=cta_tiler, coord=cta_coord, proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=cta_tiler, coord=cta_coord, proj=(1, 1, None))

    # This lane's plan of MMA
    thr_mma = tiled_mma.get_slice(tidx)

    # Apply lane's mma plan onto (M, K, Rest_K)
    # (MMA, MMA_M, MMA_K, K_tiles)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)

    # Prepare registers for MMA
    # (MMA, MMA_M, MMA_K)
    tCrA = tiled_mma.make_fragment_A(tCgA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCgB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)

    # Fill accumulator with zeros
    tCrC.fill(0)

    num_k_tiles = cute.size(tCgA, mode=[3])
    num_k_blocks = cute.size(tCrA, mode=[2])

    for k_tile in range(num_k_tiles):
        # Fill the registers with global memory values for MMA
        cute.autovec_copy(tCgA[None, None, None, k_tile], tCrA)
        cute.autovec_copy(tCgB[None, None, None, k_tile], tCrB)

        # Loop for 32/16 K blocks
        for k_block in cutlass.range(num_k_blocks, unroll_full=True):
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_block],
                tCrB[None, None, k_block],
                tCrC,
            )

    tCrD = cute.make_fragment_like(tCrC, mC.element_type)
    tCrD[None] = tCrC.load().to(mC.element_type)

    cute.autovec_copy(tCrD, tCgC)


@cute.jit
def cute_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))

    # two warps along M and N directions
    atom_layout_mnk = cute.make_layout((2, 2, 1))

    tiled_mma = cute.make_tiled_mma(mma_op, atom_layout_mnk)

    grid_m = cute.ceil_div(mC.shape[0], 128)
    grid_n = cute.ceil_div(mC.shape[1], 128)

    # reshape tensor to (N, K) from (K, N)
    mB_nk = cute.make_tensor(mB.iterator, cute.select(mB.layout, mode=[1, 0]))

    gemm_kernel(mA, mB_nk, mC, tiled_mma).launch(
        grid=(grid_m, grid_n, 1), block=(128, 1, 1)
    )


a = torch.rand(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.rand(1024, 1024, device="cuda", dtype=torch.float16)
c = torch.empty_like(a)

a_cute = from_dlpack(a)
b_cute = from_dlpack(b)
c_cute = from_dlpack(c)


def benchmark(fn, warmup=5, iterations=100):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    samples_ms = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000)

    return statistics.median(samples_ms), statistics.quantiles(
        samples_ms, n=100, method="inclusive"
    )[94]


torch_c = torch.empty_like(c)

torch_median_ms, torch_p95_ms = benchmark(lambda: torch.mm(a, b, out=torch_c))

cute_median_ms, cute_p95_ms = benchmark(lambda: cute_gemm(a_cute, b_cute, c_cute))

torch.testing.assert_close(
    c,
    torch_c,
    rtol=1e-2,
    atol=1e-2,
)

print(f"Torch Matmul: Median {torch_median_ms:.4f} ms | P95 {torch_p95_ms:.4f} ms")
print(f"CuTe Matmul:  Median {cute_median_ms:.4f} ms | P95 {cute_p95_ms:.4f} ms")
