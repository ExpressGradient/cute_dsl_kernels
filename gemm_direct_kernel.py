import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma
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
                tCrC
            )

    tCrD = cute.make_fragment_like(tCrC, mC.element_type)
    tCrD[None] = tCrC.load().to(mC.element_type)

    cute.autovec_copy(tCrD, tCgC)



@cute.jit
def cute_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: cuda.CUstream,
):
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16, cutlass.Float32, (16, 8, 16))

    # two warps along M and N directions
    atom_layout_mnk = cute.make_layout((2, 2, 1))

    tiled_mma = cute.make_tiled_mma(mma_op, atom_layout_mnk)

    grid_m = cute.ceil_div(mC.shape[0], 128)
    grid_n = cute.ceil_div(mC.shape[1], 128)

    # reshape tensor to (N, K) from (K, N)
    mB_nk = cute.make_tensor(
        mB.iterator, cute.select(mB.layout, mode=[1, 0]))

    gemm_kernel(mA, mB_nk, mC, tiled_mma).launch(
        grid=(grid_m, grid_n, 1), block=(128, 1, 1), stream=stream)
