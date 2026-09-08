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

    # Allocate Shared Memory for entire CTA for A and B
    smem = cutlass.utils.SmemAllocator()

    num_stages = 2

    # swizzle address for 8x32 of sA
    sA_layout_atom = cute.make_layout((8, 32), stride=(32, 1))
    sA_layout = cute.tile_to_shape(sA_layout_atom, (128, 32, num_stages), (0, 1, 2))
    sA_swizzle = cute.make_swizzle(2, 4, 3)
    sA = smem.allocate_tensor(mA.element_type, sA_layout, byte_alignment=16, swizzle=sA_swizzle)

    # N Contiguous Layout
    sB_layout_atom = cute.make_layout((64, 8), stride=(1, 64))
    sB_layout = cute.tile_to_shape(sB_layout_atom, (128, 32, num_stages), (0, 1, 2))
    sB_swizzle = cute.make_swizzle(3, 4, 3)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, byte_alignment=16, swizzle=sB_swizzle)

    # Create 128-bit wide copy for Gmem to Smem
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        cutlass.Float16,
        num_bits_per_copy=128
    )

    # Each thread pulls 8 FP16 values, arrange each thread contiguous, 4 threads per row
    thread_layout_A = cute.make_layout((32, 4), stride=(4, 1))
    value_layout_A = cute.make_layout((1, 8))

    # For B, 16 contiguous threads do 128-bit loads
    thread_layout_B = cute.make_layout((16, 8), stride=(1, 16))
    value_layout_B = cute.make_layout((8, 1))

    # Create Tiled Copies
    g2s_copy_A = cute.make_tiled_copy_tv(copy_atom, thread_layout_A, value_layout_A)
    g2s_copy_B = cute.make_tiled_copy_tv(copy_atom, thread_layout_B, value_layout_B)

    # This lane's plan of Copies
    thr_copy_A = g2s_copy_A.get_slice(tidx)
    thr_copy_B = g2s_copy_B.get_slice(tidx)

    # Apply Copy plan to Source and Destination
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)

    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    # Prologue: first load into smem
    cute.copy(g2s_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
    cute.copy(g2s_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])

    cute.arch.cp_async_commit_group()

    # This lane's plan of MMA
    thr_mma = tiled_mma.get_slice(tidx)

    # (MMA, MMA_M, MMA_K)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)

    # Prepare registers for MMA
    # (MMA, MMA_M, MMA_K) = (8, 4, 2)
    tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = thr_mma.make_fragment_C(tCgC)

    # Fill accumulator with zeros
    tCrC.fill(0)

    # Copy Atoms for Smem to Rmem
    ldmatrix_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mA.element_type)
    ldmatrix_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), mB.element_type)

    s2r_copy_A = cute.make_tiled_copy_A(ldmatrix_A, tiled_mma)
    s2r_copy_B = cute.make_tiled_copy_B(ldmatrix_B, tiled_mma)

    # Lane's copy plan of Smem to Rmem
    thr_s2r_A = s2r_copy_A.get_slice(tidx)
    thr_s2r_B = s2r_copy_B.get_slice(tidx)

    tCsA_copy = thr_s2r_A.partition_S(sA)
    tCsB_copy = thr_s2r_B.partition_S(sB)

    # Retile fragment for S2R Copy
    tCrA_copy = thr_s2r_A.retile(tCrA)
    tCrB_copy = thr_s2r_B.retile(tCrB)

    num_k_tiles = cute.size(gA, mode=[2])
    num_k_blocks = cute.size(tCrA, mode=[2])

    read_stage = 0
    write_stage = 1

    for k_tile in range(num_k_tiles):
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if k_tile + 1 < num_k_tiles:
            cute.copy(
                g2s_copy_A, 
                tAgA[None, None, None, k_tile + 1], 
                tAsA[None, None, None, write_stage]
            )
            cute.copy(
                g2s_copy_B, 
                tBgB[None, None, None, k_tile + 1], 
                tBsB[None, None, None, write_stage]
            )
            cute.arch.cp_async_commit_group()

        # Warp-wide loads from Smem to Rmem
        cute.copy(s2r_copy_A, tCsA_copy[None, None, None, read_stage], tCrA_copy)
        cute.copy(s2r_copy_B, tCsB_copy[None, None, None, read_stage], tCrB_copy)

        # Loop for 32/16 K blocks
        for k_block in cutlass.range(num_k_blocks, unroll_full=True):
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_block],
                tCrB[None, None, k_block],
                tCrC
            )

        # swap stages
        read_stage, write_stage = write_stage, read_stage

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

    mB_nk = cute.make_tensor(
        mB.iterator,
        cute.select(mB.layout, mode=[1, 0]),
    )

    gemm_kernel(mA, mB_nk, mC, tiled_mma).launch(
        grid=(grid_m, grid_n, 1), block=(128, 1, 1), stream=stream)
