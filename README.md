# cute_dsl_kernels

CuTe DSL kernels built incrementally to learn and optimize GPU matrix multiplication.

## Kernels

Both are handwritten SM80 FP16 GEMMs using `m16n8k16` warp MMA, FP32
accumulation, a `128 x 128 x 32` CTA tile, and four warps per block.

- `gemm_direct_kernel.py`: GMEM to RMEM, then warp MMA.
- `gemm_smem_kernel.py`: single-stage SMEM staging with a swizzled layout.

Kernel code and benchmark code are kept separate.

## Benchmark

```bash
uv run benchmark_gemm.py gemm_direct_kernel --output gemm_direct_benchmark.csv
uv run benchmark_gemm.py gemm_smem_kernel --output gemm_smem_benchmark.csv
```

Results are written to `results/` automatically.

The benchmark covers nine square and rectangular shapes. It compiles outside
the timed region, preallocates outputs, uses one CUDA stream and CUDA events,
warms up both implementations, alternates their order across seven rounds, and
checks every result against `torch.mm`.

On the RTX 3060 Laptop GPU, both CuTe kernels beat `torch.mm` at
`1024 x 1024 x 1024`. PyTorch is faster on the other tested shapes. See the
[direct results](results/gemm_direct_benchmark.md) and
[single-stage SMEM results](results/gemm_smem_benchmark.md).
