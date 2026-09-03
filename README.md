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

### RTX 3060 Laptop GPU summary

| M×N×K | Direct ms | Direct/Torch | SMEM ms | SMEM/Torch |
|---:|---:|---:|---:|---:|
| 256×256×256 | 0.0286 | 0.532x | 0.0213 | 0.716x |
| 512×512×512 | 0.0490 | 0.763x | 0.0397 | 0.936x |
| 1024×1024×1024 | 0.1882 | **1.139x** | 0.1868 | **1.153x** |
| 2048×2048×2048 | 1.0732 | 0.847x | 0.9990 | 0.916x |
| 4096×4096×4096 | 9.3098 | 0.739x | 7.0768 | 0.950x |
| 256×1024×512 | 0.0503 | 0.735x | 0.0393 | 0.908x |
| 1024×256×512 | 0.0496 | 0.696x | 0.0705 | 0.874x |
| 512×2048×1024 | 0.1805 | 0.764x | 0.3406 | 0.709x |
| 2048×512×1024 | 0.3292 | 0.793x | 0.3392 | 0.769x |

Ratios above `1.0x` mean the CuTe kernel is faster than its paired Torch run.

On the RTX 3060 Laptop GPU, both CuTe kernels beat `torch.mm` at
`1024 x 1024 x 1024`. PyTorch is faster on the other tested shapes. See the
[direct results](results/gemm_direct_benchmark.md) and
[single-stage SMEM results](results/gemm_smem_benchmark.md).
