# cute_dsl_kernels

A growing collection of handwritten GPU kernels built while learning kernel
engineering with NVIDIA CuTe DSL.

This repository is a progress log, not a production library. Each kernel keeps
the implementation and comments from its learning stage. Benchmarking is kept
separate.

## Kernels

### GEMM

The current GEMM kernels target SM80 FP16 tensor cores with FP32 accumulation,
a `128 x 128 x 32` CTA tile, and four warps per block.

- `gemm.py`: original GEMM and its historical in-file benchmark.
- `gemm_direct_kernel.py`: direct GMEM-to-RMEM loads followed by warp MMA.
- `gemm_smem_kernel.py`: single-stage, swizzled SMEM staging.
- `gemm_smem_pipeline.py`: two-stage GMEM-to-SMEM pipeline.

## Benchmarking

`benchmark_gemm.py` imports a kernel and tests representative modern LLM
inference projection shapes. It compiles outside the timed region, preallocates
outputs, uses one CUDA stream and CUDA events, warms up both implementations,
alternates their order across seven rounds, reports the median, and checks every
result against `torch.mm`.

```bash
uv run benchmark_gemm.py gemm_direct_kernel --output gemm_direct_benchmark.csv
uv run benchmark_gemm.py gemm_smem_kernel --output gemm_smem_benchmark.csv
uv run benchmark_gemm.py gemm_smem_pipeline --output gemm_pipeline_benchmark.csv
```

Results are written to `results/` automatically.

### RTX 3060 Laptop GPU summary

| Workload | M x N x K | Direct | Single-stage SMEM | Two-stage pipeline |
|---|---:|---:|---:|---:|
| Batched decode | 128 x 4096 x 4096 | 0.5150 ms / 0.490x | 0.4111 ms / 0.720x | 0.4102 ms / 0.730x |
| Short prefill | 512 x 4096 x 4096 | 0.9821 ms / **1.029x** | 0.8703 ms / **1.088x** | 0.8686 ms / **1.108x** |
| Long prefill | 4096 x 4096 x 4096 | 8.5310 ms / 0.723x | 6.9407 ms / 0.882x | 7.0273 ms / 0.869x |
| QKV projection | 1024 x 6144 x 4096 | 2.6428 ms / **1.054x** | 2.3103 ms / **1.160x** | 2.3149 ms / **1.178x** |
| FFN gate+up | 1024 x 24576 x 4096 | 11.5133 ms / 0.899x | 10.2446 ms / **1.014x** | 10.0078 ms / **1.023x** |
| FFN down | 1024 x 4096 x 12288 | 5.9140 ms / 0.918x | 5.1083 ms / **1.042x** | 5.3400 ms / **1.022x** |
| Wide QKV projection | 1024 x 8192 x 3072 | 2.7893 ms / 0.956x | 2.3736 ms / **1.073x** | 2.5103 ms / **1.011x** |
| Sliding QKV projection | 1024 x 10240 x 2048 | 2.2702 ms / **1.018x** | 1.9915 ms / **1.092x** | 2.0308 ms / **1.073x** |
| Compact QKV projection | 1024 x 3072 x 2048 | 0.7806 ms / 0.918x | 0.6442 ms / **1.036x** | 0.6414 ms / **1.046x** |

Each ratio is `Torch time / CuTe time`; values above `1.0x` favor CuTe. The
geometric-mean speedups are `0.869x` for direct loads, `1.003x` for single-stage
SMEM, and `0.998x` for the two-stage pipeline.

Full results: [direct](results/gemm_direct_benchmark.csv),
[single-stage SMEM](results/gemm_smem_benchmark.csv), and
[two-stage pipeline](results/gemm_pipeline_benchmark.csv).
