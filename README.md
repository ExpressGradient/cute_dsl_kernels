# cute_dsl_kernels

CuTe DSL kernels built incrementally to learn and optimize GPU matrix multiplication.

## Current kernel

Naive SM80 FP16 GEMM using `m16n8k16` warp MMA:

```text
GMEM A/B -> RMEM fragments -> warp MMA -> FP32 accumulator -> GMEM C
```

- Matrix size: `1024 x 1024`
- CTA tile: `128 x 128 x 32`
- MMA atom layout: `(2, 2, 1)` — four warps per block
- FP16 inputs/output with FP32 accumulation
- Direct global-memory-to-register loads; no shared-memory staging yet
- Checked against `torch.mm`

## Current benchmark

| Implementation | Median | P95 |
| --- | ---: | ---: |
| PyTorch `torch.mm` | 0.2516 ms | 0.2659 ms |
| Naive CuTe GEMM | 16.7678 ms | 17.7127 ms |

Measured with 5 warmup runs and 100 synchronized iterations.

## Run

```bash
uv run gemm.py
```
