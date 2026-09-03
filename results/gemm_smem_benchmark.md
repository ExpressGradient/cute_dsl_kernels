# GEMM benchmark: gemm_smem_kernel

Device: NVIDIA GeForce RTX 3060 Laptop GPU

Final VM run. CUDA events, same stream, 20 warmups, alternating order, seven rounds.

| M×N×K | Torch ms | CuTe ms | Speedup | Check |
|---:|---:|---:|---:|:---:|
| 256×256×256 | 0.0153 | 0.0213 | 0.716x | PASS |
| 512×512×512 | 0.0372 | 0.0397 | 0.936x | PASS |
| 1024×1024×1024 | 0.2154 | 0.1868 | 1.153x | PASS |
| 2048×2048×2048 | 0.9155 | 0.9990 | 0.916x | PASS |
| 4096×4096×4096 | 6.7230 | 7.0768 | 0.950x | PASS |
| 256×1024×512 | 0.0357 | 0.0393 | 0.908x | PASS |
| 1024×256×512 | 0.0616 | 0.0705 | 0.874x | PASS |
| 512×2048×1024 | 0.2415 | 0.3406 | 0.709x | PASS |
| 2048×512×1024 | 0.2608 | 0.3392 | 0.769x | PASS |
