# GEMM benchmark: gemm_direct_kernel

Device: NVIDIA GeForce RTX 3060 Laptop GPU

Final VM run. CUDA events, same stream, 20 warmups, alternating order, seven rounds.

| M×N×K | Torch ms | CuTe ms | Speedup | Check |
|---:|---:|---:|---:|:---:|
| 256×256×256 | 0.0152 | 0.0286 | 0.532x | PASS |
| 512×512×512 | 0.0374 | 0.0490 | 0.763x | PASS |
| 1024×1024×1024 | 0.2143 | 0.1882 | 1.139x | PASS |
| 2048×2048×2048 | 0.9092 | 1.0732 | 0.847x | PASS |
| 4096×4096×4096 | 6.8803 | 9.3098 | 0.739x | PASS |
| 256×1024×512 | 0.0370 | 0.0503 | 0.735x | PASS |
| 1024×256×512 | 0.0346 | 0.0496 | 0.696x | PASS |
| 512×2048×1024 | 0.1379 | 0.1805 | 0.764x | PASS |
| 2048×512×1024 | 0.2609 | 0.3292 | 0.793x | PASS |
