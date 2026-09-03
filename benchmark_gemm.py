import argparse
import csv
import importlib
import statistics
from functools import partial
from pathlib import Path

import cuda.bindings.driver as cuda
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream

SHAPES = (
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (256, 1024, 512),
    (1024, 256, 512),
    (512, 2048, 1024),
    (2048, 512, 1024),
)


def measure(fn, stream, count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(count):
        fn()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    parser.add_argument("--output")
    args = parser.parse_args()

    kernel = importlib.import_module(args.module).cute_gemm
    torch_stream = torch.cuda.current_stream()
    cute_stream = cuda.CUstream(torch_stream.cuda_stream)
    rows = []

    for m, n, k in SHAPES:
        a = torch.rand(m, k, device="cuda", dtype=torch.float16)
        b = torch.rand(k, n, device="cuda", dtype=torch.float16)
        torch_c = torch.empty((m, n), device="cuda", dtype=torch.float16)
        cute_c = torch.empty_like(torch_c)

        a_ = from_dlpack(a, assumed_align=16)
        b_ = from_dlpack(b, assumed_align=16)
        c_ = from_dlpack(cute_c, assumed_align=16)
        compiled = cute.compile(kernel, a_, b_, c_, make_fake_stream())
        torch_fn = partial(torch.mm, a, b, out=torch_c)
        cute_fn = partial(compiled, a_, b_, c_, cute_stream)

        for _ in range(20):
            torch_fn()
            cute_fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(cute_c, torch_c, rtol=1e-2, atol=1e-2)

        count = max(10, min(200, round(100 * 1024**3 / (m * n * k))))
        samples = {"torch": [], "cute": []}
        for round_index in range(7):
            order = (("torch", torch_fn), ("cute", cute_fn))
            if round_index % 2:
                order = reversed(order)
            for name, fn in order:
                samples[name].append(measure(fn, torch_stream, count))

        torch_ms = statistics.median(samples["torch"])
        cute_ms = statistics.median(samples["cute"])
        speedup = torch_ms / cute_ms
        rows.append((m, n, k, torch_ms, cute_ms, speedup))
        print(
            f"{m}x{n}x{k}: Torch {torch_ms:.4f} ms | "
            f"CuTe {cute_ms:.4f} ms | {speedup:.3f}x | PASS",
            flush=True,
        )

    output = Path("results") / (args.output or f"{args.module}_benchmark.csv")
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("M", "N", "K", "Torch ms", "CuTe ms", "Speedup"))
        writer.writerows(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
