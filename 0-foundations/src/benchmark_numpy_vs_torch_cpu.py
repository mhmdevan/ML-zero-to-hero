"""
Benchmark: NumPy vs PyTorch CPU for matrix multiplication.

Measures:
- elapsed time
- approximate peak memory (RSS)

Run:
    python -m src.benchmark_numpy_vs_torch_cpu
    python -m src.benchmark_numpy_vs_torch_cpu --matrix-size 512 --repetitions 20
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BenchmarkConfig:
    matrix_size: int = 1024
    repetitions: int = 15
    seed: int = 42
    torch_num_threads: int = 1
    output_dir: Path = Path("output") / "benchmarks"


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    matrix_size: int
    repetitions: int
    elapsed_seconds: float
    avg_seconds_per_iter: float
    peak_memory_mb: float
    checksum: float


def _normalize_max_rss_to_mb(raw_rss: int) -> float:
    """
    Normalize ru_maxrss to MB.

    On Linux: ru_maxrss is in KB.
    On macOS: ru_maxrss is in bytes.
    """
    if sys.platform == "darwin":
        return raw_rss / (1024.0 * 1024.0)
    return raw_rss / 1024.0


def _peak_rss_mb() -> float:
    if resource is None:
        return float("nan")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return _normalize_max_rss_to_mb(usage.ru_maxrss)


def _compute_matmul(backend: str, cfg: BenchmarkConfig) -> tuple[float, float]:
    rng = np.random.default_rng(seed=cfg.seed)
    a_np = rng.normal(loc=0.0, scale=0.1, size=(cfg.matrix_size, cfg.matrix_size)).astype(
        np.float32
    )
    b_np = rng.normal(loc=0.0, scale=0.1, size=(cfg.matrix_size, cfg.matrix_size)).astype(
        np.float32
    )

    if backend == "numpy":
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            _ = a_np @ b_np  # warm-up

        checksum = 0.0
        start = time.perf_counter()
        for _ in range(cfg.repetitions):
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                c_np = a_np @ b_np
            checksum += float(c_np[0, 0])
        elapsed = time.perf_counter() - start
        return elapsed, checksum

    if backend == "torch":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for backend='torch'.") from exc

        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.torch_num_threads)

        a_t = torch.from_numpy(a_np)
        b_t = torch.from_numpy(b_np)
        _ = torch.matmul(a_t, b_t)  # warm-up

        checksum = 0.0
        start = time.perf_counter()
        for _ in range(cfg.repetitions):
            c_t = torch.matmul(a_t, b_t)
            checksum += float(c_t[0, 0].item())
        elapsed = time.perf_counter() - start
        return elapsed, checksum

    raise ValueError(f"Unsupported backend: {backend}")


def benchmark_once(backend: str, cfg: BenchmarkConfig) -> BenchmarkResult:
    """Run a benchmark backend in the current process (useful for tests)."""
    elapsed, checksum = _compute_matmul(backend, cfg)
    peak_mb = _peak_rss_mb()

    return BenchmarkResult(
        backend=backend,
        matrix_size=cfg.matrix_size,
        repetitions=cfg.repetitions,
        elapsed_seconds=elapsed,
        avg_seconds_per_iter=elapsed / cfg.repetitions,
        peak_memory_mb=peak_mb,
        checksum=checksum,
    )


def _worker(backend: str, cfg: BenchmarkConfig, queue: mp.Queue) -> None:
    try:
        result = benchmark_once(backend, cfg)
        queue.put(asdict(result))
    except Exception as exc:  # pragma: no cover
        queue.put({"error": repr(exc)})


def benchmark_in_subprocess(backend: str, cfg: BenchmarkConfig) -> BenchmarkResult:
    """
    Run benchmark in a fresh subprocess to get isolated peak RSS.
    """
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(backend, cfg, queue))
    process.start()
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(f"Benchmark process for {backend} failed with exit code {process.exitcode}")

    payload = queue.get_nowait()
    if "error" in payload:
        raise RuntimeError(f"Benchmark worker error for {backend}: {payload['error']}")

    return BenchmarkResult(**payload)


def run_benchmark(cfg: BenchmarkConfig, backend_mode: str = "both") -> list[BenchmarkResult]:
    backends: tuple[str, ...]
    if backend_mode == "both":
        backends = ("numpy", "torch")
    elif backend_mode == "numpy":
        backends = ("numpy",)
    elif backend_mode == "torch":
        backends = ("torch",)
    else:
        raise ValueError(f"Unsupported backend_mode: {backend_mode}")

    results: list[BenchmarkResult] = []
    for backend in backends:
        results.append(benchmark_in_subprocess(backend, cfg))
    return results


def save_results(results: list[BenchmarkResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "numpy_vs_torch_cpu_benchmark.json"
    csv_path = output_dir / "numpy_vs_torch_cpu_benchmark.csv"

    rows = [asdict(result) for result in results]

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def print_summary(results: list[BenchmarkResult]) -> None:
    print("backend | total_sec | sec_per_iter | peak_mem_mb")
    print("-" * 52)
    for result in results:
        print(
            f"{result.backend:7s} | {result.elapsed_seconds:9.4f} | "
            f"{result.avg_seconds_per_iter:12.6f} | {result.peak_memory_mb:11.2f}"
        )

    has_numpy = any(result.backend == "numpy" for result in results)
    has_torch = any(result.backend == "torch" for result in results)
    if has_numpy and has_torch:
        numpy_result = next(result for result in results if result.backend == "numpy")
        torch_result = next(result for result in results if result.backend == "torch")
        speedup = torch_result.elapsed_seconds / numpy_result.elapsed_seconds
        print()
        print(f"time_ratio_torch_over_numpy: {speedup:.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs PyTorch CPU matmul.")
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument(
        "--backend",
        type=str,
        choices=("both", "numpy", "torch"),
        default="both",
        help="Which backend(s) to benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/benchmarks",
        help="Directory to save CSV/JSON benchmark results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = BenchmarkConfig(
        matrix_size=args.matrix_size,
        repetitions=args.repetitions,
        seed=args.seed,
        torch_num_threads=args.torch_num_threads,
        output_dir=Path(args.output_dir),
    )

    print(
        "Running CPU benchmark with "
        f"matrix_size={cfg.matrix_size}, repetitions={cfg.repetitions}, "
        f"torch_num_threads={cfg.torch_num_threads}"
    )

    results = run_benchmark(cfg, backend_mode=args.backend)
    print_summary(results)

    json_path, csv_path = save_results(results, cfg.output_dir)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
