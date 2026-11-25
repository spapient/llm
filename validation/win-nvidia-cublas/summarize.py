import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
BASELINE = json.loads((ROOT.parent / "win-rtx-a3000/summary.json").read_text())


def runs(prefix):
    tables = []
    for path in sorted(ROOT.glob(f"{prefix}_run_*.csv")):
        with path.open(newline="") as handle:
            tables.append(list(csv.DictReader(handle)))
    assert len(tables) == 5, (prefix, len(tables))
    return tables


def values(tables, row, column, positive=False):
    result = [float(table[row][column]) for table in tables]
    return [value for value in result if value >= 0] if positive else result


def stats(items):
    return {"median": statistics.median(items), "min": min(items), "max": max(items)}


validation = runs("validation")
main = runs("benchmark")
vendor = runs("cublas")
assert all(len(table) == 15 for table in validation)
assert all(len(table) == 4 for table in main + vendor)
errors = {
    "validation": max(float(row["max_scaled_error"]) for table in validation for row in table),
    "benchmark": max(float(row["max_scaled_error"]) for table in main for row in table),
    "cublas_comparison": max(float(row["max_scaled_error"]) for table in vendor for row in table),
}
assert all(value <= 1e-6 for value in errors.values())

source = WORKSPACE / "crates/prof/src/nvidia/kernels/matmul_f32.k"
source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
rendered_sha256 = hashlib.sha256((ROOT / "matmul_f32_cublas.cu").read_bytes()).hexdigest()
assert source_sha256 == "af1d35440124945d771e1992deca3b0ff5108598f86192ab68cd385826ae99b3"
assert rendered_sha256 == "78fcb87bb852ff6f2e5d511aa59a9e807428ec79e84af8640773177fcbb76a66"

rows = []
for index, first in enumerate(main[0]):
    M, N, K = (int(first[name]) for name in ("M", "N", "K"))
    assert tuple(int(vendor[0][index][name]) for name in ("M", "N", "K")) == (M, N, K)
    old = BASELINE["results"][index]
    assert (old["M"], old["N"], old["K"]) == (M, N, K)
    row = {"M": M, "N": N, "K": K, "runs": 5, "reps_per_run": int(first["reps"])}
    for prefix, tables, column in [
        ("tiled_kernel", main, "candidate_ms"),
        ("naive_cuda", main, "naive_cuda_ms"),
        ("cublas_abi_first", main, "public_abi_first_ms"),
        ("cublas_abi_e2e", main, "public_abi_e2e_ms"),
        ("direct_cublas", vendor, "cublas_fp32_ms"),
    ]:
        row[f"{prefix}_ms"] = stats(values(tables, index, column))
    cpu = values(main, index, "cpu_ms", positive=True)
    row["cpu_ms"] = stats(cpu) if cpu else None
    row["previous_tiled_abi_e2e_ms"] = old["public_abi_e2e_ms"]["median"]
    e2e = row["cublas_abi_e2e_ms"]["median"]
    direct = row["direct_cublas_ms"]["median"]
    work = 2 * M * N * K
    row["cublas_abi_gflops_from_median"] = work / (e2e * 1e6)
    row["direct_cublas_gflops_from_median"] = work / (direct * 1e6)
    row["abi_speedup_vs_previous_tiled"] = old["public_abi_e2e_ms"]["median"] / e2e
    row["abi_overhead_vs_direct_cublas"] = e2e / direct
    row["first_call_overhead_ratio"] = row["cublas_abi_first_ms"]["median"] / e2e
    row["tiled_kernel_speedup_vs_naive"] = (
        row["naive_cuda_ms"]["median"] / row["tiled_kernel_ms"]["median"]
    )
    if cpu:
        row["abi_speedup_vs_cpu"] = row["cpu_ms"]["median"] / e2e
    rows.append(row)

summary = {
    "schema": 1,
    "scope": "NVIDIA cuBLAS specialization compiled and executed with CUDA; not MUSA evidence.",
    "profile_note": "The built-in nvidia profile targets T600/sm75. This remote A3000 validation used the exact backend source in a standalone sm_86 harness, not profile selection.",
    "timestamp": (ROOT / "environment.txt").read_text().splitlines()[0].split("=", 1)[1],
    "device": "NVIDIA RTX A3000 12GB Laptop GPU",
    "compute_capability": "8.6",
    "driver": "621.14",
    "cuda": "13.3.33",
    "cpu": "11th Gen Intel Core i7-11700 @ 2.50GHz",
    "compile": "nvcc -std=c++17 -O3 -arch=sm_86 -lineinfo -Xcompiler /O2 -lcublas",
    "math_mode": "CUBLAS_PEDANTIC_MATH",
    "source_sha256": source_sha256,
    "rendered_sha256": rendered_sha256,
    "implementation": {
        "row_major_abi": True,
        "cublas_handle_cached": True,
        "device_buffers_cached_at_high_water": True,
        "cache_checked_against_current_device": True,
        "cache_serialized_with_mutex": True,
        "host_to_device_and_device_to_host_copies_retain_per_call": True,
        "silent_cpu_fallback": False,
    },
    "validation": {
        "cases_per_run": 15,
        "runs": 5,
        "total_case_executions": 75,
        "max_scaled_error": errors["validation"],
        "pass": True,
    },
    "benchmark": {
        "runs": 5,
        "cases_per_run": 4,
        "max_scaled_error_vs_cpu_or_naive": errors["benchmark"],
        "max_scaled_error_vs_direct_cublas": errors["cublas_comparison"],
        "timing": "CUDA events for kernel-only; steady_clock for public ABI and CPU",
        "statistics": "Median, minimum and maximum across five independent process executions",
    },
    "results": rows,
}
(ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

fields = [
    "M",
    "N",
    "K",
    "runs",
    "reps_per_run",
    "cublas_abi_gflops_from_median",
    "direct_cublas_gflops_from_median",
    "abi_speedup_vs_previous_tiled",
    "abi_overhead_vs_direct_cublas",
    "first_call_overhead_ratio",
    "tiled_kernel_speedup_vs_naive",
    "abi_speedup_vs_cpu",
    "previous_tiled_abi_e2e_ms",
]
for name in [
    "tiled_kernel",
    "naive_cuda",
    "cublas_abi_first",
    "cublas_abi_e2e",
    "direct_cublas",
    "cpu",
]:
    fields += [f"{name}_ms_median", f"{name}_ms_min", f"{name}_ms_max"]
with (ROOT / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        flat = {key: row.get(key) for key in fields}
        for name in [
            "tiled_kernel",
            "naive_cuda",
            "cublas_abi_first",
            "cublas_abi_e2e",
            "direct_cublas",
            "cpu",
        ]:
            for statistic in ["median", "min", "max"]:
                flat[f"{name}_ms_{statistic}"] = (
                    row[f"{name}_ms"][statistic] if row[f"{name}_ms"] else None
                )
        writer.writerow(flat)

for row in rows:
    print(
        f"{row['M']}x{row['N']}x{row['K']}: "
        f"ABI={row['cublas_abi_e2e_ms']['median']:.6f} ms "
        f"(previous={row['previous_tiled_abi_e2e_ms']:.6f} ms, "
        f"speedup={row['abi_speedup_vs_previous_tiled']:.3f}x), "
        f"direct cuBLAS={row['direct_cublas_ms']['median']:.6f} ms"
    )
print(f"PASS: 75/75 validation executions; max error={errors['validation']}")
