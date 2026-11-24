from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def runs(prefix):
    tables = []
    for path in sorted(ROOT.glob(f"{prefix}_run_*.csv")):
        with path.open(newline="") as handle:
            tables.append(list(csv.DictReader(handle)))
    assert len(tables) == 5, (prefix, len(tables))
    return tables


def values(tables, row, column, positive=False):
    result = [float(table[row][column]) for table in tables]
    if positive:
        result = [value for value in result if value >= 0]
    return result


def stats(values_):
    return {
        "median": statistics.median(values_),
        "min": min(values_),
        "max": max(values_),
    }


validation = runs("validation")
main = runs("benchmark")
vendor = runs("cublas")
assert all(len(table) == 15 for table in validation)
assert all(len(table) == 4 for table in main + vendor)
validation_error = max(float(row["max_scaled_error"]) for table in validation for row in table)
main_error = max(float(row["max_scaled_error"]) for table in main for row in table)
vendor_error = max(float(row["max_scaled_error"]) for table in vendor for row in table)
assert validation_error <= 1e-6
assert main_error <= 1e-6
assert vendor_error <= 1e-6

source = WORKSPACE / "crates/prof/src/musa/kernels/matmul_f32.k"
source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
rendered_sha256 = hashlib.sha256((ROOT / "matmul_f32.cu").read_bytes()).hexdigest()
assert source_sha256 == "6b72eb9a82af65435a7e3771b1e231a9aaafc02759259b9bfc52a5d16890c433"
assert rendered_sha256 == "e8a102bf3c6c9c0232df4b10e4b1f6e2af7d41fd0fbea733931c3579c1474484"

rows = []
for index, first in enumerate(main[0]):
    key = tuple(int(first[name]) for name in ("M", "N", "K"))
    assert key == tuple(int(vendor[0][index][name]) for name in ("M", "N", "K"))
    M, N, K = key
    row = {"M": M, "N": N, "K": K, "runs": 5, "reps_per_run": int(first["reps"])}
    for prefix, tables, column in [
        ("candidate", vendor, "candidate_ms"),
        ("naive_cuda", main, "naive_cuda_ms"),
        ("public_abi_e2e", main, "public_abi_e2e_ms"),
        ("cublas_fp32", vendor, "cublas_fp32_ms"),
    ]:
        row[f"{prefix}_ms"] = stats(values(tables, index, column))
    cpu = values(main, index, "cpu_ms", positive=True)
    row["cpu_ms"] = stats(cpu) if cpu else None
    candidate_ms = row["candidate_ms"]["median"]
    row["candidate_gflops_from_median"] = 2 * M * N * K / (candidate_ms * 1e6)
    row["speedup_vs_naive_cuda"] = row["naive_cuda_ms"]["median"] / candidate_ms
    row["cublas_speedup_over_candidate"] = candidate_ms / row["cublas_fp32_ms"]["median"]
    row["candidate_percent_of_cublas"] = 100 * row["cublas_fp32_ms"]["median"] / candidate_ms
    row["public_abi_overhead_ratio"] = row["public_abi_e2e_ms"]["median"] / candidate_ms
    if cpu:
        row["public_abi_speedup_vs_cpu"] = row["cpu_ms"]["median"] / row["public_abi_e2e_ms"]["median"]
    rows.append(row)

summary = {
    "schema": 1,
    "scope": "Same candidate source compiled by NVIDIA CUDA; this is not MUSA hardware evidence.",
    "timestamp": (ROOT / "environment.txt").read_text().splitlines()[0].split("=", 1)[1],
    "device": "NVIDIA RTX A3000 12GB Laptop GPU",
    "compute_capability": "8.6",
    "driver": "621.14",
    "cuda": "13.3.33",
    "cpu": "11th Gen Intel Core i7-11700 @ 2.50GHz",
    "compile": "nvcc -std=c++17 -O3 -arch=sm_86 -lineinfo -Xcompiler /O2",
    "source_sha256": source_sha256,
    "rendered_sha256": rendered_sha256,
    "validation": {
        "cases_per_run": 15,
        "runs": 5,
        "total_case_executions": 75,
        "max_scaled_error": validation_error,
        "pass": True,
    },
    "benchmark": {
        "runs": 5,
        "cases_per_run": 4,
        "max_scaled_error_vs_cpu_or_naive": main_error,
        "max_scaled_error_vs_cublas": vendor_error,
        "timing": "CUDA events for kernel-only; steady_clock for public ABI and CPU",
        "statistics": "Median, minimum and maximum across five independent process executions",
    },
    "musa": {
        "mcc_on_windows_path": False,
        "mcc_on_ubuntu_wsl_path": False,
        "musa_device_detected": False,
        "musa_evidence_obtained": False,
    },
    "results": rows,
}

(ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
fields = [
    "M", "N", "K", "runs", "reps_per_run", "candidate_gflops_from_median",
    "speedup_vs_naive_cuda", "cublas_speedup_over_candidate", "candidate_percent_of_cublas",
    "public_abi_overhead_ratio", "public_abi_speedup_vs_cpu",
]
for name in ["candidate", "naive_cuda", "public_abi_e2e", "cpu", "cublas_fp32"]:
    fields += [f"{name}_ms_median", f"{name}_ms_min", f"{name}_ms_max"]
with (ROOT / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        flat = {key: row.get(key) for key in fields}
        for name in ["candidate", "naive_cuda", "public_abi_e2e", "cpu", "cublas_fp32"]:
            for statistic in ["median", "min", "max"]:
                flat[f"{name}_ms_{statistic}"] = row[f"{name}_ms"][statistic] if row[f"{name}_ms"] else None
        writer.writerow(flat)

for row in rows:
    cpu = row["cpu_ms"]["median"] if row["cpu_ms"] else None
    print(
        f"{row['M']}x{row['N']}x{row['K']}: "
        f"candidate={row['candidate_ms']['median']:.6f} ms, "
        f"naive={row['naive_cuda_ms']['median']:.6f} ms, "
        f"cuBLAS={row['cublas_fp32_ms']['median']:.6f} ms, "
        f"e2e={row['public_abi_e2e_ms']['median']:.6f} ms, "
        f"CPU={cpu if cpu is not None else 'skipped'}"
    )
print(f"PASS: 75/75 validation executions; max error={validation_error}")
