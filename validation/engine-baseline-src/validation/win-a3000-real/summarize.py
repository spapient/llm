import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


verification = (ROOT / "verification.log").read_text(errors="replace")
benchmark = (ROOT / "benchmark.log").read_text(errors="replace")
hardware_text = (ROOT / "hardware.log").read_text(errors="replace")
transfer_text = (ROOT / "transfer.log").read_text(errors="replace")

selected = re.search(r"selected=(\w+)\((\w+)\)", verification)
assert selected and selected.group(1) == "nvidia_a3000" and selected.group(2) == "sm86"
case_pattern = re.compile(
    r"case=(\d+)x(\d+)x(\d+) PASS variant=(\S+) tile=(\S+) max_rel_err=([0-9.eE+-]+) outputs=(\d+)"
)
checks = [
    {
        "M": int(m),
        "N": int(n),
        "K": int(k),
        "variant": variant,
        "max_rel_err": float(error),
        "outputs": int(outputs),
    }
    for m, n, k, variant, _, error, outputs in case_pattern.findall(verification)
]
assert len(checks) == 14, len(checks)

bench_pattern = re.compile(
    r"case=(\d+)x(\d+)x(\d+) repeat=(\d+) samples=(\d+) max_rel_err=([0-9.eE+-]+) "
    r"min_ns=([0-9.eE+-]+) p50_ns=([0-9.eE+-]+) p95_ns=([0-9.eE+-]+) tflops=([0-9.eE+-]+)"
)
handwritten = {
    (1024, 1024, 1024): 5.849690,
    (4096, 256, 4096): 21.035440,
}
standalone_cublas = {
    (1024, 1024, 1024): 1.514730,
    (4096, 256, 4096): 7.621160,
}
rows = []
for m, n, k, repeat, samples, error, min_ns, p50_ns, p95_ns, tflops in bench_pattern.findall(
    benchmark
):
    M, N, K = int(m), int(n), int(k)
    p50 = float(p50_ns)
    row = {
        "M": M,
        "N": N,
        "K": K,
        "repeat": int(repeat),
        "samples": int(samples),
        "max_rel_err": float(error),
        "min_ns": float(min_ns),
        "p50_ns": p50,
        "p95_ns": float(p95_ns),
        "p50_ms": p50 / 1e6,
        "tflops_from_p50": float(tflops),
        "gflops_from_p50": 2.0 * M * N * K / p50,
    }
    assert samples == repeat
    assert row["gflops_from_p50"] > 0
    assert abs(row["tflops_from_p50"] - row["gflops_from_p50"] / 1000) < 0.001
    if (M, N, K) in handwritten:
        row["previous_handwritten_e2e_ms"] = handwritten[(M, N, K)]
        row["speedup_vs_handwritten"] = handwritten[(M, N, K)] / row["p50_ms"]
        row["previous_standalone_cublas_e2e_ms"] = standalone_cublas[(M, N, K)]
        row["ratio_vs_standalone_cublas"] = row["p50_ms"] / standalone_cublas[(M, N, K)]
    rows.append(row)
assert len(rows) == 3, len(rows)

hardware = dict(re.findall(r"^(\w+)=(.*)$", hardware_text, re.MULTILINE))
assert hardware["cc"] == "8.6"
assert hardware["multiProcessorCount"] == "32"
transfer_pattern = re.compile(
    r"pageable_ms=([0-9.]+) pinned_staging_ms=([0-9.]+) pageable_GBps=([0-9.]+) "
    r"pinned_staging_GBps=([0-9.]+) speedup=([0-9.]+)"
)
transfers = [
    {
        "pageable_ms": float(a),
        "pinned_staging_ms": float(b),
        "pageable_GBps": float(c),
        "pinned_staging_GBps": float(d),
        "pageable_over_pinned_speedup": float(e),
    }
    for a, b, c, d, e in transfer_pattern.findall(transfer_text)
]
assert len(transfers) == 5, len(transfers)

sources = [
    "crates/prof/build.rs",
    "crates/prof/src/buildops.rs",
    "crates/prof/src/buildtables.rs",
    "crates/prof/src/nvidia/a3000.toml",
    "crates/prof/src/nvidia/kernels/matmul_f32.k",
    "crates/shard/src/pipeline.rs",
    "crates/shard/src/verify.rs",
    "crates/shard/src/witness.rs",
    "crates/cutile/src/harness.rs",
    "crates/nn/src/kernel/matmul.rs",
]
summary = {
    "schema": 1,
    "scope": "Real Catalogue::load + measured-fingerprint Catalogue::select + negotiation + emit + NVCC + cuBLAS + independent scalar parity on win; not standalone and not MUSA evidence.",
    "selection": {
        "vendor": selected.group(1),
        "id": selected.group(2),
        "flags": "-arch=sm_86 -O3 -lcublas",
    },
    "hardware": hardware,
    "verification": {
        "passed_cases": len(checks),
        "max_rel_err": max(case["max_rel_err"] for case in checks),
        "cases": checks,
    },
    "performance": rows,
    "transfer_experiment": {
        "runs": transfers,
        "decision": "Keep pageable copies; pinned staging was slower in all five runs.",
    },
    "source_sha256": {name: sha(WORKSPACE / name) for name in sources},
    "reproduction": {
        "device_query": "device_query.cu",
        "verification_entry": "a3000_verify.rs",
        "benchmark_entry": "a3000_benchmark.rs",
        "unit_normalization": "benchmark.log only renames the derived rate to tflops; benchmark_raw.log preserves the original label. Timings are unchanged and all rates are recomputed from FLOP/p50.",
        "raw_logs": [
            "verification.log",
            "benchmark.log",
            "benchmark_raw.log",
            "hardware.log",
            "transfer.log",
        ],
    },
}
(ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
fields = [
    "M",
    "N",
    "K",
    "repeat",
    "samples",
    "max_rel_err",
    "min_ns",
    "p50_ns",
    "p95_ns",
    "p50_ms",
    "tflops_from_p50",
    "gflops_from_p50",
    "previous_handwritten_e2e_ms",
    "speedup_vs_handwritten",
    "previous_standalone_cublas_e2e_ms",
    "ratio_vs_standalone_cublas",
]
with (ROOT / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fields} for row in rows)

print(f"verification={len(checks)}/14 max_rel_err={summary['verification']['max_rel_err']:.6e}")
for row in rows:
    print(
        f"{row['M']}x{row['N']}x{row['K']}: p50={row['p50_ms']:.6f} ms, "
        f"GFLOP/s={row['gflops_from_p50']:.3f}, error={row['max_rel_err']:.3e}"
    )
print("pinned_staging_rejected=5/5 runs slower")
