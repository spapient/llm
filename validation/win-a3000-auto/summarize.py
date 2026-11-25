import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PREVIOUS = json.loads((ROOT.parent / "win-a3000-real/summary.json").read_text())
verification = (ROOT / "verification.log").read_text(errors="replace")
benchmark = (ROOT / "benchmark.log").read_text(errors="replace")

selected = re.search(r"selected=(\w+)\((\w+)\)", verification)
assert selected and selected.group(1) == "nvidia_a3000" and selected.group(2) == "sm86"
case_pattern = re.compile(
    r"case=(\d+)x(\d+)x(\d+) PASS variant=(\S+) tile=(\S+) max_rel_err=([0-9.eE+-]+) outputs=(\d+)"
)
cases = case_pattern.findall(verification)
assert len(cases) == 14, len(cases)
validation_error = max(float(case[5]) for case in cases)

bench_pattern = re.compile(
    r"case=(\d+)x(\d+)x(\d+) repeat=(\d+) samples=(\d+) max_rel_err=([0-9.eE+-]+) "
    r"min_ns=([0-9.eE+-]+) p50_ns=([0-9.eE+-]+) p95_ns=([0-9.eE+-]+) tflops=([0-9.eE+-]+)"
)
previous = {(row["M"], row["N"], row["K"]): row for row in PREVIOUS["performance"]}
rows = []
for match in bench_pattern.findall(benchmark):
    m, n, k, repeat, samples, error, min_ns, p50_ns, p95_ns, tflops = match
    M, N, K = int(m), int(n), int(k)
    p50 = float(p50_ns)
    calculated = 2.0 * M * N * K / (p50 * 1e3)
    assert abs(calculated - float(tflops)) < 0.001
    assert samples == repeat
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
        "tflops_from_p50": calculated,
    }
    old = previous[(M, N, K)]
    row["manual_fingerprint_p50_ms"] = old["p50_ms"]
    row["ratio_vs_manual_fingerprint"] = row["p50_ms"] / old["p50_ms"]
    rows.append(row)
assert len(rows) == 3, len(rows)

sources = [
    "crates/cutile/src/probe.rs",
    "crates/cutile/src/builtin.rs",
    "crates/cutile/tests/device_probe_tests.rs",
    "crates/prof/src/nvidia/a3000.toml",
    "crates/shard/src/pipeline.rs",
]
summary = {
    "schema": 1,
    "scope": 'Real auto selection through nvidia-smi enumeration and Catalogue::resolve("auto"), followed by the production pipeline on win. Not standalone and not MUSA evidence.',
    "selection": {"vendor": selected.group(1), "id": selected.group(2), "mode": "auto"},
    "verification": {"passed_cases": len(cases), "max_rel_err": validation_error},
    "performance": rows,
    "source_sha256": {
        name: hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest() for name in sources
    },
    "reproduction": {
        "verification_entry": "a3000_auto_verify.rs",
        "benchmark_entry": "a3000_auto_benchmark.rs",
        "raw_logs": ["verification.log", "benchmark.log"],
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
    "manual_fingerprint_p50_ms",
    "ratio_vs_manual_fingerprint",
]
with (ROOT / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fields} for row in rows)

print(f"auto verification={len(cases)}/14 max_rel_err={validation_error:.6e}")
for row in rows:
    print(
        f"{row['M']}x{row['N']}x{row['K']}: p50={row['p50_ms']:.6f} ms, "
        f"TFLOP/s={row['tflops_from_p50']:.3f}, manual_ratio={row['ratio_vs_manual_fingerprint']:.4f}"
    )
