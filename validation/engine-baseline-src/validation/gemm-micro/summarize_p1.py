import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
baseline = json.loads((HERE / "baseline.json").read_text())


def test_totals(name):
    text = (HERE / name).read_text()
    matches = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
    assert matches and all(int(failed) == 0 for _, failed, _ in matches)
    return {
        "passed": sum(int(passed) for passed, _, _ in matches),
        "failed": 0,
        "ignored": sum(int(ignored) for _, _, ignored in matches),
    }


def file_fact(name):
    raw = (ROOT / name).read_bytes()
    return {
        "lines": len(raw.decode().splitlines()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


source_names = [
    "crates/prof/src/ops/gemm_micro.k",
    "crates/prof/src/ops/kernels.toml",
    "crates/prof/src/build.rs",
    "crates/prof/src/buildops.rs",
    "crates/prof/src/runtime.rs",
    "crates/prof/tests/embedded.rs",
    "crates/prof/Cargo.toml",
]
sources = {name: file_fact(name) for name in source_names}
assert sources[source_names[0]]["lines"] <= 400
assert sources["crates/prof/src/runtime.rs"]["lines"] <= 200
assert sources["crates/prof/src/buildops.rs"]["lines"] <= 600
performance = [
    line.strip()
    for line in (HERE / "performance.log").read_text().splitlines()
    if re.match(r"\s+(i8|f32) \d+x\d+x\d+:", line)
]
assert len(performance) == 6
result = {
    "schema": 1,
    "status": "P0/P1 complete; P2/P3 open",
    "production_switched": False,
    "legacy_native_retained": True,
    "file_reduction_claimed": False,
    "performance_improvement_claimed": False,
    "baseline_head": baseline["head"],
    "current_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "baseline": {
        "native_lines": baseline["native_lines"],
        "native_bytes": baseline["native_bytes"],
        "public_lines": len(baseline["public_lines"]),
        "performance_samples": performance,
    },
    "sources": sources,
    "tests": {
        "prof_p1": test_totals("prof-p1.log"),
        "cutile_p1": test_totals("cutile-p1.log"),
        "nn_p1": test_totals("nn-p1.log"),
        "micro_parity": test_totals("micro-p1.log"),
        "sme_parity": test_totals("sme-p1.log"),
        "prof_baseline": test_totals("prof-tests.log"),
        "cutile_baseline": test_totals("cutile-tests.log"),
        "nn_baseline": test_totals("nn-tests.log"),
        "sme_baseline": test_totals("sme-tests.log"),
    },
    "evidence_logs": [
        "prof-tests.log",
        "cutile-tests.log",
        "nn-tests.log",
        "sme-tests.log",
        "performance.log",
        "prof-p1.log",
        "cutile-p1.log",
        "nn-p1.log",
        "micro-p1.log",
        "sme-p1.log",
    ],
}
(HERE / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": result["status"],
            "tests": result["tests"],
            "template_lines": sources[source_names[0]]["lines"],
            "runtime_lines": sources["crates/prof/src/runtime.rs"]["lines"],
            "buildops_lines": sources["crates/prof/src/buildops.rs"]["lines"],
        },
        ensure_ascii=False,
    )
)
