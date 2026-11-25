import hashlib
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
LOG = HERE / "performance-p2-paired-30.log"
EXPECTED_ROUNDS = 30
OLD = "target/release/build/llm-nn/5dce165d3cac3288/out/llm_nn-5dce165d3cac3288"
NEW = "target/release/build/llm-nn/cafccc36c1dd2436/out/llm_nn-cafccc36c1dd2436"


def parse_performance():
    context = None
    samples = defaultdict(list)
    seen = Counter()
    result_pattern = re.compile(
        r"^\s+(i8|f32) (\d+x\d+x\d+): fast ([0-9.]+) GF/s \| fast单 ([0-9.]+) GF/s"
    )
    for line in LOG.read_text().splitlines():
        marker = re.fullmatch(r"ROUND (\d+) (OLD|P2)", line)
        if marker:
            context = (int(marker.group(1)), marker.group(2))
            continue
        result = result_pattern.match(line)
        if not result:
            continue
        assert context is not None, "性能行缺少轮次／版本"
        round_number, version = context
        dtype, shape, fast, single = result.groups()
        for mode, value in (("fast", fast), ("fast_single", single)):
            key = (version, dtype, shape, mode, round_number)
            seen[key] += 1
            samples[(version, dtype, shape, mode)].append((round_number, float(value)))
    assert all(count == 1 for count in seen.values()), "同一轮出现重复指标"
    rounds = {key[-1] for key in seen}
    expected = set(range(1, EXPECTED_ROUNDS + 1))
    assert rounds == expected, f"轮次不完整：{sorted(rounds)}"
    assert len(seen) == 24 * EXPECTED_ROUNDS, (
        f"应为 2 版本×12 指标×{EXPECTED_ROUNDS} 轮，实得 {len(seen)} 项"
    )

    metrics = []
    for dtype, shape, mode in sorted({key[1:] for key in samples}):
        old = sorted(value for _, value in samples[("OLD", dtype, shape, mode)])
        new = sorted(value for _, value in samples[("P2", dtype, shape, mode)])
        old_median = statistics.median(old)
        new_median = statistics.median(new)
        metrics.append(
            {
                "dtype": dtype,
                "shape": shape,
                "mode": mode,
                "old": {
                    "median": old_median,
                    "min": min(old),
                    "max": max(old),
                    "p95_of_round_best": old[int(0.95 * len(old))],
                    "round_best_samples": old,
                },
                "p2": {
                    "median": new_median,
                    "min": min(new),
                    "max": max(new),
                    "p95_of_round_best": new[int(0.95 * len(new))],
                    "round_best_samples": new,
                },
                "median_delta_percent": (new_median / old_median - 1.0) * 100.0,
            }
        )
    assert len(metrics) == 12
    return {
        "schema": 1,
        "unit": "GF/s",
        "rounds_per_version": EXPECTED_ROUNDS,
        "old_binary": OLD,
        "p2_binary": NEW,
        "methodology": (
            f"每个版本连续运行 {EXPECTED_ROUNDS} 轮，保留前 5 轮并扩展样本，不剔除离群；"
            "fast 每轮报告内部 20 次最短耗时换算的吞吐，fast_single 每轮报告内部 5 次最短耗时换算的吞吐。"
            "这里汇总每轮最快值的中位数、最小值、最大值及 nearest-rank p95，不冒充全部原始调用的 p50/p95。"
        ),
        "p50_regression_gate_percent": -3.0,
        "median_gate_passed": all(metric["median_delta_percent"] >= -3.0 for metric in metrics),
        "metrics": metrics,
        "final_raw_sample_gate": "最终仍需原始调用级样本的 min/p50/p95、反汇编和探针位置证据。",
    }


def test_totals(name):
    text = (HERE / name).read_text()
    assert "FAILED" not in text and "error:" not in text, f"{name} 含失败"
    matches = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
    assert matches and all(int(failed) == 0 for _, failed, _ in matches), f"{name} 未完整通过"
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


def summarize(performance):
    native = (ROOT / "crates/prof/src/native.rs").read_text()
    assert "runtime::gemm_micro::fma_micro" in native
    assert "runtime::gemm_micro::sdot_micro" in native
    deleted = [
        "crates/prof/src/native/gemm8x8.rs",
        "crates/prof/src/native/i8.rs",
    ]
    assert all(not (ROOT / name).exists() for name in deleted)
    win_text = (HERE / "win-p2.log").read_text()
    assert "x86_avx2=true" in win_text and "x86_fma=true" in win_text
    sources = {
        name: file_fact(name)
        for name in (
            "crates/prof/src/ops/gemm_micro.k",
            "crates/prof/src/runtime.rs",
            "crates/prof/src/buildops.rs",
            "crates/prof/src/native.rs",
        )
    }
    budgets = {
        "gemm_micro_k_max_400": sources["crates/prof/src/ops/gemm_micro.k"]["lines"] <= 400,
        "runtime_rs_max_200_final": sources["crates/prof/src/runtime.rs"]["lines"] <= 200,
        "buildops_rs_max_600_production": sources["crates/prof/src/buildops.rs"]["lines"] <= 600,
    }
    tests = {
        "prof_p2": test_totals("prof-p2.log"),
        "cutile_p2": test_totals("cutile-p2.log"),
        "nn_p2": test_totals("nn-p2.log"),
        "testkit_boundary_p2": test_totals("testkit-boundary-p2.log"),
        "win_gemm_micro_p2": test_totals("win-p2.log"),
    }
    p2_complete = performance["median_gate_passed"] and all(
        item["failed"] == 0 for item in tests.values()
    )
    return {
        "schema": 1,
        "status": "P2 complete; P3/P4 open" if p2_complete else "P2 open",
        "production_switched": {"fma": True, "sdot": True, "sme": False},
        "deleted_files": deleted,
        "retained_files": ["crates/prof/src/native/sme.rs", "crates/cutile/src/native.rs"],
        "current_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tests": tests,
        "strict_clippy_log": "prof-p2-clippy.log",
        "win": {"host": "win", "x86_avx2": True, "x86_fma": True, "log": "win-p2.log"},
        "performance": performance,
        "sources": sources,
        "final_budgets_not_p2_signoff": budgets,
        "remaining": [
            "P3：SME 生产整铺迁移、性能和探针验收。",
            "P4：删除剩余旧文件，达到最终文件／行数预算并执行全仓门禁。",
            "30 轮每轮最快值统计不冒充原始调用级 min/p50/p95；最终仍需该口径、反汇编和探针位置证据。",
        ],
    }


performance = parse_performance()
(HERE / "performance-p2.json").write_text(
    json.dumps(performance, ensure_ascii=False, indent=2) + "\n"
)
for metric in performance["metrics"]:
    print(
        f"{metric['dtype']:>3} {metric['shape']:>15} {metric['mode']:>11} | "
        f"old {metric['old']['median']:7.1f} | P2 {metric['p2']['median']:7.1f} | "
        f"{metric['median_delta_percent']:+6.2f}%"
    )
print(f"median_gate_passed={performance['median_gate_passed']}")
if (HERE / "win-p2.log").exists():
    summary = summarize(performance)
    (HERE / "summary-p2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "tests": summary["tests"],
                "final_budgets": summary["final_budgets_not_p2_signoff"],
            },
            ensure_ascii=False,
        )
    )
