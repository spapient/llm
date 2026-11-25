import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROUNDS = 30
PATTERN = re.compile(r"^\s+(i8|f32) (\d+x\d+x\d+): fast ([0-9.]+) GF/s \| fast单 ([0-9.]+) GF/s")


def read_samples():
    context = None
    samples = defaultdict(list)
    for line in (HERE / "sme-performance-p3-final.log").read_text().splitlines():
        marker = re.fullmatch(r"ROUND (\d+) (OLD|NEW)", line)
        if marker:
            context = (marker.group(2), int(marker.group(1)))
            continue
        match = PATTERN.match(line)
        if not match:
            continue
        version, round_number = context
        dtype, shape, fast, single = match.groups()
        for mode, value in (("fast", fast), ("fast_single", single)):
            samples[(version, dtype, shape, mode)].append((round_number, float(value)))
    assert len(samples) == 24
    for key, values in samples.items():
        assert len(values) == ROUNDS, f"{key} 样本数不是 {ROUNDS}"
        assert {round_number for round_number, _ in values} == set(range(1, ROUNDS + 1))
    return samples


def summary(values):
    values = sorted(values)
    return {
        "min": values[0],
        "median": statistics.median(values),
        "p95_of_round_best": values[int(0.95 * len(values))],
        "max": values[-1],
        "round_best_samples": values,
    }


def compare(samples):
    metrics = []
    keys = sorted({key[1:] for key in samples})
    for dtype, shape, mode in keys:
        old = summary(value for _, value in samples[("OLD", dtype, shape, mode)])
        new = summary(value for _, value in samples[("NEW", dtype, shape, mode)])
        metrics.append(
            {
                "dtype": dtype,
                "shape": shape,
                "mode": mode,
                "old": old,
                "new": new,
                "median_delta_percent": (new["median"] / old["median"] - 1.0) * 100.0,
            }
        )
    return metrics


def main():
    metrics = compare(read_samples())
    result = {
        "schema": 1,
        "unit": "GF/s",
        "rounds_per_version": ROUNDS,
        "old": "RUSTC_BOOTSTRAP=1，旧 Rust SME，release",
        "new": "stable Rust + 静态 C SME，release",
        "old_binary": ".tmp/sme-p3-old-target/release/build/llm-nn/cafccc36c1dd2436/out/llm_nn-cafccc36c1dd2436",
        "new_binary": "target/release/build/llm-nn/8eee59d47eda84dc/out/llm_nn-8eee59d47eda84dc",
        "methodology": "同机同会话交替 30 轮；每轮 fast 取内部 20 次、fast_single 取内部 5 次最短耗时。统计每轮最快值，不冒充原始调用级分布。",
        "sme_median_gate_passed": all(
            item["median_delta_percent"] >= -3.0 for item in metrics if item["dtype"] == "f32"
        ),
        "median_gate_passed": all(item["median_delta_percent"] >= -3.0 for item in metrics),
        "metrics": metrics,
    }
    (HERE / "sme-performance-p3-final.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    for item in metrics:
        print(
            f"{item['dtype']:>3} {item['shape']:>15} {item['mode']:>11} | old {item['old']['median']:7.1f} | new {item['new']['median']:7.1f} | {item['median_delta_percent']:+6.2f}%"
        )
    print(f"sme_median_gate_passed={result['sme_median_gate_passed']}")
    print(f"all_median_gate_passed={result['median_gate_passed']}")


if __name__ == "__main__":
    main()
