#!/usr/bin/env bash
# 本地与 CI 共用的有界 libFuzzer + ASan 入口；参数为目标名，默认跑全部目标。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 目标清单唯一事实源；调用方可通过 FUZZ_TARGETS 或位置参数收窄。
read -r -a targets <<< "${FUZZ_TARGETS:-fuzz_cli fuzz_weights_json fuzz_weights_safetensors fuzz_weights_archive fuzz_train_snapshot fuzz_serve_frames fuzz_wire_sealed fuzz_wire_leb fuzz_http_head fuzz_lex fuzz_crc fuzz_leb_roundtrip fuzz_shape fuzz_cutile_negotiate fuzz_dist}"
if [ "$#" -gt 0 ]; then
    targets=("$@")
fi
seconds="${FUZZ_SECONDS:-30}"
timeout="${FUZZ_TIMEOUT:-10}"
max_len="${FUZZ_MAX_LEN:-4096}"
rss_limit="${FUZZ_RSS_MB:-2048}"
toolchain="${FUZZ_TOOLCHAIN:-nightly}"
corpus="${FUZZ_CORPUS:-crates/fuzz/corpus}"
artifacts="${FUZZ_ARTIFACTS:-fuzz-artifacts}"
mkdir -p "$artifacts"

rustc "+$toolchain" --version
cargo "+$toolchain" test -p llm-fuzz --test gen_corpus --locked --no-run
gen="$(find target/debug/build/llm-fuzz -type f -name 'gen_corpus'-* ! -name '*.d' ! -name '*.o' -print -quit)"
[ -x "$gen" ] || { echo "找不到语料生成器" >&2; exit 1; }
"$gen" "$corpus"

# harness = false 的目标在 engine feature 下编译；覆盖插桩与 ASan 只进入这条专用构建。
for target in "${targets[@]}"; do
    cargo "+$toolchain" rustc -p llm-fuzz --features engine --test "$target" --release --locked -- \
        -C passes='sancov-module' \
        -C llvm-args='-sanitizer-coverage-level=3' \
        -C llvm-args='-sanitizer-coverage-inline-8bit-counters' \
        -Z sanitizer=address
done

for target in "${targets[@]}"; do
    bin="$(find target/release/build/llm-fuzz -type f -name "$target"-* ! -name '*.d' ! -name '*.o' -print -quit)"
    [ -x "$bin" ] || { echo "找不到 fuzz 二进制：$target" >&2; exit 1; }
    dict_args=()
    dict="$corpus/dict/$target.dict"
    [ ! -f "$dict" ] || dict_args=("-dict=$dict")
    # 固定资源上限；值画像增强比较导向，输入缩减与泄漏检测显式开启。
    "$bin" "$corpus/$target" ${dict_args[@]+"${dict_args[@]}"} \
        -max_total_time="$seconds" -timeout="$timeout" -max_len="$max_len" \
        -rss_limit_mb="$rss_limit" -use_value_profile=1 -reduce_inputs=1 \
        -detect_leaks=1 -report_slow_units="$timeout" -print_final_stats=1 \
        -artifact_prefix="./$artifacts/"
done
