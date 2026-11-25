#!/usr/bin/env bash
# Miri 纯计算核心：文件同步等宿主能力由普通测试与 CI 覆盖，不伪装成 Miri 支持。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

toolchain="${MIRI_TOOLCHAIN:-nightly}"
package="${MIRI_PACKAGE:-llm-storage}"
output="${MIRI_OUTPUT:-target/miri}"
flags="${MIRIFLAGS:--Zmiri-disable-isolation -Zmiri-backtrace=full}"
read -r -a modules <<< "${MIRI_MODULES:-alloc crc leb}"
mkdir -p "$output"
cargo "+$toolchain" miri --version | tee "$output/version.txt"

for module in "${modules[@]}"; do
    # proptest 的持久化需要关闭宿主隔离；逐模块日志必须保留，pipefail 传播测试失败。
    MIRIFLAGS="$flags" cargo "+$toolchain" miri test -p "$package" --lib "${module}::" --locked \
        2>&1 | tee "$output/$module.log"
done
