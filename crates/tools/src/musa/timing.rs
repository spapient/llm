//! 计时注入：把首次发射到末次回读之间的整条链路包进事件计时循环。
//!
//! 设备源不动一行，只在两个行首各插一段：源程序因此还是「编译 → 运行 → 打印」那一个，
//! 计时只是多出来的一个 `stderr` 行。口径是**一条完整链路**（发射 + 同步 + 回读），
//! 不是单次 kernel 时间——两者不能混着比。

use diagnostic::{Code, Error, Result};

/// 首次发射的判据：内核对 `<<<`。设备源里的第一处即首段。
const LAUNCH: &str = "<<<";

/// 末次回读的判据：host 驱动最后一步把输出拷回里的那一句（`norm` / `cuda` 驱动同形）。
const READ_BACK: &str = "cudaMemcpy(h_y, d_y,";

/// 在 `source` 的首次发射与末次回读之间夹入 `reps` 次迭代的事件计时，返回新源。
///
/// `reps == 0` 会被拒：那会写出 `bench_ms / 0`，得到一个 `inf` 而不报错。
pub fn timed(source: &str, reps: usize) -> Result<String> {
    if reps == 0 {
        return Err(Error::new(Code::Invalid, "计时次数不能是 0"));
    }
    let (start, stop) = body(source)?;
    Ok([
        &source[..start],
        &head(reps),
        &source[start..stop],
        &tail(reps),
        &source[stop..],
    ]
    .concat())
}

/// 计时循环要包住的区间：首次发射所在行首 → 末次回读所在行首。
fn body(source: &str) -> Result<(usize, usize)> {
    let launch = source.find(LAUNCH).ok_or_else(|| absent(LAUNCH))?;
    let read = source.rfind(READ_BACK).ok_or_else(|| absent(READ_BACK))?;
    let (start, stop) = (line_start(source, launch), line_start(source, read));
    if stop <= start {
        return Err(Error::new(
            Code::Invalid,
            format!("回读在发射之前：`{READ_BACK}` 不在 `{LAUNCH}` 之后，无法计时"),
        ));
    }
    Ok((start, stop))
}

/// 循环头：建事件、记起点、开 `reps` 次迭代。
fn head(reps: usize) -> String {
    format!(
        "    cudaEvent_t t0, t1;\n    cudaEventCreate(&t0); cudaEventCreate(&t1);\n    \
         cudaEventRecord(t0, 0);\n    for (int rep = 0; rep < {reps}; ++rep) {{\n"
    )
}

/// 循环尾：记终点、同步、算平均、写 `stderr`（stdout 留给位模式，对拍脚本按行读）。
fn tail(reps: usize) -> String {
    format!(
        "    }}\n    cudaEventRecord(t1, 0); cudaEventSynchronize(t1);\n    \
         float bench_ms = 0; cudaEventElapsedTime(&bench_ms, t0, t1);\n    \
         fprintf(stderr, \"bench: %.4f ms/iter\\n\", bench_ms / {reps}.0f);\n"
    )
}

/// `at` 所在行的行首字节偏移（`at` 之前没有换行就是 0）。
fn line_start(source: &str, at: usize) -> usize {
    source[..at].rfind('\n').map_or(0, |newline| newline + 1)
}

/// 缺判据点：设备源里找不到 `marker`。
fn absent(marker: &str) -> Error {
    Error::new(
        Code::Invalid,
        format!("设备源里找不到 `{marker}`；它可能不是本仓驱动拼出来的程序"),
    )
}
