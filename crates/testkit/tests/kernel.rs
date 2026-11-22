//! `cutile_x_kernel` 的签名与配套样本：三处白盒对拍共用一份取值，免得各写一份后静默漂移。
//!
//! 调用点（都以 `#[path]` 取用本文件）：
//!
//! - `crates/cutile/src/harness.rs`：只测 C 宿主基准封装把样本渲染成数组、把签名拼成调用；
//! - `crates/device/src/program.rs`：测「声明 → 并行程序」装配，拿同一份签名/样本覆盖可拒绝族；
//! - `crates/device/tests/schedule.rs`：测 vendor 生产态 lowering 的源文本（`__device__` /
//!   `kernel_main` / `<<<1, 1>>>` 一类判据）。
//!
//! 夹具共用不改这两件事：两边测的仍是不同层的产物。取值被下游的文本断言**逐字**依赖
//! （`cutile_x_kernel(x, w, scales, y, 4, 2, 4);`、`cudaMemcpy(h_y, d_y, 2 * sizeof(float)`），
//! 改这里的数等于改那些断言。
//!
//! 每个条目都带 `#[cfg(test)]`：这是**纯测试夹具**，没有运行时形态。`testkit` 的库体零依赖
//! （不把被测层拉进依赖表），所以它不能是常驻模块；三条用到它的边（`cutile` 与 `device` 都已
//! 声明 `+dev: testkit`）按 `#[path]` 把它编进各自的测试口径。条目级 `cfg` 让静态门的
//! 「生产面零 `unwrap`/`expect`」口径直接看不到本文件，而不必在门禁里为它开例外。

#[cfg(test)]
use nn::model::{Buf, Sample, Values};
#[cfg(test)]
use nn::sig::Signature;

/// 签名文本：解析与装配吃同一份源，源码不抄第二遍。
#[cfg(test)]
pub const SOURCE: &str = "void cutile_x_kernel(const float *x, const unsigned char *w, const float *scales,\n\
         float *y, int M, int N, int gsz) {}\n";

/// 三缓冲 GEMV 形签名：`x` / `w` / `scales` 进、`y` 出，另有 `M` / `N` / `gsz` 三个整数标量。
#[cfg(test)]
pub fn signature() -> Signature {
    Signature::parse("x", SOURCE).expect("签名")
}

/// 与 [`signature`] 配套的样本：`m = 4`、`n = 2`，四个缓冲各自给出确定取值。
#[cfg(test)]
pub fn sample() -> Sample {
    let bufs = [
        ("x", Values::F32(vec![1.0, 1.0, 1.0, 1.0])),
        ("w", Values::U8(vec![0x98, 0x76])),
        ("scales", Values::F32(vec![0.5, 0.5])),
        ("y", Values::F32(vec![0.0, 0.0])),
    ];
    Sample {
        m: 4,
        n: 2,
        scalars: vec![("M", 4), ("N", 2), ("gsz", 4)],
        bufs: bufs
            .into_iter()
            .map(|(name, values)| Buf { name, values })
            .collect(),
    }
}
