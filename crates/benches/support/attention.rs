//! attention 基准的样例张量：`q` / `k` / `v` 的取值与 `out` 的初值只写一处。
//!
//! attention 的三组基准（decode、prefill 的 online 与 dense）压的是同一套 `s·d` 缓冲，
//! 样例取值也相同；取值留在各文件里，"样例是什么"就会有第二、第三份说法。

/// 一组等长样例：`q` / `k` / `v` 各 `len` 个标量，外加同样长的 `out`。
pub(crate) struct Sample {
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) out: Vec<f32>,
}

impl Sample {
    /// 按 `len = s·d` 造一组：三条输入是互不相同的线性序列，`out` 全零。
    pub(crate) fn new(len: usize) -> Self {
        Self {
            q: (0..len).map(|index| -0.5 + index as f32 * 0.13).collect(),
            k: (0..len).map(|index| 0.4 - index as f32 * 0.09).collect(),
            v: (0..len).map(|index| -0.2 + index as f32 * 0.11).collect(),
            out: vec![0.0f32; len],
        }
    }
}
