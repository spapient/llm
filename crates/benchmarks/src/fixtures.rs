//! 基准与测试共用的夹具：attention 的样例张量、一步 decode 的合成模型、夹具失败的统一出口。

use nn::{Embedding, Linear, Stack, WeightFormat};
use serve::model::{CpuModel, Params};
use std::fmt;
use testkit::Rng;

/// 夹具失败：把原因写到 stderr，再以非零码退出。
pub(crate) fn checked_or_exit<T, E: fmt::Display>(result: Result<T, E>, what: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("基准夹具失败（{what}）：{error}");
            std::process::exit(1)
        }
    }
}

/// 夹具查表失败：`Option` 版本，语义与 [`checked`] 相同（同样是唯一失败出口）。
/// 测量路径上的 `Result` 出口：同一几何已在装配阶段由 [`checked`] 校验过一次，这里只把
/// 状态压成 `bool` 交给 `black_box`——不在计时路径上第二次解释失败，也不吞掉错误。
pub(crate) fn ok<E>(r: Result<(), E>) -> bool {
    r.is_ok()
}

pub(crate) fn found<T>(option: Option<T>, what: &str) -> T {
    match option {
        Some(value) => value,
        None => {
            eprintln!("基准夹具失败（{what}）：查表未命中");
            std::process::exit(1)
        }
    }
}

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
            q: (0..len).map(|idx| -0.5 + idx as f32 * 0.13).collect(),
            k: (0..len).map(|idx| 0.4 - idx as f32 * 0.09).collect(),
            v: (0..len).map(|idx| -0.2 + idx as f32 * 0.11).collect(),
            out: vec![0.0f32; len],
        }
    }
}

/// 词表大小。
const VOCAB: usize = 16;
/// 嵌入与隐藏宽度。
const DIM: usize = 8;
/// 投影输出宽度（末层 logits 宽度）。
const OUT: usize = 8;

/// 提示长度：一步 decode，序列只带一个 token。
pub(crate) const PROMPT: &[u32] = &[3];

/// 确定性有限权重：±[0.5, 2]，远离 0，避免平方类差分为 0。
fn finite_values(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.value() as f32).collect()
}

/// 合成栈：`tok_emb` 查表首层 + `proj` f32 投影，两层权重都按固定种子生成。
pub(crate) fn synthetic_model() -> CpuModel {
    let stack = checked(
        Stack::of(vec![
            Box::new(checked(Embedding::new("tok_emb", VOCAB, DIM, 1), "嵌入层")),
            Box::new(checked(
                Linear::with_format("proj", WeightFormat::F32, DIM, OUT),
                "投影层",
            )),
        ]),
        "栈",
    );
    let mut params = Params::new();
    params.insert("tok_emb.weight", finite_values(VOCAB * DIM, 7));
    params.insert("proj.weight", finite_values(OUT * DIM, 11));
    checked(CpuModel::new(&stack, &params), "装配")
}
