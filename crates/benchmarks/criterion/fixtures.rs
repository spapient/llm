//! 基准与测试共用的夹具：attention 的样例张量与一步 decode 的合成模型。
//!
//! 只写一处，是因为 attention 的三组基准、`model` 的两组基准与 `tests/cases.rs` 压的是同一套
//! 缓冲与同一个栈；夹具各留一份，"样例是什么"就会有第二、第三份说法。

use nn::{Embedding, Linear, Stack, WeightFormat};
use serve::model::{CpuModel, Params};
use testkit::Rng;

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
    let stack = Stack::of(vec![
        Box::new(Embedding::new("tok_emb", VOCAB, DIM, 1).expect("层")),
        Box::new(Linear::with_format("proj", WeightFormat::F32, DIM, OUT).expect("层")),
    ])
    .expect("栈");
    let mut params = Params::new();
    params.insert("tok_emb.weight", finite_values(VOCAB * DIM, 7));
    params.insert("proj.weight", finite_values(OUT * DIM, 11));
    CpuModel::new(&stack, &params).expect("装配")
}
