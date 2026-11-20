//! 一步 decode 的合成模型：查表首层 + 一层 f32 投影，权重由确定性 PRNG 生成。
//!
//! 基准（`criterion/model_decode.rs`）与卡片测试（`tests/cards.rs`）压的是同一个栈，只有
//! 批的规模不同；装配因此只写一处，序列与页池各自留在自己的文件里。

use nn::{Embedding, Linear, Stack, WeightFormat};
use serve::model::{CpuModel, Params};
use testkit::Rng;

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
