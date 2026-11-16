//! 生成的两个协作者与它的结局：[`Predict`]、[`Sampling`]、[`Greedy`]、[`Generation`]、[`Ending`]。
//!
//! 这里只把两个接口定下来，好让端点把它们与调度器接起来；模型实现收在 [`crate::model`]
//! 模块内（CPU 内核桥），选 token 的策略与接口解耦：
//!
//! - [`Predict`] 给一步 logits（序列进、分数出）；
//! - [`Sampling`] 从分数里选一个 token，或说"到此为止"（`None`）。
//!
//! 把采样放成 trait 而不是一个 `argmax` 调用，是因为"贪心"只是其中一种：温度、top-k、重复
//! 惩罚都会改变"什么时候停"。端点因此不需要知道策略，策略也不需要知道页账。

use diagnostic::{Code, Error, Result};

/// 一步前向：给定序列，给出下一个位置的 logits。
///
/// 序列只读：前向不该改状态（页账与计数都由调度器推进）。长度与形状由实现按
/// [`engine::Sequence::len`] 自行决定——本层不假设词表大小。
pub trait Predict {
    /// 计算下一个位置的 logits；返回值长度由模型决定。
    ///
    /// # Errors
    ///
    /// 模型自己失败（权重缺失、形状不符）时返回错误，服务层会把它当成服务端故障。
    fn logits(&mut self, sequence: &engine::Sequence) -> Result<Vec<f32>>;
}

/// 采样策略：从一步 logits 里选一个 token；`None` 表示遇到停止符、序列到此结束。
pub trait Sampling {
    /// 选下一个 token；`None` 让序列提前结束（页仍由调度器归还）。
    ///
    /// # Errors
    ///
    /// logits 不可用（空、含 NaN / 无穷）时返回错误，而不是猜一个 token。
    fn sample(&mut self, logits: &[f32]) -> Result<Option<u32>>;
}

/// 贪心采样：取最大 logit，等值时取编号最小的那个。
///
/// "等值取最小下标"是确定性要求：否则同一份 logits 在不同实现上会给出不同 token，
/// 对拍与回归都会变成"有时通过"。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Greedy;

impl Sampling for Greedy {
    fn sample(&mut self, logits: &[f32]) -> Result<Option<u32>> {
        best(logits).map(Some)
    }
}

/// 一步生成的结局：采到的 token + 它为什么停下。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    tokens: Vec<u32>,
    ending: Ending,
}

impl Generation {
    /// 由 token 与结局构造：端点在生成循环里逐 token 收集，跑完才知道结局。
    pub fn new(tokens: Vec<u32>, ending: Ending) -> Self {
        Self { tokens, ending }
    }

    /// 采到的 token 编号，按生成顺序。
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// 为什么停下。
    pub fn ending(&self) -> Ending {
        self.ending
    }

    /// 线格式：每个 token 4 字节小端，顺序与 [`Generation::tokens`] 一致。
    ///
    /// 与 `serve::Response::ok(sequence.id().to_le_bytes())` 用同一套字节序：端点返回的
    /// 载荷不需要第二种编码。输出长度已知（`tokens.len() * 4`），一次性预分配，不再逐元素扩容。
    pub fn frame(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.tokens.len() * 4);
        for token in &self.tokens {
            out.extend_from_slice(&token.to_le_bytes());
        }
        out
    }

    /// 用词表把 token 还原成文本（越界编号由分词器报错，不在这里替换）。
    pub fn text(&self, tokenizer: &dyn tokenizer::Tokenizer) -> Result<String> {
        let tokens: Vec<tokenizer::Token> = self
            .tokens
            .iter()
            .map(|id| tokenizer::Token::new(*id))
            .collect();
        tokenizer.decode(&tokens)
    }
}

/// 生成为什么停下：到上限，还是采到了停止符。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ending {
    /// 跑满了 `max_new`。
    Limit,
    /// 采到了停止符（会由调度器归还页）。
    Stop,
}

impl Ending {
    /// 全部结局，供日志与测试遍历。
    pub const ALL: [Self; 2] = [Self::Limit, Self::Stop];

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Limit => "limit",
            Self::Stop => "stop",
        }
    }
}

/// 最大 logit 的最小下标；空输入与非有限值都报错。
fn best(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "logits 不可用（{} 个元素，含非有限值或为空）：不猜 token",
                logits.len()
            ),
        )
        .with("serve::Greedy"));
    }
    // 一趟 argmax：比较链有串行依赖，保留折叠；严格大于才更新，等值时先出现（编号更小）的下标胜出。
    let (index, _) = logits.iter().copied().enumerate().fold(
        (0usize, f32::NEG_INFINITY),
        |(best_index, best_value), (index, value)| {
            if value > best_value {
                (index, value)
            } else {
                (best_index, best_value)
            }
        },
    );
    u32::try_from(index).map_err(|_| {
        Error::new(Code::Invalid, format!("token 编号 {index} 超出 u32")).with("serve::Greedy")
    })
}

#[cfg(test)]
#[path = "../tests/generate_tests.rs"]
mod tests;
