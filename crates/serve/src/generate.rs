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
use rng::Rng;
use std::cmp::Ordering;

/// 一步前向：给定序列与批页账视图，给出下一个位置的 logits。
///
/// 序列本身只读：前向不该推进计数（页账与计数都由调度器推进）。历史 K/V 经 `batch` 从页池
/// 读回，当前步 K/V 写回对应页行；一次调用内只持一个 `&mut Batch`，只读的序列事实由调用方
/// 先取走再进入可变借用。长度与形状由实现按 [`engine::Sequence::len`] 自行决定——本层不假设
/// 词表大小。
pub trait Predict {
    /// 计算下一个位置的 logits；返回值长度由模型决定。
    ///
    /// # Errors
    ///
    /// 模型自己失败（权重缺失、形状不符、页账读不出历史）时返回错误，服务层会把它当成服务端故障。
    fn logits(
        &mut self,
        sequence: &engine::Sequence,
        batch: &mut engine::Batch,
    ) -> Result<Vec<f32>>;

    /// 批量一步前向：一次收多条只读序列快照，返回每行 logits，行序与 `sequences` 一一对应。
    ///
    /// 默认实现逐条回退 [`logits`]，与逐串前向行为等价；内核整批通路落地后由实现覆写本方法。
    /// 序列只读、前向不推进计数，页账与计数仍由调度器推进，口径同 [`logits`]。
    ///
    /// # Errors
    ///
    /// 同 [`logits`]；任一条失败即上抛，已记账序列的页账不在此回滚，归调度器统一处理。
    fn logits_batch(
        &mut self,
        sequences: &[engine::Sequence],
        batch: &mut engine::Batch,
    ) -> Result<Vec<Vec<f32>>> {
        sequences
            .iter()
            .map(|sequence| self.logits(sequence, batch))
            .collect()
    }
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

/// 未启用温度时的缩放系数：1.0 表示不缩放 logits（与贪心同口径）。
const DEFAULT_TEMPERATURE: f32 = 1.0;

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

/// 经典采样参数闭集档案：温度 / top-k / top-p；`None` 表示该档不启用。
///
/// 参数在装配采样器时一次性给定（闭集档案），不随步变化；越界值在采样时对照候选条数校验。
/// 档案只有类型化槽位，不持字符串键名表，热路径也不做查表。随机采样不在本刀：截断之后仍是
/// 确定性 argmax（等值取最小编号），与 [`Greedy`] 同一套可复现口径。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// 温度；`None` 等价 `1.0`。
    temperature: Option<f32>,
    /// top-k：保留最大的 k 个候选；`None` 不截断。
    top_k: Option<usize>,
    /// top-p：保留累计概率到 `p` 的最小候选集；`None` 不核采样。
    top_p: Option<f32>,
}

impl SamplingParams {
    /// 空档案：三档都不启用。
    pub const fn new() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
        }
    }

    /// 温度：必须是大于 0 的有限值。
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// top-k：保留最大的 k 个候选；条数在采样时对照候选个数校验。
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// top-p：必须在 `(0, 1]` 内的有限值。
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// 温度档案值（未启用为 `None`）。
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// top-k 档案值（未启用为 `None`）。
    pub fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    /// top-p 档案值（未启用为 `None`）。
    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    /// 按候选条数校验三档参数；非有限、越界、空候选报 `Invalid`。
    fn validate(&self, candidates: usize) -> Result<()> {
        self.check_temperature()?;
        self.check_top_k(candidates)?;
        self.check_top_p()?;
        Ok(())
    }

    /// 温度必须是大于 0 的有限值；未启用不校验。
    fn check_temperature(&self) -> Result<()> {
        if let Some(temperature) = self.temperature
            && (!temperature.is_finite() || temperature <= 0.0)
        {
            return Err(Error::new(
                Code::Invalid,
                format!("温度 {temperature} 必须是大于 0 的有限值"),
            )
            .with("serve::SamplingParams"));
        }
        Ok(())
    }

    /// top-k 必须落在 `1..=候选条数`；未启用不校验。
    fn check_top_k(&self, candidates: usize) -> Result<()> {
        if let Some(k) = self.top_k
            && (k == 0 || k > candidates)
        {
            return Err(Error::new(
                Code::Invalid,
                format!("top-k {k} 不在 1..={candidates} 范围内"),
            )
            .with("serve::SamplingParams"));
        }
        Ok(())
    }

    /// top-p 必须在 `(0, 1]` 内且有限；未启用不校验。
    fn check_top_p(&self) -> Result<()> {
        if let Some(p) = self.top_p
            && (!p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(
                Error::new(Code::Invalid, format!("top-p {p} 必须在 (0, 1] 内"))
                    .with("serve::SamplingParams"),
            );
        }
        Ok(())
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::new()
    }
}

/// 经典采样：温度缩放 → top-k 截断 → top-p 核采样 → 保留集 argmax。
///
/// 不接随机数：截断后的选择仍是确定性 argmax（等值取最小编号）；随机采样留后续档位，
/// 本刀不预埋 RNG。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Classical {
    params: SamplingParams,
}

impl Classical {
    /// 由参数闭集档案构造。
    pub fn new(params: SamplingParams) -> Self {
        Self { params }
    }

    /// 不启用任何截断：温度 1、不核采样，选择口径与贪心一致。
    pub fn greedy_like() -> Self {
        Self::new(SamplingParams::new())
    }

    /// 参数档案（只读）。
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }
}

impl Sampling for Classical {
    fn sample(&mut self, logits: &[f32]) -> Result<Option<u32>> {
        pipeline(
            &self.params,
            logits,
            "serve::Classical",
            |scaled, retained| argmax_index(scaled, retained, "serve::Classical"),
        )
    }
}

/// 多项式采样：截断保留集上按 softmax 概率逆变换抽一个 token。
///
/// 与 [`Classical`] 共用温度 / top-k / top-p 截断；差别在保留集的选择：经典采样是
/// 确定性 argmax，本刀按归一化概率抽样。随机流由构造时给定的种子经显式 [`Rng`]
/// 派生，不读全局熵源；同种子、同 logits 序列逐位可复现。
///
/// 并行后端的等价口径是 Gumbel-max（逐候选 `logit/T + gumbel` 后 argmax，不建累积表）；
/// 本轮 CPU 路径用累积概率逆变换，二者概率分布等价。
#[derive(Debug, Clone, PartialEq)]
pub struct Multinomial {
    params: SamplingParams,
    rng: Rng,
}

impl Multinomial {
    /// 由参数档案与种子构造；种子派生独立流，不持有全局 RNG。
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            rng: Rng::new(seed),
        }
    }

    /// 参数档案（只读）。
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }
}

impl Sampling for Multinomial {
    fn sample(&mut self, logits: &[f32]) -> Result<Option<u32>> {
        pipeline(
            &self.params,
            logits,
            "serve::Multinomial",
            |scaled, retained| sample_categorical(scaled, retained, &mut self.rng),
        )
    }
}

/// 保留集上按 softmax 概率逆变换抽样：减 max 求指数并累计，抽一个均匀量落进对应段。
///
/// 指数与累计有串行依赖，保留折叠与遍历；`scaled` 整块读入，不写回调用方缓冲。
/// 浮点兜底：阈值被累积误差推到末尾时取最后一个候选。
fn sample_categorical(scaled: &[f32], candidates: Vec<usize>, rng: &mut Rng) -> Result<usize> {
    if candidates.is_empty() {
        return Err(empty("serve::Multinomial"));
    }
    let (weights, total) = exp_weights(scaled, &candidates);
    let threshold = rng.next_f32() * total;
    let mut cumulative = 0.0;
    for (position, &index) in candidates.iter().enumerate() {
        cumulative += weights[position];
        if cumulative >= threshold {
            return Ok(index);
        }
    }
    Ok(candidates[candidates.len() - 1])
}

/// 采样公共管线：预检 → 参数校验 → 温度缩放 → 截断 → 选择器；返回选中 token 编号。
///
/// 两种选择器（确定性 argmax、概率逆变换）只在保留集上的取法不同，预检、校验、缩放与
/// 截断序列共用这一份；选择器以编译期闭包注入，单态化无 dyn 开销。
fn pipeline(
    params: &SamplingParams,
    logits: &[f32],
    scope: &str,
    choose: impl FnOnce(&[f32], Vec<usize>) -> Result<usize>,
) -> Result<Option<u32>> {
    usable(logits)?;
    params.validate(logits.len())?;
    let temperature = params.temperature().unwrap_or(DEFAULT_TEMPERATURE);
    // 温度缩放是逐元素映射，整块切片一次算完。
    let scaled: Vec<f32> = logits.iter().map(|&logit| logit / temperature).collect();
    let retained = retain(params, &scaled, (0..logits.len()).collect());
    let index = choose(&scaled, retained)?;
    token(index, scope).map(Some)
}

/// 候选下标映射到 token 编号；越出 u32 报 `Invalid`，`scope` 标记产生错误的采样器。
fn token(index: usize, scope: &str) -> Result<u32> {
    u32::try_from(index)
        .map_err(|_| Error::new(Code::Invalid, format!("token 编号 {index} 超出 u32")).with(scope))
}

/// 温度之后的 top-k → top-p 截断：从全候选集裁到概率保留集；argmax 与逆变换两种选择器共用这一份序列。
fn retain(params: &SamplingParams, scaled: &[f32], mut retained: Vec<usize>) -> Vec<usize> {
    if let Some(k) = params.top_k() {
        retained = top_k(scaled, retained, k);
    }
    if let Some(p) = params.top_p() {
        retained = nucleus(scaled, retained, p);
    }
    retained
}

/// 分数降序排序键：分数大者在前，等值时编号小者在前（确定性平局口径）。
fn descending(a_score: f32, b_score: f32, a: usize, b: usize) -> Ordering {
    b_score
        .partial_cmp(&a_score)
        .unwrap_or(Ordering::Equal)
        .then(a.cmp(&b))
}

/// top-k 截断：按缩放分数降序（等值取最小编号）排序后保留前 `k` 个候选。
fn top_k(scaled: &[f32], mut candidates: Vec<usize>, k: usize) -> Vec<usize> {
    candidates.sort_by(|&a, &b| descending(scaled[a], scaled[b], a, b));
    candidates.truncate(k);
    candidates
}

/// 保留集上确定性 argmax：严格大于才更新，等值时编号更小者胜出；候选为空报错。
fn argmax_index(scaled: &[f32], candidates: Vec<usize>, scope: &str) -> Result<usize> {
    let best = candidates.into_iter().fold(None, |best, index| match best {
        None => Some(index),
        Some(best_index) => {
            if scaled[index] > scaled[best_index] {
                Some(index)
            } else {
                Some(best_index)
            }
        }
    });
    best.ok_or_else(|| empty(scope))
}

/// 截断后候选为空：无 token 可选；`scope` 标记产生错误的采样器。
fn empty(scope: &str) -> Error {
    Error::new(Code::Invalid, "截断后候选为空：无 token 可选").with(scope)
}

/// 保留集减最大值的指数权重与未归一化总和；top-p 排序与多项式逆变换共用同一份 softmax 权重。
///
/// 减最大值只为数值稳定，不改变相对权重；返回未归一化指数，归一化由消费方按用途决定。
fn exp_weights(scaled: &[f32], candidates: &[usize]) -> (Vec<f32>, f32) {
    let max = candidates
        .iter()
        .map(|&index| scaled[index])
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = candidates
        .iter()
        .map(|&index| (scaled[index] - max).exp())
        .collect();
    let total = weights.iter().sum();
    (weights, total)
}

/// top-p 核采样：按 softmax 概率降序累计，保留累计概率不超过 `p` 的最小前缀（含越出阈值的第一项）。
///
/// 归约与排序有串行依赖，保留遍历；`scaled` 整块读入，指数与累计不写回调用方缓冲。
fn nucleus(scaled: &[f32], mut candidates: Vec<usize>, p: f32) -> Vec<usize> {
    let (weights, total) = exp_weights(scaled, &candidates);
    let mut ranked: Vec<(usize, f32)> = candidates.drain(..).zip(weights).collect();
    ranked.sort_by(|&(a, ea), &(b, eb)| descending(ea, eb, a, b));
    let mut kept = Vec::with_capacity(ranked.len());
    let mut cumulative = 0.0;
    for (index, exponential) in ranked {
        kept.push(index);
        cumulative += exponential / total;
        if cumulative >= p {
            break;
        }
    }
    kept
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

name_set! {
    /// 生成为什么停下：到上限，还是采到了停止符。
    pub enum Ending {
        /// 跑满了 `max_new`。
        Limit => "limit",
        /// 采到了停止符（会由调度器归还页）。
        Stop => "stop",
    }
}

/// logits 可用性预检：空输入与非有限值都报错；所有采样策略共用同一份预检。
fn usable(logits: &[f32]) -> Result<()> {
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
    Ok(())
}

/// 最大 logit 的最小下标；空输入与非有限值都报错。
fn best(logits: &[f32]) -> Result<u32> {
    usable(logits)?;
    let index = argmax_index(logits, (0..logits.len()).collect(), "serve::Greedy")?;
    token(index, "serve::Greedy")
}

#[cfg(test)]
#[path = "../tests/generate_tests.rs"]
mod tests;
