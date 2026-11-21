//! byte-latent 的逐字节自回归循环（§5.17，第 11 款标杆的生成侧）。
//!
//! 一次 byte-latent 前向是三段：**字节 → 动态 patch → latent transformer → 逐字节解码**。
//! [`crate::Boundary`] 回答"哪些字节同属一次全局前向"，[`crate::ByteEncoder`] 与
//! [`crate::ByteDecoder`] 管两端与字节直接打交道的那两层；本模块把它们串成**一次生成**：边生成边判
//! 边界、已收口的 patch 不重算、一步只走一个字节。
//!
//! # 条件是什么，为什么只取已收口的历史
//!
//! 解码第 `j` 个字节时喂给输出头的状态是
//! `上一条收口 patch 的表示 + 第 j−1 个字节的 N-gram 向量`：
//!
//! - **上一条收口 patch**：当前这个 patch 还长着的时候，它的表示会随字节增加而变——池化以边界字节为
//!   查询，而"断在哪"正是收口那一刻才定的。拿一个尚未收口的表示当条件，等于让同一个字节的分布在
//!   切分定下来之前与之后不一样，也就把"边界只看过去"这条性质丢在了生成侧。所以条件只取**已经
//!   收口**的那些。
//! - **上一个字节的 N-gram 向量**：patch 内的本地上下文。它走的是池化用的同一份逐字节向量、同宽，
//!   于是是残差相加，而不是又一个投影——本层不持参数、不新增权重。
//!
//! 两者都不存在时（空历史）条件是全零：与 [`nn::Bypass`]"历史不足就不加、不拿零向量补位"是同一
//! 条判据的两个方向——那里是"不假装有记忆"，这里是"还没有记忆可加"。
//!
//! # 边界分数从哪来
//!
//! 一个字节的分数就是它自己的编码代价 `−ln p`，`p` 是输出头在这个字节上给出的概率
//! （[`crate::ByteDecoder::pick`]）：一次生成里只有一份分布，分数不另立来源。BLT 用一个独立的字节级
//! 自回归模型估熵，那属装配侧——换掉它就是换掉本层的分布，而不是在这里加第二个入口。于是
//! [`crate::Boundary::Rate`] 的秩判定与 [`crate::Boundary::Entropy`] 的阈值作用在同一个实数序列上。
//!
//! # 停止
//!
//! [`Halt`] 是闭集：`Ceiling`（字节预算用完）与 `Stop`（产出了调用方给的停止字节）。预算为零就不
//! 生成、记 `Ceiling`，而不是"永不停止"——没有上限的循环不叫生成。
//!
//! # 与全局 transformer 的分工
//!
//! [`Emitter::reprs`] 是已收口 patch 的表示（紧凑，行宽 `dim`），全局 transformer 就是在这一列上跑的
//! [`crate::Global`]：**每 patch 一次**，遍数是行数而不是字节数。它把表示变成更好的条件，却不改变
//! 本层的循环结构——本层因此可以在没有它的情况下逐字节走完，装上它就是换条件。解码侧与之对称：
//! [`crate::Backread`] 让每个字节回读自己及更早的表示，可见集正好是本层已经收口的那些。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::codec::{ByteDecoder, ByteEncoder};
use crate::patch::{Boundary, Patches};

/// 报错出处：宽度与种子在入口失败，不留到生成中途。
const SITE: &str = "algorithm::emit";

/// 生成停下来的原因。
///
/// 闭集而不是 `Option`：`None` 说不清"预算用完"与"一个字节都没走"的区别，而这两件事对上游的处置
/// 不同——一个是截断，一个是空结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Halt {
    /// 字节预算用完。
    Ceiling,
    /// 产出了调用方给的停止字节。
    Stop,
}

impl Halt {
    /// 稳定的短名，写进报告与配置回显。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ceiling => "ceiling",
            Self::Stop => "stop",
        }
    }
}

/// 一次生成的结果：这次新产出的字节，以及它为什么停下。
///
/// 只装**这一次**新产出的字节（不含预填充）；停止字节含在里面——它是模型真的产出的那一个，从结果
/// 里抹掉会让"产出了什么"与"为什么停"互相对不上。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emission {
    bytes: Vec<u8>,
    halt: Halt,
}

impl Emission {
    /// 这次新产出的字节。
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// 停下来的时候发生了哪种停止。
    pub fn halt(&self) -> Halt {
        self.halt
    }

    /// 产出的字节数（含停止字节）。
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// 一个字节都没产出：预算为零，或第一步就产出了停止字节。
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// 逐字节自回归循环：一条切分规则 + 表示侧 + 输出头。
///
/// 状态只有"已产出的字节、它们的分数、patch 起点、已收口 patch 的表示"，没有可训练参数：本层是装配面
/// （谁在什么时候调哪一段），不是又一层权重。
#[derive(Debug, Clone)]
pub struct Emitter<S: Scalar = f32> {
    rule: Boundary<S>,
    encoder: ByteEncoder<S>,
    decoder: ByteDecoder<S>,
    /// 预填充 + 已生成的字节，按顺序；只追加。
    bytes: Vec<u8>,
    /// 每个字节的分数（自信息 `−ln p`），与 [`Emitter::bytes`] 等长。
    scores: Vec<S>,
    /// patch 起点，严格递增；最后一个是**仍在长**的那个 patch。
    starts: Vec<usize>,
    /// 已收口 patch 的表示（紧凑，行宽 `dim`）：只追加，不重算。
    reprs: Vec<S>,
}

impl<S: Scalar> Emitter<S> {
    /// 由切分规则、表示侧与输出头装配。
    ///
    /// # Errors
    ///
    /// 两侧宽度不等时返回 `Code::Invalid`：表示侧给出 `dim`，输出头吃 `width`，不等的内积照算不误，
    /// 结果却是另一个数（与 [`ByteDecoder::logits`] 同一条判据，只是提前到装配时）。
    pub fn new(
        rule: Boundary<S>,
        encoder: ByteEncoder<S>,
        decoder: ByteDecoder<S>,
    ) -> Result<Self> {
        if encoder.dim() != decoder.width() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "表示宽度 {} 与输出头宽度 {} 不等：条件喂不进这一个头",
                    encoder.dim(),
                    decoder.width()
                ),
            )
            .with(SITE));
        }
        Ok(Self {
            rule,
            encoder,
            decoder,
            bytes: Vec::new(),
            scores: Vec::new(),
            starts: Vec::new(),
            reprs: Vec::new(),
        })
    }

    /// 预填充：把一段字节当成已有历史（prompt），`scores` 是逐字节的分数，与 `prompt` 等长。
    ///
    /// 边界边读边判（[`Boundary::opens`]），收口的 patch 立刻编码——与后续生成走的是同一条路径，所以
    /// "预填充后再生成"与"连着生成"只差条件里的历史，不差切分口径。再调一次是**替换**历史而不是追加：
    /// 换 prompt 不该保留上一次的切分与表示。
    ///
    /// # Errors
    ///
    /// 字节数与分数数不等、某个分数非有限、规则参数越界、或单 patch 区间不合法——返回
    /// `Code::Invalid`。失败时旧状态原样保留：先在局部队列里攒完再一次性换上。
    pub fn seed(&mut self, prompt: &[u8], scores: &[S]) -> Result<()> {
        if prompt.len() != scores.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "预填充 {} 个字节却给了 {} 个分数：分数是逐字节的",
                    prompt.len(),
                    scores.len()
                ),
            )
            .with(SITE));
        }
        let replay = replay(&self.rule, &self.encoder, prompt, scores)?;
        self.bytes = replay.bytes;
        self.scores = replay.scores;
        self.starts = replay.starts;
        self.reprs = replay.reprs;
        Ok(())
    }

    /// 生成并追加**一个**字节，返回它。
    ///
    /// 一步 = 条件（上一条收口 patch 的表示 + 上一个字节的向量）→ 贪心字节与它的概率 → 分数
    /// （`−ln p`）→ 问边界 → 收口的那个 patch 就地编码。已收口的表示不会再算，这正是逐字节生成没有
    /// 被 patch 数拖成平方的原因。
    ///
    /// # Errors
    ///
    /// 条件里的几何或分数出界（例如概率下溢到 `0` 让自信息成无穷）——返回 `Code::Invalid`。贪心保证
    /// 概率不小于 `1 / BYTES`，所以这条路径只在参数与状态不一致时才会被走到。
    pub fn step(&mut self) -> Result<u8> {
        let state = self.condition()?;
        let (byte, probability) = self.decoder.pick(&state)?;
        self.append(byte, S::ZERO.sub(probability.ln()))?;
        Ok(byte)
    }

    /// 追加一个字节与它的分数，并问边界：收口的那一刻就地编码。
    ///
    /// 生成与预填充共用这一处状态机：留一份、改写一份，`starts` 与 `reprs` 就会有一条路径忘了同步。
    fn append(&mut self, byte: u8, score: S) -> Result<()> {
        let index = self.bytes.len();
        self.bytes.push(byte);
        self.scores.push(score);
        if self.rule.opens(&self.scores, index)? {
            close(
                &self.encoder,
                &self.bytes,
                &mut self.starts,
                &mut self.reprs,
                index,
            )?;
        }
        Ok(())
    }

    /// 连走到字节预算用完或产出停止字节。
    ///
    /// `ceiling` 是**这一次**的预算（不含预填充）：零就不生成，记 [`Halt::Ceiling`]——没有上限的循环
    /// 不叫生成。`stop` 是可选的停止字节（EOS）：它真的被产出的那一步也算在产出里。
    ///
    /// # Errors
    ///
    /// 同 [`Emitter::step`]。
    pub fn run(&mut self, ceiling: usize, stop: Option<u8>) -> Result<Emission> {
        let from = self.bytes.len();
        let mut halt = Halt::Ceiling;
        while self.bytes.len() - from < ceiling {
            if Some(self.step()?) == stop {
                halt = Halt::Stop;
                break;
            }
        }
        Ok(Emission {
            bytes: self.bytes[from..].to_vec(),
            halt,
        })
    }

    /// 解码下一个字节的条件：上一条收口 patch 的表示 + 上一个字节的 N-gram 向量。
    ///
    /// 两项都只依赖已收口的历史，所以同一个字节的分布在"patch 又长了几个字节"前后不变——"边界只看
    /// 过去"这条性质因此延伸到了生成上。空历史给全零：还没有记忆可加，不拿零向量假装有一条。
    ///
    /// # Errors
    ///
    /// 已有字节的向量取不到（几何与状态不一致）——返回 `Code::Invalid`。
    pub fn condition(&self) -> Result<Vec<S>> {
        let dim = self.dim();
        let mut state = vec![S::ZERO; dim];
        if self.reprs.len() >= dim {
            state.copy_from_slice(&self.reprs[self.reprs.len() - dim..]);
        }
        if self.bytes.is_empty() {
            return Ok(state);
        }
        let glyph = self.encoder.glyph(&self.bytes, self.bytes.len() - 1)?;
        for (acc, &value) in state.iter_mut().zip(&glyph) {
            *acc = acc.add(value);
        }
        Ok(state)
    }

    /// 已产出的字节（预填充 + 生成），按顺序。
    ///
    /// 只追加：一旦产出的字节不再被改写，这与"已收口的表示不重算"是同一件事的两面。
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// 当前的切分：把逐字节分数重放一遍判定。
    ///
    /// [`Boundary::split`] 与逐字节判定共用 [`Boundary::opens`]，所以重放结果与边生成边判的内部起点
    /// 逐位相同。末段是仍在长的那个 patch：它算得出表示，但还没收口。
    ///
    /// # Errors
    ///
    /// 分数非有限或规则参数越界——返回 `Code::Invalid`（内部分数由贪心概率给出，恒有限）。
    pub fn patches(&self) -> Result<Patches> {
        self.rule.split(&self.scores)
    }

    /// 已收口 patch 的表示，紧凑排列、行宽 `dim`：行数 = [`Emitter::patches`] 的 `len` 减一（末段还没
    /// 收口）。
    ///
    /// 全局 transformer 就运行在这一列上：遍数是行数，不是字节数。
    pub fn reprs(&self) -> &[S] {
        &self.reprs
    }

    /// 逐字节的分数（自信息 `−ln p`），与 [`Emitter::bytes`] 等长。
    pub fn scores(&self) -> &[S] {
        &self.scores
    }

    /// 表示宽度：一行表示的元素数，也是条件的维数。
    pub fn dim(&self) -> usize {
        self.encoder.dim()
    }
}

/// 重放一段字节的切分：边追加边问边界，收口的 patch 立刻编码。
///
/// 预填充与生成走同一条判定，于是“先预填再生成”与“连着生成”只差条件里的历史，不差切分口径。
fn replay<S: Scalar>(
    rule: &Boundary<S>,
    encoder: &ByteEncoder<S>,
    prompt: &[u8],
    scores: &[S],
) -> Result<Replay<S>> {
    let mut replay = Replay {
        bytes: Vec::with_capacity(prompt.len()),
        scores: Vec::with_capacity(prompt.len()),
        starts: Vec::new(),
        reprs: Vec::new(),
    };
    for (index, &byte) in prompt.iter().enumerate() {
        replay.bytes.push(byte);
        replay.scores.push(scores[index]);
        if rule.opens(&replay.scores, index)? {
            close(
                encoder,
                &replay.bytes,
                &mut replay.starts,
                &mut replay.reprs,
                index,
            )?;
        }
    }
    Ok(replay)
}

/// 重放的四段状态：字节、分数、patch 起点与已收口 patch 的表示。
struct Replay<S: Scalar> {
    bytes: Vec<u8>,
    scores: Vec<S>,
    starts: Vec<usize>,
    reprs: Vec<S>,
}

/// 在 `index` 处开一个新 patch：上一条（若开着）就地收口，只编码它自己那一段。
///
/// 收口是**唯一**算表示的时刻：一条 patch 从开到最后一次追加字节，表示只在它被关掉时算一遍。首字节
/// 时还没有开着的 patch，于是只记起点、什么都不编码。
fn close<S: Scalar>(
    encoder: &ByteEncoder<S>,
    bytes: &[u8],
    starts: &mut Vec<usize>,
    reprs: &mut Vec<S>,
    index: usize,
) -> Result<()> {
    if let Some(&from) = starts.last() {
        reprs.extend_from_slice(&encoder.patch(bytes, from..index)?);
    }
    starts.push(index);
    Ok(())
}

#[cfg(test)]
#[path = "../tests/emit_tests.rs"]
mod tests;
