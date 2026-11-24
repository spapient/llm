//! 优化器矩状态、累积窗口梯度与权重绑定表的快照类型、编码与解码。

use crate::codec::{Cursor, frame, invalid};
use crate::snapshot;
use crate::snapshot::PREALLOC;
use diagnostic::{Code, Error, Result};
use crate::sink::Sink;

/// 一个矩状态/累积梯度槽最少占 1 字节（槽长 varint）；解码据此把声明槽位与剩余字节对账。
pub(crate) const MIN_MOMENT_SLOT: usize = 1;

/// f32 小端字节数。
pub(crate) const F32_BYTES: usize = 4;

/// 一个参数槽的优化器矩状态：一阶矩 `m` 与二阶矩 `v`，长度一致。
#[derive(Debug, Clone, PartialEq)]
pub struct SlotMoments {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl SlotMoments {
    /// 由一阶/二阶矩构造；两张表长度必须一致，否则 [`Code::Invalid`]。
    pub fn new(m: Vec<f32>, v: Vec<f32>) -> Result<Self> {
        if m.len() != v.len() {
            let message = format!(
                "矩状态长度不一致：一阶矩 {} 元素，二阶矩 {} 元素",
                m.len(),
                v.len()
            );
            return Err(
                Error::new(Code::Invalid, message).with("storage::snapshot::SlotMoments::new")
            );
        }
        Ok(Self { m, v })
    }

    /// 一阶矩。
    pub fn m(&self) -> &[f32] {
        &self.m
    }

    /// 二阶矩。
    pub fn v(&self) -> &[f32] {
        &self.v
    }
}

/// 优化器矩状态表：按槽位编号排列，外加优化器全局步数（偏差修正分母）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Moments {
    slots: Vec<SlotMoments>,
    step: u64,
}

impl Moments {
    /// 空矩状态：无矩优化器或尚未走过一步。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 由矩表与优化器步数构造。
    pub fn new(slots: Vec<SlotMoments>, step: u64) -> Self {
        Self { slots, step }
    }

    /// 各槽矩状态。
    pub fn slots(&self) -> &[SlotMoments] {
        &self.slots
    }

    /// 优化器全局步数。
    pub fn step(&self) -> u64 {
        self.step
    }
}

/// 累积窗口内尚未步进的梯度汇总：按槽位编号排列的 f32 表，外加窗口内已汇总的 micro-batch 数。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GradAccum {
    slots: Vec<Vec<f32>>,
    count: u32,
}

impl GradAccum {
    /// 空累积：窗口从第一批 micro-batch 开始。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 由各槽汇总梯度与已汇总批数构造。
    pub fn new(slots: Vec<Vec<f32>>, count: u32) -> Self {
        Self { slots, count }
    }

    /// 各槽累积梯度。
    pub fn slots(&self) -> &[Vec<f32>] {
        &self.slots
    }

    /// 当前窗口已汇总的 micro-batch 数。
    pub fn count(&self) -> u32 {
        self.count
    }
}

/// 权重绑定表：`(源槽下标, 回填槽下标)` 对，与会话登记的 `register_tie` 同口径。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ties {
    pairs: Vec<(usize, usize)>,
}

impl Ties {
    /// 空绑定表：没有源/回填槽共享权重。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 由绑定对构造；每对为 `(源槽下标, 回填槽下标)`。
    pub fn new(pairs: Vec<(usize, usize)>) -> Self {
        Self { pairs }
    }

    /// 已登记的绑定对。
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }
}

/// 一张 f32 表：长度 varint + 各元素小端；矩状态与累积梯度共用这条写法。
fn write_f32_values(sink: &mut impl Sink, values: &[f32]) {
    for value in values {
        sink.put(&value.to_le_bytes());
    }
}

/// 带长度的 f32 表：累积梯度槽与矩状态一阶矩用它。
fn write_f32_table(sink: &mut impl Sink, values: &[f32]) {
    snapshot::varint(sink, values.len() as u64);
    write_f32_values(sink, values);
}

/// 矩状态段：槽位数 → 每槽长度 + 一阶/二阶矩 f32 小端 → 优化器步数（两表等长，长度只写一次）。
pub(crate) fn write_moments(sink: &mut impl Sink, moments: &Moments) {
    snapshot::varint(sink, moments.slots().len() as u64);
    for slot in moments.slots() {
        snapshot::varint(sink, slot.m().len() as u64);
        write_f32_values(sink, slot.m());
        write_f32_values(sink, slot.v());
    }
    snapshot::varint(sink, moments.step());
}

/// 累积梯度段：窗口批数 → 槽位数 → 每槽长度 + f32 小端；布局与矩状态段平行。
pub(crate) fn write_grad_accum(sink: &mut impl Sink, accum: &GradAccum) {
    snapshot::varint(sink, u64::from(accum.count()));
    snapshot::varint(sink, accum.slots().len() as u64);
    for slot in accum.slots() {
        write_f32_table(sink, slot);
    }
}

/// 权重绑定段：绑定对数 → 每对源槽/回填槽下标（varint）；布局与矩状态/累积段平行。
pub(crate) fn write_ties(sink: &mut impl Sink, ties: &Ties) {
    snapshot::varint(sink, ties.pairs().len() as u64);
    for &(source, tied) in ties.pairs() {
        snapshot::varint(sink, source as u64);
        snapshot::varint(sink, tied as u64);
    }
}

/// 槽位表：先与剩余字节对账（伪造的计数撑不大分配），再逐槽重建。
///
/// 矩状态、累积梯度与绑定表的槽形状不同，收集循环却相同——差别只在"读一个槽"，
/// 于是它成为参数，三处解码共用这一份对账与预分配。
fn read_slots<T>(
    cursor: &mut Cursor<'_>,
    count: usize,
    label: &str,
    slot: impl Fn(&mut Cursor<'_>, usize) -> Result<T>,
) -> Result<Vec<T>> {
    bound_slots(count, cursor.remaining(), label)?;
    let mut slots = Vec::with_capacity(count.min(PREALLOC));
    for idx in 0..count {
        slots.push(slot(cursor, idx)?);
    }
    Ok(slots)
}

/// 矩状态段解码：长度先对账，再逐槽重建两张 f32 表。
pub(crate) fn read_moments(cursor: &mut Cursor<'_>) -> Result<Moments> {
    let count = cursor.count("矩状态槽位数")?;
    let slots = read_slots(cursor, count, "矩状态", read_moment_slot)?;
    let step = cursor.varint()?;
    Ok(Moments::new(slots, step))
}

/// 一个矩状态槽：长度对账后读一阶、二阶两张表。
fn read_moment_slot(cursor: &mut Cursor<'_>, idx: usize) -> Result<SlotMoments> {
    let len = cursor.count("矩状态长度")?;
    let m = read_f32_table(cursor, len).map_err(frame(idx))?;
    let v = read_f32_table(cursor, len).map_err(frame(idx))?;
    SlotMoments::new(m, v).map_err(frame(idx))
}

/// 累积梯度段解码：长度先对账，再逐槽重建 f32 表。
pub(crate) fn read_grad_accum(cursor: &mut Cursor<'_>) -> Result<GradAccum> {
    let count = cursor.count("累积窗口批数")?;
    let slot_count = cursor.count("累积梯度槽位数")?;
    let slots = read_slots(cursor, slot_count, "累积梯度", read_accum_slot)?;
    Ok(GradAccum::new(slots, count as u32))
}

/// 一个累积梯度槽：长度对账后读 f32 表。
fn read_accum_slot(cursor: &mut Cursor<'_>, idx: usize) -> Result<Vec<f32>> {
    let len = cursor.count("累积梯度长度")?;
    read_f32_table(cursor, len).map_err(frame(idx))
}

/// 权重绑定段解码：对数先与剩余字节对账，再逐对重建下标表。
pub(crate) fn read_ties(cursor: &mut Cursor<'_>) -> Result<Ties> {
    let count = cursor.count("权重绑定对数")?;
    let pairs = read_slots(cursor, count, "权重绑定", |cursor, _| {
        read_tie_pair(cursor)
    })?;
    Ok(Ties::new(pairs))
}

/// 一对绑定下标：源槽与回填槽。
fn read_tie_pair(cursor: &mut Cursor<'_>) -> Result<(usize, usize)> {
    let source = cursor.count("绑定源槽")?;
    let tied = cursor.count("绑定回填槽")?;
    Ok((source, tied))
}

/// 槽位数与剩余字节对账：伪造的计数撑不大分配。
fn bound_slots(count: usize, remaining: usize, label: &str) -> Result<()> {
    if count > remaining / MIN_MOMENT_SLOT {
        return Err(invalid(format!(
            "检查点声明了 {count} 个{label}槽，但剩余 {remaining} 字节最多容纳 {} 个",
            remaining / MIN_MOMENT_SLOT
        )));
    }
    Ok(())
}

/// 读一张 `len` 个小端 f32 表；字节不足先在这一层报错，不把尾巴当零填充。
fn read_f32_table(cursor: &mut Cursor<'_>, len: usize) -> Result<Vec<f32>> {
    let bytes = cursor
        .take(len * F32_BYTES)
        .map_err(|error| invalid(format!("矩状态表字节不足：{error}")))?;
    let (words, rest) = bytes.as_chunks::<F32_BYTES>();
    debug_assert!(rest.is_empty(), "`take` 已按整字节对齐");
    Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
}
