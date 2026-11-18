//! 分配与复用：等长空闲表 + [`Buffers`]，把 [`autograd::Store`] 落到真实内存。
//!
//! 块本身来自内部空闲表，稳定态不再向系统要内存；分配走 `try_reserve_exact`，形状可能来自
//! 不受信任的权重文件头，分配失败是一条错误而不是 abort。

use crate::kernel::Kernel;
use crate::shape::{Data, Kind, View, no_storage, zeroed};
use crate::store::SITE;
use autograd::{Sizes, Store, Var};
use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

/// 同尺寸缓冲的空闲表：按**正好相等的长度**分档，只复用等长的块（池容量永不超峰值）。
#[derive(Debug, Default)]
pub(crate) struct Pool {
    f32s: Vec<(usize, Vec<Vec<f32>>)>,
    u8s: Vec<(usize, Vec<Vec<u8>>)>,
}

/// 一条 lane 的四件套：取、还、数块数、数字节（查找是一次线性扫描，档数由形状的个数决定，通常是个位数）。
macro_rules! buckets {
    ($(($acquire:ident, $give:ident, $pooled:ident, $bytes:ident, $lane:ty, $pile:ident)),* $(,)?) => {$(
        impl Pool {
            /// 取一块长度正好是 `len` 的缓冲：池里有就复用（只对齐长度），否则零值分配。
            fn $acquire(&mut self, len: usize) -> Result<Vec<$lane>> {
                let pile = &mut self.$pile;
                let free = pile.iter_mut().find(|(size, _)| *size == len);
                match free.and_then(|(_, blocks)| blocks.pop()) {
                    Some(mut values) => {
                        values.truncate(len);
                        Ok(values)
                    }
                    None => zeroed(len, Default::default()),
                }
            }

            /// 还回一块空闲块，按它的长度归到对应档。
            fn $give(&mut self, buffer: Vec<$lane>) {
                let len = buffer.len();
                let pile = &mut self.$pile;
                match pile.iter_mut().find(|(size, _)| *size == len) {
                    Some((_, blocks)) => blocks.push(buffer),
                    None => pile.push((len, vec![buffer])),
                }
            }

            /// 档里现成可复用的块数。
            fn $pooled(&self) -> usize {
                self.$pile.iter().map(|(_, blocks)| blocks.len()).sum()
            }

            /// 档里空闲块占用的字节数，与 [`Buffers::live`] 同口径对账。
            fn $bytes(&self) -> u64 {
                self.$pile
                    .iter()
                    .flat_map(|(_, blocks)| blocks.iter())
                    .map(|block| (block.len() * size_of::<$lane>()) as u64)
                    .sum()
            }
        }
    )*};
}

buckets!(
    (acquire_f32, give_f32, pooled_f32, bytes_f32, f32, f32s),
    (acquire_u8, give_u8, pooled_u8, bytes_u8, u8, u8s),
);

impl Pool {
    /// 一块 `dtype` 的零值缓冲：先试空闲表，没有再向分配器要（"取多少元素"只在这里算一次）。
    pub(crate) fn block(&mut self, dtype: DType, shape: &Shape) -> Result<Data> {
        match Kind::of(dtype) {
            Some(Kind::F32) => Ok(Data::F32(self.acquire_f32(shape.numel())?)),
            Some(Kind::U8) => Ok(Data::U8(self.acquire_u8(shape.element_bytes(dtype))?)),
            None => Err(no_storage(dtype)),
        }
    }

    /// 把一块缓冲放回空闲表。
    fn recycle(&mut self, block: Data) {
        match block {
            Data::F32(values) => self.give_f32(values),
            Data::U8(values) => self.give_u8(values),
        }
    }

    /// 空闲表里的块数。
    fn pooled(&self) -> usize {
        self.pooled_f32() + self.pooled_u8()
    }

    /// 空闲表里的字节数。
    fn pooled_bytes(&self) -> u64 {
        self.bytes_f32() + self.bytes_u8()
    }
}

/// 以 [`Var`] 为下标的真实缓冲：`autograd::Store` 在训练侧的实现。
///
/// 一个训练步里只有一个实例，逐图复用：`load` 装常驻值、`compute` 算激活、`free` 还激活，
/// 步与步之间用 [`Buffers::clear`] 一次性归还；块本身来自内部空闲表，稳定态不再向系统要内存。
#[derive(Debug)]
pub struct Buffers<K: Kernel> {
    sizes: Sizes,
    kernel: K,
    slots: Vec<Option<Data>>,
    /// 梯度缓冲：autograd 只出 tape / plan / Replay，梯度存储由训练侧自持，与激活槽平行按变量编号。
    gradients: Vec<Option<Vec<f32>>>,
    pool: Pool,
    live: u64,
    peak: u64,
}

impl<K: Kernel> Buffers<K> {
    /// 由形状表与内核构造：槽位数与 `sizes.len()` 同口径（即 `Tape::vars()`）。
    pub fn new(sizes: Sizes, kernel: K) -> Self {
        let slots = sizes.len();
        Self {
            slots: vec![None; slots],
            gradients: vec![None; slots],
            sizes,
            kernel,
            pool: Pool::default(),
            live: 0,
            peak: 0,
        }
    }

    /// 装入一个常驻值（参数、输入、或外部算好的结果）。
    ///
    /// 登记的元数据必须与实际数据一致：装错类型不会在训练中途才暴露成一串无意义的数；同一个
    /// 变量装第二次会先把旧的还回空闲表。
    ///
    /// # Errors
    ///
    /// 变量超出形状表的槽位、该变量没有登记形状与元素类型（都是 [`Code::Internal`]：调用方漏了
    /// `Sizes::set`）、数据与登记的（形状，元素类型）不相容（[`Code::Invalid`]）时返回说明。
    pub fn load(&mut self, var: Var, data: Data) -> Result<()> {
        let slot = self.slot(var)?;
        let Some((shape, dtype)) = self.sizes.get(var) else {
            let message = format!(
                "变量 {} 没有登记形状与元素类型：装入的数据无从校验（`Sizes` 必须按 `Tape::vars()` 构造并逐项 `set`）",
                var.id()
            );
            return Err(Error::new(Code::Internal, message).with(SITE));
        };
        if !data.fits(dtype) || data.bytes() != shape.element_bytes(dtype) as u64 {
            return Err(mismatch(var, shape, dtype, &data));
        }
        self.install(slot, data);
        Ok(())
    }

    /// 读一个变量当前的值；不在场（从未算过、或已被交还）时为 `None`。
    pub fn value(&self, var: Var) -> Option<View<'_>> {
        self.slots
            .get(var.id() as usize)
            .and_then(Option::as_ref)
            .map(Data::view)
    }

    /// 装入一个参数槽的当前梯度；长度必须与登记形状一致，梯度 lane 只有 f32。
    ///
    /// 梯度存储在训练侧：autograd 只给 tape / plan / Replay，不持有梯度；这里是梯度唯一落点。
    ///
    /// # Errors
    ///
    /// 变量未登记形状（[`Code::Internal`]）、登记类型不是 f32 或长度不符（[`Code::Invalid`]）。
    pub fn load_gradient(&mut self, var: Var, grads: &[f32]) -> Result<()> {
        self.validate_gradient(var, grads)?;
        self.gradients[var.id() as usize] = Some(grads.to_vec());
        Ok(())
    }

    /// 把一批微批梯度累加进现有梯度缓冲；缓冲为空时先装入副本。
    ///
    /// 累积窗口跨多个 micro-batch：梯度逐批相加而不是覆盖，窗口满了才由优化器步进。
    /// 长度校验与 [`load_gradient`](Self::load_gradient) 同口径。
    ///
    /// # Errors
    ///
    /// 变量未登记形状（[`Code::Internal`]）、登记类型不是 f32 或长度不符（[`Code::Invalid`]）。
    pub fn accumulate_gradient(&mut self, var: Var, grads: &[f32]) -> Result<()> {
        self.validate_gradient(var, grads)?;
        let slot = var.id() as usize;
        match &mut self.gradients[slot] {
            Some(existing) => {
                // 逐元素相加：长度已在上面校验一致，串行依赖 `a += b`，无法整块搬运。
                for (accumulated, incoming) in existing.iter_mut().zip(grads) {
                    *accumulated += *incoming;
                }
            }
            None => self.gradients[slot] = Some(grads.to_vec()),
        }
        Ok(())
    }

    /// 梯度长度/类型校验：装入与累加共用同一条判据。
    fn validate_gradient(&self, var: Var, grads: &[f32]) -> Result<()> {
        let slot = self.slot(var)?;
        let Some((shape, dtype)) = self.sizes.get(var) else {
            return Err(unregistered(var));
        };
        if dtype != DType::F32 {
            let message = format!("变量 {} 登记为 {dtype:?}，梯度 lane 只存 f32", var.id());
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        if grads.len() != shape.numel() {
            let message = format!(
                "梯度长度 {} 与变量 {} 的形状 {:?}（{} 元素）不一致",
                grads.len(),
                var.id(),
                shape.dims(),
                shape.numel()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        debug_assert!(slot < self.gradients.len());
        Ok(())
    }

    /// 读一个变量当前的梯度；未装入时为 `None`。
    pub fn gradient(&self, var: Var) -> Option<&[f32]> {
        self.gradients
            .get(var.id() as usize)
            .and_then(Option::as_deref)
    }

    /// 全部梯度槽的可变表：批量取视图用一条借用，避免逐槽重复可变借用。
    pub fn gradients_mut(&mut self) -> &mut [Option<Vec<f32>>] {
        &mut self.gradients
    }

    /// 取走一个变量的梯度向量：整块移交所有权，避免同一缓冲重复可变借用。
    ///
    /// 调用方在优化器/同步结束后把结果放回 [`Self::put_back_gradient`]；中途失败也必须归还，
    /// 否则该槽梯度会丢失。
    pub fn take_gradient(&mut self, var: Var) -> Option<Vec<f32>> {
        self.gradients.get_mut(var.id() as usize).and_then(Option::take)
    }

    /// 放回 [`Self::take_gradient`] 取走的梯度向量；空向量不写回（与未装入同口径）。
    pub fn put_back_gradient(&mut self, var: Var, grads: Vec<f32>) {
        if let Some(slot) = self.gradients.get_mut(var.id() as usize) {
            *slot = (!grads.is_empty()).then_some(grads);
        }
    }

    /// 清空全部梯度缓冲：参数/激活槽不动，只归还梯度 lane 的向量。
    pub fn clear_gradients(&mut self) {
        // 槽位长度保留（索引对应变量编号），只把梯度 lane 的向量整块置空。
        self.gradients.fill(None);
    }

    /// 当前占用的字节数（含常驻值）。
    pub fn live(&self) -> u64 {
        self.live
    }

    /// 占用字节数的峰值：与 [`autograd::Plan::peak`] 同口径之处是"都按真实占用量统计"，
    /// 差一处是这里也含常驻的参数与输入。
    pub fn peak(&self) -> u64 {
        self.peak
    }

    /// 把峰值重新对齐到当前占用：分段测量（前向一段、重放一段）时各测各的。
    pub fn reset_peak(&mut self) {
        self.peak = self.live;
    }

    /// 空闲表里现成可复用的块数（诊断用：稳定态应当不增长）。
    pub fn pooled(&self) -> usize {
        self.pool.pooled()
    }

    /// 空闲表里可复用的字节数，与 [`live`](Self::live) / [`peak`](Self::peak) 同口径。
    pub fn pooled_bytes(&self) -> u64 {
        self.pool.pooled_bytes()
    }

    /// 一次性归零：换训练步、或换图时调用。在场块回到空闲表留待等长复用，只清占用与峰值记账，
    /// 稳定态不再向分配器要内存。
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            if let Some(block) = slot.take() {
                self.pool.recycle(block);
            }
        }
        self.live = 0;
        self.peak = 0;
    }

    /// 形状表。
    pub fn sizes(&self) -> &Sizes {
        &self.sizes
    }

    /// 变量对应的槽位；越界说明 `Sizes` 与 `Tape` 不同源。
    fn slot(&self, var: Var) -> Result<usize> {
        let index = var.id() as usize;
        if index >= self.slots.len() {
            let message = format!(
                "变量 {} 超出形状表的 {} 个槽位：Sizes 必须按 Tape::vars() 构造",
                var.id(),
                self.slots.len()
            );
            return Err(Error::new(Code::Internal, message).with(SITE));
        }
        Ok(index)
    }

    /// 输出槽位与登记的（形状，元素类型）：两者缺一都说明 `Sizes` 与 `Tape` 不同源。
    fn target(&self, var: Var) -> Result<(usize, Shape, DType)> {
        let slot = self.slot(var)?;
        let (shape, dtype) = self.sizes.get(var).ok_or_else(|| unregistered(var))?;
        Ok((slot, shape.clone(), dtype))
    }

    /// 装入一块缓冲：交还旧的、记账、占住槽位。
    fn install(&mut self, slot: usize, block: Data) {
        self.release(slot);
        self.live += block.bytes();
        self.peak = self.peak.max(self.live);
        self.slots[slot] = Some(block);
    }

    /// 交还一个槽位：`live` 当场降下来，块回到空闲表。
    fn release(&mut self, slot: usize) {
        if let Some(block) = self.slots[slot].take() {
            self.live -= block.bytes();
            self.pool.recycle(block);
        }
    }
}

impl<K: Kernel> Store for Buffers<K> {
    fn compute(&mut self, op: &str, inputs: &[Var], output: Var) -> Result<()> {
        let (slot, shape, dtype) = self.target(output)?;
        let views = views(&self.slots, output, inputs)?;
        let mut block = self.pool.block(dtype, &shape)?;
        if let Err(error) = self.kernel.compute(op, &views, &mut block.view_mut()) {
            self.pool.recycle(block);
            return Err(error);
        }
        drop(views);
        self.install(slot, block);
        Ok(())
    }

    fn free(&mut self, var: Var) {
        let slot = var.id() as usize;
        // 计划/磁带错位把槽位之外的变量交回来是内部不变量破坏；release 下 Store::free 没有
        // 返回通道，只能在 debug 构建里尽早暴露，release 维持静默跳过以免污染训练步。
        debug_assert!(
            slot < self.slots.len(),
            "free 交还了槽位 {} 之外的变量（共 {} 槽）",
            slot,
            self.slots.len()
        );
        if slot < self.slots.len() {
            self.release(slot);
        }
    }
}

/// 一次重算的全部输入视图：缺一个就是"重放读到了已释放的值"。
fn views<'a>(slots: &'a [Option<Data>], output: Var, inputs: &[Var]) -> Result<Vec<View<'a>>> {
    inputs
        .iter()
        .map(|input| view_of(slots, output, *input))
        .collect()
}

/// 一个输入视图；不在场时把两个编号都写进错误。
fn view_of<'a>(slots: &'a [Option<Data>], output: Var, input: Var) -> Result<View<'a>> {
    slots
        .get(input.id() as usize)
        .and_then(Option::as_ref)
        .map(Data::view)
        .ok_or_else(|| missing_input(output, input))
}

/// 没登记形状就分配不了缓冲：前向记录时漏了 `Sizes::set`。
fn unregistered(var: Var) -> Error {
    let message = format!(
        "变量 {} 没有登记形状，无法为它分配缓冲：前向记录时漏了 Sizes::set",
        var.id()
    );
    Error::new(Code::Internal, message).with(SITE)
}

/// 装入的数据与登记的（形状，元素类型）不相容。
fn mismatch(var: Var, shape: &Shape, dtype: DType, data: &Data) -> Error {
    let label = match data {
        Data::F32(_) => " f32",
        Data::U8(_) => " u8",
    };
    let message = format!(
        "变量 {} 登记为 {dtype:?} 的 {:?}（{} 字节），装入的却是 {} 字节的{label:?}",
        var.id(),
        shape.dims(),
        shape.element_bytes(dtype),
        data.bytes()
    );
    Error::new(Code::Invalid, message).with(SITE)
}

/// 一个输入不在场：重放读到了已交还（或从未算过）的值。
fn missing_input(output: Var, input: Var) -> Error {
    let message = format!(
        "重算 {} 时输入 {} 不在场（已被交还或从未算过）",
        output.id(),
        input.id()
    );
    Error::new(Code::Missing, message).with(SITE)
}
