//! 分配与复用：等长空闲表 + [`Bufs`]，把 [`autograd::Store`] 落到真实内存。
//!
//! **拥有**：`Bufs`、`Pool`、`Kernel`、`Reference`、`Data`、`View`、`ViewMut`，以及承载静态契约的
//! `Kind`、`Lane`、`Inputs`。**不拥有**：显存计划（`autograd::Plan`）、反向调度、形状表、算子参考算术
//! （`nn`）。**依赖**：`autograd`、`tensor`、`nn`、`prof`、`diagnostic`。
//!
//! 三条硬约束：缓冲只有 `f32` 与按位打包的 `u8` 两种形态（`unsafe_code = deny`，"把字节重解释成
//! 浮点"这条路关着，`F16` / `BF16` 于是在编译期就被拒绝）；内核表与 [`nn::registry::REGISTRY`] 同名同序，
//! "注册了算子却没有参考实现"是编译错误；`live` / `peak` 都是实测值，与 [`autograd::Plan`] 的承诺
//! 对账，而计划保留的检查点必须在重放开始前就在场。
//!
//! 两条语义轴平铺在同一条轴里：[`crate::kernel`] 是形态/视图/拆包契约与标量参考内核适配，本模块是
//! 等长空闲表与 [`Bufs`] 落地——持有该层统一的诊断站点，并把 `nn` 的算子清单展开成
//! [`OperatorSetOps`] 的默认方法。
//!
//! # 例子
//!
//! ```
//! use autograd::{Sizes, Store, Tape};
//! use llm_train::{Bufs, Data, Reference};
//! use tensor::{DType, Shape};
//!
//! let mut tape = Tape::new();
//! let (x, w) = (tape.leaf(), tape.leaf());
//! let y = tape.record("gemv_f32", &[w, x]).expect("记录");
//! let mut sizes = Sizes::new(tape.vars());
//! sizes.set(x, Shape::new([2]), DType::F32);
//! sizes.set(w, Shape::new([2, 2]), DType::F32);
//! sizes.set(y, Shape::new([2]), DType::F32);
//!
//! let mut store = Bufs::new(sizes, Reference);
//! store.load(x, Data::F32(vec![1.0, 2.0])).expect("输入");
//! store.load(w, Data::F32(vec![1.0, 0.0, 0.0, 1.0])).expect("权重");
//! store.compute_step("gemv_f32", &[w, x], y).expect("可算");
//! assert!(store.value(y).and_then(|view| view.f32()).is_some());
//! ```

use crate::geometry::{Geometry, Running};
pub use crate::kernel::{Data, Inputs, Kernel, Kind, Lane, Reference, View, ViewMut};
use crate::kernel::{fits, lane_of, no_storage, zeroed};
use autograd::{Sizes, Store, Var};
use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

/// 本模块的诊断站点：出错时用它标明"是这一层拒绝了这次请求"。
pub(crate) const SITE: &str = "train::Buffers";

/// 一条 lane 的泛型空闲表：按恰好相等的长度分档，容量只复用不膨胀。
#[derive(Debug, Default)]
struct LanePool<T> {
    piles: Vec<(usize, Vec<Vec<T>>)>,
}

impl<T: Clone + Default> LanePool<T> {
    /// 取等长空块；没有则走可失败的零值分配。
    fn acquire(&mut self, len: usize) -> Result<Vec<T>> {
        let free = self.piles.iter_mut().find(|(size, _)| *size == len);
        match free.and_then(|(_, blocks)| blocks.pop()) {
            Some(values) => Ok(values),
            None => zeroed(len, T::default()),
        }
    }

    /// 按当前长度归还空块。
    fn give(&mut self, buf: Vec<T>) {
        let len = buf.len();
        match self.piles.iter_mut().find(|(size, _)| *size == len) {
            Some((_, blocks)) => blocks.push(buf),
            None => self.piles.push((len, vec![buf])),
        }
    }

    fn pooled(&self) -> usize {
        self.piles.iter().map(|(_, blocks)| blocks.len()).sum()
    }

    fn bytes(&self) -> u64 {
        self.piles
            .iter()
            .flat_map(|(_, blocks)| blocks)
            .map(|block| (block.len() * size_of::<T>()) as u64)
            .sum()
    }
}

/// 两种存储 lane 共用 [`LanePool`] 的同一实现。
#[derive(Debug, Default)]
pub(crate) struct Pool {
    f32s: LanePool<f32>,
    u8s: LanePool<u8>,
}

impl Pool {
    /// 一块 `dtype` 的零值缓冲：先试空闲表，没有再向分配器要（"取多少元素"只在这里算一次）。
    pub(crate) fn block(&mut self, dtype: DType, shape: &Shape) -> Result<Data> {
        match lane_of(dtype) {
            Some(Kind::F32) => Ok(Data::F32(self.f32s.acquire(shape.numel())?)),
            Some(Kind::U8) => Ok(Data::U8(self.u8s.acquire(shape.element_bytes(dtype))?)),
            None => Err(no_storage(dtype)),
        }
    }

    /// 把一块缓冲放回空闲表。
    fn recycle(&mut self, block: Data) {
        match block {
            Data::F32(values) => self.f32s.give(values),
            Data::U8(values) => self.u8s.give(values),
        }
    }

    /// 空闲表里的块数。
    fn pooled(&self) -> usize {
        self.f32s.pooled() + self.u8s.pooled()
    }

    /// 空闲表里的字节数。
    fn pooled_bytes(&self) -> u64 {
        self.f32s.bytes() + self.u8s.bytes()
    }
}

/// 以 [`Var`] 为下标的真实缓冲：`autograd::Store` 在训练侧的实现。
#[derive(Debug)]
pub struct Bufs<K: Kernel> {
    sizes: Sizes,
    kernel: K,
    slots: Vec<Option<Data>>,
    /// 梯度缓冲：autograd 只出 tape / plan / Replay，梯度存储由训练侧自持，与激活槽平行按变量编号。
    gradients: Vec<Option<Vec<f32>>>,
    /// 前向/输出值的等长空闲表。
    pool: Pool,
    /// 清零后的梯度向量独立回收，下一批按等长直接复用。
    gradient_pool: Pool,
    /// 每个输出变量登记的算子实例几何：反向按同一个实例回传，前向/反向不各持一套几何。
    instances: Vec<Option<Geometry>>,
    /// 归一化的滚动统计：按输出变量常驻，训练期逐步递推（推理期直接取这一份）。
    running: Vec<Option<Running>>,
    live: u64,
    peak: u64,
    /// 反向逐节点复用的暂存：由 [`backprop`](crate::backprop) 借出、用完归还，
    /// 使反传在稳态不再逐节点向分配器要内存。
    pub(crate) scratch: crate::backprop::Scratch,
}

impl<K: Kernel> Bufs<K> {
    /// 由形状表与内核构造：槽位数与 `sizes.len()` 同口径（即 `Tape::vars()`）。
    pub fn new(sizes: Sizes, kernel: K) -> Self {
        let slots = sizes.len();
        Self {
            slots: vec![None; slots],
            gradients: vec![None; slots],
            instances: vec![None; slots],
            running: vec![None; slots],
            sizes,
            kernel,
            pool: Pool::default(),
            gradient_pool: Pool::default(),
            live: 0,
            peak: 0,
            scratch: Default::default(),
        }
    }

    /// 带实例几何的一步前向：几何登记在输出变量上，反向按同一个实例回传。
    ///
    /// # Errors
    ///
    /// 形状不符、或内核执行失败时返回对应诊断。
    pub fn compute_shaped_step(
        &mut self,
        op: &str,
        inputs: &[Var],
        output: Var,
        geometry: Geometry,
    ) -> Result<()> {
        self.run(op, inputs, output, Some(geometry))
    }

    /// 读某个输出变量的实例几何；未登记即该节点按声明档语义执行。
    pub fn geometry(&self, var: Var) -> Option<Geometry> {
        self.instances.get(var.id() as usize).copied().flatten()
    }

    /// 读某个输出变量上归一化的滚动统计；没跑过带几何的归一化即为 `None`。
    pub fn running(&self, var: Var) -> Option<&Running> {
        self.running.get(var.id() as usize).and_then(Option::as_ref)
    }

    /// 前向的单一实现：登记几何、取目标槽、算、装回。声明档与实例化只差 `geometry`。
    fn run(
        &mut self,
        op: &str,
        inputs: &[Var],
        output: Var,
        geometry: Option<Geometry>,
    ) -> Result<()> {
        register_geometry(&mut self.instances, output, geometry);
        let (slot, shape, dtype) = self.target(output)?;
        // 滚动统计先取出来（裸值），内核算完再写回：内核调用期间不与 `running` 槽位交叠借用。
        let mut stats = take_stats(&mut self.running, output, geometry);
        let views = views(&self.slots, output, inputs)?;
        let mut block = self.pool.block(dtype, &shape)?;
        let result = self.kernel.compute_shaped_step(
            op,
            &views,
            &mut block.view_mut(),
            geometry.as_ref(),
            stats.as_mut(),
        );
        restore_stats(&mut self.running, output, stats);
        if let Err(error) = result {
            self.pool.recycle(block);
            return Err(error);
        }
        drop(views);
        self.install(slot, block);
        Ok(())
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
        if !fits(&data, dtype) || data.bytes() != shape.element_bytes(dtype) as u64 {
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

    /// 原位镜像一段 f32 参数；已装配的常驻槽只覆盖内容，不交还再分配缓冲。
    pub(crate) fn overwrite_f32(&mut self, var: Var, values: &[f32]) -> Result<()> {
        self.validate_gradient(var, values)?;
        let slot = self.slot(var)?;
        self.overwrite_slot(slot, var, values)
    }

    /// 命中常驻 f32 槽原地覆盖；类型不符拒绝，空槽装入副本。
    fn overwrite_slot(&mut self, slot: usize, var: Var, values: &[f32]) -> Result<()> {
        match self.slots[slot].as_mut() {
            Some(Data::F32(existing)) => existing.copy_from_slice(values),
            Some(Data::U8(_)) => {
                return Err(Error::new(Code::Invalid, "参数镜像的目标不是 f32").with(SITE));
            }
            None => self.load(var, Data::F32(values.to_vec()))?,
        }
        Ok(())
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
        self.copy_gradient(var.id() as usize, grads)
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
        if let Some(existing) = &mut self.gradients[slot] {
            // 逐元素相加：长度已在上面校验一致，串行依赖 `a += b`，无法整块搬运。
            for (accumulated, incoming) in existing.iter_mut().zip(grads) {
                *accumulated += *incoming;
            }
            Ok(())
        } else {
            self.copy_gradient(slot, grads)
        }
    }

    /// 覆盖一条梯度：优先保留在场块，否则取同长度空闲块。
    fn copy_gradient(&mut self, slot: usize, grads: &[f32]) -> Result<()> {
        if let Some(existing) = &mut self.gradients[slot] {
            existing.clear();
            existing.extend_from_slice(grads);
            return Ok(());
        }
        let mut held = self.gradient_pool.f32s.acquire(grads.len())?;
        held.copy_from_slice(grads);
        self.gradients[slot] = Some(held);
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

    /// 取走一个变量的梯度向量：整块移交所有权，避免同一缓冲重复可变借用。
    pub fn take_gradient(&mut self, var: Var) -> Option<Vec<f32>> {
        self.gradients
            .get_mut(var.id() as usize)
            .and_then(Option::take)
    }

    /// 放回 [`Self::take_gradient`] 取走的梯度向量；空向量不写回（与未装入同口径）。
    pub fn put_back_gradient(&mut self, var: Var, grads: Vec<f32>) {
        if let Some(slot) = self.gradients.get_mut(var.id() as usize) {
            if grads.is_empty() {
                self.gradient_pool.f32s.give(grads);
                *slot = None;
            } else {
                *slot = Some(grads);
            }
        }
    }

    /// 用恢复计划预建的整表原子替换梯度；旧块回收，提交本身不再分配或校验失败。
    pub(crate) fn replace_gradients(&mut self, gradients: Vec<Option<Vec<f32>>>) {
        debug_assert_eq!(gradients.len(), self.gradients.len());
        self.clear_gradients();
        self.gradients = gradients;
    }

    /// 清空全部梯度缓冲：参数/激活槽不动，梯度向量进入独立空闲表供下一批复用。
    pub fn clear_gradients(&mut self) {
        for slot in &mut self.gradients {
            if let Some(grads) = slot.take() {
                self.gradient_pool.f32s.give(grads);
            }
        }
    }

    getters! {
        /// 当前占用的字节数（含常驻值）。
        live => live: u64,
        /// 峰值与激活计划同按真实占用量统计，但这里也含常驻参数与输入。
        peak => peak: u64,
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

    ref_getters! {
        /// 形状表。
        sizes => sizes: &Sizes,
    }

    /// 变量对应的槽位；越界说明 `Sizes` 与 `Tape` 不同源。
    fn slot(&self, var: Var) -> Result<usize> {
        let idx = var.id() as usize;
        if idx >= self.slots.len() {
            let message = format!(
                "变量 {} 超出形状表的 {} 个槽位：Sizes 必须按 Tape::vars() 构造",
                var.id(),
                self.slots.len()
            );
            return Err(Error::new(Code::Internal, message).with(SITE));
        }
        Ok(idx)
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

impl<K: Kernel> Store for Bufs<K> {
    fn compute_step(&mut self, op: &str, inputs: &[Var], output: Var) -> Result<()> {
        self.run(op, inputs, output, None)
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

/// 把带实例几何的一步前向里的几何登记在输出变量上；未带几何（声明档）即跳过。
fn register_geometry(instances: &mut [Option<Geometry>], output: Var, geometry: Option<Geometry>) {
    if let Some(geometry) = geometry
        && let Some(slot) = instances.get_mut(output.id() as usize)
    {
        *slot = Some(geometry);
    }
}

/// 取出归一化输出的滚动统计（裸值）：内核调用期间不与 `running` 槽位交叠借用。
fn take_stats(
    running: &mut [Option<Running>],
    output: Var,
    geometry: Option<Geometry>,
) -> Option<Running> {
    match geometry {
        Some(Geometry::Norm { channels, .. }) => Some(
            running
                .get_mut(output.id() as usize)
                .and_then(Option::take)
                .unwrap_or_else(|| Running::fresh(channels)),
        ),
        _ => None,
    }
}

/// 把内核算完后的滚动统计写回原槽位。
fn restore_stats(running: &mut [Option<Running>], output: Var, stats: Option<Running>) {
    if let Some(stats) = stats
        && let Some(held) = running.get_mut(output.id() as usize)
    {
        *held = Some(stats);
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

// ---- 类型化算子入口 --------------------------------------------------------
//
// 从 `nn::for_each_operator!` 的同一张清单展开，没有第二份装配表：后端只需实现
// `OperatorSetOps::apply` 一个原语，其余方法都是默认方法，名字取 `Operator::NAME`、
// 输入个数取 `Operator::ARITY`。方法名与 `Operator::NAME` 在 `nn` 里编译期逐字节对拍，
// 因此注册表、API 面与样本不会各自漂移。

/// 把算子清单展开成 trait：一个原语 + 每个算子一条默认方法。
macro_rules! typed_ops {
    ($($name:ident: $ty:ty),* $(,)?) => {
        /// 由 `nn::for_each_operator!` 展开的类型化算子入口。
        pub trait OperatorSetOps {
            /// 唯一原语：按注册名派发一次。
            ///
            /// # Errors
            ///
            /// 算子未注册、输入个数或形状不符时返回说明。
            fn apply(&mut self, name: &'static str, inputs: &[Var], output: Var) -> Result<()>;

            $(
                #[doc = concat!(
                    "`", stringify!($name),
                    "`：名字取 `Operator::NAME`，输入个数取 `Operator::ARITY`。"
                )]
                #[inline]
                fn $name(
                    &mut self,
                    inputs: &[Var; <$ty as ::nn::operator::Operator>::ARITY],
                    output: Var,
                ) -> Result<()> {
                    <Self as OperatorSetOps>::apply(
                        self,
                        <$ty as ::nn::operator::Operator>::NAME,
                        inputs.as_slice(),
                        output,
                    )
                }
            )*
        }
    };
}

::nn::for_each_operator!(typed_ops);

/// 运行时缓冲实现唯一原语：按注册名派发到既有内核表。
impl<K: Kernel> OperatorSetOps for Bufs<K> {
    fn apply(&mut self, name: &'static str, inputs: &[Var], output: Var) -> Result<()> {
        Store::compute_step(self, name, inputs, output)
    }
}

#[cfg(test)]
#[path = "../tests/store_tests.rs"]
mod tests;
