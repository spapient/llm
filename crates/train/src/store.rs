//! 训练侧的激活缓冲：[`Buffers`] 把 [`Store`] 落到真实内存，算的部分交给 [`Kernel`]。
//!
//! **拥有**：`Buffers`、`Kernel`、`Reference`、`Data`、`View`、`ViewMut`。**不拥有**：显存计划
//! （`autograd::Plan`）、反向调度（`autograd::Schedule`）、形状表（`autograd::Sizes`）、算子
//! 语义（`op`，标量参考就在那里）。**依赖**：`autograd`、`tensor`、`op`、`diagnostic`。
//!
//! `autograd::Replay` 只决定"哪一步之前重算、什么时候交还"，值从哪来它不管：[`Store`] 是它
//! 唯一的口子。本模块把这个口子接到真的内存上——
//!
//! - [`Buffers::compute`] 按 [`Sizes`] 登记的（形状，元素类型）取一块缓冲，把输入视图交给
//!   [`Kernel`]，结果写进去；
//! - [`Buffers::free`] 把块还给池子，`live` 立刻降下来——**显存收益发生在这一刻**，不在作用域结束。
//!
//! # 为什么缓冲只有两种形态
//!
//! workspace 里 `unsafe_code = deny`，"把 `Vec<u8>` 重解释成 `&[f32]`"这条最省事的路是关着的，
//! 所以 [`Data`] 只有 `f32` 与按位打包的 `u8` 两种。这不是妥协：它把"哪种元素能算"提前到
//! 编译期，`F16` / `BF16` 在这里直接报 `unsupported`，而不是在运行时把一段字节当浮点读。
//!
//! # 数字必须对得上
//!
//! [`Buffers::live`] 是当前被占用的字节数，[`Buffers::peak`] 是它的峰值；两者都是**实测**，
//! 与 [`autograd::Plan`] 给出的承诺放在一起就能对账。口径差一处：计划只算激活，本层把常驻的
//! 参数与输入也算进去（它们确实占着内存），所以对账时要减掉常驻量。[`Buffers::free`] 之后
//! `live` 必须回落到只剩常驻，否则就是有检查点没还。
//!
//! 还有一条时序上的契约：**计划保留的检查点必须在重放开始前就在场**。它们不是重放算出来的，
//! 而是前向没丢掉的那一批；`Replay` 见到 `plan.keep(var)` 就当作已有，不会为它补算。所以
//! "前向结束 → 按计划还掉被丢弃的 → 开始重放"这个顺序不能反，[`Buffers::reset_peak`] 就是为
//! 了把两段的峰值分开量。
//!
//! # 例子
//!
//! ```
//! use autograd::{Sizes, Store, Tape};
//! use llm_train::{Buffers, Data, Reference};
//! use tensor::{DType, Shape};
//!
//! let mut tape = Tape::new();
//! let x = tape.leaf();
//! let w = tape.leaf();
//! let scales = tape.leaf();
//! let y = tape.record("gemv_i4", &[x, w, scales]).expect("记录");
//!
//! let mut sizes = Sizes::new(tape.vars());
//! sizes.set(x, Shape::new([4]), DType::F32);
//! sizes.set(w, Shape::new([4]), DType::U4); // 4 个 int4 = 2 字节
//! sizes.set(scales, Shape::new([1]), DType::F32);
//! sizes.set(y, Shape::new([1]), DType::F32);
//!
//! let mut store = Buffers::new(sizes, Reference);
//! store.load(x, Data::F32(vec![1.0; 4])).expect("常驻");
//! store.load(w, Data::U8(vec![0x98, 0x88])).expect("常驻，解包后是 [0, 1, 0, 0]");
//! store.load(scales, Data::F32(vec![0.5])).expect("常驻");
//!
//! store.compute("gemv_i4", &[x, w, scales], y).expect("可算");
//! assert_eq!(store.value(y).and_then(|view| view.f32()), Some(&[0.5f32][..]));
//!
//! store.free(y);
//! assert!(store.value(y).is_none(), "交还之后值就没了");
//! ```

use autograd::{Sizes, Store, Var};
use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

/// 本模块的诊断站点：出错时用它标明"是这一层拒绝了这次请求"。
const SITE: &str = "train::Buffers";

/// 一块真实缓冲：与登记的 [`DType`] 一一对应的两种形态。
///
/// 常驻值（参数、输入）与激活共用它，区别只在谁来放：参数由 [`Buffers::load`] 装进来，
/// 激活由 [`Buffers::compute`] 算出来。
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    /// 32 位浮点。
    F32(Vec<f32>),
    /// 按位打包的整型（`I8` 或 `U4`），字节序与内核一致。
    U8(Vec<u8>),
}

impl Data {
    /// 字节数。
    pub fn bytes(&self) -> u64 {
        match self {
            Self::F32(values) => (values.len() * size_of::<f32>()) as u64,
            Self::U8(values) => values.len() as u64,
        }
    }

    /// 只读视图：[`Kernel`] 看到的输入。
    pub fn view(&self) -> View<'_> {
        match self {
            Self::F32(values) => View::F32(values),
            Self::U8(values) => View::U8(values),
        }
    }

    /// 可变视图：[`Kernel`] 写的输出。
    pub fn view_mut(&mut self) -> ViewMut<'_> {
        match self {
            Self::F32(values) => ViewMut::F32(values),
            Self::U8(values) => ViewMut::U8(values),
        }
    }

    /// 与登记的元数据是否相容：浮点只装 `f32`，整型只装按位打包的 `u8`。
    fn fits(&self, dtype: DType) -> bool {
        match self {
            Self::F32(_) => dtype == DType::F32,
            Self::U8(_) => !dtype.is_float(),
        }
    }

    /// 按形状与元素类型要一块零值缓冲。
    fn empty(dtype: DType, shape: &Shape) -> Result<Self> {
        match dtype {
            DType::F32 => Ok(Self::F32(vec![0.0; shape.numel()])),
            DType::I8 | DType::U4 => Ok(Self::U8(vec![0; shape.element_bytes(dtype)])),
            DType::F16 | DType::BF16 => Err(Error::new(
                Code::Unsupported,
                format!("{dtype:?} 没有标量存储：本层的缓冲只有 f32 与按位打包的 u8"),
            )
            .with(SITE)),
        }
    }
}

/// 只读视图：一次 [`Kernel::compute`] 的输入。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum View<'a> {
    /// 32 位浮点。
    F32(&'a [f32]),
    /// 按位打包的整型。
    U8(&'a [u8]),
}

impl<'a> View<'a> {
    /// 当成 `f32` 读；不是浮点缓冲时为 `None`。
    pub fn f32(&self) -> Option<&'a [f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::U8(_) => None,
        }
    }

    /// 当成按位打包的整型读；不是 `u8` 缓冲时为 `None`。
    pub fn u8(&self) -> Option<&'a [u8]> {
        match self {
            Self::U8(values) => Some(values),
            Self::F32(_) => None,
        }
    }
}

/// 可写视图：一次 [`Kernel::compute`] 的输出。
#[derive(Debug, PartialEq)]
pub enum ViewMut<'a> {
    /// 32 位浮点。
    F32(&'a mut [f32]),
    /// 按位打包的整型。
    U8(&'a mut [u8]),
}

impl<'a> ViewMut<'a> {
    /// 当成 `f32` 写；不是浮点缓冲时为 `None`。
    pub fn f32(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::U8(_) => None,
        }
    }
}

/// 一步前向：`op(inputs) → output`。
///
/// 契约只有两条：**输入此刻都在**（重放已经先跑完依赖），**输出要被完整写入**（缓冲是复用的，
/// 没写到的位置留着上一轮的值）。参数顺序由"记录前向的那一侧"与实现共同约定，`Store` 只按
/// [`autograd::Node::inputs`] 的顺序原样转发，不猜语义。
///
/// 为什么是 trait 而不是函数表：真实内核按设备/后端不同（标量参考、`cutile` 产物、将来的
/// 设备流），把"怎么算"参数化之后，缓冲与寿命这一半只写一次。
pub trait Kernel {
    /// 用 `inputs` 算出 `output`。
    ///
    /// # Errors
    ///
    /// 算子未知、输入的类型/长度不合、设备失败——实现决定，[`Buffers::compute`] 原样向上抛；
    /// 出错时这块输出缓冲会被还回池子，不会留下半个值。
    fn compute(&mut self, op: &str, inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()>;
}

/// 标量参考内核：不碰设备，逐元素算，用于对拍与没有加速器时的正确性基线。
///
/// 算术不在本模块：`gemv_i4` 交给 [`op::gemv_i4::forward`]，与 `op` 的参考实现是同一段循环。
/// 于是"重放出来的值"与"参考值"比较的不是两份手写代码，而是同一份语义的两次执行。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reference;

impl Kernel for Reference {
    fn compute(&mut self, op: &str, inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
        match op {
            "gemv_i4" => gemv_i4(inputs, output),
            other => Err(Error::new(
                Code::Unsupported,
                format!("`{other}` 没有标量内核；`train::Reference` 目前只实现 gemv_i4"),
            )
            .with(SITE)),
        }
    }
}

/// `[x: f32, w: u8, scales: f32] → y: f32`，与 `op::gemv_i4::forward` 同一段算术。
fn gemv_i4(inputs: &[View<'_>], output: &mut ViewMut<'_>) -> Result<()> {
    let [View::F32(x), View::U8(w), View::F32(scales)] = inputs else {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "gemv_i4 的输入必须是 [x: f32, w: u8, scales: f32]，实际是 {:?}",
                inputs
            ),
        )
        .with(SITE));
    };
    let ViewMut::F32(y) = output else {
        return Err(Error::new(Code::Invalid, "gemv_i4 的输出必须是 f32").with(SITE));
    };
    op::gemv_i4::forward(x, w, scales, y)
}

/// 同尺寸缓冲的空闲表：按**正好相等的长度**分档。
///
/// 只复用等长的块，所以"池里占着的容量"永远不会超过某一档曾经的峰值，记账也不必打折；查找是
/// 一次线性扫描，档数由模型里出现过的形状决定（通常是个位数），比哈希更可预测。
#[derive(Debug, Default)]
struct Buckets<T> {
    by_len: Vec<(usize, Vec<Vec<T>>)>,
}

impl<T> Buckets<T> {
    /// 取一块长度正好是 `len` 的空闲块。
    fn take(&mut self, len: usize) -> Option<Vec<T>> {
        self.by_len
            .iter_mut()
            .find(|(size, _)| *size == len)
            .and_then(|(_, free)| free.pop())
    }

    /// 还回一块空闲块，按它的长度归到对应档。
    fn give(&mut self, buffer: Vec<T>) {
        let len = buffer.len();
        match self.by_len.iter_mut().find(|(size, _)| *size == len) {
            Some((_, free)) => free.push(buffer),
            None => self.by_len.push((len, vec![buffer])),
        }
    }

    /// 档里现成可复用的块数。
    fn pooled(&self) -> usize {
        self.by_len.iter().map(|(_, free)| free.len()).sum()
    }
}

/// 两种元素形态各自的空闲表。
#[derive(Debug, Default)]
struct Pool {
    f32s: Buckets<f32>,
    u8s: Buckets<u8>,
}

impl Pool {
    /// 一块 `dtype` 的零值缓冲：先试空闲表，没有再向分配器要。
    fn block(&mut self, dtype: DType, shape: &Shape) -> Result<Data> {
        match dtype {
            DType::F32 => {
                let numel = shape.numel();
                Ok(Data::F32(match self.f32s.take(numel) {
                    Some(mut values) => {
                        // 复用块里可能留着上一轮的值；内核承诺写满输出，这里只把长度对齐。
                        values.truncate(numel);
                        values
                    }
                    None => vec![0.0; numel],
                }))
            }
            DType::I8 | DType::U4 => {
                let len = shape.element_bytes(dtype);
                Ok(Data::U8(match self.u8s.take(len) {
                    Some(mut values) => {
                        values.truncate(len);
                        values
                    }
                    None => vec![0; len],
                }))
            }
            DType::F16 | DType::BF16 => Data::empty(dtype, shape),
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
}

/// 以 [`Var`] 为下标的真实缓冲：`autograd::Store` 在训练侧的实现。
///
/// 一个训练步里只有一个实例，逐图复用：`load` 装常驻值、`compute` 算激活、`free` 还激活，
/// 步与步之间用 [`Buffers::clear`] 一次性归还。缓冲块本身来自内部的空闲表，所以稳定态里
/// "算一块、还一块"不再向系统分配器要内存。
#[derive(Debug)]
pub struct Buffers<K: Kernel> {
    sizes: Sizes,
    kernel: K,
    slots: Vec<Option<Data>>,
    pool: Pool,
    live: u64,
    peak: u64,
}

impl<K: Kernel> Buffers<K> {
    /// 由形状表与内核构造：槽位数与 `sizes.len()` 同口径（即 `Tape::vars()`）。
    pub fn new(sizes: Sizes, kernel: K) -> Self {
        Self {
            slots: vec![None; sizes.len()],
            sizes,
            kernel,
            pool: Pool::default(),
            live: 0,
            peak: 0,
        }
    }

    /// 装入一个常驻值（参数、输入、或外部算好的结果）。
    ///
    /// 登记的元数据（形状、元素类型）与实际数据必须一致：装错类型不会在训练中途才暴露成
    /// 一串无意义的数，而是在这里就失败。同一个变量装第二次会先把旧的还回空闲表。
    ///
    /// # Errors
    ///
    /// 变量超出形状表的槽位、数据与登记的（形状，元素类型）不相容时返回说明。
    pub fn load(&mut self, var: Var, data: Data) -> Result<()> {
        let slot = self.slot(var)?;
        if let Some((shape, dtype)) = self.sizes.get(var) {
            let expected = shape.element_bytes(dtype) as u64;
            if !data.fits(dtype) || data.bytes() != expected {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "变量 {} 登记为 {dtype:?} 的 {:?}（{expected} 字节），装入的却是 {} 字节的{:?}",
                        var.id(),
                        shape.dims(),
                        data.bytes(),
                        match data {
                            Data::F32(_) => " f32",
                            Data::U8(_) => " u8",
                        }
                    ),
                )
                .with(SITE));
            }
        }
        self.release(slot);
        self.live += data.bytes();
        self.peak = self.peak.max(self.live);
        self.slots[slot] = Some(data);
        Ok(())
    }

    /// 读一个变量当前的值；不在场（从未算过、或已被交还）时为 `None`。
    pub fn value(&self, var: Var) -> Option<View<'_>> {
        self.slots
            .get(var.id() as usize)
            .and_then(Option::as_ref)
            .map(Data::view)
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
    ///
    /// 只影响 [`Buffers::peak`]，不动 `live`，也不动任何缓冲。
    pub fn reset_peak(&mut self) {
        self.peak = self.live;
    }

    /// 空闲表里现成可复用的块数（诊断用：稳定态应当不增长）。
    pub fn pooled(&self) -> usize {
        self.pool.pooled()
    }

    /// 一次性归零：换训练步、或换图时调用，空闲表与峰值统计一起重置。
    pub fn clear(&mut self) {
        self.slots.fill(None);
        self.pool = Pool::default();
        self.live = 0;
        self.peak = 0;
    }

    /// 形状表。
    pub fn sizes(&self) -> &Sizes {
        &self.sizes
    }

    /// 变量对应的槽位；超出登记范围说明 `Sizes` 与 `Tape` 不同源。
    fn slot(&self, var: Var) -> Result<usize> {
        let index = var.id() as usize;
        if index >= self.slots.len() {
            return Err(Error::new(
                Code::Internal,
                format!(
                    "变量 {} 超出形状表的 {} 个槽位：Sizes 必须按 Tape::vars() 构造",
                    var.id(),
                    self.slots.len()
                ),
            )
            .with(SITE));
        }
        Ok(index)
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
        let slot = self.slot(output)?;
        let Some((shape, dtype)) = self
            .sizes
            .get(output)
            .map(|(shape, dtype)| (shape.clone(), dtype))
        else {
            return Err(Error::new(
                Code::Internal,
                format!(
                    "变量 {} 没有登记形状，无法为它分配缓冲：前向记录时漏了 Sizes::set",
                    output.id()
                ),
            )
            .with(SITE));
        };

        // 输入必须在场。缺一个就是"重放读到了已释放的值"——这类错误一旦被算成一块假值，
        // 后面所有梯度都错得没有痕迹，所以在这里一次也不放过。
        let mut views = Vec::with_capacity(inputs.len());
        for input in inputs {
            let Some(block) = self.slots.get(input.id() as usize).and_then(Option::as_ref) else {
                return Err(Error::new(
                    Code::Missing,
                    format!(
                        "重算 {} 时输入 {} 不在场（已被交还或从未算过）",
                        output.id(),
                        input.id()
                    ),
                )
                .with(SITE));
            };
            views.push(block.view());
        }

        let mut block = self.pool.block(dtype, &shape)?;
        if let Err(error) = self.kernel.compute(op, &views, &mut block.view_mut()) {
            self.pool.recycle(block);
            return Err(error);
        }
        drop(views);

        self.release(slot);
        self.live += block.bytes();
        self.peak = self.peak.max(self.live);
        self.slots[slot] = Some(block);
        Ok(())
    }

    fn free(&mut self, var: Var) {
        let slot = var.id() as usize;
        if slot < self.slots.len() {
            self.release(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{Budget, Footprint, Plan, Replay, Tape};

    /// 每层都是 `y = gemv_i4(x, w, scales)`：`m = n = 8`，`scales` 只有一组（组大小 8）。
    const WIDTH: usize = 8;

    /// 三个常驻值的字节数：`x`（8×f32）、`w`（8×8 个 int4 = 32 B）、`scales`（8×f32）。
    const RESIDENT: u64 = 96;

    fn input() -> Data {
        Data::F32(
            (0..WIDTH)
                .map(|index| ((index * 7) % 13) as f32 * 0.25 - 1.0)
                .collect(),
        )
    }

    fn weights() -> Data {
        Data::U8(
            (0..WIDTH * (WIDTH / 2))
                .map(|index| ((index * 37) % 251) as u8)
                .collect(),
        )
    }

    fn scales() -> Data {
        Data::F32(
            (0..WIDTH)
                .map(|index| 0.031_25 * (1 + (index % 4)) as f32)
                .collect(),
        )
    }

    /// 一条 `layers` 层的链：形状表、叶子与 loss 一起给出，测试不必各自重算。
    struct Fixture {
        tape: Tape,
        sizes: Sizes,
        x: Var,
        w: Var,
        scales: Var,
        loss: Var,
    }

    impl Fixture {
        fn new(layers: usize) -> Self {
            let mut tape = Tape::new();
            let x = tape.parameter();
            let w = tape.leaf();
            let scales = tape.leaf();
            let mut loss = x;
            for _ in 0..layers {
                loss = tape.record("gemv_i4", &[loss, w, scales]).expect("记录");
            }
            let mut sizes = Sizes::new(tape.vars());
            sizes.set(x, Shape::new([WIDTH]), DType::F32);
            sizes.set(w, Shape::new([WIDTH, WIDTH]), DType::U4);
            sizes.set(scales, Shape::new([WIDTH]), DType::F32);
            for node in tape.nodes() {
                sizes.set(node.output(), Shape::new([WIDTH]), DType::F32);
            }
            Self {
                tape,
                sizes,
                x,
                w,
                scales,
                loss,
            }
        }

        fn store(&self) -> Buffers<Reference> {
            let mut store = Buffers::new(self.sizes.clone(), Reference);
            store.load(self.x, input()).expect("输入");
            store.load(self.w, weights()).expect("权重");
            store.load(self.scales, scales()).expect("scale");
            store
        }

        /// 按记录顺序算一遍全图，返回每个变量当时的值（叶子也在内）。
        fn forward(&self, store: &mut Buffers<Reference>) -> Vec<Option<Data>> {
            for node in self.tape.nodes() {
                store
                    .compute(node.op(), node.inputs(), node.output())
                    .expect("前向可算");
            }
            (0..self.tape.vars())
                .map(|id| {
                    store
                        .value(self.tape.var(id).expect("编号有效"))
                        .map(|view| match view {
                            View::F32(values) => Data::F32(values.to_vec()),
                            View::U8(values) => Data::U8(values.to_vec()),
                        })
                })
                .collect()
        }

        /// 把前向留下的激活收敛到计划保留的那一份：重放开始前的真实状态。
        ///
        /// 检查点不是重放算出来的，它们就是前向没丢掉的那些值——所以只能在旁边留、不能"重算
        /// 出来"。这里把计划丢弃的那些还掉，留下的正好是 `Plan::retained`。
        fn rewind(&self, store: &mut Buffers<Reference>, plan: &Plan) {
            for node in self.tape.nodes() {
                if !plan.keep(node.output()) {
                    store.free(node.output());
                }
            }
            assert_eq!(
                store.live(),
                RESIDENT + plan.retained(),
                "重放开始前只剩常驻值与计划的检查点"
            );
        }
    }

    #[test]
    fn compute_matches_the_operator_reference_bit_for_bit() {
        let fixture = Fixture::new(1);
        let mut store = fixture.store();
        let y = fixture.loss;
        store
            .compute("gemv_i4", &[fixture.x, fixture.w, fixture.scales], y)
            .expect("可算");

        let x = input().view().f32().expect("f32").to_vec();
        let w = weights().view().u8().expect("u8").to_vec();
        let scales = scales().view().f32().expect("f32").to_vec();
        let mut expected = vec![0.0f32; WIDTH];
        op::gemv_i4::forward(&x, &w, &scales, &mut expected).expect("参考可算");

        assert_eq!(
            store
                .value(y)
                .and_then(|view| view.f32())
                .map(<[f32]>::to_vec),
            Some(expected),
            "本层的缓冲只负责放，算术必须与 np 的参考同一份"
        );
    }

    #[test]
    fn freeing_returns_the_buffer_and_lowers_live_immediately() {
        let fixture = Fixture::new(1);
        let mut store = fixture.store();
        let y = fixture.loss;
        let activation = (WIDTH * 4) as u64;

        assert_eq!(store.live(), RESIDENT, "只有常驻值");
        store
            .compute("gemv_i4", &[fixture.x, fixture.w, fixture.scales], y)
            .expect("可算");
        assert_eq!(store.live(), RESIDENT + activation);
        assert_eq!(store.peak(), RESIDENT + activation);

        store.free(y);
        assert_eq!(store.live(), RESIDENT, "交还当场生效，不等作用域结束");
        assert_eq!(store.pooled(), 1, "块进了空闲表");

        store
            .compute("gemv_i4", &[fixture.x, fixture.w, fixture.scales], y)
            .expect("可算");
        assert_eq!(store.pooled(), 0, "第二次算复用同一块");
        assert_eq!(store.live(), RESIDENT + activation);
    }

    #[test]
    fn replaying_into_a_reused_buffer_reproduces_the_same_values() {
        let fixture = Fixture::new(1);
        let mut store = fixture.store();
        let y = fixture.loss;
        let run = |store: &mut Buffers<Reference>| {
            store
                .compute("gemv_i4", &[fixture.x, fixture.w, fixture.scales], y)
                .expect("可算");
            store
                .value(y)
                .and_then(|view| view.f32())
                .map(<[f32]>::to_vec)
                .expect("有值")
        };

        let first = run(&mut store);
        store.free(y);
        let second = run(&mut store);
        assert_eq!(first, second, "复用块不得留下上一轮的任何一项");
    }

    #[test]
    fn a_released_input_is_never_read_as_a_value() {
        let fixture = Fixture::new(2);
        let mut store = fixture.store();
        let (first, second) = (fixture.tape.nodes()[0].output(), fixture.loss);
        store
            .compute("gemv_i4", &[fixture.x, fixture.w, fixture.scales], first)
            .expect("可算");
        store.free(first);

        let error = store
            .compute("gemv_i4", &[first, fixture.w, fixture.scales], second)
            .expect_err("输入已释放");
        assert_eq!(error.code(), Code::Missing);
        assert_eq!(store.live(), RESIDENT, "失败的这一次没有留下任何占用");
    }

    #[test]
    fn a_variable_without_a_registered_shape_is_rejected() {
        let fixture = Fixture::new(1);
        let mut store = Buffers::new(Sizes::new(fixture.tape.vars()), Reference);
        let error = store
            .compute(
                "gemv_i4",
                &[fixture.x, fixture.w, fixture.scales],
                fixture.loss,
            )
            .expect_err("没有形状");
        assert_eq!(error.code(), Code::Internal);
    }

    #[test]
    fn an_element_type_without_scalar_storage_is_unsupported() {
        let error = Data::empty(DType::F16, &Shape::new([4])).expect_err("没有 f16 存储");
        assert_eq!(error.code(), Code::Unsupported);
    }

    #[test]
    fn a_loaded_block_must_match_what_the_shape_table_says() {
        let fixture = Fixture::new(1);
        let mut store = fixture.store();
        let too_short = Data::U8(vec![0; WIDTH]); // 登记的是 8×8 个 int4 = 32 B
        assert_eq!(
            store.load(fixture.w, too_short).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            store
                .load(fixture.w, Data::F32(vec![0.0; 64]))
                .unwrap_err()
                .code(),
            Code::Invalid,
            "浮点缓冲不能装进整型槽位"
        );
        assert_eq!(store.live(), RESIDENT, "失败没有破坏原来的常驻值");
    }

    #[test]
    fn an_unknown_operator_is_unsupported_rather_than_silently_zero() {
        let fixture = Fixture::new(1);
        let mut store = fixture.store();
        let error = store
            .compute("relu", &[fixture.x], fixture.loss)
            .expect_err("没有 relu 的标量内核");
        assert_eq!(error.code(), Code::Unsupported);
        assert_eq!(store.live(), RESIDENT);
    }

    #[test]
    fn a_plan_drives_the_store_and_every_replay_matches_the_forward() {
        let fixture = Fixture::new(12);
        let mut store = fixture.store();
        let truth = fixture.forward(&mut store);
        let total: u64 = fixture
            .tape
            .nodes()
            .iter()
            .map(|node| fixture.sizes.bytes(node.output()))
            .sum();
        let schedule = fixture.tape.backward(fixture.loss).expect("在图上");
        let plan = Plan::of(
            &fixture.tape,
            &schedule,
            &fixture.sizes,
            Budget::new(total * 2 / 3),
        )
        .expect("预算装得下分段方案");
        fixture.rewind(&mut store, &plan);
        // 前向自己也占过内存；峰值要按"重放这一段"量，才对得上计划承诺的数。
        store.reset_peak();
        let replay = Replay::of(&fixture.tape, &schedule, &plan).expect("可执行");

        let mut checked = 0usize;
        let computed = replay
            .run(&fixture.tape, &schedule, &mut store, |store, step| {
                for input in step.inputs() {
                    let actual = store.value(*input).expect("重放后输入必须在场");
                    let expected = truth[input.id() as usize].clone().expect("前向算过");
                    let same = match (actual, expected) {
                        (View::F32(actual), Data::F32(expected)) => actual == expected.as_slice(),
                        (View::U8(actual), Data::U8(expected)) => actual == expected.as_slice(),
                        _ => false,
                    };
                    assert!(
                        same,
                        "第 {} 步的输入 {} 与前向不一致",
                        step.output().id(),
                        input.id()
                    );
                    checked += 1;
                }
                Ok(())
            })
            .expect("重放可执行");

        assert!(computed > 0, "紧预算必须真的重算");
        assert_eq!(computed, plan.replayed(), "实测与计划承诺必须一致");
        assert!(checked > 0);
        assert_eq!(
            store.live(),
            RESIDENT,
            "执行完只该剩常驻值：检查点与重算出来的临时值都到了释放点"
        );
        assert!(
            store.peak() - RESIDENT <= plan.peak(),
            "实测峰值 {} 超过计划 {}",
            store.peak() - RESIDENT,
            plan.peak()
        );
    }

    #[test]
    fn many_rounds_do_not_grow_the_pool() {
        let fixture = Fixture::new(1);
        let mut store = fixture.store();
        let y = fixture.loss;
        for round in 0..50 {
            store
                .compute("gemv_i4", &[fixture.x, fixture.w, fixture.scales], y)
                .unwrap_or_else(|error| panic!("第 {round} 轮：{error}"));
            store.free(y);
            assert_eq!(store.live(), RESIDENT, "第 {round} 轮");
            assert_eq!(store.pooled(), 1, "第 {round} 轮：空闲表不增长");
        }
    }

    #[test]
    fn clear_returns_everything_and_resets_the_peak() {
        let fixture = Fixture::new(1);
        let mut store = fixture.store();
        store
            .compute(
                "gemv_i4",
                &[fixture.x, fixture.w, fixture.scales],
                fixture.loss,
            )
            .expect("可算");
        assert!(store.peak() > RESIDENT);

        store.clear();
        assert_eq!((store.live(), store.peak(), store.pooled()), (0, 0, 0));
        assert!(store.value(fixture.x).is_none(), "常驻值也要还回去");

        // 归零之后同一个实例可以接着用：形状表还在，值要重新装。
        store.load(fixture.x, input()).expect("重新装入");
        assert_eq!(store.live(), WIDTH as u64 * 4, "只算刚装进来的那一块");
    }
}
