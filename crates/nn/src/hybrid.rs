//! 装配：把 MLA 注意力、MoE 前馈与 mHC 多流残差串成**一层**可跑的前向。
//!
//! 前面各模块都给的是「一件机制的数学 + 它的可测不变量」——潜在 KV 只存 `(c^KV, k^R)`、
//! 稀疏专家只激活一部分 θ、多流残差只做双随机混合。各自单测都在，但**没有任何一次前向同时用到
//! 它们**，于是「机制齐了」与「能跑一层」之间还差一格。本模块就是那一格：一层 = 混流 → 注意力
//! 残差 → 前馈残差，三件都调各自模块的唯一入口，本层不重写任何一条算式。
//!
//! # 一栈：把 3:1 配比变成构造条件
//!
//! [`Tower`] 在本层之上只做装配，不加算术：K3 的「每三个通道记忆层配一个混合层」由 [`Tower::new`]
//! 校验、由 [`Tower::order`] 逐格读出，于是配比不成立就构不出栈，而不是靠注释提醒。两处结构决定：
//!
//! - **记忆状态与 KV 缓存不共用容身之处**：[`State`] 随序列长累积、[`Latent`] 按页账走，因此
//!   [`Tower::step`] 的入参里没有页号，缓存也仍由调用方在推进前 `push`——与一层同一条次序。
//! - **一次推进要么整栈走一步，要么一格都不动**：流与记忆都在副本上算、末尾一次性提交，
//!   于是第 5 个记忆层的几何错不会留下「前 4 层已写、后 2 层未写」的半截状态。
//!
//! 记忆层按 [`crate::kda`] 的口径**写进去当步就读得出来**，读数只加回主路（与下面第 5 条同一条
//! 理由）；各层持自己写入的那一份记忆，互不可见。
//!
//! # 次序与每处取舍的出处
//!
//! 1. **层首混流**：`B_l` 作用在进入本层的 `M` 条流上。这与「层末混流」（`x_{l+1} = B_l x_l`）
//!    是同一件事——边界只是画在哪一步。放在层首有两个具体好处：[`Mhc::mix_streams`] 自带
//!    「条数 = `M`、各条等宽」的入口校验，于是几何错在**任何写入之前**就报掉；而且它返回新向量、
//!    不改输入，天然就是「算完再提交」的那一半。
//! 2. **注意力只读缓存**：本 token 的 `(c^KV, k^R)` 由调用方在 [`Hybrid::step`] 之前 `push`。
//!    缓存的行数归页账（`engine::kv`）所有，本层既不持有也不推进，所以 `step` 失败不会让序列
//!    多一行或少一行——「失败不写半截」在这里是结构性的，不是靠回滚。
//! 3. **先注意力、后前馈，两者都是加**：残差用 [`crate::csa::accumulate`] 加权累加、权重恒
//!    `S::ONE`，不另写一遍逐元素加法。前馈吃的是**加过注意力的主路**，与
//!    「`AttentionBlock` → FFN」的串行是同一个次序。
//! 4. **只有主路进注意力与前馈**：KV 只有一份。多流是**残差的副本**，不是 `M` 份 KV——否则 mHC
//!    会把 MLA 的行宽收益按 `M` 乘回去，而 V3 的 `576` 字节/token 是「每层」的账，不是「每流」的。
//! 5. **算出的一份增量只加回主路**：其余流带走混流后的值。同一份增量加到每条流上会让它们从此
//!    完全相同，多流当场退化成单流；只加一条、层间靠 `B_l` 再把它们摊开，流之间才始终留着可被
//!    读取的差异（[`crate::attnres`] 读的正是这份差异）。
//!
//! # 本模块不拥有什么
//!
//! 投影矩阵（`q`/`k`/`v` 的 `W` 与查询的两段都从外面递进来）、归一化与残差缩放（`nn` 里
//! `RmsNorm`/`Scale` 是**声明**而非算术，执行侧才有实现）、通道记忆与跨层读取（[`crate::kda`]、
//! [`crate::attnres`] 各自是自己的入口，本层不转包同名方法——多一层转发只会让「这一层拥有什么」
//! 变模糊），以及训练状态。
//!
//! 也不声称这是 K3 的完整逐层配方：3:1 的 KDA:MLA 配比、归一化的位置与投影形状都属装配侧。
//! 本层承诺的只有一件事——**这五件里被选中的三件，按上面这个次序，能跑出一层前向**，且每一处
//! 连接点的理由都写在上面的清单里。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::accumulate;
use crate::mhc::Mhc;
use crate::mla::{Latent, Mla, Path};
use crate::moe::Moe;

/// 报错出处：几何与位置判定都收在装配的入口，不留到矩阵乘的中途。
const SITE: &str = "nn::hybrid";

/// 本 token 的查询两段。分开传是因为它们**不是**同一张投影的切片：`q^{no-nope}` 进吸收式路径，
/// `q^R` 直接与解耦 RoPE 键点积（见 [`Mla::attend`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Query<'a, S> {
    /// `q^{no-nope}`，宽 `n_h · d_h`。
    pub nope: &'a [S],
    /// `q^R`，宽 `n_h · d_h^R`。
    pub rope: &'a [S],
}

/// 一层：MLA 注意力 + MoE 前馈 + mHC 多流残差。
///
/// 三件都是构造期定格的（不持可变参数），一层里唯一的运行期状态是调用方手里的 `streams` 与
/// 页账上的 `cache`——于是「这一步算了什么」完全由入参决定，本层不做隐藏的累积。
#[derive(Debug, Clone, PartialEq)]
pub struct Hybrid<S: Scalar = f32> {
    attention: Mla<S>,
    ffn: Moe<S>,
    residual: Mhc<S>,
}

impl<S: Scalar> Hybrid<S> {
    /// 装配一层。
    ///
    /// # Errors
    ///
    /// 注意力输出宽 `n_h · d_h` 必须等于前馈宽——不等的话每一步都会把「另一层的数」喂进专家，
    /// 内积照算、结果却是另一个数，所以在这里拒绝；`n_h · d_h` 溢出同样返回 `Code::Invalid`。
    pub fn new(attention: Mla<S>, ffn: Moe<S>, residual: Mhc<S>) -> Result<Self> {
        let width = attention
            .heads()
            .checked_mul(attention.head_dim())
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!(
                        "注意力输出宽 n_h × d_h 溢出（{} × {}）",
                        attention.heads(),
                        attention.head_dim()
                    ),
                )
                .with(SITE)
            })?;
        if width != ffn.width() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "注意力输出宽 n_h × d_h = {width}，前馈宽 {}：两者必须相等，否则残差相加会把错位的数据拼在一起",
                    ffn.width()
                ),
            )
            .with(SITE));
        }
        Ok(Self {
            attention,
            ffn,
            residual,
        })
    }

    /// 本层宽度（= `n_h · d_h` = 前馈宽）：三条流与两段查询都由它定标。
    pub fn width(&self) -> usize {
        self.ffn.width()
    }

    /// 残差流条数 `M`。
    pub fn streams(&self) -> usize {
        self.residual.streams()
    }

    /// 注意力侧（只读视图；本层不替调用方缓存，也不转包它的方法）。
    pub fn attention(&self) -> &Mla<S> {
        &self.attention
    }

    /// 前馈侧（只读视图）。
    pub fn ffn(&self) -> &Moe<S> {
        &self.ffn
    }

    /// 多流残差侧（只读视图）。
    pub fn residual(&self) -> &Mhc<S> {
        &self.residual
    }

    /// 把一条隐藏向量铺成 `M` 条流：多流的初值就是同一个值（混合之后才分开）。
    ///
    /// # Errors
    ///
    /// `hidden` 的宽度不是本层宽度时返回 `Code::Invalid`——长度不对的初值会被后面的混流当成
    /// 「另一层的数」混进来，而混流本身看不出这一点。
    pub fn seed(&self, hidden: &[S]) -> Result<Vec<Vec<S>>> {
        if hidden.len() != self.width() {
            return Err(Error::new(
                Code::Invalid,
                format!("初值宽度是 {}，本层宽度是 {}", hidden.len(), self.width()),
            )
            .with(SITE));
        }
        Ok(vec![hidden.to_vec(); self.streams()])
    }

    /// 第 `t` 个位置的一次前向：混流 → 注意力残差 → 前馈残差，末尾一次性写回 `streams`。
    ///
    /// 缓存的最后一行就是本 token 的键：位置取 `cache.seq() - 1`，于是「先 `push` 再 `step`」
    /// 由调用方的次序保证，本层无法读到还没入缓存的位置。
    ///
    /// 算出的增量只加回**主路**（第 0 条），其余流带走混流后的值：同一份增量加给每条流会让它们
    /// 从此完全相同，多流就退化成单流。
    ///
    /// # Errors
    ///
    /// 流条数/宽度不符、查询几何不符、缓存里一个位置都没有——都返回 `Code::Invalid`，且
    /// `streams` 与 `cache` **都没有被改动**（几何与位置判定全部先于第一次写入）。
    pub fn step(
        &self,
        streams: &mut [Vec<S>],
        cache: &Latent<S>,
        query: &Query<'_, S>,
        path: Path,
    ) -> Result<()> {
        let mut mixed = self.residual.mix_streams(streams)?;
        self.check_width(&mixed)?;
        let t = cursor(cache)?;
        self.attach(&mut mixed[0], cache, query, t, path)?;
        commit(streams, mixed);
        Ok(())
    }

    /// 主路的一层增量：先注意力、后前馈，两者都是加——权重恒 `S::ONE`，与「`AttentionBlock` →
    /// FFN」是同一个次序；前馈吃的是**加过注意力**的那一份。
    fn attach(
        &self,
        main: &mut [S],
        cache: &Latent<S>,
        query: &Query<'_, S>,
        t: usize,
        path: Path,
    ) -> Result<()> {
        let attended = self
            .attention
            .attend(cache, query.nope, query.rope, t, path)?;
        accumulate(main, &attended, S::ONE);
        let fed = self.ffn.forward(main)?;
        accumulate(main, &fed, S::ONE);
        Ok(())
    }

    /// 混流之后的宽度必须正好是本层宽度：宽度多一格时 [`accumulate`] 会静默只加重叠部分。
    fn check_width(&self, mixed: &[Vec<S>]) -> Result<()> {
        let width = self.width();
        match mixed.iter().position(|row| row.len() != width) {
            Some(bad) => Err(Error::new(
                Code::Invalid,
                format!(
                    "第 {bad} 条流宽度是 {}，本层宽度是 {width}（= n_h · d_h = 前馈宽）",
                    mixed[bad].len()
                ),
            )
            .with(SITE)),
            None => Ok(()),
        }
    }
}

/// 本 token 的位置：缓存的最后一行。
fn cursor<S: Scalar>(cache: &Latent<S>) -> Result<usize> {
    cache.seq().checked_sub(1).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            "缓存里没有任何位置：本 token 的 (c^KV, k^R) 必须先 push 再 step，否则注意力会对着空历史算",
        )
        .with(SITE)
    })
}

/// 算完之后一次性写回调用方的流：中途失败过的调用不会留下「一半新一半旧」。
fn commit<S: Scalar>(streams: &mut [Vec<S>], mixed: Vec<Vec<S>>) {
    for (slot, row) in streams.iter_mut().zip(mixed) {
        *slot = row;
    }
}

/// 整栈推进中途的工作区：流与记忆都是调用方状态的**副本**，只有全部走通才提交。
struct Walk<'a, S: Scalar> {
    streams: &'a mut Vec<Vec<S>>,
    state: &'a mut State<S>,
    cache: &'a Latent<S>,
    query: &'a Query<'a, S>,
    memories: &'a [MemoryStep<'a, S>],
    path: Path,
    /// 已用掉的记忆输入份数：每遇到一个记忆层加一。
    used: usize,
}

impl<'a, S: Scalar> Walk<'a, S> {
    /// 按层序走一遍。
    fn run(&mut self, layers: &[Gear<S>]) -> Result<()> {
        layers.iter().try_for_each(|gear| self.advance(gear))
    }

    /// 一格：记忆层写后读、混合层走完整一层。
    fn advance(&mut self, gear: &Gear<S>) -> Result<()> {
        match gear {
            Gear::Memory(kda) => self.memory(kda),
            Gear::Attend(layer) => layer.step(self.streams, self.cache, self.query, self.path),
        }
    }

    /// 一个记忆层：先写进**该层自己**的记忆，再读，读数只加回主路。
    fn memory(&mut self, kda: &crate::kda::Kda) -> Result<()> {
        let slot = self.used;
        let input = self.next()?;
        let memory = &mut self.state.memories[slot];
        kda.step(memory, input.gate, input.write, input.key, input.value)?;
        let read = kda.read(memory, input.query)?;
        accumulate(&mut self.streams[0], &read, S::ONE);
        Ok(())
    }

    /// 取下一份记忆输入：少一份就是错位，必须在写任何东西之前报掉。
    fn next(&mut self) -> Result<MemoryStep<'a, S>> {
        let used = self.used;
        let input = self.memories.get(used).copied().ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("第 {used} 个记忆层没有输入：每一格要一份门/键/值/查询，少一份就是错位"),
            )
            .with(SITE)
        })?;
        self.used += 1;
        Ok(input)
    }
}

/// 记忆输入必须与记忆层一一对应：多的那份没有任何一格会读它。
fn check_count(used: usize, given: usize) -> Result<()> {
    if used == given {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("记忆输入比记忆层多：本栈 {used} 个记忆层，收到 {given} 份输入"),
    )
    .with(SITE))
}

/// 混合层的几何：至少一层，且各层宽度与流条数同源。
fn geometry<S: Scalar>(attention: &[Hybrid<S>]) -> Result<(usize, usize)> {
    let first = attention.first().ok_or_else(|| {
        Error::new(
            Code::Invalid,
            "层栈至少需要一个混合层：没有注意力就没有可推进的主路",
        )
        .with(SITE)
    })?;
    let (width, streams) = (first.width(), first.streams());
    match attention
        .iter()
        .find(|layer| layer.width() != width || layer.streams() != streams)
    {
        Some(bad) => Err(Error::new(
            Code::Invalid,
            format!(
                "混合层几何不一致：本栈是宽 {width} × {streams} 流，遇到宽 {} × {} 流",
                bad.width(),
                bad.streams()
            ),
        )
        .with(SITE)),
        None => Ok((width, streams)),
    }
}

/// 配比：每 `MEMORY_PER_ATTENTION` 个记忆层配 1 个混合层。
fn check_ratio(attention: usize, memory: usize) -> Result<()> {
    let want = attention * MEMORY_PER_ATTENTION;
    if memory == want {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "配比必须是每 {MEMORY_PER_ATTENTION} 个记忆层配 1 个混合层：{attention} 个混合层要 {want} 个记忆层，得到 {memory}"
        ),
    )
    .with(SITE))
}

/// 记忆层的读出宽必须等于主路宽：读数要加回主路，宽度不齐就是两张不同的表。
fn check_memory_width(memory: &[crate::kda::Kda], width: usize) -> Result<()> {
    match memory
        .iter()
        .enumerate()
        .find(|(_, kda)| kda.value_dim() != width)
    {
        Some((index, bad)) => Err(Error::new(
            Code::Invalid,
            format!(
                "第 {index} 个记忆层读出宽 {} ≠ 混合层宽 {width}：读数要加回主路，宽度不齐会拼起两张不同的表",
                bad.value_dim()
            ),
        )
        .with(SITE)),
        None => Ok(()),
    }
}

/// 按「三记忆 + 一混合」把两份清单交错成层序。
fn interleave<S: Scalar>(attention: Vec<Hybrid<S>>, memory: Vec<crate::kda::Kda>) -> Vec<Gear<S>> {
    let mut layers = Vec::with_capacity(memory.len() + attention.len());
    let mut memory = memory.into_iter();
    for layer in attention {
        layers.extend(memory.by_ref().take(MEMORY_PER_ATTENTION).map(Gear::Memory));
        layers.push(Gear::Attend(layer));
    }
    layers
}

/// K3 的层配比：每 `MEMORY_PER_ATTENTION` 个通道记忆层配 1 个混合层。
///
/// 配比是**结构**而不是超参：三个记忆层读到的东西进同一条主路，第四个混合层才有注意力与专家，
/// 于是「这一组里出现过几次注意力」在 [`Tower::order`] 上可以逐格读出来，不靠注释承诺。
pub const MEMORY_PER_ATTENTION: usize = 3;

/// 一个记忆层一步的输入：逐通道遗忘门、写入强度、键、值与读出查询。
///
/// 五段都由装配侧的投影给出，本层不持参数——与 [`crate::kda::Kda`] 的分工一样：它算「一步怎么写」，
/// 谁给出这几段数不归它管。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStep<'a, S> {
    /// 逐通道遗忘门，宽 `d_k`，取值 `[0, 1]`。
    pub gate: &'a [S],
    /// 写入强度 `β ∈ [0, 1]`：0 只遗忘不写。
    pub write: S,
    /// 写入的键，宽 `d_k`（入口归一，零键拒绝）。
    pub key: &'a [S],
    /// 写入的值，宽 `d_v`。
    pub value: &'a [S],
    /// 读出查询，宽 `d_k`。
    pub query: &'a [S],
}

/// 层栈里的一格形态（闭集，码位稳定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// 通道记忆层：状态在 [`State`] 里，只把读数加回主路。
    Memory = 0,
    /// 混合层：MLA 注意力 + MoE 前馈 + mHC 多流残差。
    Attend = 1,
}

/// 层栈的随序列状态：每个记忆层一份，没有别的。
///
/// KV 缓存**不在**这里——它住 [`Latent`]，按页账走；本状态只跟着序列长度累积，与页无关。两份状态
/// 分开的后果是：`Tower::step` 不需要页号，`write_kv` 不需要记忆几何。
#[derive(Debug, Clone, PartialEq)]
pub struct State<S: Scalar = f32> {
    memories: Vec<crate::kda::Memory<S>>,
}

impl<S: Scalar> State<S> {
    /// 某记忆层的当前记忆（只读）。
    pub fn memory(&self, layer: usize) -> Option<&crate::kda::Memory<S>> {
        self.memories.get(layer)
    }

    /// 记忆层个数。
    pub fn layers(&self) -> usize {
        self.memories.len()
    }

    /// 全部记忆都还是一个 token 都没写过：装配后的初始态。
    pub fn is_pristine(&self) -> bool {
        self.memories
            .iter()
            .all(|memory| memory.cells().iter().all(|&cell| cell == S::ZERO))
    }
}

/// 层栈里的一格：记忆层持有几何（`Kda` 不持参数），混合层持有三件。
#[derive(Debug, Clone, PartialEq)]
enum Gear<S: Scalar> {
    Memory(crate::kda::Kda),
    Attend(Hybrid<S>),
}

/// 按 K3 配比堆起来的层栈：每 [`MEMORY_PER_ATTENTION`] 个记忆层后接 1 个混合层。
///
/// 本层只做装配，算术全部落在被调的三处唯一入口上（[`crate::kda::Kda::step`]/[`read`]、[`Hybrid::step`]）。
/// 一次 [`Tower::step`] 要么让整栈推进一个 token，要么一格都不动——流与记忆都在副本上算、末尾一次性提交，
/// 于是「第 5 个记忆层参数错了」不会留下前 4 层已写过、后 2 层没写的状态。
///
/// [`read`]: crate::kda::Kda::read
#[derive(Debug, Clone, PartialEq)]
pub struct Tower<S: Scalar = f32> {
    layers: Vec<Gear<S>>,
    width: usize,
    streams: usize,
}

impl<S: Scalar> Tower<S> {
    /// 按配比装配：`memory.len()` 必须是 `MEMORY_PER_ATTENTION × attention.len()`。
    ///
    /// 记忆层按给出的次序落在每组的前三格，混合层落在第四格；层序只由这里决定一次。
    ///
    /// # Errors
    ///
    /// - 混合层一个都没有（没有宽度可依，也没有注意力可跑）；
    /// - 记忆层个数不是 `3 × 混合层个数`（配比不是注释而是构造条件）；
    /// - 任一记忆层的读出宽 `d_v` 不等于混合层宽度（读数要加回主路，宽度不齐就是两张不同的表）；
    /// - 混合层之间宽度或流条数不一致（同一栈里各层几何必须同源）。
    pub fn new(attention: Vec<Hybrid<S>>, memory: Vec<crate::kda::Kda>) -> Result<Self> {
        let (width, streams) = geometry(&attention)?;
        check_ratio(attention.len(), memory.len())?;
        check_memory_width(&memory, width)?;
        let layers = interleave(attention, memory);
        Ok(Self {
            layers,
            width,
            streams,
        })
    }

    /// 本栈宽度（= 混合层的 `n_h · d_h` = 记忆层读出宽 = 主路宽）。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 残差流条数 `M`。
    pub fn streams(&self) -> usize {
        self.streams
    }

    /// 层序的形态视图：`Memory` 与 `Attend` 的序列。
    pub fn order(&self) -> impl Iterator<Item = Kind> + '_ {
        self.layers.iter().map(|layer| match layer {
            Gear::Memory(_) => Kind::Memory,
            Gear::Attend(_) => Kind::Attend,
        })
    }

    /// 层数。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 装配后的初始状态：每个记忆层一份零记忆。
    pub fn state(&self) -> State<S> {
        State {
            memories: self
                .layers
                .iter()
                .filter_map(|layer| match layer {
                    Gear::Memory(kda) => Some(kda.memory::<S>()),
                    Gear::Attend(_) => None,
                })
                .collect(),
        }
    }

    /// 本 token 的查询铺成 `M` 条流：初值就是同一个值，混合之后才分开。
    ///
    /// # Errors
    ///
    /// `hidden` 宽度不是本栈宽度时返回 `Code::Invalid`。
    pub fn seed(&self, hidden: &[S]) -> Result<Vec<Vec<S>>> {
        if hidden.len() != self.width {
            return Err(Error::new(
                Code::Invalid,
                format!("初值宽度是 {}，本栈宽度是 {}", hidden.len(), self.width),
            )
            .with(SITE));
        }
        Ok(vec![hidden.to_vec(); self.streams])
    }

    /// 一次推进：按层序把记忆层与混合层各走一遍，末尾一次性提交。
    ///
    /// 记忆层先把本 token 写进**该层自己**的记忆，再读——与 [`crate::kda::Kda::read`] 的口径一致：
    /// 单个 token 写进去、立刻读得出来。读数只加回主路（第 0 条流），理由与 [`Hybrid::step`] 相同：
    /// 同一份增量加给每条流会让多流退化成单流。
    ///
    /// 缓存在整次推进里**只读**：位置取 `cache.seq() - 1`，先 `push` 再 `step` 由调用方保证。
    ///
    /// # Errors
    ///
    /// 流几何不符、记忆输入条数与记忆层不等、任一记忆层的门/键/值/查询几何或取值越界、查询几何不符、
    /// 缓存为空——都返回 `Code::Invalid`，且 `streams` 与 `state` **都没有被改动**。
    pub fn step(
        &self,
        streams: &mut [Vec<S>],
        state: &mut State<S>,
        cache: &Latent<S>,
        query: &Query<'_, S>,
        memories: &[MemoryStep<'_, S>],
        path: Path,
    ) -> Result<()> {
        cursor(cache)?;
        let mut staged = streams.to_vec();
        self.check_streams(&staged)?;
        let mut staged_state = state.clone();
        let mut walk = Walk {
            streams: &mut staged,
            state: &mut staged_state,
            cache,
            query,
            memories,
            path,
            used: 0,
        };
        walk.run(&self.layers)?;
        check_count(walk.used, memories.len())?;
        commit(streams, staged);
        *state = staged_state;
        Ok(())
    }

    /// 流条数与各流宽度必须正好是本栈的：多一条会在混流里被当成第 `M+1` 条流，窄一格会让
    /// [`accumulate`] 静默只加重叠部分。
    fn check_streams(&self, streams: &[Vec<S>]) -> Result<()> {
        if streams.len() != self.streams {
            return Err(Error::new(
                Code::Invalid,
                format!("流条数是 {}，本栈是 {}", streams.len(), self.streams),
            )
            .with(SITE));
        }
        match streams.iter().position(|row| row.len() != self.width) {
            Some(bad) => Err(Error::new(
                Code::Invalid,
                format!(
                    "第 {bad} 条流宽度是 {}，本栈宽度是 {}",
                    streams[bad].len(),
                    self.width
                ),
            )
            .with(SITE)),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "hybrid_tests.rs"]
mod tests;
