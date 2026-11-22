//! 17 族 → 生产层栈的接线台账。
//!
//! 台账存在的理由：这 17 族都能在自己的测试里自证，但"它接到了 `nn` 的哪一层"过去只是口头
//! 事实——只有把文件删掉才发现没人依赖它。于是每一族在这里留一行，写明它落成几个 `nn` 层、
//! 还缺什么。缺的是什么必须具名，而且分三种，因为三种的补法完全不同：
//!
//! - [`Blocked::Operator`]：算子层没有（要被 `op` 注册 + `ops/` 语料 + 内核模板），[`ledger`] 会用
//!   [`op::builtin_named`] 复核——声明挡住了而算子其实已经存在时，台账就是过期了，测试会失败。
//! - [`Blocked::Layer`]：算子有，但 `nn` 里没有能承载它的层，几何喂不进去。
//! - [`Blocked::Geometry`]：层与算子都有，但这一族的某条路径与现有几何冲突（例如 MLA 的
//!   `d_c + d_h^R` 不是头宽整数倍，`AttentionBlock` 的 `kv_width % head_dim == 0` 直接拒收）。
//!
//! 不写"差不多等价"的替身：三种阻塞都不许拿形状相近的层顶替，否则台账会变成一句空话。

use diagnostic::Result;
use nn::{Contract, LatentAttention, MatMul};

use crate::{Hybrid, Mhc, Mla};

/// 一族还缺什么。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blocked {
    /// 缺算子：名字是 `op` 注册表里**尚未**存在的那个。
    Operator(&'static str),
    /// 缺层：算子有，`nn` 没有能承载它的层。
    Layer(&'static str),
    /// 缺几何：层与算子都有，但这族的某条路径喂不进去。
    Geometry(&'static str),
}

/// 一族的接线状态：落了几个层，以及还剩什么。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wiring {
    layers: usize,
    blocked: Option<Blocked>,
}

impl Wiring {
    /// 整族接通。
    pub const fn wired(layers: usize) -> Self {
        Self {
            layers,
            blocked: None,
        }
    }

    /// 一个层都没落成。
    pub const fn blocked(why: Blocked) -> Self {
        Self {
            layers: 0,
            blocked: Some(why),
        }
    }

    /// 落成了部分层，剩下的路径被 `why` 挡住。
    pub const fn partial(layers: usize, why: Blocked) -> Self {
        Self {
            layers,
            blocked: Some(why),
        }
    }

    /// 每个结构块落成的层数（`tower` 这类可重复的族按单块计，整塔层数 = 块数 × 本值）。
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// 剩余阻塞；`None` 表示整族接通。
    pub fn blocked_on(&self) -> Option<Blocked> {
        self.blocked
    }

    /// 是否整族接通。
    pub fn is_wired(&self) -> bool {
        self.blocked.is_none()
    }
}

/// 一族：名字与它的接线状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Family {
    /// 族名（与 `src/<name>.rs` 一致）。
    pub name: &'static str,
    /// 接线状态。
    pub wiring: Wiring,
}

const fn f(name: &'static str, wiring: Wiring) -> Family {
    Family { name, wiring }
}

/// 17 族台账。顺序与 `src/` 的文件序一致，方便对照。
pub fn ledger() -> [Family; 17] {
    // 两个口径各一个短名：挡住的理由是"缺算子"或"缺层"，写成一行才看得出各族差在哪。
    let by_layer = |why: &'static str| Wiring::blocked(Blocked::Layer(why));
    let by_op = |name: &'static str| Wiring::blocked(Blocked::Operator(name));
    [
        f("mla", Wiring::wired(1)),
        f(
            "hybrid",
            Wiring::partial(2, Blocked::Layer("nn 无专家路由层（Moe 不给 FFN 宽）")),
        ),
        f("tower", Wiring::partial(2, Blocked::Operator("kda"))),
        f("mamba", by_op("ssm_scan")),
        f("spike", by_op("lif")),
        f("mix", by_op("lif")),
        f("pulse", by_op("lif")),
        f("kda", by_op("kda")),
        f("csa", by_op("csa")),
        f("mhc", Wiring::wired(1)),
        f("attnres", by_layer("nn 无流内残差混合层")),
        f("moe", by_layer("nn 无专家路由层")),
        f("patch", by_layer("nn 无图像切块层")),
        f("codec", by_layer("nn 无字节编解码层")),
        f("latent", by_layer("nn 无潜变量采样层")),
        f("policy", by_layer("nn 无动作采样层")),
        f("emit", by_layer("nn 无生成回路")),
    ]
}

/// `mhc` 的流重组落成 `streams × streams` 的常量矩阵：双随机混合就是作用在流轴上的一次乘，
/// `width` 只是批轴，运行时不需要分支。名字由调用方给，因为同一个混合块会在塔里出现多次。
///
/// # Errors
///
/// 流数为零或矩阵层拒绝该几何。
pub fn mhc_layer(m: &Mhc<f32>, name: &str) -> Result<MatMul> {
    MatMul::new(name, m.streams(), m.streams())
}

/// `mla` 的注意力段落成 [`LatentAttention`]：压缩宽度就是 `latent`，头数把行数摊成 `seq · h`。
///
/// # Errors
///
/// 任一宽度为零或算子缺语义表。
pub fn mla_layer(m: &Mla<f32>, seq: usize, name: &str) -> Result<LatentAttention> {
    LatentAttention::new(name, seq, m.latent(), m.heads())
}

/// `hybrid` 的接线：注意力段 + 流重组，两段都是真层；专家 FFN 仍缺层。
///
/// # Errors
///
/// 任一宽度为零或算子缺语义表。
pub fn hybrid_layer(h: &Hybrid<f32>, seq: usize, idx: usize) -> Result<Vec<Box<dyn Contract>>> {
    // 层名带位置：同一个混合块会在一座塔里出现多次，重名会被 `nn::Stack::of` 拒收。
    let attn = mla_layer(h.attention(), seq, &format!("hybrid.{idx}.attn"))?;
    Ok(vec![
        Box::new(attn),
        Box::new(mhc_layer(h.residual(), &format!("hybrid.{idx}.mix"))?),
    ])
}

/// `tower` 的接线：每层压栈的 `hybrid` 各自两段。入参是构造塔时交进去的注意力层——塔内部按
/// `tower::Gear` 交错成私有列表，只吐 `tower::Kind` 序，不吐层引用，所以从不
/// 变量更强的入口取。KDA 记忆段不贡献层：算子还没有，所以只是部分接通。
///
/// # Errors
///
/// 任一宽度为零或算子缺语义表。
pub fn tower_layers(attention: &[Hybrid<f32>], seq: usize) -> Result<Vec<Box<dyn Contract>>> {
    let mut layers = Vec::new();
    for (idx, hybrid) in attention.iter().enumerate() {
        layers.extend(hybrid_layer(hybrid, seq, idx)?);
    }
    Ok(layers)
}
