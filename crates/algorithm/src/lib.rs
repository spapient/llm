//! 架构变体的参考语义：每一族只回答"结构是什么、判据是什么"，不注册算子、不写 kernel、不产层声明。
//!
//! **拥有**：注意力变体（[`Mla`]、[`Latent`]、[`BlockTable`]）、状态空间与记忆（[`Mamba`]、[`Kda`]）、
//! 深度轴（[`AttnRes`]）、路由与残差（[`Moe`]、[`Mhc`]、[`Hybrid`]、[`Tower`]）、脉冲三段
//! （[`Spiking`]、[`Membrane`]、[`Surrogate`]）、byte-latent 三段（[`Patches`]、[`ByteEncoder`]、
//! [`Emitter`]）与因果决策律（[`pi`] / [`gamma`] / [`Calib`]）。
//! **不拥有**：层声明与下界（`nn`）、算子语义（`op`）、权重字节（`weights`）、kernel（`cutile`）。
//! **依赖**：`nn`（只取 `engram` 的 [`Table`](nn::Table) 与 [`Bypass`](nn::Bypass)）、`op`、`diagnostic`。
//! **被依赖**：无——没有二进制链接本 crate（见下）。
//!
//! # 与 `nn` 分开的判据是产出，不是主题
//!
//! `nn` 的每一项最终产出一份 `ContractSpec`：算子名写错在构造期就失败，下界由算子语义推出。本 crate
//! 的项**一个 `ContractSpec` 都不产**——它们是那些声明背后的算术与结构（压缩、路由、槽表寻址、
//! 脉冲动力学、边界切分），唯一的例外 `engram::MemoryBlock` 连同它的槽表留在 `nn`，因为它有一个
//! `Contract` 实现：`Contract::memory` 的类型化访问器不能反过来让 `nn` 依赖本 crate。
//!
//! 合在一个 crate 里时，`nn` 的公开面混着 15 个与"层声明"无关的族，只读公开面判不出它做的是哪一件
//! 事；分开后两侧的职责句都是单一动作。
//!
//! # 为什么没有二进制消费者
//!
//! 生产路径只用到 `nn` 的层声明与 `engram` 的 [`slot_of`](nn::slot_of)：本 crate 各族的判据由
//! `tests/` 里的逐族对拍保持可执行，装配侧尚未接线。这不是"历史遗留"——每一族的入口都被本 crate
//! 的用例直接调用。
//!
//! 但"改语义会让对拍变红"只挡得住改，挡不住**删**：整目录删掉时对拍随之消失，全仓不会有一条用例
//! 变红。所以存在性不靠本文件自证，而由 `crates/tools/tests/gate.rs` 的
//! `every_architecture_family_keeps_its_conformance_suite` 从外部守：每份对拍必须被 `src` 的
//! `#[path]` 真挂载，用例总数不得低于 248。

#![deny(unsafe_code)]

use diagnostic::{Code, Error};

/// 入口拒绝：几何、取值、顺序任一不合。
///
/// 全 crate 只这一处绑定 `Code::Invalid` 与站点标签，各族只负责说清「为什么」——`site` 是
/// `algorithm::<族>`，让诊断能指回是哪一族拒绝的。在此之前 15 个文件各自写
/// `Error::new(Code::Invalid, …).with(SITE)`，84 处、每处 3–6 行，其中 2 处还漏掉了站点标签。
pub(crate) fn invalid(site: &'static str, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(site)
}

mod attnres;
mod codec;
mod csa;
mod emit;
mod hybrid;
mod kda;
mod latent;
mod mamba;
mod mhc;
mod mix;
mod mla;
mod moe;
mod patch;
mod policy;
mod spike;

// 台账与落层入口原在一个独立文件里，内联后它们直接住在 crate 根，少一层模块跳转；
// `wire` 只留 4 行别名，保住 `algorithm::wire::…` 这条既有路径。

use diagnostic::Result;
use nn::{Contract, LatentAttention, MatMul};

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

/// 17 族 → 生产层栈的接线台账。
///
/// 台账存在的理由：这 17 族都能在自己的测试里自证，但"它接到了 `nn` 的哪一层"过去只是口头
/// 事实——只有把文件删掉才发现没人依赖它。于是每一族在这里留一行，写明它落成几个 `nn` 层、
/// 还缺什么。缺的是什么必须具名，而且分三种，因为三种的补法完全不同：
///
/// - [`Blocked::Operator`]：算子层没有（要被 `op` 注册 + `ops/` 语料 + 内核模板），[`ledger`] 会用
///   [`op::builtin_named`] 复核——声明挡住了而算子其实已经存在时，台账就是过期了，测试会失败。
/// - [`Blocked::Layer`]：算子有，但 `nn` 里没有能承载它的层，几何喂不进去。
/// - [`Blocked::Geometry`]：层与算子都有，但这一族的某条路径与现有几何冲突（例如 MLA 的
///   `d_c + d_h^R` 不是头宽整数倍，`AttentionBlock` 的 `kv_width % head_dim == 0` 直接拒收）。
///
/// 不写"差不多等价"的替身：三种阻塞都不许拿形状相近的层顶替，否则台账会变成一句空话。
/// 顺序与 `src/` 的文件序一致，方便对照。
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

/// 台账的旧路径：内联后各项在 crate 根，别名让 `algorithm::wire::…` 继续成立。
pub mod wire {
    pub use crate::{
        Blocked, Family, Wiring, hybrid_layer, ledger, mhc_layer, mla_layer, tower_layers,
    };
}

#[doc(inline)]
pub use attnres::{AttnRes, Sources, splice};

#[doc(inline)]
pub use codec::{BYTES, ByteDecoder, ByteEncoder, ORDERS};

#[doc(inline)]
pub use csa::{
    BlockTable, CSA_M, CSA_TOPK_FLASH, CSA_TOPK_PRO, HCA_M, WIN, compress, dense_softmax, dsa_topk,
    effective_len, is_doubly_stochastic, sinkhorn, spectral_norm,
};

#[doc(inline)]
pub use emit::{Emission, Emitter, Halt};

#[doc(inline)]
pub use hybrid::{Hybrid, Kind, MEMORY_PER_ATTENTION, MemoryStep, Query, State, Tower};

#[doc(inline)]
pub use kda::{Kda, Memory};

#[doc(inline)]
pub use latent::{Backread, Global, LOCAL_MAX, LOCAL_MIN, Layer};

#[doc(inline)]
pub use mamba::{Carry, Mamba};

#[doc(inline)]
pub use mhc::{MHC_STREAMS, Mhc};

#[doc(inline)]
pub use mix::{
    Audit, Encoder, Events, Lif, accumulate, cont, encode_id, gate, steepest, synthesize,
};

#[doc(inline)]
pub use mla::{
    Latent, Mla, Path, V3_HEAD_DIM, V3_HEADS, V3_LATENT, V3_LAYERS, V3_ROPE, dense_width, ratio,
    row_width,
};

#[doc(inline)]
pub use moe::{K3_ACTIVE, K3_EXPERTS, Moe};

#[doc(inline)]
pub use patch::{Boundary, Patches};

#[doc(inline)]
pub use policy::{
    Calib, Channel, Gating, HEAD_LAYERS_TOKEN, MissRate, N_REF, TAU_SCALE, W, brier, gamma, nu, pi,
    rms, sparsity, tau,
};

#[doc(inline)]
pub use spike::{Gradients, Master, Membrane, Spiking, Surrogate};

#[cfg(test)]
#[path = "../tests/prop_tests.rs"]
mod prop_tests;

#[cfg(test)]
#[path = "../tests/wire_tests.rs"]
mod wire_tests;
