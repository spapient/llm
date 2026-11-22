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
//! `#[path]` 真挂载，用例总数不得低于 245。

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

// 模块说明只在 `wire.rs` 头部的 `//!` 里：外层 `///` 会被并进同一个 doc 块，并按 crate 根
// 作用域解析其中的内链，于是同模块的 `Blocked` / `ledger` 反而"不在作用域"。
pub mod wire;

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
