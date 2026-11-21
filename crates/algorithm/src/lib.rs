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
//! 的用例直接调用，删除或改语义都会先让对拍变红。

#![deny(unsafe_code)]

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
