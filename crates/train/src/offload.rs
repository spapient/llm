//! 参数 offload 接缝（**暂缓**）。
//!
//! 本模块只保留接口形状：把权重在驻留内存与后备存储之间换入/换出的通路**尚未建立**。前置
//! 依赖——页池（块/页编号与驻留表）、权重驻留记账、换入换出搬运与驻留命中判定——都未落地。
//! 本刀不造假实现、不接虚假换入：接口方法统一返回 [`Code::Unsupported`]，等页池与驻留表
//! 落地后再填真实通路。
//!
//! 暂缓清单（ADR 0037 §不做）：ZeRO、分布式优化器、序列并行、真实权重落盘端到端、设备后端
//! 均不在本轨；offload 真实通路随后续里程碑接入。

use diagnostic::{Code, Error, Result};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::offload";

/// 参数换入/换出接缝：槽位编号标识一块权重；方法签名是后续真实通路的形状占位。
///
/// # 暂缓前置
///
/// - **页池**：槽位/块编号 → 驻留页或后备页的映射表尚未建立；
/// - **权重驻留**：哪一块权重此刻在场、哪一块被换出的记账不存在；
/// - **换入/换出**：后备存储搬运、驻留命中判定与换入后的缓冲镜像都未接。
///
/// 因此当前实现一律返回 [`Code::Unsupported`]，不做任何搬运，也不登记驻留状态；训练闭环
/// 不调用本 trait 的任何方法。
pub trait WeightOffload {
    /// 把一块权重换出驻留内存到后备存储（接缝，暂缓）。
    ///
    /// # Errors
    ///
    /// 通路未建立：[`Code::Unsupported`]。
    fn swap_out(&mut self, slot: usize) -> Result<()>;

    /// 把一块权重换回驻留内存（接缝，暂缓）。
    ///
    /// # Errors
    ///
    /// 通路未建立：[`Code::Unsupported`]。
    fn swap_in(&mut self, slot: usize) -> Result<()>;
}

/// 未接通路的占位实现：所有方法返回 [`Code::Unsupported`]，不持页池、不搬内存。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DeferredOffload;

impl WeightOffload for DeferredOffload {
    fn swap_out(&mut self, slot: usize) -> Result<()> {
        deferred(slot)
    }

    fn swap_in(&mut self, slot: usize) -> Result<()> {
        deferred(slot)
    }
}

/// 统一的暂缓错误：把前置缺口写进消息，避免调用方误以为"换出成功"。
fn deferred(slot: usize) -> Result<()> {
    Err(Error::new(
        Code::Unsupported,
        format!(
            "参数槽 {slot} 的 offload 换入/换出通路暂缓：页池、权重驻留记账与换入换出搬运尚未建立"
        ),
    )
    .with(SITE))
}
