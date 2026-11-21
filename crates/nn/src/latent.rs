//! 潜空间注意力层：KV 按 `d_c` 压缩存量，解耦 RoPE 键单头叠加。
//!
//! 与 [`Attention`](crate::Attention) 的区别只在 KV 的口径：本层不认识 `h · d_h` 的稠密宽度，
//! 只认识压缩宽度 `d_c`。形状仍是 `[seq, d_c]`，头数进代价（行数摊成 `seq · h`），不进形状——
//! 这与算子 `latent_attention_f32` 的几何完全一致：算子按压平行算，输出宽就是压缩宽度。
//!
//! 上投影 `W^{UK}` / `W^{UV}` 不在本层：吸收式把它们折进查询与输出投影，稠密式则整块展开，
//! 两种折法都在装配侧，层只管压缩存量与读法。

use crate::{Bounds, BoundsResult, Contract, spec_accessors};
use diagnostic::{Code, Error, Result};

/// 潜空间注意力层。
#[crate::spec(name = $name, out = [$seq, $latent], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatentAttention {
    seq: usize,
    latent: usize,
    heads: usize,
}

impl LatentAttention {
    /// 算子名。
    pub const OPERATOR: &str = "latent_attention_f32";

    /// 由层名、查询行数、压缩宽度与头数构造。
    ///
    /// # Errors
    ///
    /// `seq` / `latent` / `heads` 任一为零（[`Code::Invalid`]）；算子未注册或没有语义表
    /// （[`crate::ContractSpec::new`] 上报）。
    pub fn new(name: impl Into<String>, seq: usize, latent: usize, heads: usize) -> Result<Self> {
        for (label, value) in [("seq", seq), ("latent", latent), ("heads", heads)] {
            if value == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("潜空间注意力层 {label} 必须非零（{label} = {value}）"),
                )
                .with("nn::LatentAttention::new"));
            }
        }
        Ok(Self {
            spec: __layer_spec_LatentAttention!(name, seq, latent, Self::OPERATOR)?,
            seq,
            latent,
            heads,
        })
    }

    /// 序列长度（轴 `s`）；摊平后喂给算子的行数是 `seq · heads`。
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// 压缩宽度（轴 `c`），也是单头的键值宽度。
    pub fn latent(&self) -> usize {
        self.latent
    }

    /// 头数：不进形状，但行数与代价都按它摊开。
    pub fn heads(&self) -> usize {
        self.heads
    }
}

impl Contract for LatentAttention {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        // 行数是摊平后的 `seq · h`；宽度只认压缩宽度 `d_c`，解耦 RoPE 那段属于装配侧的几何，
        // 由运行期行宽给出，层不假装知道它——否则下界会把上投影的活算进来，重复计价。
        Bounds::of_op(Self::OPERATOR, self.seq * self.heads, self.latent).map(Some)
    }
}
