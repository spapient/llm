//! 后端能力表：`骨架族 × 元素档` 查表裁决，不写第二张 `match` 分支（ADR 0054 §3）。
//!
//! 一张行表登记「某副骨架支持哪个精度档」；[`supports`] 按两维查表，缺项返回 `false`，
//! 由上层按缺的是哪一族、哪一档定位，而不是回一个泛化 `Unsupported`。
//!
//! ## 边界（ADR 0050）
//!
//! 本层是 L2 契约层，**不认识** L3/L6 的设备字面量：第三维 `backend`（哪台机器能跑）
//! 留到 P3 接 `Executor`/后端真实实现逐条对账后再补，不在此引入设备名。表内只放骨架族与
//! 元素档，二者都来自算子目录行表，不另造一份清单。

use crate::model::DType;

/// 一条能力登记：一副骨架族支持一个元素档。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capability {
    /// 骨架族标签：与算子目录 `[op].skeleton` 同一字符串（`unary` / `elementwise` / `reduce`）。
    pub skeleton: &'static str,
    /// 该骨架当前落地的元素档。
    pub dtype: DType,
}

/// 能力行表：顺序无关，[`supports`] 全表扫描；条目只有三族一档，列表即查表，不引入索引结构。
///
/// 收敛方向（P3）：本表从算子目录行表派生，避免与 `REGISTRY` 漂移；backend 维对账后在此扩列。
pub const CAPABILITIES: &[Capability] = &[
    Capability {
        skeleton: "unary",
        dtype: DType::F32,
    },
    Capability {
        skeleton: "elementwise",
        dtype: DType::F32,
    },
    Capability {
        skeleton: "reduce",
        dtype: DType::F32,
    },
];

/// `skeleton` × `dtype` 是否登记为受支持；未登记即缺项，调用方据两维定位，不静默回退。
pub fn supports(skeleton: &str, dtype: DType) -> bool {
    CAPABILITIES
        .iter()
        .any(|cap| cap.skeleton == skeleton && cap.dtype == dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DType;

    #[test]
    fn registered_f32_families_supported() {
        for skeleton in ["unary", "elementwise", "reduce"] {
            assert!(supports(skeleton, DType::F32), "{skeleton}×f32 应在表内");
        }
    }

    #[test]
    fn unknown_family_or_dtype_is_a_missing_capability() {
        assert!(!supports("compare", DType::F32), "五新族尚未登记");
        assert!(!supports("unary", DType::I8), "骨架当前只有 f32 档");
    }
}
