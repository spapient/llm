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
//!
//! 骨架键取 [`Skeleton`] 闭集枚举而非自由字符串：族名只有声明层一张表可解析，编译期就
//! 堵死悬空族名；未落地的族（`Scan`）与未核通的精度档不登记，不凭目录名单签发能力。

use crate::decl::Skeleton;
use crate::model::DType;

/// 一条能力登记：一副骨架族支持一个元素档。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capability {
    /// 骨架族标签：闭集变体，与算子目录 `[op].skeleton` 同源解析。
    pub skeleton: Skeleton,
    /// 该骨架当前落地的元素档。
    pub dtype: DType,
}

/// 能力行表：顺序无关，[`supports`] 全表扫描；条目只有三族一档，列表即查表，不引入索引结构。
///
/// 收敛方向（P3）：本表从算子目录行表派生，避免与 `REGISTRY` 漂移；backend 维对账后在此扩列。
pub const CAPABILITIES: &[Capability] = &[
    Capability {
        skeleton: Skeleton::Elementwise,
        dtype: DType::F32,
    },
    Capability {
        skeleton: Skeleton::Reduce,
        dtype: DType::F32,
    },
    Capability {
        skeleton: Skeleton::Gemm,
        dtype: DType::F32,
    },
];

/// `skeleton` × `dtype` 是否登记为受支持；未登记即缺项，调用方据两维定位，不静默回退。
pub fn supports(skeleton: Skeleton, dtype: DType) -> bool {
    CAPABILITIES
        .iter()
        .any(|cap| cap.skeleton == skeleton && cap.dtype == dtype)
}

/// 一副骨架族的编译期默认派发几何：网格与线程块各三轴。
///
/// 形状相关的网格大小待 G2 几何通道按 `extent` 实算；当前只是族级保守占位种子。
/// 它不参与 tile 打分、不改 `select` 的声明顺序（ADR 0056 G4/G5），真机计时择优（autotune）
/// 仍 opt-in 留 G5。未落地的族（`Scan`）不登记，与 [`CAPABILITIES`] 同口径。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GeometrySeed {
    /// 骨架族标签。
    pub skeleton: Skeleton,
    /// 网格三轴占位；按形状展开的网格推导待几何通道接入。
    pub grid: [u32; 3],
    /// 线程块三轴占位。
    pub block: [u32; 3],
}

/// 族级默认几何种子行表：顺序无关，[`geometry_seed`] 全表扫描；列表即查表，不引入索引结构。
pub const GEOMETRY_SEEDS: &[GeometrySeed] = &[
    GeometrySeed {
        skeleton: Skeleton::Elementwise,
        grid: [1, 1, 1],
        block: [256, 1, 1],
    },
    GeometrySeed {
        skeleton: Skeleton::Reduce,
        grid: [1, 1, 1],
        block: [256, 1, 1],
    },
    GeometrySeed {
        skeleton: Skeleton::Gemm,
        grid: [1, 1, 1],
        block: [16, 16, 1],
    },
];

/// 取一族的默认几何种子；未登记族返回 `None`，不猜一个泛化块大小。
pub fn geometry_seed(skeleton: Skeleton) -> Option<&'static GeometrySeed> {
    GEOMETRY_SEEDS.iter().find(|seed| seed.skeleton == skeleton)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DType;

    #[test]
    fn registered_f32_families_supported() {
        for skeleton in [Skeleton::Elementwise, Skeleton::Reduce, Skeleton::Gemm] {
            assert!(supports(skeleton, DType::F32), "{skeleton:?}×f32 应在表内");
        }
    }

    #[test]
    fn unknown_family_or_dtype_is_a_missing_capability() {
        // Scan 族与低比特档尚未核通实现覆盖，不登记为能力。
        assert!(!supports(Skeleton::Scan, DType::F32), "scan 族尚未登记");
        assert!(
            !supports(Skeleton::Elementwise, DType::I8),
            "骨架当前只有 f32 档"
        );
    }

    #[test]
    fn table_keys_stay_within_the_skeleton_closed_set() {
        // 骨架键已是枚举，编译期不可悬空；运行期再逐条回指闭集全量，锁住表与声明同表。
        for cap in CAPABILITIES {
            assert!(
                Skeleton::ALL.contains(&cap.skeleton),
                "表内骨架 {} 不在闭集内",
                cap.skeleton.as_str()
            );
        }
    }

    #[test]
    fn geometry_seed_keys_stay_within_the_closed_set() {
        for seed in GEOMETRY_SEEDS {
            assert!(
                Skeleton::ALL.contains(&seed.skeleton),
                "种子表骨架 {} 不在闭集内",
                seed.skeleton.as_str()
            );
        }
    }

    #[test]
    fn landed_families_have_a_default_seed_scan_does_not() {
        for skeleton in [Skeleton::Elementwise, Skeleton::Reduce, Skeleton::Gemm] {
            assert!(
                geometry_seed(skeleton).is_some(),
                "已落地族 {skeleton:?} 应有默认几何种子"
            );
        }
        assert!(
            geometry_seed(Skeleton::Scan).is_none(),
            "Scan 族尚未落地，不登记占位几何"
        );
    }
}
