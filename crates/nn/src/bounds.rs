//! 层的性能下界：任何实现都不可能低于的流量与运算次数。
//!
//! 这些数**不是手写的**：它们由层声明的算子在 `op` 里的语义（一条求和表达式 + 每个操作数的
//! 位宽）推出。因此新增算子不会多出第二条公式，`nn` 也不必认识除算子名以外的任何领域知识。
//!
//! # 为什么要有它
//!
//! `cutile` 的 roofline 在 **kernel** 层回答"这条片段至少要搬多少、算多少"；本模块把同一件事
//! 抬到**层**：下界是声明，`bench` 的实测是事实，两者之差正是融合、量化与布局真正赚到的部分。
//! 没有层的下界，"这一层优化了多少"就只能靠人肉估算。
//!
//! 下界**与设备无关**：本模块不认识峰值带宽、厂商或存储层。需要与设备比较时，由调用方把峰值
//! 传进来（[`Bounds::calls_per_second_ceiling`]）——这与 `cutile::plan` 的做法一致：峰值归
//! `prof`，编排归调用方，本层只给数对。

use diagnostic::{Code, Error, Result};

/// 一次调用的性能下界：最少流量（读 + 写）与最少浮点运算次数。
///
/// 下界**与实现无关**：它既不看片段怎么写，也不看设备有多少单元，因此可以在任何机器上离线
/// 复现（与 `einsum` 的代价同源）。它也不承诺"能做到"——那是 `bench` 的事。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    bytes: u64,
    flops: f64,
}

impl Bounds {
    /// 零：还没有声明任何算子。
    pub const ZERO: Self = Self {
        bytes: 0,
        flops: 0.0,
    };

    /// 由算子语义推出下界。
    ///
    /// # Errors
    ///
    /// - 算子没有语义表（[`Code::Missing`]）：声明了却不能对拍，也就无法定义它的下界；
    /// - `(m, n)` 不是该算子的合法形状（[`Code::Invalid`]）：由算子自己的代价模型报出，
    ///   例如量化 GEMV 的规约轴必须是分组因子的整数倍。
    pub fn of_op(op: &str, m: usize, n: usize) -> Result<Self> {
        let model = op::model(op).ok_or_else(|| {
            Error::new(
                Code::Missing,
                format!("算子 `{op}` 没有语义表，无法推出下界"),
            )
            .with("nn::Bounds::of_op")
        })?;
        let cost = (model.cost)(m, n)?;
        Ok(Self {
            bytes: cost.bytes,
            flops: cost.flops,
        })
    }

    /// 两笔下界相加：一串层（或一个复合算子）的总下界。
    pub fn plus(self, other: Self) -> Self {
        Self {
            bytes: self.bytes.saturating_add(other.bytes),
            flops: self.flops + other.flops,
        }
    }

    /// 最少流量（字节）：读与写合计。
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// 最少浮点运算次数。
    pub fn flops(&self) -> f64 {
        self.flops
    }

    /// 算术强度（flop/字节）：roofline 的横轴。
    pub fn intensity(&self) -> f64 {
        if self.bytes == 0 {
            0.0
        } else {
            self.flops / self.bytes as f64
        }
    }

    /// 给定峰值带宽时，每秒调用次数的上限：内存受限一侧的 roofline。
    ///
    /// 峰值由调用方给（`prof` 拥有设备峰值，本层不认识设备）。峰值非正或流量为零时返回
    /// `0.0`：没有带宽就没有上限，没有流量则不存在这条约束。
    pub fn calls_per_second_ceiling(&self, peak_bytes_per_second: f64) -> f64 {
        let usable = peak_bytes_per_second.is_finite() && peak_bytes_per_second > 0.0;
        if self.bytes == 0 || !usable {
            0.0
        } else {
            peak_bytes_per_second / self.bytes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_projection_declares_its_floor() {
        // 4096 方阵的 int4 权重：光是"读一遍权重"就已经这么多字节，
        // 再加上输入、每个 32 元素的缩放因子与输出。
        let floor = Bounds::of_op("gemv_i4", 4096, 4096).expect("生产形状");
        assert_eq!(
            floor.bytes(),
            4096 * 4096 / 2 + 4096 * 4 + 4096 * (4096 / 32) * 4 + 4096 * 4
        );
        assert_eq!(floor.flops(), 2.0 * 4096.0 * 4096.0);
        // 量化让"每搬一个字节能算几次"落在 3.19，而不是同一层用 f32 权重时的 1.0。
        assert_eq!((floor.intensity() * 100.0).round() / 100.0, 3.19);
    }

    #[test]
    fn the_floor_is_exactly_the_operator_semantics() {
        let model = op::model("gemv_i4").expect("有语义表");
        for (m, n) in [(64, 4), (32, 1), (4096, 64)] {
            let floor = Bounds::of_op("gemv_i4", m, n).expect("合法形状");
            let cost = (model.cost)(m, n).expect("合法形状");
            assert_eq!(floor.bytes(), cost.bytes, "{m}x{n}");
            assert_eq!(floor.flops(), cost.flops, "{m}x{n}");
        }
    }

    #[test]
    fn a_shape_the_operator_rejects_has_no_floor() {
        // 层本身可以构造（宽度非零），但 8 不是量化分组的整数倍：
        // 这个形状根本没有合法内核，因此谈不上"至少搬多少"。
        let error = Bounds::of_op("gemv_i4", 8, 4).expect_err("形状非法");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("整数倍"), "{error}");
    }

    #[test]
    fn an_operator_without_semantics_has_no_floor() {
        let error = Bounds::of_op("flash_attention", 64, 4).expect_err("没有语义表");
        assert_eq!(error.code(), Code::Missing);
        assert!(error.to_string().contains("flash_attention"), "{error}");
    }

    #[test]
    fn floors_add_up_across_a_stack() {
        let one = Bounds::of_op("gemv_i4", 64, 4).expect("合法");
        let two = one.plus(one);
        assert_eq!(two.bytes(), 2 * one.bytes());
        assert_eq!(two.flops(), 2.0 * one.flops());
        assert_eq!(two.intensity(), one.intensity(), "同构的两层强度不变");
        assert_eq!(Bounds::ZERO.plus(one), one, "零元");
    }

    #[test]
    fn the_ceiling_divides_peak_by_traffic() {
        let floor = Bounds::of_op("gemv_i4", 64, 4).expect("合法");
        assert_eq!(
            floor.calls_per_second_ceiling(1_000_000_000.0),
            1e9 / floor.bytes() as f64
        );
        assert_eq!(
            floor.calls_per_second_ceiling(0.0),
            0.0,
            "没有带宽就没有上限"
        );
        assert_eq!(Bounds::ZERO.calls_per_second_ceiling(1e9), 0.0);
        assert_eq!(Bounds::ZERO.intensity(), 0.0, "除零不返回 NaN");
    }
}
