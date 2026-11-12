//! 层的性能下界：任何实现都不可能低于的流量与运算次数。
//!
//! 这些数**不是手写的**：它们由层声明的算子在 `op` 里的语义（一条求和表达式 + 每个操作数的
//! 位宽与方向）推出。因此新增算子不会多出第二条公式，`nn` 也不必认识除算子名以外的任何领域知识。
//!
//! # 为什么要有它
//!
//! `cutile` 的 roofline 在 **kernel** 层回答"这条片段至少要搬多少、算多少"；本模块把同一件事
//! 抬到**层**：下界是声明，`bench` 的实测是事实，两者之差正是融合、量化与布局真正赚到的部分。
//! 没有层的下界，"这一层优化了多少"就只能靠人肉估算。
//!
//! # 口径：按方向分账
//!
//! 流量分成 [`Bounds::read`] 与 [`Bounds::write`] 两个方向，而不是一个 `bytes`：设备的读、写
//! 上限不是同一个数（只流式读权重的 GEMV 可以跑过拷贝上限），合成一个总量之后"这层卡在哪一侧"
//! 就再也判断不了。[`Bounds::bytes`] 仍然给出合计，但它是**派生**量。
//!
//! 这个区分不是为了好看：[`Bounds::moves_within`] 靠它识别"总量不变、只是把读挪成写"——那不是
//! 融合省下的流量，只是换了一个瓶颈。
//!
//! # 与设备无关
//!
//! 本模块不认识峰值带宽、厂商或存储层，也不做 roofline 编排：需要与设备比较时由调用方拿峰值
//! 去比（`cutile::plan` 就是这么做的）。这里只给数对，而且每个方向各给一份。

use diagnostic::{Code, Error, Result};

/// 一次调用的性能下界：读、写两个方向的最少流量与最少浮点运算次数。
///
/// 下界**与实现无关**：它既不看片段怎么写，也不看设备有多少单元，因此可以在任何机器上离线
/// 复现（与 `einsum` 的代价同源）。它也不承诺"能做到"——那是 `bench` 的事。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    read: u64,
    write: u64,
    flops: f64,
}

impl Bounds {
    /// 零：还没有声明任何算子。
    pub const ZERO: Self = Self {
        read: 0,
        write: 0,
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
        // 投影的 `(输入宽度, 输出宽度)` 由算子自己的轴角色翻译成具名轴：层不必知道规约轴在
        // 表达式里叫 `m` 还是 `k`，也不必知道量化分组的派生轴怎么切。
        let shape = model.projection(m, n)?;
        let cost = (model.cost)(&shape)?;
        Ok(Self {
            read: cost.read,
            write: cost.write,
            flops: cost.flops,
        })
    }

    /// 一串算子在**同一个形状**上的下界之和：组合层自己的下界。
    ///
    /// 空列表是 [`Bounds::ZERO`]——"没有可计价的算子"与"代价为零"在数值上一致，两者的区别由
    /// 调用方的清单（`Stack` 的未覆盖/无法计价条目）表达，而不是靠一个特殊值。
    ///
    /// 每个算子都用同一组 `(m, n)`：这只对"同一条投影上叠了几个算子"成立。轴映射不同的复合层
    /// （RMSNorm 的 `[batch, seq, d_model]`、Attention 的 `seq²`）需要先把形状上下文抬进算子
    /// 语义表，本函数不替它们伪造一对 `(m, n)`。
    ///
    /// # Errors
    ///
    /// 任一算子没有语义表或该形状对它不合法时，错误原样上报并点名该算子。
    pub fn of_ops(ops: &[String], m: usize, n: usize) -> Result<Self> {
        let mut total = Self::ZERO;
        for op in ops {
            total = total
                .plus(Self::of_op(op, m, n).map_err(|error| error.with(format!("算子 `{op}`")))?);
        }
        Ok(total)
    }

    /// 两笔下界相加：一串层（或一个复合算子）的总下界。
    ///
    /// 两个方向各自饱和相加；`flops` 用普通加法。这不是疏漏：`u64` 溢出时静默截断会给出
    /// **偏小**的下界（"偏小"比"没有"更危险，它看起来仍然合理），所以宁可饱和；`f64` 溢出成
    /// `inf` 本身就是响亮的信号，再套一层饱和只会掩盖它。
    pub fn plus(self, other: Self) -> Self {
        Self {
            read: self.read.saturating_add(other.read),
            write: self.write.saturating_add(other.write),
            flops: self.flops + other.flops,
        }
    }

    /// 最少读入字节。
    pub fn read(&self) -> u64 {
        self.read
    }

    /// 最少写出字节。
    pub fn write(&self) -> u64 {
        self.write
    }

    /// 最少流量（字节）：读与写合计的**派生**量。
    ///
    /// 判断 roofline 卡在哪一侧必须分别拿 [`Bounds::read`] 与 [`Bounds::write`] 去比峰值。
    pub fn bytes(&self) -> u64 {
        self.read.saturating_add(self.write)
    }

    /// 最少浮点运算次数。
    pub fn flops(&self) -> f64 {
        self.flops
    }

    /// 算术强度（flop/字节）：roofline 的横轴。用合计流量，因为横轴本来就只关心"每字节算几次"。
    pub fn intensity(&self) -> f64 {
        let bytes = self.bytes();
        if bytes == 0 {
            0.0
        } else {
            self.flops / bytes as f64
        }
    }

    /// 本下界是否落在那笔下界之内：**每个方向**都不更高。
    ///
    /// 这是"融合到底赚了多少"的可检查形式：一个复合组件的下界只有在读、写、算三项上都不高于
    /// 子组件之和，才谈得上更省；差额就是完美融合能省掉的中间流量。
    ///
    /// 反过来，总量相同但把读挪成写**不**落于其中——那不是省流量，只是换了一个瓶颈。
    pub fn moves_within(self, other: Self) -> bool {
        self.read <= other.read && self.write <= other.write && self.flops <= other.flops
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
            floor.read(),
            4096 * 4096 / 2 + 4096 * 4 + 4096 * (4096 / 32) * 4
        );
        assert_eq!(floor.write(), 4096 * 4, "写方向只有输出那一份");
        assert_eq!(
            floor.bytes(),
            4096 * 4096 / 2 + 4096 * 4 + 4096 * (4096 / 32) * 4 + 4096 * 4,
            "合计与改造前逐字节相同"
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
            let shape = model.projection(m, n).expect("合法形状");
            let cost = (model.cost)(&shape).expect("合法形状");
            assert_eq!(floor.read(), cost.read, "{m}x{n}");
            assert_eq!(floor.write(), cost.write, "{m}x{n}");
            assert_eq!(floor.flops(), cost.flops, "{m}x{n}");
        }
    }

    #[test]
    fn the_two_directions_are_not_collapsed() {
        let floor = Bounds::of_op("gemv_i4", 4096, 4096).expect("生产形状");
        assert!(
            floor.read() > 8 * 1024 * 1024,
            "读侧是权重主导：{}",
            floor.read()
        );
        assert!(
            floor.write() < floor.read() / 100,
            "写侧只有一个 f32 输出向量：{}",
            floor.write()
        );
    }

    #[test]
    fn shifting_traffic_between_directions_does_not_move_within() {
        // 总量相同，只是把读挪成写：这不是"更省"，只是换了一个瓶颈。
        let read_heavy = Bounds {
            read: 100,
            write: 0,
            flops: 1.0,
        };
        let write_heavy = Bounds {
            read: 0,
            write: 100,
            flops: 1.0,
        };
        assert_eq!(read_heavy.bytes(), write_heavy.bytes());
        assert!(!read_heavy.moves_within(write_heavy));
        assert!(!write_heavy.moves_within(read_heavy));
        assert!(read_heavy.moves_within(read_heavy), "自己落于自己之内");
    }

    #[test]
    fn a_fused_layer_moves_within_the_sum_of_its_parts() {
        let weights = Bounds::of_op("gemv_i4", 4096, 4096).expect("合法");
        let sum = weights.plus(weights);
        let fused = Bounds {
            read: weights.read,
            write: weights.write,
            flops: 2.0 * weights.flops,
        };
        assert!(fused.moves_within(sum), "融合省掉了中间那份流量");
        assert!(!sum.moves_within(fused), "反过来不成立：和不会更小");
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
        assert_eq!(two.read(), 2 * one.read());
        assert_eq!(two.write(), 2 * one.write());
        assert_eq!(two.bytes(), 2 * one.bytes());
        assert_eq!(two.flops(), 2.0 * one.flops());
        assert_eq!(two.intensity(), one.intensity(), "同构的两层强度不变");
        assert_eq!(Bounds::ZERO.plus(one), one, "零元");
        assert_eq!(Bounds::ZERO.intensity(), 0.0, "除零不返回 NaN");
    }

    #[test]
    fn a_composite_layer_is_the_sum_of_its_operators() {
        let one = vec!["gemv_i4".to_string()];
        let twice = vec!["gemv_i4".to_string(), "gemv_i4".to_string()];
        assert_eq!(Bounds::of_ops(&[], 64, 4).expect("空列表"), Bounds::ZERO);
        assert_eq!(
            Bounds::of_ops(&twice, 64, 4).expect("合法"),
            Bounds::of_ops(&one, 64, 4)
                .expect("合法")
                .plus(Bounds::of_ops(&one, 64, 4).expect("合法"))
        );
        let error = Bounds::of_ops(&twice, 8, 4).expect_err("形状非法");
        assert!(error.to_string().contains("gemv_i4"), "{error}");
    }
}
