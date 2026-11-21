//! 通信后端要实现的东西：[`Collective`]。
//!
//! 只有 `all_reduce` 一个方法：它是 TP/DP 的全部需求（`all_gather` / `reduce_scatter` 都能由它
//! 与切分拓扑组合出来）。后端实现不同，上层调用点不变。

use diagnostic::Result;

pub use group::Group;
pub use reduce::Reduce;
pub use remote::Remote;

/// 通信后端要实现的东西：对组内缓冲做一次集合操作。
pub trait Collective {
    /// 对 `buffer` 做原地归约；返回时表示"已完成"。
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buffer: &mut [f32]) -> Result<()>;
}

mod group {
    //! 参与一次通信的进程组：[`Group`]。
    //!
    //! 构造时要求**升序、去重、非空**：集合通信里"组内下标"必须稳定，否则归约顺序不固定，
    //! 浮点求和的最后几位会随启动顺序变化。

    use diagnostic::{Code, Error, Result};

    /// 参与一次通信的进程组。
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Group {
        ranks: Vec<u32>,
    }

    impl Group {
        /// 校验并构造。
        pub fn new(ranks: impl Into<Vec<u32>>) -> Result<Self> {
            let ranks = ranks.into();
            if ranks.is_empty() {
                return Err(Error::new(Code::Invalid, "通信组不能为空").with("shard::Group::new"));
            }
            for pair in ranks.windows(2) {
                if pair[0] >= pair[1] {
                    return Err(Error::new(
                        Code::Invalid,
                        format!("通信组必须升序且不重复：{} >= {}", pair[0], pair[1]),
                    )
                    .with("shard::Group::new"));
                }
            }
            Ok(Self { ranks })
        }

        /// 组内全部 rank。
        pub fn ranks(&self) -> &[u32] {
            &self.ranks
        }

        /// 组大小。
        pub fn size(&self) -> usize {
            self.ranks.len()
        }

        /// 组内下标（用于切片"我负责哪一段"）。
        pub fn index_of(&self, rank: u32) -> Option<usize> {
            self.ranks.iter().position(|item| *item == rank)
        }

        /// 是否包含某个 rank。
        pub fn contains(&self, rank: u32) -> bool {
            self.index_of(rank).is_some()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_group_must_be_sorted_and_unique() {
            let group = Group::new([0, 2, 4]).expect("升序");
            assert_eq!(group.ranks(), [0, 2, 4]);
            assert_eq!(group.size(), 3);
            assert_eq!(group.index_of(4), Some(2));
            assert!(!group.contains(1));
            assert_eq!(Group::new([]).unwrap_err().code(), Code::Invalid);
            assert_eq!(Group::new([2, 1]).unwrap_err().code(), Code::Invalid);
            assert_eq!(Group::new([1, 1]).unwrap_err().code(), Code::Invalid);
        }
    }
}

mod reduce {
    //! 归约种类闭集：[`Reduce`] 与它的算术。
    //!
    //! 归约的算术只有一份：[`Reduce::identity`]（单位元）与 [`Reduce::apply`]（二元运算）在这里定义，
    //! `fold` 与集合通信实现都从这两个派生。分开写会漂移——把 `Max` 的单位元写成 `f32::MIN`、
    //! 或在 `fold` 里对空输入返回单位元，都是只在全负输入或空缓冲上才露头的错。
    //!
    //! NaN 的口径写死为**传播**：`Max` / `Min` 用 `f32::max` / `f32::min` 会静默吃掉 NaN，
    //! 而"某一路算出 NaN，另一路把它抹掉"在梯度同步里是最难查的一类静默错误。

    /// 归约种类闭集。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Reduce {
        /// 求和（梯度同步的默认选择）。
        Sum,
        /// 取最大（张量并行里的数值稳定选项）。
        Max,
        /// 取最小。
        Min,
        /// 取乘积。
        Prod,
    }

    impl Reduce {
        /// 全部种类，供日志与测试遍历。
        pub const ALL: [Self; 4] = [Self::Sum, Self::Max, Self::Min, Self::Prod];

        /// 稳定的短名。
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Sum => "sum",
                Self::Max => "max",
                Self::Min => "min",
                Self::Prod => "prod",
            }
        }

        /// 单位元：与任何值 `apply` 之后都不改变它。
        ///
        /// `Max` / `Min` 的单位元是无穷而不是 `f32::MAX` / `f32::MIN`：后者会把"所有值都比它大"
        /// 这一正常情形误判成一个具体的有限数。
        pub fn identity(self) -> f32 {
            match self {
                Self::Sum => 0.0,
                Self::Max => f32::NEG_INFINITY,
                Self::Min => f32::INFINITY,
                Self::Prod => 1.0,
            }
        }

        /// 二元运算：满足结合律，且对 [`Reduce::identity`] 中立。
        pub fn apply(self, left: f32, right: f32) -> f32 {
            match self {
                Self::Sum => left + right,
                Self::Prod => left * right,
                Self::Max => extreme(left, right, true),
                Self::Min => extreme(left, right, false),
            }
        }

        /// 把一个非空切片折成一个值；空切片返回 `None`。
        ///
        /// 空输入不给单位元：`Max` 的空归约在数学上没有答案，返回 `-inf` 会让调用方以为
        /// "这一组确实取到了负无穷"，而不是"没有数据"。
        pub fn fold(self, values: &[f32]) -> Option<f32> {
            let (first, rest) = values.split_first()?;
            Some(
                rest.iter()
                    .fold(*first, |acc, value| self.apply(acc, *value)),
            )
        }
    }

    /// 极值归约：任一侧是 NaN 就传播 NaN，否则取 `max` / `min`。
    fn extreme(left: f32, right: f32, upper: bool) -> f32 {
        if left.is_nan() || right.is_nan() {
            return f32::NAN;
        }
        if upper {
            left.max(right)
        } else {
            left.min(right)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn every_kind_names_itself() {
            let names: Vec<&str> = Reduce::ALL.iter().map(|kind| kind.as_str()).collect();
            assert_eq!(names, ["sum", "max", "min", "prod"]);
        }

        /// 单位元必须中立：拿它去 `apply` 任何值都不能改变那个值。
        ///
        /// `Max` 的单位元写成 `f32::MIN` 时这条会在负值上失败——那正是最容易被漏掉的输入。
        #[test]
        fn the_identity_is_neutral_for_every_kind() {
            let samples = [
                -1.5e30,
                -3.0,
                -0.0,
                0.0,
                1.0e-30,
                7.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ];
            for kind in Reduce::ALL {
                for value in samples {
                    assert_eq!(
                        kind.apply(kind.identity(), value),
                        value,
                        "{kind:?} 的单位元吃掉了 {value}"
                    );
                    assert_eq!(
                        kind.apply(value, kind.identity()),
                        value,
                        "{kind:?} 的单位元吃掉了 {value}"
                    );
                }
            }
        }

        #[test]
        fn apply_is_the_expected_arithmetic() {
            assert_eq!(Reduce::Sum.apply(2.0, 3.0), 5.0);
            assert_eq!(Reduce::Prod.apply(2.0, 3.0), 6.0);
            assert_eq!(Reduce::Max.apply(2.0, 3.0), 3.0);
            assert_eq!(Reduce::Min.apply(2.0, 3.0), 2.0);
            // 负值上的极值：`0.0` 不能是任何一个极值的单位元。
            assert_eq!(Reduce::Max.apply(-9.0, -2.0), -2.0);
            assert_eq!(Reduce::Min.apply(-9.0, -2.0), -9.0);
        }

        /// NaN 必须传播，而不是被 `f32::max` / `f32::min` 静默吃掉。
        #[test]
        fn a_non_finite_value_is_never_silently_dropped() {
            assert!(Reduce::Max.apply(f32::NAN, 1.0).is_nan());
            assert!(Reduce::Min.apply(1.0, f32::NAN).is_nan());
            assert!(Reduce::Sum.apply(f32::NAN, 1.0).is_nan());
            assert!(Reduce::Prod.apply(1.0, f32::NAN).is_nan());
        }

        /// 空输入没有答案：不许拿单位元冒充"这一组算完了"。
        #[test]
        fn folding_nothing_is_not_a_value() {
            for kind in Reduce::ALL {
                assert_eq!(kind.fold(&[]), None, "{kind:?} 给空输入编了一个值");
            }
            assert_eq!(Reduce::Sum.fold(&[1.0]), Some(1.0));
            assert_eq!(Reduce::Max.fold(&[-4.0, -7.0]), Some(-4.0));
            assert_eq!(Reduce::Prod.fold(&[2.0, 3.0, 4.0]), Some(24.0));
        }
    }
}

mod remote {
    //! 远端集合通信占位：[`Remote`]。
    //!
    //! [`Collective`] 的进程内实现是 [`crate::Local`]；跨进程归约要走 NCCL / MPI / Gloo 等远端后端。
    //! 本刀只落占位骨架：不持网络 / 通信句柄，不接 FFI，不写 `unsafe`；`all_reduce` 入口统一返回
    //! [`Code::Unsupported`]。真实远端实现（组内握手、通信句柄池、拓扑绑定与心跳）随后续后端刀
    //! 填入本模块；上层调用点仍是同一个 [`Collective`] 端口。

    use super::{Collective, Group, Reduce};
    use diagnostic::{Code, Error, Result};

    /// 远端集合通信占位后端：与 [`crate::Local`] 同一 [`Collective`] 端口，未挂载真实通信栈。
    ///
    /// 不保存进程拓扑、通信句柄或 rank 镜像；构造是无状态占位，只证明第二后端能挂到同一契约上。
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct Remote;

    impl Remote {
        /// 构造占位远端后端；不绑定任何进程组、通信句柄或拓扑。
        pub fn new() -> Self {
            Self
        }
    }

    impl Collective for Remote {
        fn all_reduce(
            &mut self,
            _group: &Group,
            _reduce: Reduce,
            _buffer: &mut [f32],
        ) -> Result<()> {
            Err(Error::new(
                Code::Unsupported,
                "远端集合通信未挂载：跨进程 all-reduce 在占位适配上不可用",
            )
            .with("shard::Remote::all_reduce"))
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::{Collective, Group, Local, Parallel, Reduce, Remote, Topology};
        use diagnostic::Code;

        /// 占位远端后端拒绝任何集合操作；缓冲原样保留，不产生半个归约结果。
        #[test]
        fn all_reduce_is_unsupported_and_leaves_the_buffer_untouched() {
            let mut remote = Remote::new();
            let group = Group::new([0, 1]).expect("双进程组");
            let mut buffer = [1.0f32, 2.0, 3.0];

            let error = remote
                .all_reduce(&group, Reduce::Sum, &mut buffer)
                .expect_err("远端未挂载应拒绝归约");
            assert_eq!(error.code(), Code::Unsupported);
            assert_eq!(buffer, [1.0, 2.0, 3.0], "拒绝时不得改动调用方缓冲");
        }

        /// [`Local`] 与 [`Remote`] 能放进同一个 `Box<dyn Collective>`：端口契约不随后端切换。
        #[test]
        fn local_and_remote_share_the_collective_port() {
            let topology = Topology::new(2, 1, 1).expect("tp 2");
            let mut local: Box<dyn Collective> =
                Box::new(Local::new(topology, 2).expect("宽度非零"));
            let mut remote: Box<dyn Collective> = Box::new(Remote::new());

            let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
            let mut buffer = [0.0f32; 2];
            local
                .all_reduce(&group, Reduce::Sum, &mut buffer)
                .expect("进程内归约");
            assert_eq!(
                remote
                    .all_reduce(&group, Reduce::Sum, &mut buffer)
                    .unwrap_err()
                    .code(),
                Code::Unsupported
            );
        }
    }
}

#[cfg(test)]
#[path = "../tests/collective_tests.rs"]
mod tests;
