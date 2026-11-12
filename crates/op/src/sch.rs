//! 调度：4 个模板（DESIGN §7）。
//!
//! 与 [`crate::layout`] 一样，这是**声明层的词汇表**：`[[variants]]` 的 `schedule` 取值就是
//! 这里的闭集，声明解析（[`crate::decl`]）要用它，因此事实源在本 crate，机制层只消费。
//!
//! 调度只产出模板文本里的骨架槽位，不产出设备指令；设备差异不进入这一层，S1–S4 才能
//! 在八台设备上成立。

use core::fmt;
use diagnostic::{Code, Error};
use prof::atom::Prim;

/// 调度模板闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// 单级、无重叠。
    Direct,
    /// N 级预取 + 双缓冲。
    Pipe {
        /// 预取级数。
        stages: u32,
    },
    /// warp 专门化：搬运 / 计算分工。
    WarpSpecialized {
        /// 生产者组数。
        producer: u32,
        /// 消费者组数。
        consumer: u32,
    },
    /// 两组交替，用另一组的非 MMA 时间填自己的 MMA 空档。
    PingPong {
        /// 组数（P0 固定 2）。
        groups: u32,
    },
}

impl Schedule {
    /// 解析声明层写法：`direct`、`pipe<3>`、`ws<2,2>`、`pingpong<2>`。
    pub fn parse(text: &str) -> diagnostic::Result<Self> {
        let text = text.trim();
        if text == "direct" {
            return Ok(Self::Direct);
        }
        if let Some((name, args)) = split_args(text) {
            let numbers = parse_numbers(&args)?;
            return match (name, numbers.as_slice()) {
                ("pipe", [stages]) if *stages >= 2 => Ok(Self::Pipe { stages: *stages }),
                ("pingpong", [groups]) if *groups >= 2 => Ok(Self::PingPong { groups: *groups }),
                ("ws", [producer, consumer]) => Ok(Self::WarpSpecialized {
                    producer: *producer,
                    consumer: *consumer,
                }),
                _ => Err(Error::new(
                    Code::Invalid,
                    format!("`{text}` 的参数不合法（pipe<N≥2> | ws<prod,cons> | pingpong<N≥2>）"),
                )),
            };
        }
        Err(Error::new(
            Code::Invalid,
            format!(
                "`{text}` 不在调度闭集内（闭集：direct | pipe<N> | ws<prod,cons> | pingpong<N>）"
            ),
        ))
    }

    /// 渲染回声明层写法。
    pub fn as_str(&self) -> String {
        match self {
            Self::Direct => "direct".to_string(),
            Self::Pipe { stages } => format!("pipe<{stages}>"),
            Self::WarpSpecialized { producer, consumer } => format!("ws<{producer},{consumer}>"),
            Self::PingPong { groups } => format!("pingpong<{groups}>"),
        }
    }

    /// 该调度需要的原语（协商时逐条检查）。
    pub fn required_prims(&self) -> Vec<Prim> {
        match self {
            Self::Direct => Vec::new(),
            Self::Pipe { .. } => vec![Prim::Bar],
            Self::WarpSpecialized { .. } => vec![Prim::Bar, Prim::LoadAsync],
            Self::PingPong { .. } => vec![Prim::Bar],
        }
    }

    /// 是否用到异步搬运。
    pub fn uses_async(&self) -> bool {
        matches!(self, Self::Pipe { .. } | Self::WarpSpecialized { .. })
    }

    /// 预取级数（`direct` 为 1）。
    pub fn stages(&self) -> u32 {
        match self {
            Self::Direct => 1,
            Self::Pipe { stages } => *stages,
            Self::WarpSpecialized { .. } => 2,
            Self::PingPong { groups } => *groups,
        }
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

fn split_args(text: &str) -> Option<(&str, String)> {
    let open = text.find('<')?;
    let close = text.rfind('>')?;
    if open >= close {
        return None;
    }
    Some((text[..open].trim(), text[open + 1..close].to_string()))
}

fn parse_numbers(args: &str) -> diagnostic::Result<Vec<u32>> {
    args.split(',')
        .map(|part| {
            part.trim().parse::<u32>().map_err(|_| {
                Error::new(
                    Code::Invalid,
                    format!("`{}` 不是合法的调度参数", part.trim()),
                )
            })
        })
        .collect()
}

/// 降级链的收益必须单调不增（DESIGN §12.2 判据 3）。
pub fn check_monotonic(variants: &[(String, f64)]) -> diagnostic::Result<()> {
    for pair in variants.windows(2) {
        let (first, second) = (&pair[0], &pair[1]);
        if second.1 > first.1 {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "降级链上 `{}` 的 mma_share {} 大于 `{}` 的 {}：`extra_needs` 写反了",
                    second.0, second.1, first.0, first.1
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_set_round_trip() {
        for text in ["direct", "pipe<3>", "ws<2,2>", "pingpong<2>"] {
            assert_eq!(Schedule::parse(text).unwrap().as_str(), text);
        }
    }

    #[test]
    fn closed_set_rejects_unknown() {
        let error = Schedule::parse("rs").unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("不在调度闭集内"), "{error}");
        assert!(Schedule::parse("pipe<1>").is_err(), "单级预取没有意义");
    }

    #[test]
    fn required_prims_follow_the_template() {
        assert!(
            Schedule::parse("direct")
                .unwrap()
                .required_prims()
                .is_empty()
        );
        assert_eq!(
            Schedule::parse("pipe<3>").unwrap().required_prims(),
            vec![Prim::Bar]
        );
        assert!(
            Schedule::parse("ws<2,2>")
                .unwrap()
                .required_prims()
                .contains(&Prim::LoadAsync)
        );
    }

    #[test]
    fn monotonic_chain_is_accepted() {
        let chain = vec![
            ("pingpong<2>".to_string(), 0.82),
            ("pipe<3>".to_string(), 0.74),
            ("pipe<2>".to_string(), 0.61),
            ("direct".to_string(), 0.40),
        ];
        assert!(check_monotonic(&chain).is_ok());
    }

    #[test]
    fn reversed_chain_is_rejected() {
        let chain = vec![
            ("pipe<3>".to_string(), 0.50),
            ("pingpong<2>".to_string(), 0.90),
        ];
        let error = check_monotonic(&chain).unwrap_err();
        assert!(error.to_string().contains("写反了"), "{error}");
    }
}
