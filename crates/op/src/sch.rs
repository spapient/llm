//! 调度：4 个模板（cutile README §7）。
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
        let Some((name, args)) = split_args(text) else {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{text}` 不在调度闭集内（闭集：direct | pipe<N> | ws<prod,cons> | pingpong<N>）"
                ),
            ));
        };
        let numbers = parse_numbers(&args)?;
        shape(text, name, &numbers)
    }

    /// 渲染回声明层写法。
    ///
    /// 只有这一处写对照表：`Display` 直接写进调用方的缓冲，不在中间先建一个 `String`。
    #[must_use]
    pub fn as_str(&self) -> String {
        self.to_string()
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
        match self {
            Self::Direct => f.write_str("direct"),
            Self::Pipe { stages } => write!(f, "pipe<{stages}>"),
            Self::WarpSpecialized { producer, consumer } => {
                write!(f, "ws<{producer},{consumer}>")
            }
            Self::PingPong { groups } => write!(f, "pingpong<{groups}>"),
        }
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

/// 按模板名与参数构造调度；参数个数 / 下限不合法时报同一个闭集口径错误。
fn shape(text: &str, name: &str, numbers: &[u32]) -> diagnostic::Result<Schedule> {
    match (name, numbers) {
        ("pipe", [stages]) => two(text, *stages).map(|stages| Schedule::Pipe { stages }),
        ("pingpong", [groups]) => two(text, *groups).map(|groups| Schedule::PingPong { groups }),
        ("ws", [producer, consumer]) => Ok(Schedule::WarpSpecialized {
            producer: *producer,
            consumer: *consumer,
        }),
        _ => Err(bad(text)),
    }
}

/// `pipe` / `pingpong` 的下限：一级等于没分阶段，按参数不合法处理。
fn two(text: &str, count: u32) -> diagnostic::Result<u32> {
    if count >= 2 {
        Ok(count)
    } else {
        Err(bad(text))
    }
}

/// 参数个数或下限不合法时的唯一口径。
fn bad(text: &str) -> Error {
    Error::new(
        Code::Invalid,
        format!("`{text}` 的参数不合法（pipe<N≥2> | ws<prod,cons> | pingpong<N≥2>）"),
    )
}

/// 流水掩盖的阈值（cutile README §12.2 判据 1）：未掩盖的非 MMA 时间不得超过 20%。
pub const UNMASKED_LIMIT: f64 = 0.20;

/// 一个变体的掩盖账：`mma_share`、调度级数与由此推得的未掩盖占比。
///
/// 三个数一起走，因为判据（[`Self::within_limit`]）与 roofline 的「未掩盖 x%」必须是同一组值：
/// 报告里写 19%、判定却按另一个数，等于没有判定。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Masking {
    /// 声明的 MMA 时间占比。
    pub mma_share: f64,
    /// 调度级数（`direct` 为 1）。
    pub stages: u32,
    /// 未掩盖的非 MMA 占比。
    pub unmasked: f64,
}

impl Masking {
    /// 按声明推账。
    pub fn of(mma_share: f64, schedule: &Schedule) -> Self {
        let stages = schedule.stages();
        Self {
            mma_share,
            stages,
            unmasked: unmasked_share(mma_share, stages),
        }
    }

    /// 是否声称用 MMA：占比为 0 的变体没有可掩盖的对象，判据对它不适用。
    pub fn claims_mma(&self) -> bool {
        self.mma_share > 0.0
    }

    /// 未掩盖占比是否在阈值内。
    pub fn within_limit(&self) -> bool {
        self.unmasked <= UNMASKED_LIMIT
    }
}

/// 一句话结论：阈值与取值并排，判定只有一个（cutile README §12.2 判据 1）。
impl fmt::Display for Masking {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "未掩盖 {:.0}%（阈值 {:.0}%，",
            self.unmasked * 100.0,
            UNMASKED_LIMIT * 100.0
        )?;
        if !self.claims_mma() {
            f.write_str("不适用：该变体不声称 MMA）")
        } else if self.within_limit() {
            f.write_str("达标）")
        } else {
            f.write_str("未达标：降级兜底）")
        }
    }
}

/// 未掩盖的非 MMA 占比：`(1 - mma_share) / stages`。级数越多，同一条非 MMA 用量被掩盖得越多。
pub fn unmasked_share(mma_share: f64, stages: u32) -> f64 {
    let stages = f64::from(stages.max(1));
    ((1.0 - mma_share) / stages).clamp(0.0, 1.0)
}

/// 降级链**首选**变体的掩盖判据（cutile README §12.2 判据 1）。
///
/// 只判首选：链上其余变体存在的理由就是"设备缺原语时仍能跑"，用质量判据把它们判死等于取消
/// 显式降级——它们的账照样出现在 `doctor`/`bench` 的计划行里（「未掩盖 x% > 20%，兜底」）。
/// 首选不达标则是声明写错了：要么加级数（S3/S4），要么承认这条路径根本不走 MMA。
pub fn check_masking(variant: &str, masking: Masking) -> diagnostic::Result<()> {
    if !masking.claims_mma() || masking.within_limit() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "首选变体 `{variant}` 的未掩盖占比 {:.0}%（mma_share {}，{} 级调度）超过 {:.0}%：\
             加大调度级数（S3/S4）或提高 `mma_share`",
            masking.unmasked * 100.0,
            masking.mma_share,
            masking.stages,
            UNMASKED_LIMIT * 100.0
        ),
    ))
}

/// 降级链的收益必须单调不增（cutile README §12.2 判据 3）。
///
/// 只读名字与占比，所以名字泛化到 `AsRef<str>`：装载期能直接传借用的变体名，
/// 不为了一条判据先拷一份 `String`。
pub fn check_monotonic<S: AsRef<str>>(variants: &[(S, f64)]) -> diagnostic::Result<()> {
    for pair in variants.windows(2) {
        let (first, second) = (&pair[0], &pair[1]);
        if second.1 > first.1 {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "降级链上 `{}` 的 mma_share {} 大于 `{}` 的 {}：`extra_needs` 写反了",
                    second.0.as_ref(),
                    second.1,
                    first.0.as_ref(),
                    first.1
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
    fn unmasked_share_shrinks_with_stages() {
        let one = unmasked_share(0.5, 1);
        let two = unmasked_share(0.5, 2);
        assert!(two < one, "级数越多，越能掩盖");
        assert!((one - 0.5).abs() < 1e-12);
    }

    #[test]
    fn the_preferred_variant_must_hide_its_non_mma_work() {
        let pipe = Schedule::parse("pipe<2>").unwrap();
        assert!(check_masking("pipe2", Masking::of(0.62, &pipe)).is_ok());
        let error = check_masking("direct", Masking::of(0.35, &Schedule::Direct)).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("首选变体 `direct`"), "{error}");
        assert!(error.to_string().contains("65%"), "{error}");
    }

    #[test]
    fn a_variant_without_mma_has_nothing_to_hide() {
        // 占比为 0 的变体全部时间都是非 MMA：判据不适用，不该拦住一条 SIMT 路径。
        let masking = Masking::of(0.0, &Schedule::Direct);
        assert!(!masking.claims_mma());
        assert!(!masking.within_limit());
        assert!(check_masking("scalar", masking).is_ok());
    }

    #[test]
    fn the_limit_is_inclusive() {
        // (1 - 0.6) / 2 = 0.20：正好压线算通过，判定必须与报告里那个百分比同一个数。
        let masking = Masking::of(0.6, &Schedule::parse("pipe<2>").unwrap());
        assert!((masking.unmasked - UNMASKED_LIMIT).abs() < 1e-12);
        assert!(masking.within_limit());
        assert!(check_masking("pipe2", masking).is_ok());
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
