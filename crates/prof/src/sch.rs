//! 调度：4 个模板（cutile README §7）。

use crate::atom::Prim;
use core::fmt;
use diagnostic::{Code, Error};

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
    #[must_use]
    pub fn as_str(&self) -> String {
        self.to_string()
    }

    /// 该调度需要的原语（协商时逐条检查）。返回静态切片，不为每次协商分配 `Vec`。
    pub fn required_prims(&self) -> &'static [Prim] {
        match self {
            Self::Direct | Self::Pipe { .. } => &[],
            Self::WarpSpecialized { .. } => &[Prim::Bar, Prim::LoadAsync],
            Self::PingPong { .. } => &[Prim::Bar],
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
#[path = "../tests/spec/sch_tests.rs"]
mod tests;

pub mod layout {
    //! 布局：4 个构造子的闭集与声明式签名（cutile README §6）。
    use core::fmt;
    use diagnostic::{Code, Error};

    /// 布局构造子闭集（cutile README §6.1）。`Swz`/`Pad` 只能包裹 `Row`/`Col`。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LayoutSpec {
        /// 行主序紧凑。
        Row,
        /// 列主序紧凑。
        Col,
        /// 在基布局上做 XOR swizzle（`b` 个 bank 位、`m` 个行位、`s` 个段位）。
        Swz {
            /// bank 位宽。
            b: u8,
            /// 行位宽。
            m: u8,
            /// 段位宽。
            s: u8,
        },
        /// 在最后一维插入填充元素。
        Pad(u32),
    }

    /// 布局写法不合法：闭集外的取值一律是 `invalid`。
    fn invalid(detail: impl Into<String>) -> Error {
        Error::new(Code::Invalid, detail)
    }

    /// `swz<b,m,s>`：三个参数是位宽、块宽、步长，各一个 `u8`。
    fn swz(text: &str, rest: &str) -> diagnostic::Result<LayoutSpec> {
        let [b, m, s] = widths(text, closed(text, rest, '>')?)?;
        Ok(LayoutSpec::Swz { b, m, s })
    }

    /// 去掉收尾的分隔符；缺了就是声明写坏了。
    fn closed<'a>(text: &str, rest: &'a str, end: char) -> diagnostic::Result<&'a str> {
        rest.strip_suffix(end)
            .ok_or_else(|| invalid(format!("`{text}` 缺少 `{end}`")))
    }

    /// 逗号分隔的三个位宽（顺序：位宽、块宽、步长）。
    fn widths(text: &str, inner: &str) -> diagnostic::Result<[u8; 3]> {
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        let [b, m, s] = parts
            .try_into()
            .map_err(|_| invalid(format!("`{text}` 需要三个参数：swz<b,m,s>")))?;
        Ok([width(b)?, width(m)?, width(s)?])
    }

    /// 一个位宽：`u8` 十进制。
    fn width(value: &str) -> diagnostic::Result<u8> {
        value
            .parse::<u8>()
            .map_err(|_| invalid(format!("`{value}` 不是合法的位宽")))
    }

    /// `pad(p)`：括号里是填充长度（允许首尾空白）。
    fn pad(text: &str, rest: &str) -> diagnostic::Result<LayoutSpec> {
        let trimmed = closed(text, rest, ')')?.trim();
        trimmed
            .parse::<u32>()
            .map(LayoutSpec::Pad)
            .map_err(|_| invalid(format!("`{trimmed}` 不是合法的填充长度")))
    }

    impl LayoutSpec {
        /// 解析声明层写法：`row`、`col`、`swz<3,3,3>`、`pad(4)`。
        pub fn parse(text: &str) -> diagnostic::Result<Self> {
            let text = text.trim();
            match text {
                "row" => return Ok(Self::Row),
                "col" => return Ok(Self::Col),
                _ => {}
            }
            if let Some(rest) = text.strip_prefix("swz<") {
                return swz(text, rest);
            }
            if let Some(rest) = text.strip_prefix("pad(") {
                return pad(text, rest);
            }
            Err(invalid(format!(
                "`{text}` 不在布局闭集内（闭集：row | col | swz<b,m,s> | pad(p)）"
            )))
        }

        /// 渲染回声明层写法。
        #[must_use]
        pub fn as_str(&self) -> String {
            self.to_string()
        }
    }

    impl fmt::Display for LayoutSpec {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Row => f.write_str("row"),
                Self::Col => f.write_str("col"),
                Self::Swz { b, m, s } => write!(f, "swz<{b},{m},{s}>"),
                Self::Pad(pad) => write!(f, "pad({pad})"),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/spec/layout_tests.rs");
    }
}
