//! 布局：4 个构造子 + 正规形 + 契约校验（DESIGN §6）。
//!
//! 布局不做推导，只做三件事：把构造子实例化成正规形、判断两个正规形是否相等、不相等时
//! 给出"能重组（计费）"或"不可重组（报错）"的结论。

use crate::atom::DType;
use core::fmt;

/// 布局构造子闭集（DESIGN §6.1）。`Swz`/`Pad` 只能包裹 `Row`/`Col`。
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

impl LayoutSpec {
    /// 解析声明层写法：`row`、`col`、`swz<3,3,3>`、`pad(4)`。
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if text == "row" {
            return Ok(Self::Row);
        }
        if text == "col" {
            return Ok(Self::Col);
        }
        if let Some(rest) = text.strip_prefix("swz<") {
            let inner = rest
                .strip_suffix('>')
                .ok_or_else(|| format!("`{text}` 缺少 `>`"))?;
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            if parts.len() != 3 {
                return Err(format!("`{text}` 需要三个参数：swz<b,m,s>"));
            }
            let parse = |value: &str| {
                value
                    .parse::<u8>()
                    .map_err(|_| format!("`{value}` 不是合法的位宽"))
            };
            return Ok(Self::Swz {
                b: parse(parts[0])?,
                m: parse(parts[1])?,
                s: parse(parts[2])?,
            });
        }
        if let Some(rest) = text.strip_prefix("pad(") {
            let inner = rest
                .strip_suffix(')')
                .ok_or_else(|| format!("`{text}` 缺少 `)`"))?;
            let pad = inner
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("`{}` 不是合法的填充长度", inner.trim()))?;
            return Ok(Self::Pad(pad));
        }
        Err(format!(
            "`{text}` 不在布局闭集内（闭集：row | col | swz<b,m,s> | pad(p)）"
        ))
    }

    /// 渲染回声明层写法。
    pub fn as_str(&self) -> String {
        match self {
            Self::Row => "row".to_string(),
            Self::Col => "col".to_string(),
            Self::Swz { b, m, s } => format!("swz<{b},{m},{s}>"),
            Self::Pad(pad) => format!("pad({pad})"),
        }
    }

    /// 实例化成正规形；`d0`/`d1` 是该槽位的两级形状。
    pub fn instantiate(&self, d0: u32, d1: u32) -> LayoutNF {
        match self {
            Self::Row => LayoutNF {
                order: Order::RowMajor,
                shape: [d0, d1],
                stride: [d1.max(1), 1],
                swizzle: None,
                pad: 0,
            },
            Self::Col => LayoutNF {
                order: Order::ColMajor,
                shape: [d0, d1],
                stride: [1, d0.max(1)],
                swizzle: None,
                pad: 0,
            },
            Self::Swz { b, m, s } => {
                let base = Self::Row.instantiate(d0, d1);
                LayoutNF {
                    swizzle: Some((*b, *m, *s)),
                    ..base
                }
            }
            Self::Pad(pad) => {
                let base = Self::Row.instantiate(d0, d1);
                LayoutNF {
                    stride: [d1.saturating_add(*pad).max(1), 1],
                    pad: *pad,
                    ..base
                }
            }
        }
    }
}

impl fmt::Display for LayoutSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

/// 遍历次序。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// 行主序。
    RowMajor,
    /// 列主序。
    ColMajor,
}

impl Order {
    /// 渲染名。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowMajor => "row",
            Self::ColMajor => "col",
        }
    }
}

/// 布局正规形：`(order, shape, stride, swizzle, pad)` 五元组（DESIGN §6.2）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutNF {
    /// 遍历次序。
    pub order: Order,
    /// 两级形状。
    pub shape: [u32; 2],
    /// 两级步长。
    pub stride: [u32; 2],
    /// swizzle 三元组。
    pub swizzle: Option<(u8, u8, u8)>,
    /// 最后一维的填充元素数。
    pub pad: u32,
}

impl LayoutNF {
    /// 元素个数（不含填充）。
    pub fn elements(&self) -> u64 {
        u64::from(self.shape[0]) * u64::from(self.shape[1])
    }

    /// 占用字节数（含填充）。
    pub fn bytes(&self, dt: DType) -> u64 {
        let elements = u64::from(self.shape[0]) * u64::from(self.stride[0]);
        elements * u64::from(dt.size_bits()) / 8
    }

    /// 单行签名，供 `doctor` 打印。
    pub fn summary(&self) -> String {
        let swizzle = match self.swizzle {
            None => "-".to_string(),
            Some((b, m, s)) => format!("swz<{b},{m},{s}>"),
        };
        format!(
            "{}({:?} stride {:?} swz {swizzle} pad {})",
            self.order.as_str(),
            self.shape,
            self.stride,
            self.pad
        )
    }
}

impl fmt::Display for LayoutNF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// 契约校验结论。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// 模板给的正是片段要的。
    Exact,
    /// 同序但步长/swizzle/填充不同，可插入一次 `@repack`。
    Repack {
        /// 额外搬运字节数（读 + 写）。
        extra_bytes: u64,
        /// 说明。
        reason: String,
    },
    /// 不可重组。
    Incompatible {
        /// 说明。
        reason: String,
    },
}

impl Verdict {
    /// 是否可通过（含计费的重组）。
    pub fn is_ok(&self) -> bool {
        !matches!(self, Self::Incompatible { .. })
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => f.write_str("exact"),
            Self::Repack {
                extra_bytes,
                reason,
            } => {
                write!(f, "repack(+{extra_bytes}B: {reason})")
            }
            Self::Incompatible { reason } => write!(f, "incompatible({reason})"),
        }
    }
}

/// 模板声明 vs 片段要求的契约校验；`bank` 来自档案 `[mem]`。
pub fn check(template: &LayoutNF, fragment: &LayoutNF, bank: u32, dt: DType) -> Verdict {
    if let Some((b, _, _)) = fragment.swizzle
        && bank > 0
        && (1u32 << b) > bank
    {
        return Verdict::Incompatible {
            reason: format!("片段要求 swz<{b},…>，但该设备只有 {bank} 个 bank"),
        };
    }
    if template.order != fragment.order {
        return Verdict::Incompatible {
            reason: format!(
                "模板给 {}、片段要 {}；换序会破坏操作数约定，需改模板或换片段",
                template.order.as_str(),
                fragment.order.as_str()
            ),
        };
    }
    if template.stride == fragment.stride
        && template.swizzle == fragment.swizzle
        && template.pad == fragment.pad
    {
        return Verdict::Exact;
    }
    let elements = template.elements().max(fragment.elements());
    let element_bytes = u64::from(dt.size_bits().max(8)) / 8;
    Verdict::Repack {
        extra_bytes: elements * element_bytes * 2,
        reason: format!("{} → {}", template.summary(), fragment.summary()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_set_round_trip() {
        for text in ["row", "col", "swz<3,3,3>", "pad(4)"] {
            let spec = LayoutSpec::parse(text).unwrap();
            assert_eq!(spec.as_str(), text);
        }
        let error = LayoutSpec::parse("interleaved").unwrap_err();
        assert!(error.contains("不在布局闭集内"), "{error}");
    }

    #[test]
    fn row_and_col_have_swapped_strides() {
        let row = LayoutSpec::Row.instantiate(16, 64);
        let col = LayoutSpec::Col.instantiate(16, 64);
        assert_eq!(row.stride, [64, 1]);
        assert_eq!(col.stride, [1, 16]);
        assert_eq!(row.order, Order::RowMajor);
    }

    #[test]
    fn pad_widens_the_major_stride() {
        let padded = LayoutSpec::Pad(4).instantiate(16, 64);
        assert_eq!(padded.stride, [68, 1]);
        assert_eq!(padded.pad, 4);
        assert_eq!(padded.elements(), 16 * 64);
    }

    #[test]
    fn equal_normal_forms_are_exact() {
        let a = LayoutSpec::Row.instantiate(16, 64);
        let b = LayoutSpec::Row.instantiate(16, 64);
        assert_eq!(check(&a, &b, 32, DType::F32), Verdict::Exact);
    }

    #[test]
    fn same_order_different_stride_is_repack() {
        let a = LayoutSpec::Row.instantiate(16, 64);
        let b = LayoutSpec::Pad(4).instantiate(16, 64);
        match check(&a, &b, 32, DType::F32) {
            Verdict::Repack { extra_bytes, .. } => {
                assert_eq!(extra_bytes, 16 * 64 * 4 * 2);
            }
            other => panic!("期望 repack，实际 {other}"),
        }
    }

    #[test]
    fn order_change_is_incompatible() {
        let a = LayoutSpec::Row.instantiate(16, 64);
        let b = LayoutSpec::Col.instantiate(16, 64);
        match check(&a, &b, 32, DType::F32) {
            Verdict::Incompatible { reason } => assert!(reason.contains("换序")),
            other => panic!("期望 incompatible，实际 {other}"),
        }
    }

    #[test]
    fn swizzle_beyond_bank_count_is_incompatible() {
        let a = LayoutSpec::Row.instantiate(64, 64);
        let b = LayoutSpec::Swz { b: 6, m: 3, s: 3 }.instantiate(64, 64);
        assert!(!check(&a, &b, 32, DType::F32).is_ok());
    }

    #[test]
    fn bytes_include_padding() {
        let layout = LayoutSpec::Pad(4).instantiate(16, 64);
        assert_eq!(layout.bytes(DType::F32), 16 * 68 * 4);
    }
}
