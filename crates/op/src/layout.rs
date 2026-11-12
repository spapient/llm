//! 布局：4 个构造子的闭集与声明式签名（DESIGN §6）。
//!
//! 布局是**声明层的词汇表**：`[layouts]` 的取值就是这里的闭集，声明解析（[`crate::decl`]）
//! 要用它，因此事实源在本 crate，机制层（`cutile`）只消费。
//!
//! 布局不做推导，只给出一份稳定的声明式签名：缓存指纹与 `doctor` 都用它。声明式就是布局
//! 在构建期的身份——模板里 `{{w}}` 插的是它，而不是某个具体形状下的步长。
//!
//! 正规形（`(order, shape, stride, swizzle, pad)`）需要槽位的两级形状，而构建期没有形状：
//! tile 是轴取值（`GMAX=32`），不是 `w[n, m/2]` 这样的槽位形状。用编出来的形状去实例化，
//! 只能让缓存键和诊断看起来覆盖了形状，实际上描述了一次没有发生的编译。因此 P0 只保留闭集
//! 与声明式；等有真实形状（以及片段声明的布局）时再加回正规形与判定。
//!
//! 契约判定（模板给 vs 片段要）也不在 P0：片段层至今没有声明自己要求的布局（CPU 片段只收
//! 指针），没有第二个操作数可比较。设备侧唯一的真实约束是 bank 数，由
//! `cutile::space::check_alignment` 在构建期执行。

use core::fmt;
use diagnostic::{Code, Error};

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

/// 布局写法不合法：闭集外的取值一律是 `invalid`。
fn invalid(detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, detail)
}

impl LayoutSpec {
    /// 解析声明层写法：`row`、`col`、`swz<3,3,3>`、`pad(4)`。
    pub fn parse(text: &str) -> diagnostic::Result<Self> {
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
                .ok_or_else(|| invalid(format!("`{text}` 缺少 `>`")))?;
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            if parts.len() != 3 {
                return Err(invalid(format!("`{text}` 需要三个参数：swz<b,m,s>")));
            }
            let parse = |value: &str| {
                value
                    .parse::<u8>()
                    .map_err(|_| invalid(format!("`{value}` 不是合法的位宽")))
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
                .ok_or_else(|| invalid(format!("`{text}` 缺少 `)`")))?;
            let pad = inner
                .trim()
                .parse::<u32>()
                .map_err(|_| invalid(format!("`{}` 不是合法的填充长度", inner.trim())))?;
            return Ok(Self::Pad(pad));
        }
        Err(invalid(format!(
            "`{text}` 不在布局闭集内（闭集：row | col | swz<b,m,s> | pad(p)）"
        )))
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
}

impl fmt::Display for LayoutSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
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
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("不在布局闭集内"), "{error}");
    }
}
