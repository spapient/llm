//! 确定性输入处理底座：带字节位置的扫描游标，与已解析签名序列的形状核对。
//!
//! - [`Cursor`]：逐字节文法（头部 JSON 之类）共用的最小骨架。位置既是前进下标、也是报错
//!   坐标，所以这里不缓存、不跳读：只回答「下一个字节是什么」「跳过空白」「吃一个必需字节」。
//!   文法不在这里（各层自己写），错误消息也由调用方给，本类型只补上位置与站点——同一段
//!   输入在两处被解析时，诊断格式必须一致。
//! - [`ffi_align`]：两侧已解析外部符号的名集与参数宽类序列核对；本层不解析 C／Rust 类型
//!   词法，词法映射与符号前缀由各消费方解析成 [`FfiSignature`] 后传入。
//!
//! # 例子
//!
//! ```
//! use llm_diagnostic::scan::Cursor;
//!
//! let mut cursor = Cursor::new("  {", "demo::scan");
//! cursor.skip_space();
//! cursor.eat(b'{').expect("有左花括号");
//! assert_eq!(cursor.pos(), 3);
//! assert!(cursor.eat(b'}').is_err(), "已到末尾");
//! ```

use crate::{Code, Error, Result};
use std::collections::{BTreeMap, BTreeSet};

/// 一块文本上的逐字节游标：`pos` 同时是报错位置。
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    site: &'static str,
}

impl<'a> Cursor<'a> {
    /// 在一段文本上开一个游标；`site` 会写进这里产生的每条错误的上下文。
    #[must_use]
    pub fn new(text: &'a str, site: &'static str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
            site,
        }
    }

    /// 全文的字节视图。
    #[must_use]
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// 当前下标。
    #[must_use]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 把下标移到 `pos`（调用方已经校验过的跳过）。
    pub fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// 前进 `count` 个字节。
    pub fn advance(&mut self, count: usize) {
        self.pos += count;
    }

    /// 下一个字节；已到末尾为 `None`。
    #[must_use]
    pub fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// 消费一个字节；已到末尾为 `None`。
    pub fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    /// 跳过 ASCII 空白（空格 / 制表 / 换行 / 回车）。
    pub fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// 吃一个必需的字节；不一致与已到末尾两种失败带不同文案。
    ///
    /// # Errors
    ///
    /// 下一字节不等于 `wanted`，或已到末尾时返回 [`Code::Invalid`]；两类失败
    /// 文案分别指明期望的字节与“已到末尾”。
    pub fn eat(&mut self, wanted: u8) -> Result<()> {
        let at = self.pos;
        match self.bump() {
            Some(byte) if byte == wanted => Ok(()),
            Some(_) => Err(self.fail_at(at, format!("期望 `{}`", wanted as char))),
            None => Err(self.fail_at(at, format!("期望 `{}`，但已到末尾", wanted as char))),
        }
    }

    /// 在当前下标处报错。
    pub fn fail(&self, message: impl Into<String>) -> Error {
        self.fail_at(self.pos, message)
    }

    /// 在指定下标处报错：消息前补位置，上下文补站点。
    pub fn fail_at(&self, position: usize, message: impl Into<String>) -> Error {
        let message = format!("位置 {position}：{}", message.into());
        Error::new(Code::Invalid, message).with(self.site)
    }
}

// ---- FFI 签名序列对齐 ---------------------------------------------------------

/// 外部参数的 ABI 宽类：只核指针 const 性与整数宽度，不核参数名与具体类型位宽。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FfiParam {
    /// 只读指针（`const T *` / `*const T`）。
    PtrConst,
    /// 可写指针（`T *` / `*mut T`）。
    PtrMut,
    /// 机器字整数（`size_t` / `usize`）。
    Size,
}

/// 一条外部符号：名字与按声明顺序排列的参数宽类序列。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FfiSignature {
    /// 链接符号名。
    pub name: String,
    /// 参数宽类序列。
    pub params: Vec<FfiParam>,
}

/// 核对 `expected` 符号集在两侧名集相等、共同名宽类序列一致。
///
/// `left` / `right` 是两侧各自解析出的签名；缺符号、多符号或宽类序列不一致均拒绝。
///
/// # Errors
///
/// 形状不一致返回 [`Code::Invalid`]；缺签名或出现未登记符号返回 [`Code::Missing`]，
/// 消息指明符号名与违例类型。
pub fn ffi_align(
    expected: &BTreeSet<&str>,
    left: &[FfiSignature],
    right: &[FfiSignature],
) -> Result<()> {
    let l = by_name(left);
    let r = by_name(right);
    check_expected(expected, &l, &r)?;
    check_registered(expected, &l, &r)
}

/// 核对登记符号在两侧都存在且宽类序列一致。
fn check_expected(
    expected: &BTreeSet<&str>,
    left: &BTreeMap<&str, &[FfiParam]>,
    right: &BTreeMap<&str, &[FfiParam]>,
) -> Result<()> {
    for name in expected.iter().copied() {
        let (lp, rp) = pair(name, left, right)?;
        if lp != rp {
            return Err(Error::new(
                Code::Invalid,
                format!("`{name}` 两侧参数形状不一致"),
            ));
        }
    }
    Ok(())
}

/// 取同一符号两侧的参数序列；任一侧缺签名按 [`Code::Missing`] 报错。
fn pair<'a>(
    name: &str,
    left: &'a BTreeMap<&'a str, &'a [FfiParam]>,
    right: &'a BTreeMap<&'a str, &'a [FfiParam]>,
) -> Result<(&'a [FfiParam], &'a [FfiParam])> {
    match (left.get(name), right.get(name)) {
        (None, _) => Err(Error::new(Code::Missing, format!("左侧缺签名 `{name}`"))),
        (_, None) => Err(Error::new(Code::Missing, format!("右侧缺签名 `{name}`"))),
        (Some(lp), Some(rp)) => Ok((*lp, *rp)),
    }
}

/// 核对两侧出现的符号都在登记集内，拒绝未登记的外部符号。
fn check_registered(
    expected: &BTreeSet<&str>,
    left: &BTreeMap<&str, &[FfiParam]>,
    right: &BTreeMap<&str, &[FfiParam]>,
) -> Result<()> {
    for name in left.keys().chain(right.keys()) {
        if !expected.contains(name) {
            return Err(Error::new(
                Code::Missing,
                format!("`{name}` 未登记为外部符号"),
            ));
        }
    }
    Ok(())
}

fn by_name(sigs: &[FfiSignature]) -> BTreeMap<&str, &[FfiParam]> {
    sigs.iter()
        .map(|sig| (sig.name.as_str(), sig.params.as_slice()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn sig(name: &str, params: &[FfiParam]) -> FfiSignature {
        FfiSignature {
            name: name.to_owned(),
            params: params.to_vec(),
        }
    }

    testkit::test_cases! {
    fn matching_signatures_pass() {
        let expected: BTreeSet<&str> = ["a", "b"].into_iter().collect();
        let left = [
            sig("a", &[FfiParam::PtrConst, FfiParam::Size]),
            sig("b", &[FfiParam::Size]),
        ];
        let right = [
            sig("a", &[FfiParam::PtrConst, FfiParam::Size]),
            sig("b", &[FfiParam::Size]),
        ];
        assert!(ffi_align(&expected, &left, &right).is_ok());
    }

    fn shape_mismatch_is_rejected() {
        let expected: BTreeSet<&str> = ["a"].into_iter().collect();
        let left = [sig("a", &[FfiParam::PtrConst])];
        let right = [sig("a", &[FfiParam::PtrMut])];
        let err = ffi_align(&expected, &left, &right).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
        assert!(err.to_string().contains("形状不一致"));
    }

    fn missing_on_either_side_is_rejected() {
        let expected: BTreeSet<&str> = ["a", "b"].into_iter().collect();
        let left = [sig("a", &[])];
        let right = [sig("b", &[])];
        let err = ffi_align(&expected, &left, &right).unwrap_err();
        assert_eq!(err.code(), Code::Missing);
    }

    fn unregistered_symbol_is_rejected() {
        let expected: BTreeSet<&str> = ["a"].into_iter().collect();
        let left = [sig("a", &[]), sig("z", &[])];
        let right = [sig("a", &[])];
        let err = ffi_align(&expected, &left, &right).unwrap_err();
        assert!(err.to_string().contains("未登记"));
    }
    }
}
