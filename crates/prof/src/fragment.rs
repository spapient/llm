//! 一段片段：原语名、它声明实现的那条签名、以及真正拼进模板的文本。

use crate::Prim;
use diagnostic::{Code, Error};
use std::collections::BTreeMap;

/// 签名声明的标记：片段第一行 `/* cutile-sign: <prim>/<impl> */` 里的 `cutile-sign:`。
pub const SIGN_MARKER: &str = prof::sign::MARKER;

/// 一段片段：文本，以及它声明实现的那条签名。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    source: String,
    body: String,
    implementation: String,
}

impl Fragment {
    /// 片段全文（含第一行的签名声明）。
    pub fn source(&self) -> &str {
        &self.source
    }

    /// 真正拼进模板的文本：声明行之后的部分。
    pub fn body(&self) -> &str {
        &self.body
    }

    /// 声明实现的 `impl` 名：档案里那条 `provides.impl`。
    pub fn implementation(&self) -> &str {
        &self.implementation
    }

    /// 解析第一行的签名声明。
    ///
    /// # Errors
    ///
    /// 第一行不是签名声明、声明里的原语与文件名不一致、或缺少 `impl` 名时返回 `invalid`。
    fn parse(prim: Prim, source: impl Into<String>) -> diagnostic::Result<Self> {
        let source = source.into();
        let (first, body) = split_declaration(&source, prim)?;
        let (declared_prim, implementation) = declaration_parts(first, prim)?;
        if Prim::parse(declared_prim.trim()) != Some(prim) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的签名声明写的是 `{}`：文件名即原语名，两者必须一致",
                    declared_prim.trim()
                ),
            ));
        }
        let implementation = implementation.trim();
        if implementation.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                format!("`{prim}` 的签名声明缺少 `impl` 名"),
            ));
        }
        // 先把两处都落成 `String`：它们借的是 `source`，随后 `source` 要整体移进结构体。
        let implementation = implementation.to_string();
        let body = body.to_string();
        Ok(Self {
            source,
            body,
            implementation,
        })
    }
}

/// 片段文本 → 「声明行」与「实现体」：只有声明没有实现的片段不是片段。
fn split_declaration(source: &str, prim: Prim) -> diagnostic::Result<(&str, &str)> {
    source.split_once('\n').ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("片段 `{prim}` 只有签名声明，没有实现"),
        )
    })
}

/// 声明行 → `<prim>/<impl>` 两半：注释外壳与 [`SIGN_MARKER`] 标记都要在这里对上。
fn declaration_parts(first: &str, prim: Prim) -> diagnostic::Result<(&str, &str)> {
    let declared = prof::sign::strip_marker(first).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!(
                "片段 `{prim}` 的第一行必须是 `/* {SIGN_MARKER} {prim}/<impl> */`，实际是 `{first}`"
            ),
        )
    })?;
    let (declared_prim, implementation) = prof::sign::split_slash(declared).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("`{prim}` 的签名声明 `{declared}` 不是 `<prim>/<impl>`"),
        )
    })?;
    Ok((declared_prim, implementation))
}

/// 某个 vendor 的片段集合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentSet {
    vendor: String,
    ext: String,
    fragments: BTreeMap<Prim, Vec<Fragment>>,
}

impl FragmentSet {
    /// 新建空集合。
    pub fn new(vendor: &str, ext: &str) -> Self {
        Self {
            vendor: vendor.to_string(),
            ext: ext.to_string(),
            fragments: BTreeMap::new(),
        }
    }

    /// 加入一段片段。
    ///
    /// # Errors
    ///
    /// 第一行不是签名声明、声明里的原语与文件名不一致、缺少 `impl` 名，或同一原语同名实现重复登记时
    /// 返回 `invalid`。
    pub fn insert(&mut self, prim: Prim, source: impl Into<String>) -> diagnostic::Result<()> {
        let fragment = Fragment::parse(prim, source)?;
        let list = self.fragments.entry(prim).or_default();
        if list
            .iter()
            .any(|existing| existing.implementation == fragment.implementation)
        {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 已登记 `{}` 实现：同一原语的同名片段不能放两份",
                    fragment.implementation
                ),
            ));
        }
        list.push(fragment);
        Ok(())
    }

    /// 取登记的首份片段正文（单片段设备的兼容入口）。
    pub fn get(&self, prim: Prim) -> Option<&str> {
        self.first(&prim).map(Fragment::body)
    }

    /// 取某条实现对应的片段正文；没有这条实现时返回 `None`。
    pub fn get_impl(&self, prim: Prim, implementation: &str) -> Option<&str> {
        self.fragment_impl(prim, implementation).map(Fragment::body)
    }

    /// 取登记的首份片段（含它的签名声明）。
    pub fn fragment(&self, prim: Prim) -> Option<&Fragment> {
        self.first(&prim)
    }

    /// 取某条实现对应的片段（含签名声明）。
    pub fn fragment_impl(&self, prim: Prim, implementation: &str) -> Option<&Fragment> {
        self.fragments
            .get(&prim)?
            .iter()
            .find(|fragment| fragment.implementation == implementation)
    }

    /// 片段声明实现的 `impl` 名；没有这段片段时 `None`。
    pub fn implementation(&self, prim: Prim) -> Option<&str> {
        self.first(&prim).map(Fragment::implementation)
    }

    /// 遍历某原语登记的全部片段（按登记顺序）。
    pub fn fragments_of(&self, prim: Prim) -> &[Fragment] {
        self.fragments
            .get(&prim)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// 设备名。
    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// 片段所在的扩展名。
    pub fn ext(&self) -> &str {
        &self.ext
    }

    /// 片段总数（同一原语的多条实现分别计数）。
    pub fn len(&self) -> usize {
        self.fragments.values().map(Vec::len).sum()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.fragments.values().all(Vec::is_empty)
    }

    /// 取某原语登记的首份片段。
    fn first(&self, prim: &Prim) -> Option<&Fragment> {
        self.fragments.get(prim)?.first()
    }
}
