//! 职责：⑤ emit：把拼接结果包成最终源文件（README §10.2）。

use crate::SCHEMA_VER;
use crate::decl::{OpDecl, TileCombo, Variant};
use crate::prof::{Emit, Profile};
use core::fmt::Write as _;

/// 一次 emit 的产物：语言 + 文本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    /// 输出语言。
    pub lang: Emit,
    /// 源文本。
    pub text: String,
}

impl Source {
    /// 行数（统计用，不含尾部空行）。
    #[must_use]
    pub fn lines(&self) -> usize {
        self.text.lines().count()
    }

    /// 源文件建议的扩展名。
    #[must_use]
    pub fn ext(&self) -> &'static str {
        self.lang.ext()
    }
}

/// README §16 给片段定的行数上限：超过它说明原语粒度错，先合并原语。
pub const FRAGMENT_LIMIT: usize = 400;

/// §16 给片段定的内部目标（行数）：达标余量，不是另一条判据。
pub const FRAGMENT_GOAL: usize = 240;

/// 一份生成的片段：行数与 §16 的口径。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fragment {
    /// 行数。
    pub lines: usize,
}

impl Fragment {
    /// 以行数构造（[`Source::lines`] 是唯一数据源）。
    pub const fn new(lines: usize) -> Self {
        Self { lines }
    }

    /// §16 的上限。
    pub const fn limit(&self) -> usize {
        FRAGMENT_LIMIT
    }

    /// §16 的内部目标。
    pub const fn goal(&self) -> usize {
        FRAGMENT_GOAL
    }

    /// 是否在上限内：§16 “片段 ≤ 400 行”的结论。
    #[must_use]
    pub const fn within_limit(&self) -> bool {
        self.lines <= FRAGMENT_LIMIT
    }

    /// 是否达到内部目标（与上限分开：目标未达不算失败）。
    #[must_use]
    pub const fn within_goal(&self) -> bool {
        self.lines <= FRAGMENT_GOAL
    }
}

/// FNV-1a 64：跨进程稳定的内容哈希（缓存指纹用，不使用 `DefaultHasher`）。
pub fn fnv1a(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// 组装最终源文件。
pub fn assemble(
    profile: &Profile,
    op: &OpDecl,
    variant: &Variant,
    combo: &TileCombo,
    body: &str,
) -> Source {
    // 直接写入 `text`：`format!` 会先分配再拷贝，成文路径上每次 emit 白搭两次分配。
    let mut text = String::new();
    let _ = writeln!(
        text,
        "// cutile schema v{SCHEMA_VER} | op={} | skeleton={} | device={}({}) | variant={} | schedule={} | tile=[{}]",
        op.name,
        op.skeleton,
        profile.meta.vendor,
        profile.meta.id,
        variant.name,
        variant.schedule,
        combo.summary()
    );
    let _ = writeln!(
        text,
        "// emit={} tools={} {} | 由声明层 + 档案层 + 片段层生成，不要手改",
        profile.meta.emit, profile.meta.tool, profile.meta.tool_ver
    );
    write_preamble(&mut text, profile);
    text.push('\n');
    write_float4_typedef(&mut text, profile, body);
    text.push_str(body.trim_start());
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Source {
        lang: profile.meta.emit,
        text,
    }
}

/// 档案 preamble 的换行与空值规则只在这里解释一次。
fn write_preamble(text: &mut String, profile: &Profile) {
    if profile.preamble.is_empty() {
        return;
    }
    text.push_str(&profile.preamble);
    if !profile.preamble.ends_with('\n') {
        text.push('\n');
    }
}

/// C 向量定义必须在档案 preamble 之后、内核正文之前出现且只出现一次。
fn write_float4_typedef(text: &mut String, profile: &Profile, body: &str) {
    let typedef = crate::printer::C_FLOAT4_TYPEDEF;
    let required = profile.meta.emit == Emit::C && crate::printer::has_float4(body);
    if required && !body.contains(typedef) {
        text.push_str(typedef);
    }
}

#[cfg(test)]
#[path = "../tests/emit_tests.rs"]
mod tests;
