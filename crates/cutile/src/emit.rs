//! ⑤ emit：把拼接结果包成最终源文件（README §10.2）。
//!
//! 这一层刻意只做"加头、加前导、拼正文"，不做任何重写——换 `emit` 语言时改的是片段，
//! 不是模板（cutile README 风险 R6）。

use crate::SCHEMA_VER;
use crate::decl::{OpDecl, TileCombo, Variant};
use crate::prof::{Emit, Profile};

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
///
/// 行数是 P0 里唯一能自动读出的“原语粒度”指标：超过上限不是“代码写得长”，而是这个算子的原语
/// 切得太细，模板里开始堆特化代码。判定与数字同源（[`Fragment::within_limit`] 就是拿
/// [`Fragment::lines`] 与 [`FRAGMENT_LIMIT`] 比），前端只负责选词。
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
    let mut text = String::new();
    text.push_str(&format!(
        "// cutile schema v{SCHEMA_VER} | op={} | skeleton={} | device={}({}) | variant={} | schedule={} | tile=[{}]\n",
        op.name,
        op.skeleton,
        profile.meta.vendor,
        profile.meta.id,
        variant.name,
        variant.schedule,
        combo.summary()
    ));
    text.push_str(&format!(
        "// emit={} tool={} {} | 由声明层 + 档案层 + 片段层生成，不要手改\n",
        profile.meta.emit, profile.meta.tool, profile.meta.tool_ver
    ));
    if !profile.preamble.is_empty() {
        text.push_str(&profile.preamble);
        if !profile.preamble.ends_with('\n') {
            text.push('\n');
        }
    }
    text.push('\n');
    text.push_str(body.trim_start());
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Source {
        lang: profile.meta.emit,
        text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §16 的片段判据是“同一对数与两个门槛比”：上限管成败，内部目标只是余量。
    #[test]
    fn the_fragment_verdict_reads_the_p0_limits() {
        assert_eq!(FRAGMENT_LIMIT, 400, "README §16 的上限");
        assert_eq!(FRAGMENT_GOAL, 240, "README §16 的内部目标");
        assert!(Fragment::new(FRAGMENT_LIMIT).within_limit());
        assert!(!Fragment::new(FRAGMENT_LIMIT + 1).within_limit());
        assert!(Fragment::new(FRAGMENT_GOAL).within_goal());
        assert!(
            !Fragment::new(FRAGMENT_GOAL + 1).within_goal(),
            "超出目标是余量不足，不是失败"
        );
    }

    #[test]
    fn fnv_is_stable_and_distinguishes_inputs() {
        assert_eq!(fnv1a("abc"), fnv1a("abc"));
        assert_ne!(fnv1a("abc"), fnv1a("abd"));
        assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn assemble_puts_provenance_in_the_header() {
        let profile = Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案");
        let declaration = &op::builtin_named("gemv_i4").expect("内置");
        let op = OpDecl::parse(declaration.declaration(), declaration.template()).expect("声明");
        let variant = &op.variants[0];
        let combo = crate::decl::expand(&op).expect("展开").remove(0);
        let source = assemble(&profile, &op, variant, &combo, "int x;\n");
        assert!(source.text.contains("op=gemv_i4"));
        assert!(source.text.contains("device=cpu(portable)"));
        assert!(source.text.contains("variant=pipe2"));
        assert!(
            source.text.contains("#include <stdint.h>"),
            "前导声明要带进来"
        );
        assert!(source.text.ends_with("int x;\n"));
        assert_eq!(source.lang, Emit::C);
    }
}
