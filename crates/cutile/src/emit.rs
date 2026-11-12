//! ⑤ emit：把拼接结果包成最终源文件（DESIGN §10.2）。
//!
//! 这一层刻意只做"加头、加前导、拼正文"，不做任何重写——换 `emit` 语言时改的是片段，
//! 不是模板（DESIGN 风险 R6）。

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
    pub fn lines(&self) -> usize {
        self.text.lines().count()
    }

    /// 源文件建议的扩展名。
    pub fn ext(&self) -> &'static str {
        self.lang.ext()
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
