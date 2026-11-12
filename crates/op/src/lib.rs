//! 声明层：算子声明（`ops/<name>.toml`）、模板（`ops/<name>.k`）与它们的解析。
//!
//! 这一层是**内容**而不是机制：写算子的人只改本 crate 的 `ops/` 与 `src/<name>.rs`，`cutile` 负责协商能力、
//! 拼接片段、emit 与对拍。因此新增算子不必改 `cutile` 的任何一行代码，领域数据也不会
//! 与机制代码混在同一个 crate 里。
//!
//! 两段文本的语义（骨架 / 轴 / 需求 / 布局 / 降级链，模板里的 `{{插值}}` 与 `@原语`）在这里
//! 被解析成[`decl::OpDecl`]，词汇表（[`layout`]、[`sch`]、[`tpl`]、[`sig`]）也在这里：机制层只消费
//! 解析结果，不再看 TOML。模板的入口签名是 ABI 的唯一事实源：实参顺序与输出缓冲名都从它派生。
//! 闭集的内容与书写规则见 [`DESIGN.md`](../../cutile/DESIGN.md) §4。
//!
//! 除了声明与模板，算子语义也在本 crate：求和表达式、确定性样本与标量参考（[`mod@model`]）。它们是
//! 第二份实现，与声明、模板共同构成"正确性来自双源"（DESIGN §11.1）；`cutile` 只提供对拍口径
//! 与执行环境，不认识任何算子名。
//!
//! 代价（`flops` / `bytes`）不在本 crate 里手写：算子只声明 `EXPR`（如 `nm,m,ng,n->n`）与每个操作数
//! 的位宽/方向，由 [`einsum`] 推出；[`model::Cost`] 只是它的再导出（`docs/adr/0007`）。这样新增算子
//! 不会引入第二条流量公式，roofline 的分子也只有一处来源。
//!
//! # 例子
//!
//! ```
//! let ops = llm_op::builtin();
//! assert_eq!(ops[0].name(), "gemv_i4");
//! assert!(ops[0].declaration().contains("[op]"));
//! assert!(!ops[0].template().is_empty());
//! ```

#![deny(unsafe_code)]

pub mod decl;
pub mod gemv_i4;
pub mod layout;
pub mod model;
pub mod sch;
pub mod sig;
pub mod tpl;

use diagnostic::{Code, Error};
use std::path::{Path, PathBuf};

use model::Model;

/// 算子语义注册表：声明层有的算子，这里必须有它的代价、样本与标量参考。
///
/// 新增算子 = `ops/<name>.{toml,k}` + `BUILTIN` 一行 + 本函数一行；`cutile` 一侧不动。
pub fn model(name: &str) -> Option<Model> {
    match name {
        "gemv_i4" => Some(gemv_i4::MODEL),
        _ => None,
    }
}

/// 编译期内嵌的声明层：`(名字, 声明, 模板)`。
///
/// 新增算子要在这里补一行，否则发布出去的二进制里没有它；`builtin_matches_disk` 测试会把
/// "磁盘上有、内嵌里没有"直接判为失败。
pub const BUILTIN: [(&str, &str, &str); 1] = [(
    "gemv_i4",
    include_str!("../ops/gemv_i4.toml"),
    include_str!("../ops/gemv_i4.k"),
)];

/// 一份算子声明：`<name>.toml` 的正文与同名 `<name>.k` 模板。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Op {
    name: String,
    declaration: String,
    template: String,
}

impl Op {
    /// 由三段文本构造；名字取 `<name>.toml` 的文件名。
    pub fn new(
        name: impl Into<String>,
        declaration: impl Into<String>,
        template: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            declaration: declaration.into(),
            template: template.into(),
        }
    }

    /// 算子名（与 `<name>.toml` 的文件名一致）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 声明正文：`[op]` / `[axes]` / `[needs]` / `[layouts]` / `[[variants]]`。
    pub fn declaration(&self) -> &str {
        &self.declaration
    }

    /// 模板正文：C 风格文本 + `{{插值}}` + `@原语` 槽位。
    pub fn template(&self) -> &str {
        &self.template
    }
}

/// 内嵌声明，顺序与 [`BUILTIN`] 一致。
pub fn builtin() -> Vec<Op> {
    BUILTIN
        .iter()
        .map(|(name, declaration, template)| Op::new(*name, *declaration, *template))
        .collect()
}

/// 声明层目录：本 crate 根下的 `ops/`，即磁盘上的事实源。
pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ops")
}

/// 从目录装载声明：`<name>.toml` 必须与同名 `<name>.k` 成对，结果按名字排序。
///
/// # Errors
///
/// 目录不可读、文件名不是合法 UTF-8、或缺同名模板时返回带路径的说明。
pub fn load(dir: &Path) -> diagnostic::Result<Vec<Op>> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| Error::new(Code::Io, format!("读取 {} 失败：{error}", dir.display())))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect();
    paths.sort();

    let mut ops = Vec::new();
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) != Some("toml") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!("`{}` 的文件名不是合法 UTF-8", path.display()),
                )
            })?;
        let template_path = path.with_extension("k");
        let template = std::fs::read_to_string(&template_path).map_err(|_| {
            Error::new(
                Code::Missing,
                format!(
                    "`{}` 缺少同名模板 `{}`",
                    path.display(),
                    template_path.display()
                ),
            )
        })?;
        let declaration = read(&path)?;
        ops.push(Op::new(name, declaration, template));
    }
    Ok(ops)
}

fn read(path: &Path) -> diagnostic::Result<String> {
    std::fs::read_to_string(path)
        .map_err(|error| Error::new(Code::Io, format!("读取 {} 失败：{error}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("op-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("建临时目录");
        dir
    }

    #[test]
    fn builtin_matches_disk() {
        let disk = load(&dir()).expect("磁盘上的声明层可解析");
        let embedded = builtin();
        assert_eq!(
            disk.iter().map(Op::name).collect::<Vec<_>>(),
            embedded.iter().map(Op::name).collect::<Vec<_>>(),
            "`ops/` 里新增或删除了算子，但 `BUILTIN` 没跟着改"
        );
        assert_eq!(disk, embedded);
    }

    #[test]
    fn every_builtin_entry_declares_an_operator() {
        for op in builtin() {
            assert!(!op.name().is_empty());
            assert!(op.declaration().contains("[op]"), "{}", op.name());
            assert!(!op.template().is_empty(), "{}", op.name());
        }
    }

    #[test]
    fn the_builtin_template_only_uses_declared_primitives() {
        let (name, declaration, template) = BUILTIN[0];
        let op = decl::OpDecl::parse(declaration, template).expect("内置声明必须自洽");
        assert_eq!(op.name, name);
    }

    #[test]
    fn a_signature_with_an_unsupported_buffer_type_is_rejected_at_load() {
        // ABI 的唯一事实源是 `.k` 的签名：把 `scales` 写成 `double *` 必须在装载期就失败，
        // 而不是等 `cc` 报 C 的指针类型不兼容——诊断要带算子名与形参名。
        let (name, declaration, template) = BUILTIN[0];
        let broken = template.replace("const float *scales", "double *scales");
        let error = decl::OpDecl::parse(declaration, &broken).expect_err("类型不在闭集内");
        let text = error.to_string();
        assert!(text.contains(name), "{text}");
        assert!(text.contains("scales"), "{text}");
        assert!(text.contains("double"), "{text}");
    }

    #[test]
    fn a_template_primitive_must_be_declared_in_needs() {
        let (name, declaration, template) = BUILTIN[0];
        let template = format!("{template}\n@atomic(y, n, out);\n");
        let error = decl::OpDecl::parse(declaration, &template)
            .expect_err("模板里的原语不在 needs 里必须报错");
        assert!(error.to_string().contains("atomic"), "{error}");
        assert!(error.to_string().contains(name), "{error}");
    }

    #[test]
    fn every_variant_must_cover_the_shared_template() {
        let (_, declaration, template) = BUILTIN[0];
        // 把 `bar` 从算子级需求挪到只有 `pipe2` 的增量需求里：`direct` 仍然会 emit 出 `@bar`，
        // 所以装载期必须拒绝，而不是等协商选中 `direct` 才在 substitute 里发现片段缺失。
        let broken = declaration.replace("bar         = {}\n", "").replace(
            "schedule  = \"pipe<2>\"\nmma_share = 0.62\n",
            "schedule  = \"pipe<2>\"\nmma_share = 0.62\nextra_needs.bar = {}\n",
        );
        let error =
            decl::OpDecl::parse(&broken, template).expect_err("变体没有覆盖共享模板必须报错");
        assert!(error.to_string().contains("direct"), "{error}");
        assert!(error.to_string().contains("bar"), "{error}");
    }

    #[test]
    fn a_missing_template_names_both_files() {
        let dir = temp_dir("missing");
        std::fs::write(dir.join("foo.toml"), "[op]\n").expect("写");
        let error = load(&dir).unwrap_err();
        assert_eq!(error.code(), Code::Missing);
        assert!(error.to_string().contains("缺少同名模板"), "{error}");
        assert!(error.to_string().contains("foo.k"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn files_are_loaded_in_name_order() {
        let dir = temp_dir("order");
        for name in ["b", "a"] {
            std::fs::write(dir.join(format!("{name}.toml")), "[op]\n").expect("写");
            std::fs::write(dir.join(format!("{name}.k")), "x\n").expect("写");
        }
        let names: Vec<String> = load(&dir)
            .expect("装载")
            .into_iter()
            .map(|op| op.name().to_string())
            .collect();
        assert_eq!(names, ["a", "b"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
