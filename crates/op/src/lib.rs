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
//! 同一副骨架的多种精度写在 [`gemv`]：一份泛型实现（求和、代价、样本、校验、遍历）+ 每个格式
//! 一份 [`gemv::Format`] 契约（权重元素、位宽、粒度、解包）。`gemv_i4` / `gemv_f32` 因此只写
//! "这份格式是什么"，不重写骨架。它们的 `MODEL` 由 `#[annotation::model]` 注解生成：配置在编译期
//! 定下，"注解说的是哪份格式"由类型（`Quantized<32>` / `Dense<32>`）保证，写错就编译不过。
//!
//! 代价（`flops` / `bytes`）不在本 crate 里手写：算子只声明 `EXPR`（如 `nm,m,ng,n->n`）与每个操作数
//! 的位宽/方向，由 [`einsum`] 推出；[`model::Cost`] 只是它的再导出（`docs/adr/0007`）。这样新增算子
//! 不会引入第二条流量公式，roofline 的分子也只有一处来源。
//!
//! # 例子
//!
//! ```
//! // 注册表里的每一条都自带声明与模板；要哪一条就按名字取，不靠位置。
//! for op in llm_op::builtin() {
//!     assert!(op.declaration().contains("[op]"), "{}", op.name());
//!     assert!(!op.template().is_empty(), "{}", op.name());
//! }
//! let int4 = llm_op::builtin_named("gemv_i4").expect("已注册");
//! assert_eq!(int4.name(), "gemv_i4");
//! assert_eq!(
//!     llm_op::model("gemv_f32").map(|model| model.group),
//!     Some(1),
//!     "稠密算子的形状粒度是 1"
//! );
//! ```

#![deny(unsafe_code)]

// `#[annotation::model]` 生成的代码用 `::llm_op::..` 指回本 crate（与 `nn` 的
// `extern crate self as nn` 同一条约定：注解绑定它所属契约的路径，而不是硬编码调用方的名字）。
extern crate self as llm_op;

pub mod decl;
pub mod gemv;
pub mod gemv_f32;
pub mod gemv_i4;
pub mod layout;
pub mod model;
pub mod sch;
pub mod sig;
pub mod tpl;

use diagnostic::{Code, Error};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

use model::Model;

/// 一次注册：一个算子的四件事——名字、声明、模板、语义表。
///
/// **这是唯一一处注册表**：加一个算子只改这里。[`BUILTIN`] 与 [`model()`] 都从 [`REGISTRY`]
/// 投影出来，不存在"声明层有、语义层没有"（或反过来）的第二份事实。
#[derive(Debug, Clone, Copy)]
pub struct Registration {
    /// 算子名（与 `ops/<name>.toml` 的文件名一致）。
    pub name: &'static str,
    /// `<name>.toml` 的正文。
    pub declaration: &'static str,
    /// `<name>.k` 的正文。
    pub template: &'static str,
    /// 代价、样本与标量参考。
    pub model: Model,
}

/// 编译期内嵌的算子注册表。
///
/// 新增算子在这里补一行，否则发布出去的二进制里没有它；`builtin_matches_disk` 测试会把
/// "磁盘上有、内嵌里没有"直接判为失败。
pub const REGISTRY: [Registration; 2] = [
    Registration {
        name: "gemv_f32",
        declaration: include_str!("../ops/gemv_f32.toml"),
        template: include_str!("../ops/gemv_f32.k"),
        model: gemv_f32::MODEL,
    },
    Registration {
        name: "gemv_i4",
        declaration: include_str!("../ops/gemv_i4.toml"),
        template: include_str!("../ops/gemv_i4.k"),
        model: gemv_i4::MODEL,
    },
];

/// 算子语义注册表：声明层有的算子，这里必须有它的代价、样本与标量参考。
///
/// 语义表和声明同属一条注册（[`REGISTRY`]），因此两者不会漂移，`cutile` 一侧不动。
pub fn model(name: &str) -> Option<Model> {
    REGISTRY
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| entry.model)
}

/// 编译期内嵌的声明层：`(名字, 声明, 模板)`，由 [`REGISTRY`] 投影而来。
pub const BUILTIN: [(&str, &str, &str); REGISTRY.len()] = project(&REGISTRY);

/// 把注册表投影成"名字 / 声明 / 模板"三元组，供只关心文本的一方使用。
const fn project<const N: usize>(
    registry: &[Registration; N],
) -> [(&'static str, &'static str, &'static str); N] {
    let mut out = [("", "", ""); N];
    let mut index = 0;
    while index < N {
        out[index] = (
            registry[index].name,
            registry[index].declaration,
            registry[index].template,
        );
        index += 1;
    }
    out
}

/// 一份算子声明：`<name>.toml` 的正文与同名 `<name>.k` 模板。
///
/// 字段是 [`Cow`]：内嵌的算子直接借用编译进去的文本（[`builtin`] 不拷贝一个字符），从磁盘
/// 装载的全是 `String`。两边共用同一个类型，调用方看不到这个差别。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Op {
    name: Cow<'static, str>,
    declaration: Cow<'static, str>,
    template: Cow<'static, str>,
}

impl Op {
    /// 由三段文本构造；名字取 `<name>.toml` 的文件名。
    ///
    /// 接受 `&'static str`（内嵌）与 `String`（磁盘）：前者不拷贝，后者移进来。
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        declaration: impl Into<Cow<'static, str>>,
        template: impl Into<Cow<'static, str>>,
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

/// 按名字取一条内嵌算子；名字不在注册表里时返回 `None`。
///
/// 存在的理由和 `model()` 一样：调用方要的是**某一条**算子，而不是"第一条算子"。按位置取
/// （`builtin()[0]`）在注册表从一条变成两条的那天会静默换人——而算子之间并不等价（稠密 f32
/// 与 int4 的代价差 8 倍）。
pub fn builtin_named(name: &str) -> Option<Op> {
    REGISTRY
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| Op::new(entry.name, entry.declaration, entry.template))
}

/// 内嵌声明，顺序与 [`REGISTRY`] 一致。
///
/// 文本是编译进去的常量，所以这里只有一次 `Vec` 分配：不再为每个算子把三段文本各拷一份。
pub fn builtin() -> Vec<Op> {
    REGISTRY
        .iter()
        .map(|entry| Op::new(entry.name, entry.declaration, entry.template))
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
        ops.push(Op::new(name.to_string(), declaration, template));
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

    /// 声明与语义同属一条注册：不该出现"内嵌里有、语义表没有"的算子。
    #[test]
    fn every_builtin_entry_has_a_model() {
        for op in builtin() {
            assert!(model(op.name()).is_some(), "{} 没有语义表", op.name());
        }
        assert!(model("no_such_op").is_none());
    }

    /// 测试要的是**量化那条**算子，不是"注册表里的第一条"。
    ///
    /// 按位置取会在注册表从一条变两条的那天静默换人——而两条算子并不等价（稠密 f32 与 int4
    /// 的代价差 8 倍）。名字写在这里，注册表增删都不会让这条测试去验另一个算子。
    fn int4() -> (&'static str, &'static str, &'static str) {
        BUILTIN
            .iter()
            .copied()
            .find(|(name, _, _)| *name == "gemv_i4")
            .expect("gemv_i4 已注册")
    }

    #[test]
    fn the_builtin_template_only_uses_declared_primitives() {
        let (name, declaration, template) = int4();
        let op = decl::OpDecl::parse(declaration, template).expect("内置声明必须自洽");
        assert_eq!(op.name, name);
    }

    #[test]
    fn a_signature_with_an_unsupported_buffer_type_is_rejected_at_load() {
        // ABI 的唯一事实源是 `.k` 的签名：把 `scales` 写成 `double *` 必须在装载期就失败，
        // 而不是等 `cc` 报 C 的指针类型不兼容——诊断要带算子名与形参名。
        let (name, declaration, template) = int4();
        let broken = template.replace("const float *scales", "double *scales");
        let error = decl::OpDecl::parse(declaration, &broken).expect_err("类型不在闭集内");
        let text = error.to_string();
        assert!(text.contains(name), "{text}");
        assert!(text.contains("scales"), "{text}");
        assert!(text.contains("double"), "{text}");
    }

    #[test]
    fn a_template_primitive_must_be_declared_in_needs() {
        let (name, declaration, template) = int4();
        let template = format!("{template}\n@atomic(y, n, out);\n");
        let error = decl::OpDecl::parse(declaration, &template)
            .expect_err("模板里的原语不在 needs 里必须报错");
        assert!(error.to_string().contains("atomic"), "{error}");
        assert!(error.to_string().contains(name), "{error}");
    }

    /// 模板传的能力参数必须被 `[needs]` 允许：否则设备按声明验、代码按模板生成。
    #[test]
    fn a_call_argument_must_match_the_declared_need() {
        let (name, declaration, template) = int4();
        let broken = template.replace("sat: off", "sat: on");
        let error = decl::OpDecl::parse(declaration, &broken).expect_err("模板与需求矛盾必须报错");
        let text = error.to_string();
        assert!(text.contains(name), "{text}");
        assert!(text.contains("cvt"), "{text}");
        assert!(text.contains("sat"), "{text}");
        assert!(text.contains("on"), "{text}");
    }

    /// 闭集内的能力属性传了却没声明，等于协商永远不检查它。
    #[test]
    fn a_capability_argument_must_be_declared_in_needs() {
        let (name, declaration, template) = int4();
        let broken = template.replace("sat: off", "sat: off, accuracy: approx");
        let error =
            decl::OpDecl::parse(declaration, &broken).expect_err("未声明的能力参数必须报错");
        let text = error.to_string();
        assert!(text.contains(name), "{text}");
        assert!(text.contains("cvt"), "{text}");
        assert!(text.contains("accuracy"), "{text}");
    }

    /// 闭集之外的命名参数是片段级提示，不构成能力声明，不能被这条判据误伤。
    #[test]
    fn a_fragment_only_named_argument_stays_legal() {
        let (_, declaration, template) = int4();
        let extra = template.replace("sat: off", "sat: off, along: 32");
        decl::OpDecl::parse(declaration, &extra).expect("片段级命名参数不该被能力判据拒绝");
    }

    #[test]
    fn every_variant_must_cover_the_shared_template() {
        let (_, declaration, template) = int4();
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

    /// 判据 1（DESIGN §12.2）：首选变体必须把非 MMA 工作盖住，否则该换 S3/S4。
    #[test]
    fn the_preferred_variant_must_hide_its_non_mma_work() {
        let (_, declaration, template) = int4();
        // 占比降下来链仍然单调（0.10 → 0.05），所以先失败的只会是掩盖判据。
        let hidden = declaration
            .replace("mma_share = 0.62", "mma_share = 0.10")
            .replace("mma_share = 0.35", "mma_share = 0.05");
        assert_ne!(hidden, declaration, "声明里的占比写法变了");
        let error = decl::OpDecl::parse(&hidden, template).expect_err("首选变体掩盖不足必须报错");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("首选变体 `pipe2`"), "{error}");
        assert!(error.to_string().contains("45%"), "{error}");
    }

    /// 兜底变体的掩盖账超阈值是事实，不是声明错误：它存在的理由正是"设备缺原语时仍能跑"。
    #[test]
    fn a_falling_back_variant_may_miss_the_masking_bar() {
        let (_, declaration, template) = int4();
        let op = decl::OpDecl::parse(declaration, template).expect("内置声明应当通过");
        let direct = op.variant("direct").expect("兜底变体");
        let masking = sch::Masking::of(direct.mma_share, &direct.schedule);
        assert!(masking.claims_mma(), "{masking}");
        assert!(!masking.within_limit(), "{masking}");
        assert!(masking.to_string().contains("降级兜底"), "{masking}");
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
