//! 算子目录：`src/ops/*.toml` 经 [`prof_ops!`](annotation::prof_ops) 展开的强类型视图，
//! 与 `<stem>.k` 的模板正文。
//!
//! 与 [`crate::builtin`] 的档案表并列但不同：档案表答"这台机器能跑什么"，目录答"有哪些算子"。

pub use crate::templates::TEMPLATES;

/// 按名字取模板正文；名字不在清单里时返回 `None`。表按名升序（生成序），二分查找。
pub fn template(name: &str) -> Option<&'static str> {
    let idx = TEMPLATES
        .binary_search_by_key(&name, |(entry, _)| entry)
        .ok()?;
    Some(TEMPLATES[idx].1)
}

/// 一次注册：一个算子的三件事——名字、语义表、是否可微（模板问 [`template`]）。
///
/// 这是**注册表的值类型**：`nn::REGISTRY` 按 [`for_each_template!`](crate::for_each_template)
/// 的同一顺序逐条填它，于是"有哪些算子"只有一份，`cutile` 与下游只认这个结构，不认识任何
/// 内核类型。
#[derive(Debug, Clone, Copy)]
pub struct Registration {
    /// 算子名（与 `src/ops/<name>.toml` 的文件名一致）。
    pub name: &'static str,
    /// 代价、样本与标量参考。
    pub model: crate::model::Model,
    /// 训练反向图里是否可微：`true` 的算子必须配套一条反向公式。
    pub differentiable: bool,
}

/// 一份算子声明：编译期强类型 [`OpView`](crate::decl::OpView) 与同名 `<stem>.k` 模板。
#[derive(Debug, Clone, Copy)]
pub struct Op {
    name: &'static str,
    view: &'static crate::decl::OpView,
    template: &'static str,
}

impl Op {
    /// 由名字、声明视图与模板构造；名字取 `<name>.toml` 的文件名。
    pub fn new(
        name: &'static str,
        view: &'static crate::decl::OpView,
        template: &'static str,
    ) -> Self {
        Self {
            name,
            view,
            template,
        }
    }

    /// 算子名（与 `<name>.toml` 的文件名一致）。
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// 声明视图：`[op]` / `[axes]` / `[needs]` / `[layouts]` / `[[variants]]` 的静态镜像。
    pub fn view(&self) -> &'static crate::decl::OpView {
        self.view
    }

    /// 模板正文：C 风格文本 + `{{插值}}` + `@原语` 槽位。
    pub fn template(&self) -> &'static str {
        self.template
    }

    /// owned 化成 [`OpDecl`](crate::decl::OpDecl)：视图 → 归属类型，并跑模板侧校验。
    pub fn to_decl(&self) -> diagnostic::Result<crate::decl::OpDecl> {
        crate::decl::OpDecl::from_view(self.view, self.template)
    }
}

/// 按名字取一条内嵌算子；名字不在清单里时返回 `None`。
pub fn named(name: &str) -> Option<Op> {
    let (entry, template) = TEMPLATES
        .binary_search_by_key(&name, |(entry, _)| entry)
        .ok()
        .map(|idx| TEMPLATES[idx])?;
    Some(Op::new(entry, view(entry)?, template))
}

/// 全部内嵌算子，顺序与 [`crate::OPS`] 一致（与 [`TEMPLATES`] 同序）。
pub fn all() -> Vec<Op> {
    crate::OPS
        .iter()
        .filter_map(|view| Some(Op::new(view.name, view, template(view.name)?)))
        .collect()
}

/// 按名字取编译期视图；`OPS` 按名升序，二分查找。
fn view(name: &str) -> Option<&'static crate::decl::OpView> {
    let idx = crate::OPS
        .binary_search_by_key(&name, |view| view.name)
        .ok()?;
    Some(&crate::OPS[idx])
}
