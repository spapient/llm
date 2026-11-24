//! 算子目录：`src/ops/*.toml` 经 [`prof_ops!`](annotation::prof_ops) 展开的强类型视图，
//! 与 `<stem>.k` 的模板正文。
//!
//! 与 [`crate::builtin`] 的档案表并列但不同：档案表答"这台机器能跑什么"，目录答"有哪些算子"。

use std::borrow::Cow;

use diagnostic::{Code, Error};

pub use crate::templates::{DEVICE_TEMPLATES, TEMPLATES};

/// 按名字取主机参考模板；等价于 `template_for("cpu", name)`。
pub fn template(name: &str) -> Option<&'static str> {
    let idx = TEMPLATES
        .binary_search_by_key(&name, |(entry, _)| entry)
        .ok()?;
    Some(TEMPLATES[idx].1)
}

/// 按 `(vendor, 算子名)` 取后端微内核；表按二元组升序，查询为 O(log n)。
pub fn template_for(vendor: &str, name: &str) -> Option<&'static str> {
    let idx = DEVICE_TEMPLATES
        .binary_search_by_key(&(vendor, name), |(entry, op, _)| (*entry, *op))
        .ok()?;
    Some(DEVICE_TEMPLATES[idx].2)
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

/// 按名字取一条主机参考算子；名字不在清单里时返回 `None`。
pub fn named(name: &str) -> Option<Op> {
    named_for("cpu", name)
}

/// 按后端与名字取一条内嵌算子；声明共享，正文来自该后端的 `kernels/`。
pub fn named_for(vendor: &str, name: &str) -> Option<Op> {
    let template = template_for(vendor, name)?;
    Some(Op::new(view(name)?.name, view(name)?, template))
}

/// 全部主机参考算子，顺序与 [`crate::OPS`] 一致。
pub fn all() -> Vec<Op> {
    all_for("cpu")
}

/// 一台后端的全部算子，顺序与 [`crate::OPS`] 一致；缺任一条就视为目录不完整。
pub fn all_for(vendor: &str) -> Vec<Op> {
    crate::OPS
        .iter()
        .filter_map(|view| Some(Op::new(view.name, view, template_for(vendor, view.name)?)))
        .collect()
}

/// 为实际发射设备选择微内核，同时守住跨后端 ABI：形参顺序、类型与可写性必须完全一致。
///
/// CPU 或正文相同的设备借用原声明；只有真正换实现时才克隆声明并替换正文。
///
/// # Errors
///
/// 后端没有该算子、后端正文的入口不合法，或后端 ABI 与共享契约不一致时返回诊断。
pub fn for_device<'a>(
    op: &'a crate::decl::OpDecl,
    vendor: &str,
) -> diagnostic::Result<Cow<'a, crate::decl::OpDecl>> {
    let source = template_for(vendor, &op.name).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("后端 `{vendor}` 没有算子 `{}` 的微内核", op.name),
        )
    })?;
    if source == op.template {
        return Ok(Cow::Borrowed(op));
    }
    let signature = crate::sig::Signature::parse(&op.name, source)?;
    if signature != op.signature {
        return Err(Error::new(
            Code::Invalid,
            format!("后端 `{vendor}` 的算子 `{}` 与共享契约的 ABI 不一致", op.name),
        ));
    }
    let mut specialized = op.clone();
    specialized.template = source.to_string();
    specialized.signature = signature;
    Ok(Cow::Owned(specialized))
}

/// 按名字取编译期视图；`OPS` 按名升序，二分查找。
fn view(name: &str) -> Option<&'static crate::decl::OpView> {
    let idx = crate::OPS
        .binary_search_by_key(&name, |view| view.name)
        .ok()?;
    Some(&crate::OPS[idx])
}
