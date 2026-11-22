//! 每台设备的**全部数据**：能力档案（区间、存储层、执行单元）与片段文本（原语怎么写）。
//!
//! 一个 crate 装两样东西，理由不是"都跟设备有关"，而是**它们必须一起在构建期读**：能力档案声明
//! "这台设备能拼出 `dot`"，片段文本声明"`dot` 在 `.metal` 里这么写"，两边对不上就该在构建期失败，
//! 而不是等运行期 emit 出来编译不过。合并前它们被切成三个 crate（`prof` / `frag` / `prim`），
//! 而 cargo 不允许一个包当自己的构建依赖——`frag` 存在的唯一理由就是给 `prof` 的 `build.rs` 提供
//! 目录，于是同一件事有两个 crate。现在生成器就在 [`build.rs`](../build.rs) 里。
//!
//! # 四层，各归各位
//!
//! - `src/<vendor>/` 是**内容**：`<vendor>.toml`（存储层物理名、输出语言、工具链版本、执行单元、
//!   每条能力区间）与 8 段 `<prim>.<ext>` 片段、`epilogue.<ext>`、`witness.<ext>`；
//! - [`atom`] 是**唯一的词汇表**：16 个原语、类型/存储位集、参数化签名 `Need`/`Provide`、
//!   属性键闭集 [`atom::AttrKey`]；
//! - [`prof`] 是**唯一的 schema**：解析、闭集字段、见证与 `[units]` 一致性校验；
//! - [`device`] 是**设备行的形状**：这次构建带上的元数据、见证源码、可选镜像与 [`Status`]。
//!
//! # 生成，不解析
//!
//! [`build.rs`](../build.rs) 读 `src/<vendor>/<vendor>.toml`，一次生成两棵树：
//! `$OUT_DIR/embedded.rs`（档案 → `diagnostic::data::Table` 字面量，[`embedded`] / [`names`]，
//! ADR 0022）与 `$OUT_DIR/tables.rs`（片段 → 每台设备一个模块 + `BUILTIN` / `WITNESS` /
//! `DEVICES` / `ROOTS` 四张聚合表）。运行期只遍历这两棵树：不读目录、没有 TOML 解析器。
//!
//! `lib.rs` 里**没有 vendor 清单**：`src/` 下哪个目录有 `<vendor>/<vendor>.toml`，哪台设备就存在。
//! 加一台设备 = 加一个目录，机制代码零改动——这是"设备差异只有数据"的可执行版本。
//!
//! # 依赖方向
//!
//! `cutile` 只做机制（协商、降级、拼接、emit），因此本 crate 不反向依赖它：依赖方向
//! `cutile → prof`，而 `prof` 只依赖 `std` 与同属基座的 `diagnostic`（错误类别由产生错误的那一层
//! 给，见 `docs/ARCHITECTURE.md` 第三节）。
//!
//! # 两种模式：源码调试 / 设备镜像发布
//!
//! 默认（调试）只有源码：片段文本 + 见证源码，运行期照旧 emit 后编译。打开 `device-image`
//! feature（发布）时，能用本机工具链编出设备镜像的设备（如今是 `metal`）额外带上二进制；
//! [`Device::status`] 说明每台设备这次走到了哪一步，没有镜像也只是"这一步没走"，不是配置错误。
//!
//! # 例子
//!
//! ```
//! let cpu = llm_prof::embedded("cpu").expect("内置 cpu 档案");
//! assert!(llm_prof::names().contains(&"cpu"));
//! assert!(llm_prof::prof::Profile::parse(cpu).is_ok());
//! assert_eq!(llm_prof::DEVICES.len(), llm_prof::names().len());
//!
//! let set = llm_prof::builtin("cpu", "c").expect("内置 cpu 片段");
//! assert!(set.get(llm_prof::Prim::Load).is_some());
//!
//! let device = llm_prof::device("metal").expect("内置设备");
//! assert_eq!(device.ext, "metal", "扩展名由档案的 emit 语言决定");
//! ```

#![deny(unsafe_code)]

pub mod atom;
pub mod device;
pub mod prof;
pub mod sign;
pub mod stems;

pub use atom::Prim;
pub use device::{Builtin, Device, Image, Status};
pub use builtin::{builtin, builtin_witness, device, devices, dir_of};
pub use fragment::{Fragment, FragmentSet, SIGN_MARKER};
pub use layout::{Layout, WITNESS_STEM, layer_root, load};

/// 构建期生成的档案表（`build.rs` 读 `src/<vendor>/<vendor>.toml` 展开成 `Table` 字面量）。
#[allow(clippy::all, missing_docs)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
}

// 构建期生成的片段表：每台设备一个模块 + 四张聚合表。
// `include!` 在 crate 根上、不套 `mod`：套一层就要在 `lib.rs` 里重导出每台设备，那份 vendor
// 清单会重新变成第二份真相——生成表的意义正是让它只存在于 `src/` 的目录名里。
include!(concat!(env!("OUT_DIR"), "/tables.rs"));

pub use generated::{embedded, names};

mod builtin;
mod fragment;
mod layout;

mod eval;
mod key;
mod need;
mod provide;
mod schema;
mod span;
mod witness;

#[cfg(test)]
#[path = "../tests/embedded.rs"]
mod tests;
pub mod decl;

/// 构建期生成的算子声明表（`build.rs` 读 `src/ops/*.toml` 展开成 `Table` 字面量）。
#[allow(clippy::all, missing_docs)]
mod declared_generated {
    include!(concat!(env!("OUT_DIR"), "/declared.rs"));
}

pub use declared_generated::declared;

/// 全量算子清单里「名字 → 模板 stem」的那一半：**唯一一次**声明。
///
/// 这里不含任何内核类型：本 crate 只回答"有哪些算子、各自用哪份模板"。谁实现它们由
/// `nn::registry` 回答——那张表按名字与本表逐条对齐，错位在编译期就失败。
#[macro_export]
macro_rules! for_each_template {
    ($m:ident) => {
        $m! {
            adaptive_pool_f32 => "adaptive_pool_f32",
            add_f32 => "binary",
            attention_f32 => "attention_f32",
            avg_pool2d_f32 => "avg_pool2d_f32",
            batch_norm_f32 => "batch_norm_f32",
            conv2d_f32 => "conv2d_f32",
            decode_attention_f32 => "decode_attention_f32",
            decode_flash_attention_f32 => "decode_flash_attention_f32",
            dequant_i2 => "dequant_packed",
            dequant_i4 => "dequant_packed",
            dequant_i8 => "dequant_i8",
            div_f32 => "binary",
            embed_f32 => "embed_f32",
            exp_f32 => "exp_f32",
            flash_attention_f32 => "flash_attention_f32",
            gelu_f32 => "gelu_f32",
            gemm_f32 => "gemm_f32",
            gemm_i2 => "gemm_packed",
            gemm_i4 => "gemm_packed",
            gemm_i8 => "gemm_i8",
            gemv_f32 => "gemv_f32",
            gemv_i2 => "gemv_packed",
            gemv_i4 => "gemv_packed",
            gemv_i8 => "gemv_i8",
            group_norm_f32 => "group_norm_f32",
            hard_sigmoid_f32 => "hard_sigmoid_f32",
            instance_norm_f32 => "instance_norm_f32",
            interpolate_f32 => "interpolate_f32",
            latent_attention_f32 => "latent_attention_f32",
            layer_norm_f32 => "layer_norm_f32",
            leaky_relu_f32 => "leaky_relu_f32",
            log_f32 => "log_f32",
            matmul_f32 => "matmul_f32",
            matmul_i8 => "matmul_i8",
            max_f32 => "max_f32",
            max_pool1d_f32 => "max_pool1d_f32",
            max_pool2d_f32 => "max_pool2d_f32",
            min_f32 => "min_f32",
            mul_f32 => "binary",
            reduce_max_f32 => "reduce_max_f32",
            reduce_sum_f32 => "reduce_sum_f32",
            relu_f32 => "relu_f32",
            residual_add_f32 => "binary",
            rmsnorm_f32 => "rmsnorm_f32",
            rope_f32 => "rope_f32",
            scale_f32 => "scale_f32",
            scan_f32 => "scan_f32",
            sigmoid_f32 => "sigmoid_f32",
            silu_f32 => "silu_f32",
            softmax_f32 => "softmax_f32",
            sub_f32 => "binary",
            tanh_f32 => "tanh_f32",
        }
    };
}

/// 模板表：算子名 → `src/ops/<stem>.k` 的正文，`include_str!` 在编译期读入。
mod templates {
    macro_rules! impl_templates {
        ($($name:ident => $stem:literal),* $(,)?) => {
            /// 算子名与模板正文，顺序与 [`for_each_template!`] 一致。
            pub const TEMPLATES: [(&str, &str); <[()]>::len(&[$(impl_templates!(@one $stem)),*])] = [
                $((stringify!($name), include_str!(concat!("ops/", $stem, ".k")))),*
            ];
        };
        (@one $stem:literal) => { () };
    }

    crate::for_each_template!(impl_templates);
}

pub use templates::TEMPLATES;

/// 按名字取模板正文；名字不在清单里时返回 `None`。
pub fn template(name: &str) -> Option<&'static str> {
    TEMPLATES
        .iter()
        .find(|(entry, _)| *entry == name)
        .map(|(_, text)| *text)
}

/// 一次注册：一个算子的四件事——名字、模板、语义表、是否可微。
///
/// 这是**注册表的值类型**：`nn::REGISTRY` 按 [`for_each_template!`] 的同一顺序逐条填它，
/// 于是"有哪些算子"只有一份，`cutile` 与下游只认这个结构，不认识任何内核类型。
#[derive(Debug, Clone, Copy)]
pub struct Registration {
    /// 算子名（与 `src/ops/<name>.toml` 的文件名一致）。
    pub name: &'static str,
    /// `<stem>.k` 的正文。
    pub template: &'static str,
    /// 代价、样本与标量参考。
    pub model: spec::model::Model,
    /// 训练反向图里是否可微：`true` 的算子必须配套一条反向公式。
    pub differentiable: bool,
}

/// 内核表装配：一行一个「算子名 → 适配函数」，展开成 `Entry { op, run }` 的数组字面量。
#[macro_export]
macro_rules! kernel_table {
    ($($run:ident),* $(,)?) => {
        [$( Entry { op: stringify!($run), run: $run } ),*]
    };
}

/// 一份算子声明：编译期生成的声明表与同名 `<stem>.k` 模板。
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    name: &'static str,
    declaration: &'static diagnostic::data::Table,
    template: &'static str,
}

impl Op {
    /// 由名字、声明表与模板构造；名字取 `<name>.toml` 的文件名。
    pub fn new(
        name: &'static str,
        declaration: &'static diagnostic::data::Table,
        template: &'static str,
    ) -> Self {
        Self {
            name,
            declaration,
            template,
        }
    }

    /// 算子名（与 `<name>.toml` 的文件名一致）。
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// 声明表：`[op]` / `[axes]` / `[needs]` / `[layouts]` / `[[variants]]`。
    pub fn declaration(&self) -> &'static diagnostic::data::Table {
        self.declaration
    }

    /// 模板正文：C 风格文本 + `{{插值}}` + `@原语` 槽位。
    pub fn template(&self) -> &'static str {
        self.template
    }
}

/// 按名字取一条内嵌算子；名字不在清单里时返回 `None`。
pub fn builtin_named(name: &str) -> Option<Op> {
    let (entry, template) = TEMPLATES.iter().find(|(entry, _)| *entry == name)?;
    Some(Op::new(entry, declared(entry)?, template))
}

/// 内嵌声明，顺序与 [`TEMPLATES`] 一致。
pub fn builtin() -> Vec<Op> {
    TEMPLATES
        .iter()
        .filter_map(|(name, template)| Some(Op::new(name, declared(name)?, template)))
        .collect()
}
