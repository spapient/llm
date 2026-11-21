//! 设备源生成：算子名 + 形状 → 可编译的单文件程序 + 逐元素参考值。
//!
//! 只走四步公开缝，每一步都由别的成员负责判断：
//!
//! 1. 协商（[`cutile::cap::negotiate_with`]）用**声明自述**的证明，不等本机执行证据——
//!    源文本对不对由目标机的编译器与对拍回答，本机跑没跑过验证不是生成的前提；
//! 2. 规划（[`cutile::plan::select`]）按 `op` 的代价模型选变体与 tile；
//! 3. 聚合（[`cutile::build::source`]）给设备源文本；
//! 4. 装配（[`engine::program::cuda`]）按 [`cutile::execution::launch`] 的形态拼 host 驱动。
//!
//! 因此本模块**不看算子名、不数段数、不选几何**：形态只有一个来源（`launch.kind`），
//! 在这里再写一份分派就会与 `[chunk]` / `[grid]` 声明漂移。

use cutile::builtin::Catalogue;
use cutile::cap::{Path, Proof};
use cutile::decl::OpDecl;
use cutile::dirs::Dirs;
use cutile::execution::launch;
use cutile::prof::Profile;
use diagnostic::{Code, Error, Result};
use op::model::{Model, Sample};

/// 一次测量作业：算子名 + 形状。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    /// 算子声明名（`op` 全量算子表里的一格）。
    pub op: String,
    /// 规约维度。
    pub m: usize,
    /// 输出维度。
    pub n: usize,
}

impl Job {
    /// 缺省作业：`64 × 1`，与算子语义表的 `sample` 口径一致。
    pub fn of(op: &str) -> Self {
        Self {
            op: op.to_string(),
            m: 64,
            n: 1,
        }
    }

    /// 改形状；`m` 与 `n` 一起给，避免出现半改的作业。
    pub fn shape(mut self, m: usize, n: usize) -> Self {
        self.m = m;
        self.n = n;
        self
    }
}

/// 生成结果：可编译程序 + 逐元素参考值。
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// 完整 `.cu`（设备内核 + `kernel_main` + host 驱动），可直接 `mcc` 编译。
    pub device: String,
    /// 标量参考，逐元素；对拍用。
    pub reference: Vec<f32>,
}

/// 生成一份可编译程序：锚点 → 协商 → 规划 → 聚合 → 按形态装配。
///
/// # Errors
///
/// 算子或 `nvidia` 档案缺失、档案自述与声明不自洽（协商失败）、形状非法、
/// 该形态在 CUDA 上没有并行实现，或聚合/装配失败时返回说明。
pub fn program(job: &Job, dirs: &Dirs) -> Result<Program> {
    let catalogue = Catalogue::try_builtin()?;
    let model = modelled(&job.op)?;
    let sample = (model.sample)(job.m, job.n);
    let device = emitted(&catalogue, job, &model, &sample, dirs)?;
    let reference = (model.reference)(&sample)?;
    Ok(Program { device, reference })
}

/// 设备源 + host 驱动：本模块唯一把两半拼起来的地方。
fn emitted(
    catalogue: &Catalogue,
    job: &Job,
    model: &Model,
    sample: &Sample,
    dirs: &Dirs,
) -> Result<String> {
    let (op, profile) = anchored(catalogue, &job.op)?;
    let text = source(op, profile, job, model, dirs)?;
    engine::program::cuda(&text, &op.signature, sample, &launch(op, sample)?)
}

/// 声明与档案：协商的两端，缺任一就报名字而不是退回空档案。
fn anchored<'a>(catalogue: &'a Catalogue, name: &str) -> Result<(&'a OpDecl, &'a Profile)> {
    let op = catalogue.op(name).ok_or_else(|| missing_op(name))?;
    let profile = catalogue
        .profile("nvidia")
        .ok_or_else(missing_profile)?;
    Ok((op, profile))
}

/// 聚合后的设备源文本：协商 → 规划 → 片段拼接。
fn source(
    op: &OpDecl,
    profile: &Profile,
    job: &Job,
    model: &Model,
    dirs: &Dirs,
) -> Result<String> {
    let path = negotiated(op, profile)?;
    let plan = planned(op, profile, &path, job, model)?;
    let (text, _) = cutile::build::source(op, profile, &path, &plan.tile, dirs)?;
    Ok(text.text)
}

/// 协商：只有"档案自述与声明自洽"这一条证据，因此不要求本机跑过验证。
fn negotiated(op: &OpDecl, profile: &Profile) -> Result<Path> {
    cutile::cap::negotiate_with(op, profile, &Proof::declared(profile))
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))
}

/// 规划：形状上下文 → 标量代价 → 变体与 tile。
fn planned(
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    job: &Job,
    model: &Model,
) -> Result<cutile::plan::Plan> {
    let shape = (model.shape_ctx)(job.m, job.n)?;
    cutile::plan::select(op, profile, path, (model.cost)(&shape)?)
}

/// 语义表：没有它的算子无法生成（样本、代价、参考值全在表里）。
fn modelled(name: &str) -> Result<Model> {
    op::model(name).ok_or_else(|| missing_op(name))
}

/// 没有该算子。
fn missing_op(name: &str) -> Error {
    Error::new(Code::Missing, format!("没有算子 `{name}`"))
}

/// 没有 `nvidia` 档案：生成通道只认它。
fn missing_profile() -> Error {
    Error::new(Code::Missing, "没有 `nvidia` 档案".to_string())
}

/// 参考值的文本形态：一行一个，`{:?}` 保证浮点数可被 `read` 原样读回。
pub fn lines(value: &[f32]) -> String {
    value.iter().map(|item| format!("{item:?}\n")).collect()
}
