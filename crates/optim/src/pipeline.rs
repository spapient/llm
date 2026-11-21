//! 一条流水线：找算子 → 取模型 → 建立证据 → 协商 → 选 tile → emit。

use crate::build;
use cutile::cache::CacheKey;
use cutile::cap::{self, NoPath, Path, Proof};
use cutile::dirs::Dirs;
use cutile::emit::Source;
use cutile::harness;
use cutile::plan::{self, Plan};
use cutile::prof::Profile;
use diagnostic::{Code, Error};
use op::decl::OpDecl;
use op::model::{Model, Sample};

/// 这次要跑的那条流水线：算子、语义模型、设备档案与执行证据。
#[derive(Debug)]
pub(crate) struct Run<'a> {
    /// 声明层的算子。
    pub op: &'a OpDecl,
    /// 算子的语义模型（代价、样本、标量参考）。
    pub model: Model,
    /// 设备档案。
    pub profile: &'a Profile,
    /// 这次执行得到的证据（协商只认它，不认档案里的 `grade`）。
    pub proof: Proof,
}

impl<'a> Run<'a> {
    /// 从算子、档案与证据建立一条流水线。
    pub fn new(op: &'a OpDecl, profile: &'a Profile, proof: Proof) -> diagnostic::Result<Self> {
        Ok(Self {
            model: model_for(&op.name)?,
            op,
            profile,
            proof,
        })
    }

    /// 这次协商该用的证据。
    pub fn negotiate(&self) -> Result<Path, Box<NoPath>> {
        cap::negotiate_with(self.op, self.profile, &self.proof)
    }

    /// 把 `(m, n)` 变成可执行的一批：形状、代价、计划与源文本。
    pub fn prepare(
        &self,
        path: &Path,
        dirs: &Dirs,
        m: usize,
        n: usize,
    ) -> diagnostic::Result<Prepared> {
        let shape = (self.model.shape_ctx)(m, n)?;
        let cost = (self.model.cost)(&shape)?;
        let plan = plan::select(self.op, self.profile, path, cost)?;
        let (source, key) = build::source(self.op, self.profile, path, &plan.tile, dirs)?;
        Ok(Prepared { plan, source, key })
    }
}

/// 一个 `(m, n)` 的可执行产物：计划、源文本与它的缓存键。
#[derive(Debug)]
pub(crate) struct Prepared {
    /// ③ select 的产物。
    pub plan: Plan,
    /// ⑤ emit 的产物。
    pub source: Source,
    /// 源文本的身份（产物缓存键的一半）。
    pub key: CacheKey,
}

impl Prepared {
    /// 拼出可执行程序：内核源码 + 样本 + 签名派生的实参 + 重复次数。
    pub fn harness(
        &self,
        op: &OpDecl,
        sample: &Sample,
        repeat: usize,
    ) -> diagnostic::Result<String> {
        op.signature.check(sample)?;
        harness::harness(&self.source.text, &op.signature, sample, repeat)
    }
}

/// 形状覆盖：命令行给的 `MxN`，或算子的生产形状。
pub(crate) fn shape(text: Option<&str>, model: &Model) -> diagnostic::Result<(u64, u64)> {
    match text {
        Some(text) => parse_shape(text, u64::from(model.group.max(1))),
        None => {
            let (m, n) = model.shape;
            Ok((m as u64, n as u64))
        }
    }
}

pub(crate) fn parse_shape(shape: &str, group: u64) -> diagnostic::Result<(u64, u64)> {
    let (m, n) = split(shape)?;
    let dims = (dim(m)?, dim(n)?);
    validate(shape, dims, group).map(|()| dims)
}

/// 拆成两段：缺 `x` 分隔符时把原始文本原样报回去。
fn split(shape: &str) -> diagnostic::Result<(&str, &str)> {
    shape
        .split_once('x')
        .ok_or_else(|| invalid(format!("`{shape}` 不是 `MxN` 形式")))
}

/// 单个维度：去掉两侧空白后必须是非负整数。
fn dim(text: &str) -> diagnostic::Result<u64> {
    text.trim()
        .parse()
        .map_err(|_| invalid(format!("`{text}` 不是整数")))
}

/// 形状约束：两个维度都为正，且 `M` 是组大小的整数倍。
fn validate(shape: &str, (m, n): (u64, u64), group: u64) -> diagnostic::Result<()> {
    if m > 0 && n > 0 && m.is_multiple_of(group) {
        return Ok(());
    }
    Err(invalid(format!(
        "形状 {shape} 非法：M 必须是 {group} 的正整数倍"
    )))
}

fn invalid(message: String) -> Error {
    Error::new(Code::Invalid, message)
}

/// 取算子的语义模型（代价、样本、标量参考）。声明层新增了算子却没有模型时，
/// `doctor`/`verify`/`bench` 都无法给出计划或对拍结果。
pub(crate) fn model_for(name: &str) -> diagnostic::Result<Model> {
    op::model(name).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!(
                "算子 `{name}` 没有语义模型：`crates/op/src/lib.rs` 的 `model()` 注册表里没有它"
            ),
        )
    })
}

/// `verify` 的形状矩阵：最小合法、多行非方形、生产形状。
pub(crate) fn verify_shapes(model: &Model) -> Vec<(u64, u64)> {
    let group = u64::from(model.group.max(1));
    let (m, n) = model.shape;
    vec![(group, 1), (group * 2, 3), (m as u64, n as u64)]
}

/// 必须被拒绝的形状：`M` 不是组大小的整数倍。
pub(crate) fn rejected_shapes(group: u64) -> Vec<(u64, u64)> {
    if group <= 1 {
        return Vec::new();
    }
    vec![(1, group), (group + 1, 8)]
}

#[cfg(test)]
#[path = "../tests/pipeline_tests.rs"]
mod tests;
