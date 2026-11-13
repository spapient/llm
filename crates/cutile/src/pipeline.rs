//! 一条流水线：找算子 → 取模型 → 建立证据 → 协商 → 选 tile → emit。
//!
//! `doctor` / `verify` / `bench` / `execution` 只决定"跑哪些形状、跑几次、对拍什么"，
//! 这段序列只在这里写一次——四条命令各写一遍曾让同一台设备在两个命令里得出不同结论。
//!
//! 三个类型的分工：
//!
//! - [`Run`]：这次要跑谁（算子 + 模型 + 档案 + 证据），协商也由它给出；
//! - [`Prepared`]：某个 `(m, n)` 的可执行产物（计划 + 源文本 + 缓存键）；
//! - [`Prepared::harness`]：把样本变成实参，签名与样本在这里对齐。

use crate::backend;
use crate::build;
use crate::cache::CacheKey;
use crate::cap::{self, NoPath, Path, Proof};
use crate::emit::Source;
use crate::plan::{self, Plan};
use crate::prof::Profile;
use diagnostic::{Code, Error};
use llm_config::Settings;
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
    ///
    /// 失败时原样返回 [`NoPath`]：`doctor` 的职责正是解释"为什么不行"，它把诊断当报告内容。
    pub fn negotiate(&self) -> Result<Path, Box<NoPath>> {
        cap::negotiate_with(self.op, self.profile, &self.proof)
    }

    /// 把 `(m, n)` 变成可执行的一批：形状、代价、计划与源文本。
    ///
    /// 它不碰样本：`doctor` 只解释流水线，不该为了一次解释去分配一份 `m × n` 的样本缓冲。
    /// 签名与样本的对齐在 [`Prepared::harness`] 里，仍先于编译。
    pub fn prepare(
        &self,
        path: &Path,
        settings: &Settings,
        m: usize,
        n: usize,
    ) -> diagnostic::Result<Prepared> {
        let shape = (self.model.shape_ctx)(m, n)?;
        let cost = (self.model.cost)(&shape)?;
        let plan = plan::select(self.op, self.profile, path, cost)?;
        let (source, key) = build::source(self.op, self.profile, path, &plan.tile, settings)?;
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
    ///
    /// 签名与样本先对齐：ABI 不一致时编译能过、结果会错位，因此这条检查必须先于编译。
    pub fn harness(
        &self,
        op: &OpDecl,
        sample: &Sample,
        repeat: usize,
    ) -> diagnostic::Result<String> {
        op.signature.check(sample)?;
        backend::harness(&self.source.text, &op.signature, sample, repeat)
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
    let (m, n) = shape
        .split_once('x')
        .ok_or_else(|| Error::new(Code::Invalid, format!("`{shape}` 不是 `MxN` 形式")))?;
    let m = m
        .trim()
        .parse::<u64>()
        .map_err(|_| Error::new(Code::Invalid, format!("`{m}` 不是整数")))?;
    let n = n
        .trim()
        .parse::<u64>()
        .map_err(|_| Error::new(Code::Invalid, format!("`{n}` 不是整数")))?;
    if m == 0 || n == 0 || !m.is_multiple_of(group) {
        return Err(Error::new(
            Code::Invalid,
            format!("形状 {shape} 非法：M 必须是 {group} 的正整数倍"),
        ));
    }
    Ok((m, n))
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
///
/// 三项全部来自算子模型：组大小（`model.group`）与生产形状（`model.shape`）都由算子声明，
/// 换一个分组策略或换一个算子时矩阵随它变，负例也不会跟着漂移。审计不写第二个方阵。
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
mod tests {
    use super::*;
    use crate::builtin::Catalogue;

    #[test]
    fn shape_validation_matches_the_operator_constraints() {
        let group = u64::from(model_for("gemv_i4").expect("有模型").group);
        assert_eq!(parse_shape("4096x4096", group).unwrap(), (4096, 4096));
        assert!(parse_shape("4096", group).is_err());
        assert!(parse_shape("1000x8", group).is_err());
        assert!(parse_shape("0x8", group).is_err());
    }

    #[test]
    fn the_shape_matrix_follows_the_operator_model() {
        let model = model_for("gemv_i4").expect("有模型");
        assert_eq!(verify_shapes(&model), vec![(32, 1), (64, 3), (4096, 4096)]);
        // 换一个生产形状，矩阵跟着变：审计里没有第二个方阵。
        let other = op::model::Model {
            shape: (128, 64),
            ..model
        };
        assert_eq!(verify_shapes(&other), vec![(32, 1), (64, 3), (128, 64)]);
    }

    #[test]
    fn the_model_group_matches_the_declaration() {
        // 声明里的 `needs.quant.group` 是能力请求，模型里的 `group` 是样本与代价的取值；
        // 两者漂移会让 verify 在一组原语支撑不了的组大小上"通过"。
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let declared = catalogue
            .op("gemv_i4")
            .expect("内置算子")
            .needs
            .iter()
            .filter(|need| need.prim == prof::atom::Prim::Quant)
            .flat_map(|need| need.attrs.iter())
            .find(|attr| attr.key() == prof::atom::AttrKey::Group)
            .and_then(|attr| attr.values().first())
            .expect("声明里有 quant.group")
            .parse::<u32>()
            .expect("组大小是整数");
        assert_eq!(declared, model_for("gemv_i4").expect("有模型").group);
    }
}
