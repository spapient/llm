//! ② negotiate：需求 ∩ 供给，显式降级（DESIGN §8）。
//!
//! 命中第一个可行变体即停：降级链的顺序就是人在 TOML 里写的优先级，不做自动打分——自动
//! 打分的可解释性成本高于收益。失败时给出的诊断必须包含"缺什么、供给方是什么、最近的可行解与代价"。

use crate::atom::{Grade, Mismatch, Need, Prim, Provide};
use crate::decl::{OpDecl, Variant};
use crate::prof::Profile;
use crate::sch::Schedule;
use core::fmt;
use std::collections::BTreeSet;

/// 协商结果：可行的变体 + 调度。
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    /// 变体名。
    pub variant: String,
    /// 变体在降级链里的下标。
    pub index: usize,
    /// 调度模板。
    pub schedule: Schedule,
    /// 该变体的 `mma_share`（roofline 用）。
    pub mma_share: f64,
}

impl Path {
    /// 是否命中降级链的首选变体（DESIGN §16「协商 3/3 命中首档」的判据）。
    ///
    /// 只看下标：链按 `order` 排好，`index = 0` 就是声明里最优先的那个变体。落到后面的变体
    /// 说明首选的能力没被满足——`needs` 要么写多了一条，要么这台设备真的缺。
    pub fn is_head(&self) -> bool {
        self.index == 0
    }
}

/// 这次真的执行过、并且通过了的供给：`(原语, 实现名)` 的集合。
///
/// 档案里的 [`Grade`] 是作者的自述；本类型是执行的结论。协商只有在**两者同时成立**时才认为
/// 一条供给可用——DESIGN §11.3 的"未验证视为不具备"因此对每条命令都成立，而不是只对
/// `verify` 成立（那里跑完就不再用它，别的命令却还在读档案）。
///
/// 空集不是"没有原语"，而是"这次什么都没执行"：协商照样会走完降级链，只是每一档都失败，
/// 失败原因写成"L2 但没有这次执行的见证"。这正是想要的形状——拒绝的理由必须能被读出来。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Proof {
    holds: BTreeSet<(Prim, String)>,
}

impl Proof {
    /// 什么都没执行过。
    ///
    /// 执行路径上用不到它（那里的证据来自见证），但"证据还没到"这件事必须能被表达出来：
    /// 缺省值就是"一条也不具备"，而不是"按档案全具备"。
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// 按档案自述：每条 `grade ≥ L2` 的供给都算可用。
    ///
    /// 只给**解释性**报告用——§16 的协商判据（"首选变体在声明上是否命中"）与本机执行不了的
    /// 设备（非 C 档案）。它把 TOML 当结论，所以任何声称"这台设备能跑"的路径都不许用它，
    /// 否则 `verify` 刚闭合的那条证据链又从别的门开了。用它的报告必须写明依据（`audit::witness::Basis`）。
    #[must_use]
    pub fn declared(prof: &Profile) -> Self {
        let mut proof = Self::default();
        for have in &prof.provides {
            if have.grade.is_verified() {
                proof.insert(have.prim, &have.impl_name);
            }
        }
        proof
    }

    /// 记下一条真正跑通的供给（链式，便于从执行结果直接拼）。
    #[must_use]
    pub fn with(mut self, prim: Prim, impl_name: &str) -> Self {
        self.insert(prim, impl_name);
        self
    }

    /// 记下一条真正跑通的供给。
    pub fn insert(&mut self, prim: Prim, impl_name: &str) {
        self.holds.insert((prim, impl_name.to_string()));
    }

    /// 这条供给这次是否被证明。
    ///
    /// 认 `(原语, 实现名)` 而不是整个签名：同一条供给的参数没变，变的是"这次跑没跑"。
    #[must_use]
    pub fn holds(&self, have: &Provide) -> bool {
        self.holds.contains(&(have.prim, have.impl_name.clone()))
    }

    /// 被证明的供给条数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.holds.len()
    }

    /// 一条也没有。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.holds.is_empty()
    }
}

/// 一条需求的检查结果。
#[derive(Debug, Clone, PartialEq)]
pub struct NeedReport {
    /// 需求。
    pub need: Need,
    /// 供给方（同原语的档案条目）。
    pub have: Option<Provide>,
    /// 供给存在但证据等级不足 `L2`。
    pub blocked_by_grade: bool,
    /// 供给声明了 `L2`，但这次执行没有拿到它的见证结论（[`Proof`] 里没有这一条）。
    ///
    /// 它与 `blocked_by_grade` 必须分开：前者是"档案说它具备"、后者是"档案自己就没说"。
    /// 合成一个布尔值之后，报告就再也说不出"这条能力欠一次执行"。
    pub unproven: bool,
    /// 逐项差异。
    pub mismatches: Vec<Mismatch>,
}

impl NeedReport {
    /// 是否满足。
    pub fn is_satisfied(&self) -> bool {
        self.mismatches.is_empty()
            && !self.blocked_by_grade
            && !self.unproven
            && self.have.is_some()
    }

    /// 一行原因：先报参数不匹配，再报证据。
    ///
    /// 顺序很重要——"dp4a 不支持 f32" 和"没验证"是两件事，只说后者会让人误以为补见证就能跑。
    /// 这里只给出一句话的判定，不排版：拼成哪一行、排在哪里由前端决定。
    pub fn reason(&self) -> String {
        let mut parts: Vec<String> = self.mismatches.iter().map(ToString::to_string).collect();
        if self.blocked_by_grade {
            let grade = self.have.as_ref().map_or(Grade::L0, |have| have.grade);
            parts.push(format!("{grade} < L2（未验证视为不具备）"));
        }
        if self.unproven {
            parts.push("L2 但没有这次执行的见证（未验证视为不具备）".to_string());
        }
        if parts.is_empty() {
            return "已满足".to_string();
        }
        parts.join("；")
    }
}

/// 降级链上的一步。
#[derive(Debug, Clone, PartialEq)]
pub struct ChainStep {
    /// 变体名。
    pub variant: String,
    /// 调度。
    pub schedule: Schedule,
    /// 是否可行。
    pub ok: bool,
    /// 未满足的需求（`ok` 时为空）。原因文字由 [`NeedReport::reason`] 给出，本结构不持有文本。
    pub failures: Vec<NeedReport>,
    /// `mma_share`。
    pub mma_share: f64,
}

/// 一个变体的需求复核结果，供 `doctor` 逐条解释降级链。
///
/// 与 [`negotiate`] 的分工：协商**只跑到第一个可行变体**为止（命中即停），解释则要看完整条链，
/// 因此本函数总是检查全部变体。两者都不做排版。
#[derive(Debug, Clone, PartialEq)]
pub struct VariantCheck {
    /// 变体名。
    pub name: String,
    /// 未满足的需求（空表示可行）。
    pub failures: Vec<NeedReport>,
}

/// 距离可行只差哪些能力：用来给出"补哪一个见证就能解锁"的建议。
#[derive(Debug, Clone, PartialEq)]
pub struct Unlock {
    /// 最接近可行的变体。
    pub variant: String,
    /// 还缺的原语（按原语去重：同一原语的多个 dtype 只算一条）。
    pub missing: Vec<Prim>,
    /// 该变体的 `mma_share`。
    pub mma_share: f64,
}

/// 无可行变体时的诊断。
#[derive(Debug, Clone, PartialEq)]
pub struct NoPath {
    /// 算子名。
    pub op: String,
    /// 设备标签。
    pub device: String,
    /// 最高优先级那个变体的需求检查明细。
    pub needs: Vec<NeedReport>,
    /// 完整降级链。
    pub chain: Vec<ChainStep>,
    /// 最近的解锁路径。
    pub unlock: Option<Unlock>,
}

/// 降级链一步的原因文字：`原语：原因`，多条以 `；` 连接。
///
/// 这是**错误文本**的一部分（[`NoPath`] 的 `Display` 按 DESIGN §8.2 输出它），不是排版：
/// 换行、缩进、标题由用它的地方决定。
fn describe_failures(failures: &[NeedReport]) -> String {
    failures
        .iter()
        .map(|report| format!("{}：{}", report.need.prim, report.reason()))
        .collect::<Vec<_>>()
        .join("；")
}

impl fmt::Display for NoPath {
    /// DESIGN §8.2 的失败诊断：缺什么、供给方是什么、最近的可行解与代价。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "error: {} 在 {} 上没有可行变体", self.op, self.device)?;
        for report in &self.needs {
            writeln!(f, "  need  {}", report.need.signature())?;
            match &report.have {
                Some(have) => writeln!(f, "  have  {}", have.summary())?,
                None => writeln!(f, "  have  —（该档案未声明 `{}`）", report.need.prim)?,
            }
            writeln!(f, "         ↳ 不满足：{}", report.reason())?;
        }
        writeln!(f, "  降级链：")?;
        for step in &self.chain {
            if step.ok {
                writeln!(
                    f,
                    "    {} ✓ （可用，预计 mma_share {:.2}）",
                    step.variant, step.mma_share
                )?;
            } else {
                writeln!(
                    f,
                    "    {} ✗ （{}）",
                    step.variant,
                    describe_failures(&step.failures)
                )?;
            }
        }
        match &self.unlock {
            Some(unlock) => {
                let missing: Vec<String> = unlock
                    .missing
                    .iter()
                    .map(|prim| format!("`{prim}`"))
                    .collect();
                // 档位也报：建议的解通常不是首选（首选缺得更少时才会是它），读者要能看清
                // 这个建议换来的代价是“降级到第 k 档”，而不是误以为补完就是满配。
                let step = self
                    .chain
                    .iter()
                    .position(|step| step.variant == unlock.variant)
                    .map_or_else(|| "?".to_string(), |index| format!("{}", index + 1));
                writeln!(
                    f,
                    "  建议：在 {device} 补齐 {} 的 L2 见证后可用 {}（第 {step}/{} 档，预计 mma_share {:.2}）",
                    missing.join(" "),
                    unlock.variant,
                    self.chain.len(),
                    unlock.mma_share,
                    device = self.device
                )?;
            }
            None => writeln!(
                f,
                "  建议：该设备缺失全部原语，先补 `provides` 再谈特化片段"
            )?,
        }
        Ok(())
    }
}

impl std::error::Error for NoPath {}

/// 检查一个变体的全部需求；空向量表示可行。
fn check_variant(op: &OpDecl, prof: &Profile, variant: &Variant, proof: &Proof) -> Vec<NeedReport> {
    let mut reports = Vec::new();
    for need in variant.needs(op) {
        reports.push(check_need(prof, need, proof));
    }
    for prim in variant.schedule.required_prims() {
        let need = Need::of(prim);
        let report = check_need(prof, &need, proof);
        if !report.is_satisfied() {
            reports.push(report);
        }
    }
    reports
}

/// 同一原语可能有多条签名（`dot` 的 `dp4a` 与标量 `fma`），挑一条来比对：
/// 先看谁能**覆盖**需求，再看**这次有没有执行证据**，最后看证据等级；都不覆盖时取差异最少的
/// 那条——诊断里的 `have` 行必须指向最接近的那份实现，否则打印出来的是一份与需求无关的签名。
fn select(prof: &Profile, need: &Need, proof: &Proof) -> Option<(Provide, Vec<Mismatch>)> {
    let mut best: Option<(Provide, Vec<Mismatch>)> = None;
    for have in prof.providers(need.prim) {
        let diff = have.diff(need);
        let proven = proof.holds(have);
        let better = match &best {
            None => true,
            Some((current, current_diff)) => {
                let current_proven = proof.holds(current);
                match (diff.is_empty(), current_diff.is_empty()) {
                    (true, false) => true,
                    (false, true) => false,
                    // 两条都能覆盖：先取真的跑过的，再取等级高的。反过来的话，一份自述过来的
                    // `L3` 会把本次唯一有见证的那条挤掉，协商结果取决于哪个数字写得更漂亮。
                    (true, true) => match (proven, current_proven) {
                        (true, false) => true,
                        (false, true) => false,
                        _ => have.grade > current.grade,
                    },
                    (false, false) => diff.len() < current_diff.len(),
                }
            }
        };
        if better {
            best = Some((have.clone(), diff));
        }
    }
    best
}

/// 片段层与档案的签名差异：协商会挑中的那条实现，片段真的写了吗。
///
/// 这是"签名对齐"唯一必须两边一起看的判据：`needs ↔ provides` 由 [`negotiate`] 判，
/// `provides ↔ witness` 由 `audit::witness` 判，而"档案挑中的实现"与"片段文本写的实现"
/// 之间没有任何别的检查。少了它，一份档案可以同时"协商可行"且"片段存在"，两者却指向不同的
/// 实现——错误会晚到构建期，甚至把 `dp4a` 的文本当成 `fma` 用。
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentGap {
    /// 原语。
    pub prim: Prim,
    /// 协商会挑中的那条供给（`None`：档案里没有这个原语的任何签名）。
    pub chosen: Option<Provide>,
    /// 片段声明的实现（`None`：片段层没有这段文本）。
    pub declared: Option<String>,
}

impl FragmentGap {
    /// 片段声明实现的，是否就是协商会挑中的那条。
    pub fn is_aligned(&self) -> bool {
        match (&self.chosen, &self.declared) {
            (Some(chosen), Some(declared)) => chosen.impl_name == *declared,
            _ => false,
        }
    }
}

impl fmt::Display for FragmentGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.chosen, &self.declared) {
            (Some(chosen), Some(declared)) => write!(
                f,
                "`{}` 协商会挑中 `{}`（{}），但片段声明实现了 `{declared}`：\
                 同一原语的两条签名不能互换，先让文本与档案对齐",
                self.prim, chosen.impl_name, chosen.grade
            ),
            (Some(chosen), None) => write!(
                f,
                "`{}` 协商会挑中 `{}`（{}），但片段层没有这段文本",
                self.prim, chosen.impl_name, chosen.grade
            ),
            (None, Some(declared)) => write!(
                f,
                "片段声明实现了 `{}/{declared}`，但档案里没有 `{}` 的任何签名",
                self.prim, self.prim
            ),
            (None, None) => write!(f, "`{}` 既没有档案签名也没有片段文本", self.prim),
        }
    }
}

/// 逐变体、逐条需求比对"协商会挑中的实现"与"片段声明实现的实现"。
///
/// 判据与 [`negotiate`] 同源（同一张需求表、同一个 [`plan::select`](crate::plan::select)），所以"协商能通过"与
/// "片段能生成"不会各算一份。只返回差异；同一原语在多个变体里重复出现在差异里时去重。
pub fn fragment_gaps<'a>(
    op: &OpDecl,
    prof: &Profile,
    declared: impl Fn(Prim) -> Option<&'a str>,
) -> Vec<FragmentGap> {
    let mut gaps: Vec<FragmentGap> = Vec::new();
    for variant in &op.variants {
        let mut needs: Vec<Need> = variant.needs(op).cloned().collect();
        needs.extend(variant.schedule.required_prims().into_iter().map(Need::of));
        for need in &needs {
            let gap = FragmentGap {
                prim: need.prim,
                chosen: select(prof, need, &Proof::declared(prof)).map(|(have, _)| have),
                declared: declared(need.prim).map(str::to_string),
            };
            if !gap.is_aligned() && !gaps.contains(&gap) {
                gaps.push(gap);
            }
        }
    }
    gaps
}

fn check_need(prof: &Profile, need: &Need, proof: &Proof) -> NeedReport {
    match select(prof, need, proof) {
        None => NeedReport {
            need: need.clone(),
            have: None,
            blocked_by_grade: false,
            unproven: false,
            mismatches: vec![Mismatch::PrimAbsent { prim: need.prim }],
        },
        Some((have, mismatches)) => NeedReport {
            need: need.clone(),
            blocked_by_grade: !have.grade.is_verified(),
            // 等级本来就不够时不必再说"没执行"：那是同一个结论的另一种说法，两行叠在一起只会
            // 让人以为补一条见证就能绕过 `L1`。
            unproven: have.grade.is_verified() && !proof.holds(&have),
            have: Some(have),
            mismatches,
        },
    }
}

/// ② negotiate 本体。
///
/// 按档案自述协商（[`Proof::declared`]）：调用方没有执行证据时的唯一选择，也是
/// `verify` 报告里“§16 协商判据”那一栏的口径。执行路径要用 [`negotiate_with`]。
pub fn negotiate(op: &OpDecl, prof: &Profile) -> Result<Path, Box<NoPath>> {
    negotiate_with(op, prof, &Proof::declared(prof))
}

/// ② negotiate 本体：供给必须在 `proof` 里，才算这次具备。
///
/// `proof` 空集不是错误：它表示“这次什么都没执行”，于是每一档都不可行，而失败原因会写明
/// 是“L2 但没有这次执行的见证”——拒绝的理由必须能被读出来。
pub fn negotiate_with(op: &OpDecl, prof: &Profile, proof: &Proof) -> Result<Path, Box<NoPath>> {
    let mut chain = Vec::new();
    let mut first_failure: Option<Vec<NeedReport>> = None;
    // 变体下标与它缺的原语一起记：缺项必须与建议里的变体名同源，否则“补齐这些就能用”是假话。
    let mut closest: Option<(usize, usize, Vec<Prim>)> = None;

    for (index, variant) in op.variants.iter().enumerate() {
        let reports = check_variant(op, prof, variant, proof);
        let ok = reports.iter().all(NeedReport::is_satisfied);
        let failing: Vec<NeedReport> = reports
            .into_iter()
            .filter(|report| !report.is_satisfied())
            .collect();
        chain.push(ChainStep {
            variant: variant.name.clone(),
            schedule: variant.schedule,
            ok,
            failures: failing.clone(),
            mma_share: variant.mma_share,
        });
        if ok {
            return Ok(Path {
                variant: variant.name.clone(),
                index,
                schedule: variant.schedule,
                mma_share: variant.mma_share,
            });
        }
        // 按原语去重：`needs` 允许同一原语写多个 dtype，调度也会再要一次（`bar`）；建议说的是
        // “补哪些原语”，所以同样只算一条——“缺得最少”的挑选也用这个数，否则两条 dtype 需求会
        // 让真正只差一个原语的变体看起来更远。
        let mut missing: Vec<Prim> = Vec::new();
        for report in &failing {
            if !missing.contains(&report.need.prim) {
                missing.push(report.need.prim);
            }
        }
        if closest
            .as_ref()
            .is_none_or(|(_, count, _)| missing.len() < *count)
        {
            closest = Some((index, missing.len(), missing));
        }
        if first_failure.is_none() {
            first_failure = Some(failing);
        }
    }

    let unlock = closest.and_then(|(index, _, missing)| {
        op.variants.get(index).map(|variant| Unlock {
            variant: variant.name.clone(),
            missing,
            mma_share: variant.mma_share,
        })
    });

    Err(Box::new(NoPath {
        op: op.name.clone(),
        device: format!("{}({})", prof.meta.vendor, prof.meta.id),
        needs: first_failure.unwrap_or_default(),
        chain,
        unlock,
    }))
}

/// 逐变体复核整条降级链：`doctor` 要解释“为何选它、其它变体差在哪”，
/// 而 [`negotiate`] 命中即停，不持有后面的判定。
///
/// 这里只给结构化结果；“✓ 选中/·”这类标记与缩进由前端根据 [`Path::variant`] 排版。
pub fn inspect(op: &OpDecl, prof: &Profile) -> Vec<VariantCheck> {
    inspect_with(op, prof, &Proof::declared(prof))
}

/// 逐变体复核整条降级链，判据与 [`negotiate_with`] 同源。
///
/// 这里只给结构化结果；“✓ 选中/·”这类标记与缩进由前端根据 [`Path::variant`] 排版。
pub fn inspect_with(op: &OpDecl, prof: &Profile, proof: &Proof) -> Vec<VariantCheck> {
    op.variants
        .iter()
        .map(|variant| VariantCheck {
            name: variant.name.clone(),
            failures: check_variant(op, prof, variant, proof)
                .into_iter()
                .filter(|report| !report.is_satisfied())
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::{Attr, AttrKey, DType, DTypeSet};
    use crate::decl::{Axis, Skeleton};
    use crate::sch::Schedule;

    fn op() -> OpDecl {
        let source = &op::builtin_named("gemv_i4").expect("内置");
        OpDecl::parse(source.declaration(), source.template()).expect("声明可解析")
    }

    fn cpu() -> Profile {
        Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案可解析")
    }

    fn nvidia() -> Profile {
        Profile::parse(prof::embedded("nvidia").expect("nvidia 档案")).expect("档案可解析")
    }

    /// 一个模板只用 `dot` 的合成声明：差异全部来自变体级 `extra_needs`。
    ///
    /// 内置 `gemv_i4` 的模板由所有变体共享，`load_async` / `commit_wait` / `bar`
    /// 因此是算子级需求（装载期校验），不能再拿它演示"缺一个原语就降级"。
    fn downgrade_op() -> OpDecl {
        // 合成声明只需要一份可解析的签名：协商不读 ABI，但 `OpDecl` 的不变量要求它存在。
        let template = "void cutile_probe_kernel(float *y) { @dot(acc, a, b, k); }\n";
        OpDecl {
            name: "probe".to_string(),
            skeleton: Skeleton::Gemm,
            summary: "-".to_string(),
            axes: vec![Axis {
                name: "GMAX".to_string(),
                values: vec![32],
            }],
            needs: vec![Need::of(Prim::Dot)],
            layouts: Vec::new(),
            variants: vec![
                Variant {
                    name: "pipe2".to_string(),
                    order: 1,
                    schedule: Schedule::parse("pipe<2>").expect("调度"),
                    mma_share: 0.62,
                    extra_needs: vec![Need::of(Prim::LoadAsync), Need::of(Prim::CommitWait)],
                },
                Variant {
                    name: "direct".to_string(),
                    order: 2,
                    schedule: Schedule::parse("direct").expect("调度"),
                    mma_share: 0.35,
                    extra_needs: Vec::new(),
                },
            ],
            template: template.to_string(),
            signature: op::sig::Signature::parse("probe", template).expect("签名"),
        }
    }

    #[test]
    fn cpu_hits_the_first_variant() {
        let path = negotiate(&op(), &cpu()).expect("CPU 应命中首档");
        assert_eq!(path.variant, "pipe2");
        assert_eq!(path.index, 0);
        assert!(path.schedule.uses_async());
    }

    #[test]
    fn missing_evidence_blocks_every_variant() {
        let error = negotiate(&op(), &nvidia()).unwrap_err();
        assert!(error.to_string().contains("没有可行变体"));
        assert!(error.chain.iter().all(|step| !step.ok));
        assert!(
            error.to_string().contains("L2"),
            "证据不足要明确指出：{error}"
        );
    }

    #[test]
    fn variant_level_needs_drive_downgrade() {
        let op = downgrade_op();
        let mut profile = cpu();
        // 去掉 load_async / commit_wait：pipe2 需要它们，direct 不需要 —— 这正是 extra_needs 的意义。
        profile
            .provides
            .retain(|have| have.prim != Prim::LoadAsync && have.prim != Prim::CommitWait);
        let path = negotiate(&op, &profile).expect("应降级到 direct");
        assert_eq!(path.variant, "direct");
        assert!(!path.schedule.uses_async());
    }

    #[test]
    fn diagnostics_name_the_missing_atom_and_the_unlock() {
        // dot 是算子级公共需求：缺它时整条链都不可行，诊断必须点名并给出解锁路径。
        let mut profile = cpu();
        profile.provides.retain(|have| have.prim != Prim::Dot);
        let error = negotiate(&op(), &profile).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("dot"), "{text}");
        assert!(text.contains("建议："), "{text}");
        assert!(text.contains("L2 见证"), "{text}");
    }

    /// 解锁建议必须自洽：变体名与“还缺哪些原语”来自同一个变体。
    ///
    /// 首选缺 `dot` 与 `load_async`、兜底只缺 `dot` 时，建议应指向兜底（第 2 档）且只列 `dot`。
    /// 把首选的变体名配上兜底的缺项，会让人补完证之后发现仍然不可用。
    #[test]
    fn the_unlock_advice_names_the_variant_its_missing_atoms_belong_to() {
        let op = downgrade_op();
        let mut profile = cpu();
        profile
            .provides
            .retain(|have| have.prim != Prim::Dot && have.prim != Prim::LoadAsync);
        let error = negotiate(&op, &profile).unwrap_err();
        let unlock = error.unlock.as_ref().expect("应当给出解锁路径");
        assert_eq!(unlock.variant, "direct", "{unlock:?}");
        assert_eq!(unlock.missing, vec![Prim::Dot], "{unlock:?}");
        let text = error.to_string();
        assert!(text.contains("可用 direct（第 2/2 档"), "{text}");
        assert!(!text.contains("可用 pipe2"), "{text}");
    }

    /// 缺项按原语去重：调度与 `needs` 都要 `bar` 时，建议只列一次。
    ///
    /// 真实数据里就会撞到这条（`pipe<2>` 的调度要 `bar`，声明里往往也写了一条），重复的原语会让
    /// “补哪些原语”读成“补两条需求”，也会让“缺得最少”的挑选偏向需求条数多的变体。
    #[test]
    fn the_unlock_advice_lists_each_atom_once() {
        let mut op = downgrade_op();
        op.needs = vec![Need::of(Prim::Bar)];
        op.variants.truncate(1);
        let mut profile = cpu();
        profile.provides.retain(|have| have.prim != Prim::Bar);
        let error = negotiate(&op, &profile).unwrap_err();
        let unlock = error.unlock.as_ref().expect("应当给出解锁路径");
        assert_eq!(unlock.missing, vec![Prim::Bar], "{unlock:?}");
        let text = error.to_string();
        let advice = text.split("建议：").nth(1).expect("有建议行");
        assert_eq!(advice.matches("`bar`").count(), 1, "{advice}");
    }

    #[test]
    fn missing_async_atom_only_disables_the_pipelined_variant() {
        let op = downgrade_op();
        let mut profile = cpu();
        profile.provides.retain(|have| have.prim != Prim::LoadAsync);
        let path = negotiate(&op, &profile).expect("应降级到 direct");
        assert_eq!(path.variant, "direct");
        // 解释不复用协商结果：inspect 总是看完整条链，因此被跳过的 pipe2 也在。
        let checks = inspect(&op, &profile);
        let piped = &checks[0];
        assert_eq!(piped.name, "pipe2");
        assert!(
            !piped.failures.is_empty(),
            "pipe2 应该报出缺失的 load_async"
        );
    }

    /// P0 的每条需求都必须被某一条签名**结构性**覆盖：剩下的唯一障碍只能是证据等级。
    ///
    /// 这正是"签名对齐"的定义——`dt`/`lanes`/`dst` 对不上时补见证也没用，不能混在 L0/L1 里。
    #[test]
    fn every_p0_need_is_covered_by_some_signature_on_every_accelerator() {
        for vendor in ["nvidia", "intel", "ascend"] {
            let profile =
                Profile::parse(prof::embedded(vendor).expect("内置档案")).expect("档案可解析");
            let checks = inspect(&op(), &profile);
            assert!(!checks.is_empty(), "{vendor} 应当有降级链");
            for check in &checks {
                for failure in &check.failures {
                    assert!(
                        failure.mismatches.is_empty(),
                        "{vendor}: {} 仍然不同构：{}",
                        failure.need.signature(),
                        failure.reason()
                    );
                    assert!(failure.blocked_by_grade, "{vendor}: 只应被证据等级挡住");
                }
            }
        }
    }

    #[test]
    fn the_signature_that_covers_the_need_is_the_one_compared() {
        let profile = nvidia();
        let need = Need::of(Prim::Dot)
            .with_dt(DTypeSet::of([DType::F32]))
            .with_attr(Attr::new(AttrKey::Lanes, ["1"]));
        // 这条用例只验"签名比对挑中了哪条供给"，与证据等级无关：按档案自述取证据即可。
        let report = check_need(&profile, &need, &Proof::declared(&profile));
        assert!(report.mismatches.is_empty(), "{}", report.reason());
        assert_eq!(report.have.expect("nvidia 有 dot 供给").impl_name, "fma");
    }

    #[test]
    fn inspect_reports_the_whole_chain_as_data() {
        let checks = inspect(&op(), &cpu());
        assert_eq!(checks.len(), 2);
        assert!(checks.iter().all(|check| check.failures.is_empty()));
        assert_eq!(checks[0].name, "pipe2");
        assert_eq!(checks[1].name, "direct");
    }

    /// 同一份能力签名在两层里换一种合法写法（标量/数组、`bytes`/`m`）必须仍然同构。
    ///
    /// 声明层的 `[needs]` 与档案层的 `[[provides]]` 各有一个解析器。它们若无共同词汇，同一份
    /// 签名就会在一边多出或少掉一个约束，于是"协商通过"与"档案声明"说的不是一回事。这里
    /// 直接拿协商用的覆盖关系当判据，两种写法两个方向都要成立。
    #[test]
    fn one_signature_written_in_either_layer_stays_the_same() {
        fn declaration(needs: &str) -> OpDecl {
            let text = format!(
                "[op]\n\
                 name = \"probe\"\n\
                 skeleton = \"gemm\"\n\
                 summary = \"-\"\n\
                 \n\
                 [axes]\n\
                 GMAX = [32]\n\
                 \n\
                 [needs]\n\
                 {needs}\n\
                 \n\
                 [[variants]]\n\
                 name = \"direct\"\n\
                 order = 1\n\
                 schedule = \"direct\"\n\
                 mma_share = 0.8\n"
            );
            let template = "void cutile_probe_kernel(const float *x, float *y) { @load(y, x); }\n";
            OpDecl::parse(&text, template).expect("合成声明可解析")
        }

        fn provide(entry: &str) -> Provide {
            let text = format!(
                "{}\n[[provides]]\n{entry}\ngrade = \"L0\"\n",
                prof::embedded("cpu").expect("cpu 档案")
            );
            Profile::parse(&text)
                .expect("合成档案可解析")
                .provides
                .pop()
                .expect("新加的供给")
        }

        fn load(op: &OpDecl) -> Need {
            op.needs
                .iter()
                .find(|need| need.prim == Prim::Load)
                .expect("load 需求")
                .clone()
        }

        // 声明写标量、档案写数组与 `m`。
        let scalar = declaration(
            "load = { bytes = \"<=4096\", dt = \"f32\", space = \"FAST\", src = \"GM\", dst = \"FAST\" }",
        );
        let array_provide = provide(
            "prim = \"load\"\nimpl = \"alias\"\nm = \"<=4096\"\ndt = [\"f32\"]\n\
             space = [\"FAST\"]\nsrc = [\"GM\"]\ndst = [\"FAST\"]",
        );
        let need = load(&scalar);
        assert!(
            array_provide.covers(&need),
            "档案写法必须覆盖同一条声明：{:?}",
            array_provide.diff(&need)
        );

        // 反向：声明写数组、档案写标量。
        let array = declaration(
            "load = { m = \"<=4096\", dt = [\"f32\"], space = [\"FAST\"], src = [\"GM\"], dst = [\"FAST\"] }",
        );
        let scalar_provide = provide(
            "prim = \"load\"\nimpl = \"alias\"\nbytes = \"<=4096\"\ndt = \"f32\"\n\
             space = \"FAST\"\nsrc = \"GM\"\ndst = \"FAST\"",
        );
        let need = load(&array);
        assert!(
            scalar_provide.covers(&need),
            "声明写法必须覆盖同一条档案：{:?}",
            scalar_provide.diff(&need)
        );
    }

    #[test]
    fn schedule_primitives_are_part_of_the_requirement() {
        let op = downgrade_op();
        // 只剩 direct 可用时 bar 不再被需要。
        let mut profile = cpu();
        profile.provides.retain(|have| have.prim != Prim::Bar);
        let path = negotiate(&op, &profile).expect("应降级到 direct");
        assert_eq!(path.variant, "direct");

        // 再去掉公共需求里的 dot：两条链都不可行，诊断必须点名 bar（来自 pipe<2> 调度）。
        let mut profile = cpu();
        profile
            .provides
            .retain(|have| have.prim != Prim::Bar && have.prim != Prim::Dot);
        let error = negotiate(&op, &profile).unwrap_err();
        assert!(error.to_string().contains("bar"), "{error}");
    }

    /// 空证据不是"全都具备"：链会走完，每一档都不可行，原因写的是"欠一次执行"。
    #[test]
    fn an_unexecuted_l2_offer_does_not_negotiate() {
        let op = op();
        let profile = cpu();
        let error = negotiate_with(&op, &profile, &Proof::none()).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("没有这次执行的见证"), "{text}");
        assert!(!error.needs.is_empty(), "{error}");
        assert!(
            error.needs.iter().all(|need| !need.is_satisfied()),
            "空证据下一档都不能满足：{:?}",
            error.needs
        );
        // 它不是因为"档案没写"而失败：这些供给都写着 L2，缺的是执行。
        assert!(
            error
                .needs
                .iter()
                .all(|need| !need.blocked_by_grade && need.unproven),
            "{:?}",
            error.needs
        );
    }

    /// 档案自述是解释性报告的口径：同一份档案按声明协商照样命中首档。
    #[test]
    fn the_declared_reading_is_still_available_for_reports() {
        let op = op();
        let profile = cpu();
        let declared = Proof::declared(&profile);
        assert_eq!(
            declared.len(),
            profile
                .provides
                .iter()
                .filter(|have| have.grade.is_verified())
                .count(),
            "每一条 grade ≥ L2 的供给都在里面"
        );
        let path = negotiate_with(&op, &profile, &declared).expect("按声明可行");
        assert!(path.is_head());
    }

    /// 两条覆盖同一需求的供给：先取这次真的跑过的那条，而不是等级写得漂亮的那条。
    #[test]
    fn a_proven_supply_wins_over_a_higher_grade_that_was_not_executed() {
        let op = op();
        let need = op
            .needs
            .iter()
            .find(|need| need.prim == Prim::Dot)
            .expect("dot 需求")
            .clone();
        let mut profile = cpu();
        // 一条从没被执行过的 `L3`：签名与 `scalar_f32` 逐项相同，只有等级写得更漂亮。
        let mut vendor = profile
            .providers(Prim::Dot)
            .next()
            .cloned()
            .expect("cpu 有 dot 供给");
        vendor.impl_name = "vendor_dot".to_string();
        vendor.grade = Grade::L3;
        profile.provides.push(vendor);

        let mut proof = Proof::none();
        for have in &profile.provides {
            if have.impl_name != "vendor_dot" {
                proof.insert(have.prim, &have.impl_name);
            }
        }
        let (chosen, _) = select(&profile, &need, &proof).expect("有供给");
        assert_eq!(
            chosen.impl_name, "scalar_f32",
            "不能拿没执行过的等级去挤掉跑过的实现"
        );

        // 反向：它真的跑过了，就按等级取它。
        let proven = proof.with(Prim::Dot, "vendor_dot");
        let (chosen, _) = select(&profile, &need, &proven).expect("有供给");
        assert_eq!(chosen.impl_name, "vendor_dot");
    }
}
