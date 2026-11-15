//! ② negotiate：需求 ∩ 供给，显式降级。命中第一个可行变体即停——链序就是 TOML 里写的优先级，
//! 不做自动打分（可解释性成本高于收益）；失败诊断必须给出"缺什么、供给方是什么、最近的可行解与代价"。

use crate::atom::{Grade, Mismatch, Need, Prim, Provide};
use crate::decl::{OpDecl, Variant};
use crate::prof::Profile;
use crate::sch::Schedule;
use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

/// 协商结果：命中的变体、调度与它在降级链里的下标（[`Path::is_head`] 要读下标）。
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
    /// 是否命中降级链的首选变体：落到后面的档说明首选的能力没被满足。
    #[must_use]
    pub fn is_head(&self) -> bool {
        self.index == 0
    }
}

/// 证据来源：自述（档案等级 ≥ L2 即算具备）与本次执行见证分开存，自述路径不为每次协商重建集合。
#[derive(Debug, Clone, PartialEq, Eq)]
enum ProofState {
    /// 按档案自述：`grade ≥ L2` 即视为具备；计数只用于报告条数。
    Declared(usize),
    /// 本次执行记下的见证集合（原语 → 实现名集合）。
    Holds(BTreeMap<Prim, BTreeSet<String>>),
}

/// 这次真的执行过、并且通过了的供给：档案里的 [`Grade`] 是作者自述，本类型是执行结论；两者同时成立才算可用。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof {
    state: ProofState,
}

impl Default for Proof {
    fn default() -> Self {
        Self {
            state: ProofState::Holds(BTreeMap::new()),
        }
    }
}

impl Proof {
    /// 什么都没执行过：执行路径用不到它，但"证据还没到"必须能被表达，而不是"按档案全具备"。
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// 按档案自述：每条 `grade ≥ L2` 的供给都算可用。只给解释性报告用，任何声称"能跑"的路径都不许用它。
    #[must_use]
    pub fn declared(prof: &Profile) -> Self {
        let verified = prof
            .provides
            .iter()
            .filter(|have| have.grade.is_verified())
            .count();
        Self {
            state: ProofState::Declared(verified),
        }
    }

    /// 记下一条真正跑通的供给。执行路径从 [`Proof::none`] 起步；若误从自述模式写入，先落到空见证集合。
    pub fn insert(&mut self, prim: Prim, impl_name: &str) {
        if let ProofState::Declared(_) = self.state {
            self.state = ProofState::Holds(BTreeMap::new());
        }
        let ProofState::Holds(map) = &mut self.state else {
            return;
        };
        map.entry(prim).or_default().insert(impl_name.to_string());
    }

    /// 这条供给这次是否被证明：认 `(原语, 实现名)`，因为同一供给的参数不变，变的是"跑没跑"。
    #[must_use]
    pub fn holds(&self, have: &Provide) -> bool {
        match &self.state {
            ProofState::Declared(_) => have.grade.is_verified(),
            ProofState::Holds(map) => map
                .get(&have.prim)
                .is_some_and(|impls| impls.contains(have.impl_name.as_str())),
        }
    }

    /// 被证明的供给条数。
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.state {
            ProofState::Declared(count) => *count,
            ProofState::Holds(map) => map.values().map(BTreeSet::len).sum(),
        }
    }

    /// 一条也没有。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match &self.state {
            ProofState::Declared(_) => false,
            ProofState::Holds(map) => map.is_empty(),
        }
    }
}

/// 一条需求的检查结果。`blocked_by_grade` 与 `unproven` 必须分开，否则报告说不出"这条能力欠一次执行"。
#[derive(Debug, Clone, PartialEq)]
pub struct NeedReport {
    /// 需求。
    pub need: Need,
    /// 供给方（同原语的档案条目）。
    pub have: Option<Provide>,
    /// 供给存在但证据等级不足 `L2`。
    pub blocked_by_grade: bool,
    /// 供给声明了 `L2`，但这次执行没有拿到它的见证。
    pub unproven: bool,
    /// 逐项差异。
    pub mismatches: Vec<Mismatch>,
}

impl NeedReport {
    /// 是否满足。
    #[must_use]
    pub fn is_satisfied(&self) -> bool {
        self.mismatches.is_empty()
            && !self.blocked_by_grade
            && !self.unproven
            && self.have.is_some()
    }
}

/// 降级链上的一步。`failures` 只存结构，原因文字由 [`NeedReport::reason`] 给出。
#[derive(Debug, Clone, PartialEq)]
pub struct ChainStep {
    /// 变体名。
    pub variant: String,
    /// 调度。
    pub schedule: Schedule,
    /// 是否可行。
    pub ok: bool,
    /// 未满足的需求（`ok` 时为空）。
    pub failures: Vec<NeedReport>,
    /// `mma_share`。
    pub mma_share: f64,
}

/// 一个变体的需求复核结果，供 `doctor` 逐条解释降级链：协商命中即停，解释则要看完整条链。
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

impl std::error::Error for NoPath {}

/// 一档里满足不了的需求；空向量表示这一档可行。判定只做一次，各判一次会让"协商能过"与"报告说不能过"同时成立。
fn failing_needs(op: &OpDecl, prof: &Profile, variant: &Variant, proof: &Proof) -> Vec<NeedReport> {
    let mut reports: Vec<NeedReport> = variant
        .needs(op)
        .filter_map(|need| failing_report(prof, need, proof))
        .collect();
    // 调度隐式要求的原语（如 `bar`）同样是需求，只留不满足的那几条。
    reports.extend(
        variant
            .schedule
            .required_prims()
            .iter()
            .copied()
            .map(Need::of)
            .filter_map(|need| failing_report(prof, &need, proof)),
    );
    reports
}

/// 同一原语的一条候选供给：与需求的签名差异，以及这次执行有没有留下证据；分开算会指向另一条实现。
struct Offer<'a> {
    have: &'a Provide,
    diff: Vec<Mismatch>,
    proven: bool,
}

impl<'a> Offer<'a> {
    /// 折成可比较的形态：差异只算一次，证据只查一次；候选只借档案里的供给，胜出后才克隆。
    fn new(have: &'a Provide, need: &Need, proof: &Proof) -> Self {
        Self {
            have,
            diff: have.diff(need),
            proven: proof.holds(have),
        }
    }

    /// 是否该替换已选中的候选：先看谁覆盖需求，再看"这次跑过"，最后看证据等级（自述的 `L3` 必须输给本次有见证的 `L2`）。
    fn beats(&self, current: &Self) -> bool {
        let proven_first = (self.proven, self.have.grade) > (current.proven, current.have.grade);
        match (self.diff.is_empty(), current.diff.is_empty()) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => proven_first,
            (false, false) => self.diff.len() < current.diff.len(),
        }
    }

    /// 需求被这条供给真正满足：签名全覆盖、等级已验证、这次有见证。
    fn satisfied(&self) -> bool {
        self.diff.is_empty() && self.have.grade.is_verified() && self.proven
    }
}

/// 同一原语可能有多条签名（`dot` 的 `dp4a` 与标量 `fma`），挑一条来比对：先看覆盖，再看证据，都不覆盖时取差异最少。
pub(crate) fn select(
    prof: &Profile,
    need: &Need,
    proof: &Proof,
) -> Option<(Provide, Vec<Mismatch>)> {
    select_borrowed(prof, need, proof).map(|offer| (offer.have.clone(), offer.diff))
}

/// 挑出胜出候选，只借档案里的供给、不克隆；确认为失败的报告才在 [`select`] 或构造器里克隆。
fn select_borrowed<'a>(prof: &'a Profile, need: &Need, proof: &Proof) -> Option<Offer<'a>> {
    let mut best: Option<Offer<'_>> = None;
    for have in prof.providers(need.prim) {
        let offer = Offer::new(have, need, proof);
        if best.as_ref().is_none_or(|current| offer.beats(current)) {
            best = Some(offer);
        }
    }
    best
}

/// 片段层与档案的签名差异：协商会挑中的那条实现，片段真的写了吗——缺了它，两者会各自"可行"却指向不同实现。
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
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        match (&self.chosen, &self.declared) {
            (Some(chosen), Some(declared)) => chosen.impl_name == *declared,
            _ => false,
        }
    }
}

/// 逐变体比对"协商会挑中的实现"与"片段声明的实现"，判据与 [`negotiate`] 同源，因此两边不会各算一份。
pub fn fragment_gaps<'a>(
    op: &OpDecl,
    prof: &Profile,
    declared: impl Fn(Prim) -> Option<&'a str>,
) -> Vec<FragmentGap> {
    let mut gaps: Vec<FragmentGap> = Vec::new();
    for variant in &op.variants {
        let mut needs: Vec<Need> = variant.needs(op).cloned().collect();
        needs.extend(
            variant
                .schedule
                .required_prims()
                .iter()
                .copied()
                .map(Need::of),
        );
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

/// 仅当需求不满足时构造报告；满足路径不克隆胜出供给、也不构造随后被过滤丢弃的报告。
fn failing_report(prof: &Profile, need: &Need, proof: &Proof) -> Option<NeedReport> {
    let Some(offer) = select_borrowed(prof, need, proof) else {
        return Some(NeedReport {
            need: need.clone(),
            have: None,
            blocked_by_grade: false,
            unproven: false,
            mismatches: vec![Mismatch::PrimAbsent { prim: need.prim }],
        });
    };
    let verified = offer.have.grade.is_verified();
    if offer.satisfied() {
        return None;
    }
    Some(NeedReport {
        need: need.clone(),
        blocked_by_grade: !verified,
        unproven: verified && !offer.proven,
        have: Some(offer.have.clone()),
        mismatches: offer.diff,
    })
}

/// 一条需求的检查结果。等级本来就不够时不再说"没执行"：那是同一个结论的另一种说法。
#[cfg(test)]
pub(crate) fn check_need(prof: &Profile, need: &Need, proof: &Proof) -> NeedReport {
    match select_borrowed(prof, need, proof) {
        None => NeedReport {
            need: need.clone(),
            have: None,
            blocked_by_grade: false,
            unproven: false,
            mismatches: vec![Mismatch::PrimAbsent { prim: need.prim }],
        },
        Some(offer) => NeedReport {
            need: need.clone(),
            blocked_by_grade: !offer.have.grade.is_verified(),
            unproven: offer.have.grade.is_verified() && !offer.proven,
            have: Some(offer.have.clone()),
            mismatches: offer.diff,
        },
    }
}

/// ② negotiate 本体：按档案自述协商（[`Proof::declared`]），这是没有执行证据时的唯一选择，也是 `verify` 报告的口径。
pub fn negotiate(op: &OpDecl, prof: &Profile) -> Result<Path, Box<NoPath>> {
    negotiate_with(op, prof, &Proof::declared(prof))
}

/// 一次协商的账本：整条降级链、首选那一档的缺项、缺得最少的那一档；失败后再走一遍链就不再是本次发生过的事。
#[derive(Default)]
struct Ledger {
    chain: Vec<ChainStep>,
    first_failure: Option<Vec<NeedReport>>,
    closest: Option<(usize, Vec<Prim>)>,
}

impl Ledger {
    /// 记下一档的结果，返回它是否可行；`ok` 字段与"是否命中"必须是同一个布尔。
    fn record(&mut self, index: usize, variant: &Variant, failing: Vec<NeedReport>) -> bool {
        let ok = failing.is_empty();
        self.chain.push(ChainStep {
            variant: variant.name.clone(),
            schedule: variant.schedule,
            ok,
            failures: failing.clone(),
            mma_share: variant.mma_share,
        });
        if !ok {
            self.remember_failure(index, failing);
        }
        ok
    }

    /// 记下这一档的缺项：`first_failure` 答"首选为什么不行"，`closest` 答"补哪一个就能用"，两者不能互相冒充。
    fn remember_failure(&mut self, index: usize, failing: Vec<NeedReport>) {
        let mut missing: Vec<Prim> = Vec::new();
        for report in &failing {
            if !missing.contains(&report.need.prim) {
                missing.push(report.need.prim);
            }
        }
        let fewer = self
            .closest
            .as_ref()
            .is_none_or(|(_, best)| missing.len() < best.len());
        if fewer {
            self.closest = Some((index, missing));
        }
        if self.first_failure.is_none() {
            self.first_failure = Some(failing);
        }
    }

    /// 最近的解锁路径：变体名与缺项来自同一档（只陈述这一档缺这些原语，不承诺补完必然可行）。
    fn unlock(&self, op: &OpDecl) -> Option<Unlock> {
        let (index, missing) = self.closest.as_ref()?;
        op.variants.get(*index).map(|variant| Unlock {
            variant: variant.name.clone(),
            missing: missing.clone(),
            mma_share: variant.mma_share,
        })
    }

    /// 把账本折成失败诊断；设备标签在这里成形（`厂商(编号)`），与档案同源，不接受调用方另传字符串。
    fn into_no_path(self, op: &OpDecl, prof: &Profile) -> NoPath {
        let unlock = self.unlock(op);
        NoPath {
            op: op.name.clone(),
            device: format!("{}({})", prof.meta.vendor, prof.meta.id),
            needs: self.first_failure.unwrap_or_default(),
            chain: self.chain,
            unlock,
        }
    }
}

/// ② negotiate 本体：供给必须在 `proof` 里才算这次具备；空集是"这次什么都没执行"，于是每一档都不可行。
pub fn negotiate_with(op: &OpDecl, prof: &Profile, proof: &Proof) -> Result<Path, Box<NoPath>> {
    let mut ledger = Ledger::default();
    for (index, variant) in op.variants.iter().enumerate() {
        if ledger.record(index, variant, failing_needs(op, prof, variant, proof)) {
            return Ok(Path {
                variant: variant.name.clone(),
                index,
                schedule: variant.schedule,
                mma_share: variant.mma_share,
            });
        }
    }
    Err(Box::new(ledger.into_no_path(op, prof)))
}

/// 逐变体复核整条降级链：`doctor` 要解释"为何选它、其它变体差在哪"，而 [`negotiate`] 命中即停、不持有后面的判定。
pub fn inspect(op: &OpDecl, prof: &Profile) -> Vec<VariantCheck> {
    inspect_with(op, prof, &Proof::declared(prof))
}

/// 逐变体复核整条降级链，判据与 [`negotiate_with`] 同源。
pub fn inspect_with(op: &OpDecl, prof: &Profile, proof: &Proof) -> Vec<VariantCheck> {
    op.variants
        .iter()
        .map(|variant| VariantCheck {
            name: variant.name.clone(),
            failures: failing_needs(op, prof, variant, proof),
        })
        .collect()
}

impl NeedReport {
    /// 一行原因：先报参数不匹配，再报证据——"dp4a 不支持 f32"与"没验证"是两件事。
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

/// 降级链一步的原因文字：`原语：原因`，多条以 `；` 连接；这是错误文本的一部分，排版由调用方决定。
fn describe_failures(failures: &[NeedReport]) -> String {
    failures
        .iter()
        .map(|report| format!("{}：{}", report.need.prim, report.reason()))
        .collect::<Vec<_>>()
        .join("；")
}

impl fmt::Display for NoPath {
    /// README §8.2 的失败诊断，四段固定顺序（身份、`need`/`have` 对照、整条链、建议），缺一段就不成立。
    /// 错误文本是对外接口（终端原样打印、测试按内容断言），所以措辞与判定分开审阅：改一句不该动判定。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "error: {} 在 {} 上没有可行变体", self.op, self.device)?;
        write_needs(f, &self.needs)?;
        write_chain(f, &self.chain)?;
        write_advice(f, self)
    }
}

/// 逐条需求成组输出 `need` / `have` / `↳ 不满足`：少了 `↳` 就没人把它们连成"为什么不行"。
fn write_needs(f: &mut fmt::Formatter<'_>, needs: &[NeedReport]) -> fmt::Result {
    for report in needs {
        writeln!(f, "  need  {}", report.need.signature())?;
        write_have(f, report)?;
        writeln!(f, "         ↳ 不满足：{}", report.reason())?;
    }
    Ok(())
}

/// 供给方那一行：档案里最接近的那份签名，或者"该档案未声明"（与"声明了但不够"分开）。
fn write_have(f: &mut fmt::Formatter<'_>, report: &NeedReport) -> fmt::Result {
    match &report.have {
        Some(have) => writeln!(f, "  have  {}", have.summary()),
        None => writeln!(f, "  have  —（该档案未声明 `{}`）", report.need.prim),
    }
}

/// 整条降级链每档一行，✓ 与 ✗ 的差别就是 `ok`：可用的报调度代价，不可用的报缺项。全链都印——
/// 失败诊断要给的是"这台设备上还有什么"，只印不可行的档等于把读者推回档案原文。
fn write_chain(f: &mut fmt::Formatter<'_>, chain: &[ChainStep]) -> fmt::Result {
    writeln!(f, "  降级链：")?;
    for step in chain {
        let cost = if step.ok {
            format!("✓ （可用，预计 mma_share {:.2}）", step.mma_share)
        } else {
            format!("✗ （{}）", describe_failures(&step.failures))
        };
        writeln!(f, "    {} {cost}", step.variant)?;
    }
    Ok(())
}

/// 建议行：有最近的一档就报"补什么、能用哪档、代价多少"，没有就报"先补档案"。两种形状对应两种现实
/// （只差几条见证 / 档案与算子没有交集），合成一句会让人去补一条本来不需要的见证。
fn write_advice(f: &mut fmt::Formatter<'_>, path: &NoPath) -> fmt::Result {
    let Some(unlock) = &path.unlock else {
        return writeln!(
            f,
            "  建议：该设备缺失全部原语，先补 `provides` 再谈特化片段"
        );
    };
    let missing: Vec<String> = unlock
        .missing
        .iter()
        .map(|prim| format!("`{prim}`"))
        .collect();
    // 档位从 1 起：建议换来的常是"降级到第 k 档"，读者要能看清代价，而不是误以为补完就是满配。
    // 找不到时给 `?` 而不是猜一个数，编出来的档位会让人按错误的位置去改声明。
    let step = path
        .chain
        .iter()
        .position(|step| step.variant == unlock.variant)
        .map_or_else(|| "?".to_string(), |index| format!("{}", index + 1));
    writeln!(
        f,
        "  建议：在 {device} 补齐 {} 的 L2 见证后可用 {}（第 {step}/{} 档，预计 mma_share {:.2}）",
        missing.join(" "),
        unlock.variant,
        path.chain.len(),
        unlock.mma_share,
        device = path.device
    )
}

impl fmt::Display for FragmentGap {
    /// 一句话说清"协商会挑谁、片段写了谁、差在哪一侧"。四种组合各一种措辞：两侧都有但不同、片段缺、
    /// 档案缺、两侧都没有——合成一句"未对齐"会把该改档案还是该改片段这件事重新藏起来。
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

#[cfg(test)]
#[path = "cap_tests.rs"]
mod tests;
