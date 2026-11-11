//! ② negotiate：需求 ∩ 供给，显式降级（DESIGN §8）。
//!
//! 命中第一个可行变体即停：降级链的顺序就是人在 TOML 里写的优先级，不做自动打分——自动
//! 打分的可解释性成本高于收益。失败时给出的诊断必须包含"缺什么、供给方是什么、最近的可行解与代价"。

use crate::atom::{Grade, Mismatch, Need, Prim, Provide};
use crate::lower::expand::{OpDecl, Variant};
use crate::prof::Profile;
use crate::sch::Schedule;
use core::fmt;

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

/// 一条需求的检查结果。
#[derive(Debug, Clone, PartialEq)]
pub struct NeedReport {
    /// 需求。
    pub need: Need,
    /// 供给方（同原语的档案条目）。
    pub have: Option<Provide>,
    /// 供给存在但证据等级不足 `L2`。
    pub blocked_by_grade: bool,
    /// 逐项差异。
    pub mismatches: Vec<Mismatch>,
}

impl NeedReport {
    /// 是否满足。
    pub fn is_satisfied(&self) -> bool {
        self.mismatches.is_empty() && !self.blocked_by_grade && self.have.is_some()
    }

    /// 一行原因，用于降级链：先报参数不匹配，再报证据等级。
    ///
    /// 顺序很重要——"dp4a 不支持 f32" 和"没验证"是两件事，只说后者会让人误以为补见证就能跑。
    fn reason(&self) -> String {
        let mut parts: Vec<String> = self.mismatches.iter().map(ToString::to_string).collect();
        if self.blocked_by_grade {
            let grade = self.have.as_ref().map_or(Grade::L0, |have| have.grade);
            parts.push(format!("{grade} < L2（未验证视为不具备）"));
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
    /// 不可行的原因。
    pub reason: String,
    /// `mma_share`。
    pub mma_share: f64,
}

/// 距离可行只差哪些能力：用来给出"补哪一个见证就能解锁"的建议。
#[derive(Debug, Clone, PartialEq)]
pub struct Unlock {
    /// 最接近可行的变体。
    pub variant: String,
    /// 还缺的原语。
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

impl NoPath {
    /// 按 DESIGN §8.2 的格式渲染。
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "error: {} 在 {} 上没有可行变体\n",
            self.op, self.device
        ));
        for report in &self.needs {
            out.push_str(&format!("  need  {}\n", report.need.signature()));
            match &report.have {
                Some(have) => out.push_str(&format!("  have  {}\n", have.summary())),
                None => out.push_str(&format!(
                    "  have  —（该档案未声明 `{}`）\n",
                    report.need.prim
                )),
            }
            out.push_str(&format!("         ↳ 不满足：{}\n", report.reason()));
        }
        out.push_str("  降级链：\n");
        for step in &self.chain {
            if step.ok {
                out.push_str(&format!(
                    "    {} ✓ （可用，预计 mma_share {:.2}）\n",
                    step.variant, step.mma_share
                ));
            } else {
                out.push_str(&format!("    {} ✗ （{}）\n", step.variant, step.reason));
            }
        }
        match &self.unlock {
            Some(unlock) => {
                let missing: Vec<String> = unlock
                    .missing
                    .iter()
                    .map(|prim| format!("`{prim}`"))
                    .collect();
                out.push_str(&format!(
                    "  建议：在 {device} 补齐 {} 的 L2 见证后可用 {}（预计 mma_share {:.2}）\n",
                    missing.join(" "),
                    unlock.variant,
                    unlock.mma_share,
                    device = self.device
                ));
            }
            None => out.push_str("  建议：该设备缺失全部原语，先补 `provides` 再谈特化片段\n"),
        }
        out
    }
}

impl fmt::Display for NoPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

impl std::error::Error for NoPath {}

/// 检查一个变体的全部需求；空向量表示可行。
fn check_variant(op: &OpDecl, prof: &Profile, variant: &Variant) -> Vec<NeedReport> {
    let mut reports = Vec::new();
    for need in variant.needs(op) {
        reports.push(check_need(prof, need));
    }
    for prim in variant.schedule.required_prims() {
        let need = Need::of(prim);
        let report = check_need(prof, &need);
        if !report.is_satisfied() {
            reports.push(report);
        }
    }
    reports
}

fn check_need(prof: &Profile, need: &Need) -> NeedReport {
    match prof.provider(need.prim) {
        None => NeedReport {
            need: need.clone(),
            have: None,
            blocked_by_grade: false,
            mismatches: vec![Mismatch::PrimAbsent { prim: need.prim }],
        },
        Some(have) => NeedReport {
            need: need.clone(),
            blocked_by_grade: !have.grade.is_verified(),
            mismatches: have.diff(need),
            have: Some(have.clone()),
        },
    }
}

/// ② negotiate 本体。
pub fn negotiate(op: &OpDecl, prof: &Profile) -> Result<Path, Box<NoPath>> {
    let mut chain = Vec::new();
    let mut first_failure: Option<Vec<NeedReport>> = None;
    let mut closest: Option<(usize, Vec<Prim>)> = None;

    for (index, variant) in op.variants.iter().enumerate() {
        let reports = check_variant(op, prof, variant);
        let failures: Vec<&NeedReport> = reports.iter().filter(|r| !r.is_satisfied()).collect();
        let ok = failures.is_empty();
        let reason = if ok {
            String::new()
        } else {
            failures
                .iter()
                .map(|report| format!("{}：{}", report.need.prim, report.reason()))
                .collect::<Vec<_>>()
                .join("；")
        };
        chain.push(ChainStep {
            variant: variant.name.clone(),
            schedule: variant.schedule,
            ok,
            reason,
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
        let failing: Vec<NeedReport> = reports
            .into_iter()
            .filter(|report| !report.is_satisfied())
            .collect();
        let missing: Vec<Prim> = failing.iter().map(|report| report.need.prim).collect();
        if closest
            .as_ref()
            .is_none_or(|(count, _)| missing.len() < *count)
        {
            closest = Some((missing.len(), missing));
        }
        if first_failure.is_none() {
            first_failure = Some(failing);
        }
    }

    let unlock = closest.and_then(|(_, missing)| {
        op.variants
            .iter()
            .find(|variant| variant.name == chain.first().map_or("", |step| step.variant.as_str()))
            .map(|variant| Unlock {
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

/// `doctor` 用的成功路径解释：把整条链连同判定原因打印出来。
pub fn explain(op: &OpDecl, prof: &Profile) -> String {
    match negotiate(op, prof) {
        Ok(path) => {
            let mut out = format!(
                "path: {} (schedule {}) mma_share {:.2}\n",
                path.variant, path.schedule, path.mma_share
            );
            for (index, variant) in op.variants.iter().enumerate() {
                let reports = check_variant(op, prof, variant);
                let failures: Vec<String> = reports
                    .iter()
                    .filter(|report| !report.is_satisfied())
                    .map(|report| format!("{}：{}", report.need.prim, report.reason()))
                    .collect();
                let mark = if index == path.index {
                    "✓ 选中"
                } else {
                    "·"
                };
                if failures.is_empty() {
                    out.push_str(&format!("  {mark} {}\n", variant.name));
                } else {
                    out.push_str(&format!(
                        "  {mark} {} ✗ {}\n",
                        variant.name,
                        failures.join("；")
                    ));
                }
            }
            out
        }
        Err(no_path) => no_path.render(),
    }
}

/// 某设备对某算子的全部需求是否都有 `L2` 供给（`verify` 用）。
pub fn unmet_prims(op: &OpDecl, prof: &Profile) -> Vec<Prim> {
    let mut missing = Vec::new();
    for variant in &op.variants {
        for need in variant.needs(op) {
            if !prof.satisfies(need) && !missing.contains(&need.prim) {
                missing.push(need.prim);
            }
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op() -> OpDecl {
        OpDecl::parse(
            include_str!("../ops/gemv_i4.toml"),
            include_str!("../ops/gemv_i4.k"),
        )
        .expect("声明可解析")
    }

    fn cpu() -> Profile {
        Profile::parse(include_str!("../devices/cpu.toml")).expect("档案可解析")
    }

    fn nvidia() -> Profile {
        Profile::parse(include_str!("../devices/nvidia.toml")).expect("档案可解析")
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
        assert!(error.render().contains("没有可行变体"));
        assert!(error.chain.iter().all(|step| !step.ok));
        assert!(
            error.render().contains("L2"),
            "证据不足要明确指出：{}",
            error.render()
        );
    }

    #[test]
    fn variant_level_needs_drive_downgrade() {
        let mut profile = cpu();
        // 去掉 load_async：pipe2 需要它，direct 不需要 —— 这正是 extra_needs 的意义。
        profile.provides.retain(|have| have.prim != Prim::LoadAsync);
        profile
            .provides
            .retain(|have| have.prim != Prim::CommitWait);
        let path = negotiate(&op(), &profile).expect("应降级到 direct");
        assert_eq!(path.variant, "direct");
        assert!(!path.schedule.uses_async());
    }

    #[test]
    fn diagnostics_name_the_missing_atom_and_the_unlock() {
        // dot 是算子级公共需求：缺它时整条链都不可行，诊断必须点名并给出解锁路径。
        let mut profile = cpu();
        profile.provides.retain(|have| have.prim != Prim::Dot);
        let error = negotiate(&op(), &profile).unwrap_err();
        let text = error.render();
        assert!(text.contains("dot"), "{text}");
        assert!(text.contains("建议："), "{text}");
        assert!(text.contains("L2 见证"), "{text}");
    }

    #[test]
    fn missing_async_atom_only_disables_the_pipelined_variant() {
        let mut profile = cpu();
        profile.provides.retain(|have| have.prim != Prim::LoadAsync);
        let path = negotiate(&op(), &profile).expect("应降级到 direct");
        assert_eq!(path.variant, "direct");
        let text = explain(&op(), &profile);
        assert!(text.contains("pipe2 ✗"), "{text}");
    }

    #[test]
    fn explain_prints_the_whole_chain() {
        let text = explain(&op(), &cpu());
        assert!(text.contains("path: pipe2"), "{text}");
        assert!(text.contains("✓ 选中 pipe2"), "{text}");
        assert!(text.contains("direct"), "{text}");
    }

    #[test]
    fn unmet_prims_lists_every_gap() {
        let missing = unmet_prims(&op(), &nvidia());
        assert!(missing.contains(&Prim::Dot), "{missing:?}");
    }

    #[test]
    fn schedule_primitives_are_part_of_the_requirement() {
        // 只剩 direct 可用时 bar 不再被需要；把 bar 之外的能力也拿掉，才能看到 bar 出现在诊断里。
        let mut profile = cpu();
        profile.provides.retain(|have| have.prim != Prim::Bar);
        let path = negotiate(&op(), &profile).expect("应降级到 direct");
        assert_eq!(path.variant, "direct");

        let mut profile = cpu();
        profile.provides.retain(|have| have.prim != Prim::Bar);
        profile.provides.retain(|have| have.prim != Prim::Store);
        let error = negotiate(&op(), &profile).unwrap_err();
        assert!(error.render().contains("bar"), "{}", error.render());
    }
}
