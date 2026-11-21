//! 职责：失败诊断的排版：`NoPath` 与 `FragmentGap` 的对外文本。

use super::*;
use core::fmt;

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
