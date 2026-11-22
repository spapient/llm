//! 职责：③ select 与融合打分：按声明顺序取 tile，roofline 只给估值（README §9、§12.1）。

use crate::cap::Path;
use crate::decl::{Fusion, OpDecl, TileCombo, compose, expand};
use crate::prof::Profile;
use crate::sch::Masking;
use ::prof::catalogue::Registration;
use core::cmp::Ordering;
use core::fmt;
use diagnostic::{Code, Error};
use prof::model::{Cost, Model};

/// 一次 plan 的产物。
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// 变体名。
    pub variant: String,
    /// 调度。
    pub schedule: String,
    /// 声明的 tile 取值（`[axes]` 笛卡尔积的第一个组合）。
    pub tile: TileCombo,
    /// 预测耗时（秒）。
    pub predicted_s: f64,
    /// 计算量（FLOP）。
    pub flops: f64,
    /// 流量（字节）。
    pub bytes: u64,
    /// 掩盖账：`mma_share`、级数与未掩盖占比（判据与报告用的是同一组数）。
    pub masking: Masking,
}

fn scale_flops(flops: f64) -> String {
    if flops >= 1e9 {
        format!("{:.1} GFLOP", flops / 1e9)
    } else {
        format!("{:.1} MFLOP", flops / 1e6)
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} [{}] → {:.3} ms（{}，{:.1} MB，{}）",
            self.variant,
            self.schedule,
            self.tile,
            self.predicted_s * 1e3,
            scale_flops(self.flops),
            self.bytes as f64 / 1e6,
            self.masking
        )
    }
}

/// roofline 的两侧秒数：`(算力侧 flops/峰值, 带宽侧 bytes/带宽)`。档案没给出可求值的峰值时
/// 那一侧是 `None`——峰值是能力，缺席就是"不知道"，不是 0。
fn sides(profile: &Profile, flops: f64, bytes: u64) -> (Option<f64>, Option<f64>) {
    let compute = profile
        .peak
        .flops_f32
        .as_ref()
        .filter(|constant| constant.value > 0.0)
        .map(|constant| flops / constant.value);
    let memory = profile
        .peak
        .bw
        .as_ref()
        .filter(|constant| constant.value > 0.0)
        .map(|constant| bytes as f64 / constant.value);
    (compute, memory)
}

/// roofline 预测：`max(flops/峰值, bytes/带宽) × (1 + 未掩盖占比)`。
pub fn predict(profile: &Profile, flops: f64, bytes: u64, masking: Masking) -> Option<f64> {
    let base = match sides(profile, flops, bytes) {
        (Some(compute), Some(memory)) => compute.max(memory),
        (Some(base), None) | (None, Some(base)) => base,
        (None, None) => return None,
    };
    Some(base * (1.0 + masking.unmasked))
}

/// 轴展开为空是声明解析的不变量破了：那是内部错误，不是"候选不好"。
fn no_tiles() -> Error {
    Error::new(
        Code::Internal,
        "轴展开后没有任何 tile 候选（声明解析应当保证至少一个轴）",
    )
}

/// 预测值缺席就是档案缺峰值；写成独立一步，好让 [`select`] 读成"三步取料"，而不是三层错误构造。
fn predict_or_fail(profile: &Profile, cost: Cost, masking: Masking) -> diagnostic::Result<f64> {
    predict(profile, cost.flops, cost.bytes(), masking).ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!(
                "档案 `{}` 没有可推导的 peak 常数，无法做 roofline 估算",
                profile.meta.vendor
            ),
        )
    })
}

/// ③ select 本体：取轴展开出的**第一个** tile（TOML 顺序即优先级，不在这里打分），再用算子给出
/// 的代价做 roofline 估值。峰值是能力：档案没给出可求值的峰值就无法判定，而不是"预测为 0"。
pub fn select(op: &OpDecl, profile: &Profile, path: &Path, cost: Cost) -> diagnostic::Result<Plan> {
    let masking = Masking::of(path.mma_share, &path.schedule);
    let tile = expand(op)?.into_iter().next().ok_or_else(no_tiles)?;
    let predicted_s = predict_or_fail(profile, cost, masking)?;
    Ok(Plan {
        variant: path.variant.clone(),
        schedule: path.schedule.as_str(),
        tile,
        predicted_s,
        flops: cost.flops,
        bytes: cost.bytes(),
        masking,
    })
}

/// 窗口半径由预算解出：`radius = √budget`（向上取整）。
///
/// 全目录 `N` 个算子有 `N²` 个有序对，但只有 `compose` 判“能拼”的才有资格进 roofline 打分。
/// 半径把每个生产者的邻域收窄到 `±radius` 的窗口，`N · (2r + 1)` 次判定的量级随 `√budget` 涨，
/// 而不是随 `N²` 涨——这就是压住组合爆炸的闸（cutile README §9 的“窗口半径由预算解出”）。
pub fn window_radius(budget: usize) -> usize {
    (budget as f64).sqrt().ceil() as usize
}

/// 候选剪枝：在算子表上枚举生产者→消费者对，先过 `compose`（只留"能拼"），再用窗口半径
/// 压邻域，最多接受 `budget` 条。`compose` 返回的 [`Fusion`] 才进 roofline 打分。
pub fn fusions(ops: &[OpDecl], budget: usize) -> Vec<Fusion> {
    let radius = window_radius(budget);
    let mut accepted = Vec::new();
    for (i, producer) in ops.iter().enumerate() {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(ops.len());
        // 窗口天生含生产者自己，`compose` 按槽位代数把自融合摘掉；预算为 0 时 `room` 也是 0，
        // 这一圈的 `take` 什么都不收——两条退化情形都不需要额外的守卫分支。
        let room = budget - accepted.len();
        accepted.extend(
            ops[lo..hi]
                .iter()
                .filter_map(|consumer| compose(producer, consumer))
                .take(room),
        );
        if accepted.len() == budget {
            break;
        }
    }
    accepted
}

/// 一条融合候选的 roofline 增益。
///
/// 融合只省中间量的**往返**：生产者写一次、消费者读一次，两笔都落在 DRAM 上，算力一点不减。
/// 于是 `unfused` 是两侧各自 `max(算力, 带宽)` 之和，`fused` 是复合代价的一次定界。`max` 次可加
/// （`max(a₁+a₂, b₁+b₂) ≤ max(a₁,b₁) + max(a₂,b₂)`）保证 `fused ≤ unfused`，[`Payoff::speedup`]
/// 因此恒 ≥ 1——这是代数结论，不是拟合出来的数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Payoff {
    /// 不融合：两侧各自一次 roofline 预测之和（秒）。
    pub unfused_s: f64,
    /// 融合：中间量不进 DRAM 之后的预测（秒）。
    pub fused_s: f64,
    /// 省掉的流量（字节）：中间量的写 + 读。
    pub saved_bytes: u64,
    /// 融合后算力侧定界。私有：增益是算出来的，不能从外部拼一个自相矛盾的出来。
    compute_bound: bool,
}

impl Payoff {
    /// 加速比：不融合 / 融合。两侧都是 0（没流量也没算力）时记 1：没有可省的东西就没有加速。
    pub fn speedup(&self) -> f64 {
        if self.fused_s > 0.0 {
            self.unfused_s / self.fused_s
        } else {
            1.0
        }
    }

    /// 融合后哪一侧定界。融合把字节减掉中间量的往返，常常把界从带宽挪到算力——那是融合的目的，
    /// 所以它是一个要报告的结论，不是一个中间量。
    pub fn bound(&self) -> &'static str {
        if self.compute_bound {
            "compute"
        } else {
            "memory"
        }
    }
}

/// 一条候选的判决：三种诚实结局，没有第四种。
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// 打得出分：两侧都有语义投影、接缝能归因、档案有峰值。
    Scored(Payoff),
    /// 拼不出来：几何否掉了它（长度、槽位、类型），原因来自判据本身。
    Rejected(String),
    /// 无从打分：缺语义投影、接缝上有多条输出、或档案缺峰值——都不许猜一个数顶上。
    Unscorable(String),
}

impl Verdict {
    /// 判决名，写进报告。
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scored(_) => "scored",
            Self::Rejected(_) => "rejected",
            Self::Unscorable(_) => "unscorable",
        }
    }

    /// 打不出分的原因；`Scored` 没有原因可报。
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Scored(_) => None,
            Self::Rejected(reason) | Self::Unscorable(reason) => Some(reason),
        }
    }

    /// 排序键：能打分的按加速比降序在前，其次是"没条件打分"，最后是"拼不出来"。
    fn rank(&self) -> (u8, f64) {
        match self {
            Self::Scored(payoff) => (0, -payoff.speedup()),
            Self::Unscorable(_) => (1, 0.0),
            Self::Rejected(_) => (2, 0.0),
        }
    }
}

/// 算子在自己生产形状上的代价；形状或代价不自洽就是 `None`——打分不猜数。
fn cost_of(model: &Model) -> Option<Cost> {
    let (m, n) = model.shape;
    (model.cost)(&(model.shape_ctx)(m, n).ok()?).ok()
}

/// 两侧都要：判定"融合后哪一侧定界"缺一侧就不成立。
fn both(profile: &Profile, flops: f64, bytes: u64) -> Option<(f64, f64)> {
    let (compute, memory) = sides(profile, flops, bytes);
    Some((compute?, memory?))
}

/// 三笔代价过 roofline：不融合是两侧各自定界之和，融合是复合代价一次定界。
fn payoff(profile: &Profile, before: Cost, after: Cost, fused: Cost, seam: u64) -> Option<Payoff> {
    let (before_compute, before_memory) = both(profile, before.flops, before.bytes())?;
    let (after_compute, after_memory) = both(profile, after.flops, after.bytes())?;
    let (compute_s, memory_s) = both(profile, fused.flops, fused.bytes())?;
    Some(Payoff {
        unfused_s: before_compute.max(before_memory) + after_compute.max(after_memory),
        fused_s: compute_s.max(memory_s),
        saved_bytes: seam.saturating_mul(2),
        compute_bound: compute_s >= memory_s,
    })
}

/// 在语义表里按名字取投影。表与 [`::prof::catalogue::TEMPLATES`] 同序，由持有内核的 `nn` 提供。
fn model_of<'a>(models: &'a [Registration], name: &str) -> Option<&'a Model> {
    models
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| &entry.model)
}

/// 接缝的代数：取两侧代价与中间量的字节数。
///
/// 接缝只认一条输出槽（与 [`crate::reference::verify_fusion`] 同一条判据），此时生产者的
/// `Cost::write` 就是那条槽；消费者读它一次是同样的字节，所以省下 `2 × write`。任何一步不成立都
/// 给出判决而不是猜数：缺语义投影、多槽缝隙、代价不成立是"无从打分"，代价表与缝不一致是"拼不出来"。
fn seam(fusion: &Fusion, models: &[Registration]) -> Result<(Cost, Cost, u64), Verdict> {
    let (Some(producer), Some(consumer)) = (
        model_of(models, &fusion.producer),
        model_of(models, &fusion.consumer),
    ) else {
        return Err(Verdict::Unscorable(format!(
            "`{}` → `{}` 有一侧没有语义投影，算不出 roofline",
            fusion.producer, fusion.consumer
        )));
    };
    let outputs = fusion.outputs();
    let [from] = outputs.as_slice() else {
        return Err(Verdict::Unscorable(format!(
            "接缝落在 {} 条输出槽上（`{}`），中间量无法从代价表归因",
            outputs.len(),
            outputs.join("`, `")
        )));
    };
    let (Some(before), Some(after)) = (cost_of(producer), cost_of(consumer)) else {
        return Err(Verdict::Unscorable(format!(
            "`{}` 或 `{}` 在自己的生产形状上给不出代价",
            fusion.producer, fusion.consumer
        )));
    };
    if after.read < before.write {
        return Err(Verdict::Rejected(format!(
            "缝 `{from}` 有 {} 字节，消费者只读 {} 字节：代价表与缝代数不一致",
            before.write, after.read
        )));
    }
    Ok((before, after, before.write))
}

/// 复合代价过一次 roofline：融合把中间量的往返从字节账里减掉，剩下的两项各按峰值定界。
fn merged(profile: &Profile, before: Cost, after: Cost, seam: u64) -> Verdict {
    let fused = Cost::new(
        before.flops + after.flops,
        before.read.saturating_add(after.read).saturating_sub(seam),
        before
            .write
            .saturating_add(after.write)
            .saturating_sub(seam),
    );
    payoff(profile, before, after, fused, seam).map_or_else(
        || {
            Verdict::Unscorable(format!(
                "档案 `{}` 缺算力或带宽峰值，roofline 无从定界",
                profile.meta.vendor
            ))
        },
        Verdict::Scored,
    )
}

/// 给一条融合打分（README §12.1：roofline 是判据本身，不是装饰）。
///
/// 中间量的字节取自**生产者的代价表**（[`seam`]），这条路不需要跑标量参考，也就不必为打分分配
/// 4 GiB 样本。这是结构性上界，三件事它不算：流水掩盖（按变体算，在加速比里两边相消）、融合核多
/// 出来的寄存器压力、代码变大带来的指令缓存代价。所以 [`Payoff::speedup`] 是"值不值得动手"的判据，
/// 不是实测。
pub fn score(profile: &Profile, fusion: &Fusion, models: &[Registration]) -> Verdict {
    seam(fusion, models).map_or_else(
        |verdict| verdict,
        |(before, after, seam)| merged(profile, before, after, seam),
    )
}

/// 完整打分表：[`fusions`] 给出的候选逐条过 [`score`]，按加速比降序。
///
/// 打不出分的候选留在表里并带原因——它们不是"被过滤掉"，而是"这一条今天还不成立"，报告据此才知道
/// 下一步该补形状、补样本还是补峰值。这正是 README §9.1 那条"12 条被接受、0 条可对拍"的基线从
/// 一句口头事实变成一张可执行的工作清单。
pub fn ranking(
    profile: &Profile,
    ops: &[OpDecl],
    budget: usize,
    models: &[Registration],
) -> Vec<(Fusion, Verdict)> {
    let mut ranked: Vec<(Fusion, Verdict)> = fusions(ops, budget)
        .into_iter()
        .map(|fusion| {
            let verdict = score(profile, &fusion, models);
            (fusion, verdict)
        })
        .collect();
    ranked.sort_by(|a, b| {
        a.1.rank()
            .partial_cmp(&b.1.rank())
            .unwrap_or(Ordering::Equal)
    });
    ranked
}

#[cfg(test)]
#[path = "../tests/plan_tests.rs"]
mod tests;
