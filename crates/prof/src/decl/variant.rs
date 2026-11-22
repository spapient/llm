//! 变体链：`[variants]` 的解析与链路校验（数量、`order` 连续、共享比例范围）。
use diagnostic::data::Table;

use crate::sch;
use crate::sch::Schedule;
use prof::atom::Need;

use super::build::parse_need;
use super::{MAX_VARIANTS, Variant, field};

/// 变体链：逐个解析、按 `order` 排序，再判数量与链路。
pub fn variants(table: &Table) -> diagnostic::Result<Vec<Variant>> {
    let mut variants: Vec<Variant> = table
        .table_array("variants")
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(idx, entry)| variant(idx, entry))
        .collect::<diagnostic::Result<_>>()?;
    check_count(&variants)?;
    variants.sort_by_key(|variant| variant.order);
    check_chain(&variants)?;
    Ok(variants)
}

/// 变体数：至少一个、至多 [`MAX_VARIANTS`] 个。
fn check_count(variants: &[Variant]) -> diagnostic::Result<()> {
    if variants.is_empty() {
        return Err(field("variants", "至少需要一个变体"));
    }
    if variants.len() > MAX_VARIANTS {
        return Err(field(
            "variants",
            format!("{} 个变体超过上限 {MAX_VARIANTS}", variants.len()),
        ));
    }
    Ok(())
}

/// 降级链的判据：名字唯一、非 MMA 占比单调，且首选的掩码能落地。
fn check_chain(variants: &[Variant]) -> diagnostic::Result<()> {
    for pair in variants.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(field("variants", format!("变体名 `{}` 重复", pair[0].name)));
        }
    }
    // 借用变体名：这条判据只读名字与占比，没有理由在装载期为它拷一份字符串。
    let chain: Vec<(&str, f64)> = variants
        .iter()
        .map(|variant| (variant.name.as_str(), variant.mma_share))
        .collect();
    sch::check_monotonic(&chain).map_err(|detail| field("variants", detail.message()))?;
    let Some(head) = variants.first() else {
        return Ok(());
    };
    sch::check_masking(&head.name, sch::Masking::of(head.mma_share, &head.schedule))
        .map_err(|detail| field("variants", detail.message()))
}

/// 一个 `[[variants]]` 段：名字与优先级、调度、非 MMA 占比、增量需求。
fn variant(idx: usize, entry: &Table) -> diagnostic::Result<Variant> {
    let (name, order) = identity(idx, entry)?;
    let schedule = schedule(idx, entry)?;
    let mma_share = share(idx, entry)?;
    Ok(Variant {
        name,
        order,
        schedule,
        mma_share,
        extra_needs: extra_needs(idx, entry)?,
    })
}

/// 变体名与优先级：两个键都必填，负的 `order` 按 0 处理。
fn identity(idx: usize, entry: &Table) -> diagnostic::Result<(String, u32)> {
    let name = entry
        .string("name")
        .ok_or_else(|| field(&field_path(idx, "name"), "缺失"))?
        .to_string();
    let order = entry
        .int("order")
        .ok_or_else(|| field(&field_path(idx, "order"), "缺失"))?
        .max(0) as u32;
    Ok((name, order))
}

/// 调度：`schedule` 必须落在调度闭集内。
fn schedule(idx: usize, entry: &Table) -> diagnostic::Result<Schedule> {
    let path = field_path(idx, "schedule");
    let text = entry
        .string("schedule")
        .ok_or_else(|| field(&path, "缺失"))?;
    Schedule::parse(text).map_err(|detail| field(&path, detail.message()))
}

/// 非 MMA 占比：`0..=1` 之间。
fn share(idx: usize, entry: &Table) -> diagnostic::Result<f64> {
    let path = field_path(idx, "mma_share");
    let mma_share = entry
        .float("mma_share")
        .ok_or_else(|| field(&path, "缺失"))?;
    if (0.0..=1.0).contains(&mma_share) {
        return Ok(mma_share);
    }
    Err(field(&path, "必须在 0..=1 之间"))
}

/// `extra_needs`：变体自己追加的需求。
fn extra_needs(idx: usize, entry: &Table) -> diagnostic::Result<Vec<Need>> {
    let Some(section) = entry.table("extra_needs") else {
        return Ok(Vec::new());
    };
    let prefix = field_path(idx, "");
    section
        .entries()
        .iter()
        .map(|(key, _)| parse_need(section, key, &format!("{prefix}.extra_needs.{key}")))
        .collect()
}

/// `variants[i].{name}` 形式的字段路径；`name` 为空时只剩前缀。
fn field_path(idx: usize, name: &str) -> String {
    format!("variants[{idx}].{name}")
}
