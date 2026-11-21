//! 职责：对拍口径（README §11.1）。

use crate::atom::DType;
use crate::decl::Fusion;
use crate::sig::CType;
use diagnostic::{Code, Error};
use op::model::{Model, Sample, Values};

/// 对拍口径：`max|got − want| / max|want|`。
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    // 非有限值不能走下面的 `f64::max`：NaN 会被 max 当缺席，漏报成"对上了"。逐对先定性：
    // 同是 NaN、同号 Inf 算对上，其余非有限组合一律判不一致。
    if !got.iter().zip(want).all(|(&g, &w)| nonfinite_match(g, w)) {
        return f64::INFINITY;
    }
    let max_want = want
        .iter()
        .fold(0.0f64, |acc, value| acc.max(f64::from(*value).abs()));
    let max_err = max_diff(got, want);
    if max_want == 0.0 {
        return max_err;
    }
    max_err / max_want
}

/// 非有限值配对是否一致；有限值交给数值误差比较。
fn nonfinite_match(got: f32, want: f32) -> bool {
    match (got.is_nan(), want.is_nan()) {
        (true, true) => true,
        (true, false) | (false, true) => false,
        // 非 NaN：同号 Inf 相等，或两边都是有限值；其余（Inf 对有限、异号 Inf）不一致。
        _ => got == want || (!got.is_infinite() && !want.is_infinite()),
    }
}

/// 一次比较的元素个数。
const WIDTH: usize = 8;

/// 逐元素绝对误差的最大值：8 路展开，尾部按剩余元素收口。
fn max_diff(got: &[f32], want: &[f32]) -> f64 {
    let mut lanes = [0.0f64; WIDTH];
    let block = got.len() - got.len() % WIDTH;
    for base in (0..block).step_by(WIDTH) {
        let a = &got[base..base + WIDTH];
        let b = &want[base..base + WIDTH];
        lanes[0] = lanes[0].max(diff(a[0], b[0]));
        lanes[1] = lanes[1].max(diff(a[1], b[1]));
        lanes[2] = lanes[2].max(diff(a[2], b[2]));
        lanes[3] = lanes[3].max(diff(a[3], b[3]));
        lanes[4] = lanes[4].max(diff(a[4], b[4]));
        lanes[5] = lanes[5].max(diff(a[5], b[5]));
        lanes[6] = lanes[6].max(diff(a[6], b[6]));
        lanes[7] = lanes[7].max(diff(a[7], b[7]));
    }
    let tail = got[block..]
        .iter()
        .zip(&want[block..])
        .fold(0.0f64, |acc, (a, b)| acc.max(diff(*a, *b)));
    lanes.into_iter().fold(tail, f64::max)
}

/// 一对元素的绝对误差，算在 `f64` 里：`f32` 相减在阈值附近会先丢掉要比较的那几位。
fn diff(got: f32, want: f32) -> f64 {
    (f64::from(got) - f64::from(want)).abs()
}

/// 按数据类型的阈值（README §11.1）：浮点按尾数宽度分档，整数与布尔位精确。
pub fn tolerance(dt: DType) -> f64 {
    match dt {
        DType::F16 | DType::BF16 => 1e-3,
        DType::F32 => 1e-5,
        // 浮点但当前无执行路径（MXFP4 的元素与 8 位浮点族）：阈值 0 表示"精确对拍"，
        // 接入解码/执行时与 `DTYPES` 一起改。
        DType::F4
        | DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        // 整数与布尔：位精确，任何差值都是错值。
        | DType::Bool
        | DType::U1
        | DType::U2
        | DType::U4
        | DType::U8
        | DType::U16
        | DType::U32
        | DType::U64
        | DType::I1
        | DType::I2
        | DType::I4
        | DType::I8
        | DType::I16
        | DType::I32
        | DType::I64 => 0.0,
        DType::F64 => 0.0,
    }
}

/// 是否在阈值内。
pub fn within(got: &[f32], want: &[f32], dt: DType) -> bool {
    max_rel_err(got, want) <= tolerance(dt)
}

/// 一次融合对拍的读数。
///
/// `got` 是消费者在**全部缝都桥接之后**的产出；`bridged` 是实际桥接的
/// `(生产者输出槽位, 消费者读槽位)` 对。两者的完整性由 [`verify_fusion`] 保证：
/// `bridged.len()` 恒等于 `fusion.seams.len()`——只桥一条就返回的旧口径会让对拍悄悄测另一个程序。
pub struct Oracle {
    /// 消费者产出。
    pub got: Vec<f32>,
    /// 实际桥接的缝。
    pub bridged: Vec<(String, String)>,
}

/// 融合对拍：按 [`Fusion`] 的缝**逐条**桥接，生产者参考先跑，把输出喂给消费者的每个读槽位，
/// 再跑消费者的参考。
///
/// 这是"律实例化必带对拍"里"对不对"的那一半，与 `compose()` 的"能不能拼"由同一个 [`Fusion`]
/// 值串起来：缝从证据里来，不从调用点的字符串来。因此下列情况一律拒绝，而不是算一个近似值：
///
/// - 自融合（[`Fusion::is_self`]）：循环携带依赖，`compose` 不会产出它，手搓的也不接受；
/// - 多输出生产者：标量参考只给一条输出（[`op::model::Model::reference`]），拼不出完整桥接；
/// - 非 `F32` 缝：同一原因，`U8` 槽位没有字节级 oracle 值可喂。
///
/// 每条缝的读槽位还必须在样本里存在且长度与生产者输出一致——这是"缝上的形状也确实对得上"的
/// 那一半证明，缺了它，长度不符会推迟到消费者 reference 内部以另一个错误出现。
///
/// # Errors
///
/// 融合值本身不合法（[`Code::Invalid`]）、超出本条机制能对拍的范围（[`Code::Unsupported`]）、
/// 或样本缺槽位 / 长度不符（[`Code::Missing`] / [`Code::Invalid`]）时返回说明。
pub fn verify_fusion(
    producer: &Model,
    consumer: &Model,
    fusion: &Fusion,
    sample: &mut Sample,
) -> diagnostic::Result<Oracle> {
    let bridged = plan(fusion)?;
    let got = run_fused(producer, consumer, sample, &bridged)?;
    Ok(Oracle { got, bridged })
}

/// 校验融合值本身是否可对拍，并列出要桥接的 `(生产者写槽位, 消费者读槽位)`。
/// 只看名字与类型：形状要到 `bridge_one` 里才拿得到样本，所以放在那里报。
fn plan(fusion: &Fusion) -> diagnostic::Result<Vec<(String, String)>> {
    check(fusion)?;
    let from = sole_output(fusion)?;
    let targets = seam_targets(fusion, from)?;
    Ok(targets
        .into_iter()
        .map(|to| (from.to_string(), to.to_string()))
        .collect())
}

/// 先排除不可对拍的形状：空融合与自融合。
fn check(fusion: &Fusion) -> diagnostic::Result<()> {
    if fusion.seams.is_empty() {
        return Err(reject("融合值为空：没有任何缝，就没有可对拍的程序"));
    }
    if fusion.is_self() {
        return Err(reject("自融合不是融合：生产者与消费者是同一个算子"));
    }
    Ok(())
}

/// 标量参考只产出一条输出，多输出的生产者桥接不完整。
fn sole_output(fusion: &Fusion) -> diagnostic::Result<&str> {
    let outputs = fusion.outputs();
    if outputs.len() != 1 {
        return Err(unsupported(
            "生产者有多条输出，标量参考只给一条，无法完整桥接",
        ));
    }
    Ok(outputs[0])
}

/// 至少有一条缝指向消费者，且全部是 F32——只有 F32 能由标量参考无损搬运。
fn seam_targets<'a>(fusion: &'a Fusion, from: &'a str) -> diagnostic::Result<Vec<&'a str>> {
    let targets: Vec<&str> = fusion.targets(from).collect();
    if targets.is_empty() {
        return Err(reject("融合值里没有指向消费者的缝"));
    }
    if fusion.seams.iter().any(|seam| seam.ctype != CType::F32) {
        return Err(unsupported("只有 F32 缝能由标量参考桥接"));
    }
    Ok(targets)
}

/// 跑生产者、把每条缝桥进消费者样本、再跑消费者。
fn run_fused(
    producer: &Model,
    consumer: &Model,
    sample: &mut Sample,
    bridged: &[(String, String)],
) -> diagnostic::Result<Vec<f32>> {
    let mid = (producer.reference)(sample)?;
    for (from, to) in bridged {
        bridge_one(sample, from, to, &mid)?;
    }
    (consumer.reference)(sample)
}

/// 把生产者产出写进消费者的一个槽位，长度不符就不写。
fn bridge_one(sample: &mut Sample, from: &str, to: &str, mid: &[f32]) -> diagnostic::Result<()> {
    let buffer = sample
        .buffer(to)
        .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有消费者槽位 `{to}`")))?;
    if buffer.values.len() != mid.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "缝 `{from} → {to}` 形状对不上：生产者产出 {} 个，消费者槽位声明 {} 个",
                mid.len(),
                buffer.values.len()
            ),
        ));
    }
    sample.set(to, Values::F32(mid.to_vec()))
}

/// 融合值自身不合法。
fn reject(message: &str) -> Error {
    Error::new(Code::Invalid, message.to_string()).with("cutile::reference::verify_fusion")
}

/// 融合值合法，但超出当前对拍机制能回答的范围。
fn unsupported(message: &str) -> Error {
    Error::new(Code::Unsupported, message.to_string()).with("cutile::reference::verify_fusion")
}

#[cfg(test)]
#[path = "../tests/reference_tests.rs"]
mod tests;
