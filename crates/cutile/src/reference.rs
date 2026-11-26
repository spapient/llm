//! 职责：对拍口径（README §11.1）。

use crate::atom::DType;
use crate::decl::Fusion;
use crate::sig::CType;
use diagnostic::{Code, Error};
use prof::model::{Model, Sample, Values};

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
    /// 消费者产出（按元素档存放；对拍侧按档解包）。
    pub got: Values,
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
/// - 多输出生产者：标量参考只给一条输出（[`Model::reference`]），拼不出完整桥接；
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
    check_fusion(fusion)?;
    let from = sole_output(fusion)?;
    let targets = seam_targets(fusion, from)?;
    Ok(targets
        .into_iter()
        .map(|to| (from.to_string(), to.to_string()))
        .collect())
}

/// 先排除不可对拍的形状：空融合与自融合。
fn check_fusion(fusion: &Fusion) -> diagnostic::Result<()> {
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

/// 跑生产者、把每条缝桥进消费者样本、再跑消费者；产出按原样档位返回，对拍侧解包。
fn run_fused(
    producer: &Model,
    consumer: &Model,
    sample: &mut Sample,
    bridged: &[(String, String)],
) -> diagnostic::Result<Values> {
    let mid = (producer.reference)(sample)?;
    bridge_all(sample, bridged, &mid)?;
    (consumer.reference)(sample)
}

/// 批量桥接全部缝；任一槽位失败即停止，不继续写后面的槽位。
fn bridge_all(
    sample: &mut Sample,
    bridged: &[(String, String)],
    mid: &Values,
) -> diagnostic::Result<()> {
    for (from, to) in bridged {
        bridge_one(sample, from, to, mid)?;
    }
    Ok(())
}

/// 把生产者产出写进消费者的一个槽位，长度不符就不写；档位一致性由 [`Sample::set`] 校验。
fn bridge_one(sample: &mut Sample, from: &str, to: &str, mid: &Values) -> diagnostic::Result<()> {
    let buf = sample
        .buf(to)
        .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有消费者槽位 `{to}`")))?;
    if buf.values.len() != mid.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "缝 `{from} → {to}` 形状对不上：生产者产出 {} 个，消费者槽位声明 {} 个",
                mid.len(),
                buf.values.len()
            ),
        ));
    }
    sample.set(to, mid.clone())
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
mod tests {
    //! 对拍口径的单元测试：误差量级、长度路径、阈值分档与非有限值分类。

    use super::*;

    testkit::test_cases! {
    fn relative_error_uses_the_biggest_magnitude() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        let err = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
        assert!((err - 0.05).abs() < 1e-6, "{err}");
        assert!(within(&[1.0, 2.0], &[1.0, 2.0], DType::F32));
        assert!(!within(&[1.0, 2.5], &[1.0, 2.0], DType::F32));
    }

    fn every_length_takes_the_same_path_to_the_same_answer() {
        // 8 路展开后有三条路径：整块、整块 + 尾部、只有尾部。三者对同一批数必须一致。
        let want: Vec<f32> = (0..19u32).map(|i| i as f32 * 0.5).collect();
        let mut got = want.clone();
        got[17] += 0.25;
        let expected = 0.25 / 9.0;
        assert!((max_rel_err(&got, &want) - expected).abs() < 1e-9);
        // 长度不足一组：只走尾部。
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert_eq!(max_rel_err(&[], &[]), 0.0);
        // 不是同一批数时是无穷，而不是某个"很大"的误差。
        assert_eq!(max_rel_err(&[1.0], &[1.0, 2.0]), f64::INFINITY);
    }

    fn tolerance_follows_dtype() {
        assert_eq!(tolerance(DType::F16), 1e-3);
        assert_eq!(tolerance(DType::F32), 1e-5);
        assert_eq!(tolerance(DType::I8), 0.0);
    }

    fn non_finite_values_are_classified_not_averaged_away() {
        // 同是 NaN、同号 Inf 算对上；其余非有限组合是不一致，不能被 max 当缺席吞掉。
        assert!(within(&[f32::NAN], &[f32::NAN], DType::F32));
        assert!(within(&[f32::INFINITY], &[f32::INFINITY], DType::F32));
        assert!(!within(&[f32::NAN], &[0.0], DType::F32));
        assert!(!within(&[f32::INFINITY], &[f32::NEG_INFINITY], DType::F32));
        assert!(!within(&[f32::INFINITY], &[1.0], DType::F32));
        assert_eq!(max_rel_err(&[f32::NAN], &[0.0]), f64::INFINITY);
    }

    fn every_accepted_fusion_is_either_testable_or_named_unrealizable() {
        // README §9 的契约：`plan::fusions` 接受的每一条融合都必须能生成一条对拍。这里在整张内置
        // 注册表上把契约变成可执行断言，不留"悄悄没测"的结局：
        //   Ok          → 每条缝都桥接了（`bridged.len() == seams.len()`）；
        //   Invalid     → 几何上拼不出来（gemv 的 `[N]` 喂不进 elementwise 的 `[M*N]`），必须说形状；
        //   Unsupported → 多输出 / U8 缝，标量参考答不了；
        //   跳过        → 样本面上没有生产者的输入，这条融合的 ABI 压根没构造出来。
        // 唯独 `Missing` 是 bug：样本按消费者声明构造，代数若点出样本里没有的槽位，就是发明了名字。
        let sources = prof::catalogue::all();
        let decls: Vec<crate::decl::OpDecl> = sources
            .iter()
            .map(|source| source.to_decl().expect("声明自洽"))
            .collect();
        let models: Vec<(&str, Model)> = sources
            .iter()
            .filter_map(|source| nn::registry::model(source.name()).map(|model| (source.name(), model)))
            .collect();

        let accepted = crate::plan::fusions(&decls, 12);
        assert!(!accepted.is_empty(), "整张注册表里必须有能拼的对");
        let (mut testable, mut unrealizable, mut unshaped) = (0usize, 0usize, 0usize);
        for fusion in &accepted {
            assert!(!fusion.is_self(), "自融合不该进候选：{fusion:?}");
            assert!(!fusion.seams.is_empty(), "空缝不是融合：{fusion:?}");
            let (Some((_, producer)), Some((_, consumer))) = (
                models.iter().find(|(name, _)| *name == fusion.producer),
                models.iter().find(|(name, _)| *name == fusion.consumer),
            ) else {
                continue; // 没有语义投影的算子无从对拍，代数与语义两张表的差异另有测试管。
            };
            // 样本按消费者声明构造：融合核的 ABI 就是消费者那一侧，缝写进去的也是它的槽位。
            let mut sample = (consumer.sample)(consumer.shape.0, consumer.shape.1);
            let declared = decls
                .iter()
                .find(|decl| decl.name == fusion.producer)
                .is_some_and(|decl| {
                    decl.phases
                        .iter()
                        .flat_map(|phase| phase.reads.iter())
                        .all(|name| sample.buf(name).is_some())
                });
            if !declared {
                unshaped += 1;
                continue;
            }
            match verify_fusion(producer, consumer, fusion, &mut sample) {
                Ok(oracle) => {
                    assert_eq!(
                        oracle.bridged.len(),
                        fusion.seams.len(),
                        "{} → {} 只桥了一部分缝",
                        fusion.producer,
                        fusion.consumer
                    );
                    assert_eq!(oracle.got.len(), consumer.shape.1.max(1), "产出长度");
                    testable += 1;
                }
                Err(error) => {
                    assert_ne!(
                        error.code(),
                        Code::Missing,
                        "代数发明了样本里没有的槽位（{} → {}）：{error}",
                        fusion.producer,
                        fusion.consumer
                    );
                    if error.code() == Code::Invalid {
                        assert!(
                            error.to_string().contains("形状对不上"),
                            "只有几何不匹配能算 Invalid：{error}"
                        );
                        unrealizable += 1;
                    }
                }
            }
        }
        assert_eq!(
            testable + unrealizable + unshaped,
            accepted.len(),
            "被接受的融合必须逐条落到某一种结局"
        );
        // 实测基线：内置注册表的 ABI 多数互不相同——`gemv_f32` 的 `y` 是 `[N]`，elementwise 是 `[M*N]`；
        // 代数只看名字与类型、不看形状。当前唯一几何可实现的融合是 `abs_f32 → add_f32`：同为逐元素、
        // 同秩同长，abs 产出行连续桥进 add 的 `a` 输入槽，oracle 链路由融合对拍用例独立背书。
        // 这是快照绊线：再增 elementwise 邻近对、或把形状纳入代数时，逐条复核 oracle 后再调基线。
        assert_eq!(
            testable, 1,
            "几何可实现融合条数偏离当前快照（可测 {testable} / 几何不可行 {unrealizable} / 未构造样本 {unshaped}）"
        );
    }
    }

    /// 建一条真实的声明：内置算子的 TOML + 模板，过一遍拼装期校验。
    fn declared(name: &str) -> crate::decl::OpDecl {
        prof::catalogue::named(name)
            .expect("内置算子")
            .to_decl()
            .expect("声明自洽")
    }

    testkit::test_cases! {
    fn the_fused_oracle_bridges_every_seam_not_just_one() {
        let mul = nn::registry::model("mul_f32").expect("mul_f32 已登记");
        let add = nn::registry::model("add_f32").expect("add_f32 已登记");
        let fusion = crate::decl::compose(&declared("mul_f32"), &declared("add_f32")).expect("可拼");
        assert_eq!(fusion.outputs(), ["y"], "生产者只有一条输出");
        assert_eq!(fusion.seams.len(), 2, "add 的两个读槽位各一条缝");

        let mut sample = (mul.sample)(8, 8);
        let a = sample.f32("a").expect("样本有 a").to_vec();
        let b = sample.f32("b").expect("样本有 b").to_vec();
        // 完整桥接：mul 的结果同时喂给 add 的两路输入。
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y + x * y).collect();
        let oracle = verify_fusion(&mul, &add, &fusion, &mut sample).expect("融合参考可算");
        assert_eq!(oracle.bridged.len(), fusion.seams.len(), "每条缝都必须桥接");
        let got = oracle.got.as_f32().expect("f32 产出");
        assert!(within(got, &want, DType::F32), "对拍失败");

        // 只桥一条缝（旧口径）算的是另一个程序：b 仍是原值而不是生产者产出。
        let half: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y + y).collect();
        assert!(!within(got, &half, DType::F32), "旧口径必须与完整桥接不同");
    }

    fn the_fused_oracle_refuses_what_it_cannot_bridge() {
        let mul = nn::registry::model("mul_f32").expect("mul_f32 已登记");
        let add = nn::registry::model("add_f32").expect("add_f32 已登记");
        let real = crate::decl::compose(&declared("mul_f32"), &declared("add_f32")).expect("可拼");
        let seam = |from: &str, to: &str, ctype| crate::decl::Seam {
            from: from.to_string(),
            to: to.to_string(),
            ctype,
        };
        let code_of = |fusion: &crate::decl::Fusion| {
            verify_fusion(&mul, &add, fusion, &mut (mul.sample)(8, 8))
                .err()
                .expect("必须拒绝")
                .code()
        };

        let mut self_fusion = real.clone();
        self_fusion.consumer = self_fusion.producer.clone();
        assert_eq!(code_of(&self_fusion), Code::Invalid, "自融合是循环携带依赖");

        let mut two_outputs = real.clone();
        two_outputs.seams.push(seam("z", "a", CType::F32));
        assert_eq!(
            code_of(&two_outputs),
            Code::Unsupported,
            "标量参考只给一条输出"
        );

        let mut u8_seam = real.clone();
        for seam in &mut u8_seam.seams {
            seam.ctype = CType::U8;
        }
        assert_eq!(code_of(&u8_seam), Code::Unsupported, "U8 没有字节级 oracle");

        let mut missing = real.clone();
        missing.seams = vec![seam("y", "nope", CType::F32)];
        assert_eq!(code_of(&missing), Code::Missing, "槽位必须真的在样本里");

        let mut empty = real;
        empty.seams.clear();
        assert_eq!(code_of(&empty), Code::Invalid, "没有缝就没有可对拍的程序");
    }
    }
}
