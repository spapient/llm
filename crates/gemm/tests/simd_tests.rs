use super::*;
use crate::quant::{I2, I4};

const M: usize = 256;
const N: usize = 13;
const GROUP: usize = 32;

/// 反量化表：与 [`crate::quant::Row`] 里两档打包格式同一张。
fn table(cpb: usize) -> &'static [f32] {
    match cpb {
        2 => I4.table,
        _ => I2.table,
    }
}

/// `n` 行、每行 `m / cpb` 字节的确定取值权重。
fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
    (0..n * (m / cpb)).map(|i| ((i * 37) % 251) as u8).collect()
}

/// `n` 行 × `groups` 组的标度，逐行逐组不同；2 的幂，便于逐位比较。
fn scales(n: usize, groups: usize) -> Vec<f32> {
    (0..n * groups)
        .map(|i| [0.25f32, 0.5, 1.0, 2.0][(i * 5) % 4])
        .collect()
}

fn context(m: usize, cpb: usize) -> ScaledRow {
    ScaledRow {
        groups: m / GROUP,
        group: GROUP,
        row: m / cpb,
    }
}

/// 标准样本（`M` 个元素、`N` 列、整张表）过得了 `admit`——测试从这里取通道参数。
fn admitted(cpb: usize, ctx: ScaledRow) -> Affine {
    admit(cpb, N, ctx, table(cpb)).expect("标准样本必须落在通道内")
}

/// 一条输入行：整数，`|x| <= 8`。
fn whole(m: usize) -> Vec<f32> {
    (0..m).map(|i| ((i * 7 + 3) % 17) as f32 - 8.0).collect()
}

/// 一条输入行：带小数的通用取值。
fn fractional(m: usize) -> Vec<f32> {
    (0..m)
        .map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.037_037)
        .collect()
}

/// 标量叶（[`super::packed::dot_row`]）对一行给出的值——独立于本通道的第二实现。
fn oracle(m: usize, cpb: usize, w_row: &[u8], s_row: &[f32], x: &[f32]) -> f32 {
    let ctx = context(m, cpb);
    super::super::packed::dot_row(w_row, x, s_row, ctx.group / cpb, ctx.group, cpb, table(cpb))
}

#[test]
fn the_affine_zero_is_the_table_zero() {
    assert_eq!(affine_zero(I4.table, 16), Some(8.0));
    assert_eq!(affine_zero(I2.table, 4), Some(2.0));
    // 一张不落在仿射关系上的表必须被挡在通道外。
    let bent = [0.0, 2.0, 4.0, 9.0];
    assert_eq!(affine_zero(&bent, 4), None);
    assert_eq!(affine_zero(&bent, 3), None);
}

/// 车道数只取两档，且必须整除一组；凑不满一档的形状由 [`admit`] 在进内核之前挡下。
#[test]
fn the_lane_count_only_takes_the_two_wide_tiers() {
    assert_eq!(
        lanes_of(GROUP / 2),
        Some(WIDE),
        "4 位码：一组 16 字节、16 车道"
    );
    assert_eq!(
        lanes_of(GROUP / 4),
        Some(NARROW),
        "2 位码：一组 8 字节、8 车道"
    );
    // 一组 2 个码、一组 12 个码：两档都不整除，不添第三档，也不为零头补掩码路径。
    assert_eq!(lanes_of(8 / 4), None);
    assert_eq!(lanes_of(12), None);
    // 走不到通道的形状必须在 `admit` 就被挡下，而不是让内核给出一个数。
    assert!(admit(2, N, context(M, 2), table(2)).is_some());
    assert!(admit(4, N, context(M, 4), table(4)).is_some());
    // 一组 2 个码的形状（`m = 8`、2 位码）：凑不满一条向量，`admit` 必须拒绝。
    assert!(admit(4, 2, context(8, 4), table(4)).is_none());
    // int8 不打包：码表对不上，不走本通道（它走 `elementwise`）。
    assert!(admit(1, N, context(M, 1), table(1)).is_none());
    // 输出宽度不足一列组：尾部没有 `COLS` 行可并，回退。
    assert!(admit(2, COLS - 1, context(M, 2), table(2)).is_none());
    // 表不落在仿射关系上：查表换不成算式的形状，回退。
    assert!(admit(2, N, context(M, 2), &[0.0, 2.0, 4.0, 9.0]).is_none());
}

/// 跑一遍字节内核，把每行与标量叶 [`oracle`] 逐行比对；`agree` 决定该行的容差口径。
///
/// 两个用例只差 `values`（权重取值）与 `agree`（容差口径），装配与遍历这一份。
fn compare_rows(values: fn(usize) -> Vec<f32>, agree: impl Fn(f32, f32, usize, usize)) {
    for cpb in [2, 4] {
        let ctx = context(M, cpb);
        let (w, s, x) = (weights(M, N, cpb), scales(N, ctx.groups), values(M));
        let mut out = vec![0.0f32; N];
        forward_batch(&[&x], &w, &s, admitted(cpb, ctx), &mut out);
        for (j, &got) in out.iter().enumerate() {
            let want = oracle(
                M,
                cpb,
                &w[j * ctx.row..][..ctx.row],
                &s[j * ctx.groups..][..ctx.groups],
                &x,
            );
            agree(got, want, cpb, j);
        }
    }
}

#[test]
fn exactly_representable_values_are_bit_identical() {
    // 整数码 × 整数 x × 2 的幂标度：每一步都不产生舍入，于是归约顺序无关，两条通道逐位相同。
    compare_rows(whole, |got, want, cpb, j| {
        assert_eq!(got, want, "cpb = {cpb} 第 {j} 行必须与标量叶逐位相同");
    });
}

#[test]
fn inexact_values_agree_within_rounding() {
    // 非精确取值：只有舍入关联不同，差值必须落在 f32 舍入的量级里。
    compare_rows(fractional, |got, want, cpb, j| {
        let slack = 1e-5 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= slack,
            "cpb = {cpb} 第 {j} 行：{got} vs {want}"
        );
    });
}

#[test]
fn a_row_batch_equals_row_by_row_calls() {
    let cpb = 2;
    let ctx = context(M, cpb);
    let rows = 3;
    let (w, s) = (weights(M, N, cpb), scales(N, ctx.groups));
    let base = fractional(M);
    let mut x = Vec::with_capacity(rows * M);
    for r in 0..rows {
        x.extend(base.iter().map(|v| v + r as f32));
    }
    let affine = admitted(cpb, ctx);
    let rows_of_x: Vec<&[f32]> = x.chunks(M).collect();
    let mut batched = vec![0.0f32; rows * N];
    forward_batch(&rows_of_x, &w, &s, affine, &mut batched);
    for r in 0..rows {
        let mut single = vec![0.0f32; N];
        forward_batch(&[&x[r * M..][..M]], &w, &s, affine, &mut single);
        assert_eq!(
            batched[r * N..][..N],
            *single,
            "第 {r} 行：批与逐行调用必须逐位相同"
        );
    }
}

/// 逐元素通道的一种取法与逐个展开比对：容差取 f32 舍入量级（累加顺序不同，不能逐位比）。
fn agree_elementwise<T: Load>(w: &[T], x: &[f32], one: impl Fn(T) -> f32) {
    let want: f32 = w.iter().zip(x).map(|(&elem, &v)| one(elem) * v).sum();
    let got = elementwise(w, x);
    let slack = 1e-5 * want.abs().max(1.0);
    assert!((got - want).abs() <= slack, "{got} vs {want}");
}

/// 稠密 `f32` 的逐元素通道：累加链轮转、整段与零头都得走到。
#[test]
fn the_dense_channel_covers_every_chain_and_the_tail() {
    // 一段 = `CHAINS · WIDE` 个元素；两边的零头与整段都要出现。
    for m in [
        1,
        WIDE - 1,
        WIDE,
        WIDE + 1,
        CHAINS * WIDE,
        CHAINS * WIDE + 3,
        257,
    ] {
        let w: Vec<f32> = (0..m)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        agree_elementwise(&w, &fractional(m), core::convert::identity);
    }
}

/// 对称 `int8` 的逐元素通道：`> 127` 的字节必须按二补数落到负值，而不是当无符号数乘。
#[test]
fn the_int8_channel_sign_extends_the_byte_pattern() {
    for m in [1, WIDE - 1, WIDE, WIDE + 1, CHAINS * WIDE + 5, 263] {
        // `% 251` 跨过 127，正值与负值交替出现；短样本凑不齐两侧，按整段起点判定。
        let w: Vec<u8> = (0..m).map(|i| ((i * 37) % 251) as u8).collect();
        if m >= WIDE {
            assert!(
                w.iter().any(|&b| b > 127),
                "样本必须跨过半程，否则符号没被考到"
            );
            assert!(w.iter().any(|&b| b <= 127), "样本必须同时含正码");
        }
        agree_elementwise(&w, &fractional(m), |b| (b as i8) as f32);
    }
}
