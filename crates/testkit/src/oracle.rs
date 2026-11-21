//! 对拍口径的**唯一性**门禁：夹具层的读数与生产侧 `cutile::reference` 必须是同一个判据。
//!
//! 两份实现并存是结构性的：`llm-testkit` 的库体零依赖（夹具不能把 `cutile` 拉进来），而
//! `cutile::reference` 是生产侧口径（`cli` 的 verify、`engine` 的设备基准都直接用它，不能只
//! 活在测试里）。口径只允许有一条，于是留一道门禁把两者逐位钉住——分母、非有限值的分类、
//! 长度处理、全零参考任何一处漂移都会在这里红。
//!
//! 阈值本身不在这里另设一张表：精度分档只由 `cutile::reference::tolerance` 拥有，本门禁只
//! 校验 `within_ratio(got, want, tolerance(dt))` 与该分档一致。

#[cfg(test)]
pub(crate) mod tests {
use cutile::atom::DType;
use cutile::reference;
use crate::{Scale, max_rel_err, rel_err, rel_err_with, within_ratio};

/// 参与交叉对拍的样本：覆盖空批、全零参考、跨 8 车道边界、次正规、溢出与三类非有限值。
fn samples() -> Vec<Vec<f32>> {
    vec![
        vec![],
        vec![0.0],
        vec![0.0, 0.0],
        vec![1.0, -2.0, 3.0],
        vec![1e-30, 3.0],
        vec![f32::MAX, 1.0],
        vec![f32::NAN, 1.0],
        vec![f32::INFINITY, 1.0],
        vec![f32::NEG_INFINITY, f32::INFINITY],
        (0..9).map(|i| i as f32 * 0.25 - 1.0).collect(),
    ]
}

#[test]
fn the_two_oracles_agree_bit_for_bit() {
    let samples = samples();
    for (left, got) in samples.iter().enumerate() {
        for (right, want) in samples.iter().enumerate() {
            assert_eq!(
                max_rel_err(got, want).to_bits(),
                reference::max_rel_err(got, want).to_bits(),
                "样本 {left}×{right} 的判据读数不同"
            );
            assert_eq!(
                rel_err(got, want).ratio().to_bits(),
                max_rel_err(got, want).to_bits(),
                "样本 {left}×{right}：读数与投影不是同一个数"
            );
        }
    }
}

#[test]
fn the_tolerance_ladder_is_shared_not_copied() {
    let samples = samples();
    for (left, got) in samples.iter().enumerate() {
        for (right, want) in samples.iter().enumerate() {
            for dtype in [DType::F32, DType::F64, DType::I8, DType::I4] {
                assert_eq!(
                    within_ratio(got, want, reference::tolerance(dtype)),
                    reference::within(got, want, dtype),
                    "{dtype:?} 分档下样本 {left}×{right} 的判定不同"
                );
            }
        }
    }
}

#[test]
fn the_worst_element_attains_the_reported_error() {
    let got = [1.0, 2.5, -3.0, 4.0];
    let want = [1.0, 2.0, 3.0, 4.0];
    let report = rel_err(&got, &want);
    let at = report.worst.expect("有偏差就有最差点");
    assert_eq!(at.index, 2, "−3 与 3 相差 6，是整批最大绝对误差");
    assert_eq!((at.got, at.want), (-3.0, 3.0));
    let attained = f64::from((at.got - at.want).abs());
    assert_eq!(report.err, attained, "读数必须是这个元素上的绝对误差");
    assert_eq!(report.scale, 4.0, "分母是整批最大的参考幅度");
}

/// 两条分母策略相差的**只有分母**：幅度远小于 1 的批次里单位口径给 1 的绝对地板，幅度大于 1 时
/// 单位口径反而更严。这条用例把「两种用法不是松紧之分」钉住：引擎的真机基准（`metal_tests`）
/// 用 [`Scale::Unit`]（GEMM 与注意力输出的量级差四个数量级），其余门禁与这里的交叉对拍用整批口径。
#[test]
fn the_two_denominators_differ_only_in_the_floor() {
    let got = [0.00011, 0.0002];
    let want = [0.0001, 0.0002];
    let batch = rel_err(&got, &want);
    let unit = rel_err_with(&got, &want, Scale::Unit);
    assert_eq!(batch.worst.map(|worst| worst.index), Some(0));
    assert_eq!(unit.worst.map(|worst| worst.index), Some(0));
    assert_eq!(
        batch.scale,
        f64::from(0.0002f32),
        "整批口径的分母是最大幅度"
    );
    assert_eq!(unit.scale, 1.0, "单位口径的分母在该元素上是地板 1");
    assert_eq!(unit.ratio(), f64::from((0.00011f32 - 0.0001f32).abs()));
    assert!(
        batch.ratio() > unit.ratio() * 100.0,
        "参考整体小于 1 时，整批口径把同一个绝对差放大成两个数量级以上"
    );

    let got = [1.5, 100.0];
    let want = [1.0, 100.0];
    assert!(
        rel_err_with(&got, &want, Scale::Unit).ratio() > rel_err(&got, &want).ratio() * 10.0,
        "参考幅度大于 1 时单位口径更严：两条是不同的问题，不是松紧两档"
    );
}
}
