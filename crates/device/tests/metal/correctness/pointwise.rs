//! 职责：Metal 真机逐点一元算子端到端对拍——目录 lowering 出 MSL → 真机编译 → 派发 → 回读 → CPU 参考。
//!
//! 表驱动：装配 / 编译 / 派发 / 回读 / 断言只写一次（[`run_entry`]），每个算子一行配置（[`ENTRIES`]）。
//! 参考闭包与设备 `.k` 表达式同式；域内取值由条目自带生成器。无设备时 early return，不算失败。

use super::*;
use nn::model::{Buf, Sample, Values};

/// 一条逐点算子对拍配置：目录 key、CPU 同式参考、域内输入生成器、容差（`0.0` 表示位级相等）。
struct Entry {
    op: &'static str,
    apply: fn(f32) -> f32,
    inputs: fn(usize) -> Vec<f32>,
    tol: f64,
}

/// 对拍长度：96 个元素，单线程组可覆盖；末组多出来的线程由逐点网格的长度守卫跳过。
const LEN: usize = 96;

/// 线性夹具：`off + step·i`，条目按各自取值域选 `step`/`off`。
fn linear(len: usize, step: f32, off: f32) -> Vec<f32> {
    (0..len).map(|i| off + step * i as f32).collect()
}

/// 设备 `.k` 用 exp 组合的激活族，参考按同一 exp 式给出，避免与另一条数学库实现比对齐。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

/// 逐点一元族条目：纯操作 / 取整族位级对齐（`tol = 0.0`）；库函数 / exp 组合族给相对容差。
const ENTRIES: &[Entry] = &[
    // 纯算术与三态：设备与 Rust 对同一 f32 输入做同一操作，结果位级一致。
    Entry { op: "abs_f32", apply: f32::abs, inputs: |len| linear(len, 0.125, -6.0), tol: 0.0 },
    Entry { op: "neg_f32", apply: |x| -x, inputs: |len| linear(len, 0.125, -6.0), tol: 0.0 },
    Entry { op: "square_f32", apply: |x| x * x, inputs: |len| linear(len, 0.125, -6.0), tol: 0.0 },
    Entry {
        op: "sign_f32",
        apply: |x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        inputs: |len| linear(len, 0.125, -6.0),
        tol: 0.0,
    },
    Entry {
        op: "relu_f32",
        apply: |x| x.max(0.0),
        inputs: |len| linear(len, 0.125, -6.0),
        tol: 0.0,
    },
    // exp 组合激活：设备与参考都走同一 exp 式，相对误差受 f32 舍入支配。
    Entry {
        op: "silu_f32",
        apply: |x| x * sigmoid(x),
        inputs: |len| linear(len, 0.125, -6.0),
        tol: 1e-5,
    },
    Entry {
        op: "sigmoid_f32",
        apply: sigmoid,
        inputs: |len| linear(len, 0.125, -6.0),
        tol: 1e-5,
    },
    Entry {
        op: "tanh_f32",
        apply: |x| 2.0 / (1.0 + f32::exp(-2.0 * x)) - 1.0,
        inputs: |len| linear(len, 0.125, -6.0),
        tol: 1e-5,
    },
    // 指数：输入压在不溢出 f32 的区间，参考取同一标量核。
    Entry {
        op: "exp_f32",
        apply: f32::exp,
        inputs: |len| linear(len, 0.08, -6.0),
        tol: 1e-5,
    },
    Entry {
        op: "exp2_f32",
        apply: f32::exp2,
        inputs: |len| linear(len, 0.08, -6.0),
        tol: 1e-5,
    },
    // 正域：log / sqrt / rsqrt / recip 只取正值，避开 NaN 与除零。
    Entry {
        op: "log_f32",
        apply: f32::ln,
        inputs: |len| linear(len, 0.125, 0.25),
        tol: 1e-5,
    },
    Entry {
        op: "sqrt_f32",
        apply: f32::sqrt,
        inputs: |len| linear(len, 0.125, 0.25),
        tol: 1e-5,
    },
    Entry {
        op: "rsqrt_f32",
        apply: |x| 1.0 / x.sqrt(),
        inputs: |len| linear(len, 0.125, 0.25),
        tol: 1e-5,
    },
    Entry {
        op: "recip_f32",
        apply: |x| 1.0 / x,
        inputs: |len| linear(len, 0.125, 0.25),
        tol: 1e-5,
    },
    Entry {
        op: "log1p_f32",
        apply: f32::ln_1p,
        inputs: |len| linear(len, 0.0625, -0.5),
        tol: 1e-5,
    },
    // 三角：输入限在主值区间，避开大参数下库实现分歧。
    Entry {
        op: "sin_f32",
        apply: f32::sin,
        inputs: |len| linear(len, 0.04, -1.57),
        tol: 1e-5,
    },
    Entry {
        op: "cos_f32",
        apply: f32::cos,
        inputs: |len| linear(len, 0.04, -1.57),
        tol: 1e-5,
    },
    // 取整：结果是整数，设备与 Rust 取整规则一致（round 半值远离零），位级对齐。
    Entry {
        op: "ceil_f32",
        apply: f32::ceil,
        inputs: |len| linear(len, 0.125, -3.0),
        tol: 0.0,
    },
    Entry {
        op: "floor_f32",
        apply: f32::floor,
        inputs: |len| linear(len, 0.125, -3.0),
        tol: 0.0,
    },
    Entry {
        op: "round_f32",
        apply: f32::round,
        inputs: |len| linear(len, 0.125, -3.0),
        tol: 0.0,
    },
    Entry {
        op: "trunc_f32",
        apply: f32::trunc,
        inputs: |len| linear(len, 0.125, -3.0),
        tol: 0.0,
    },
];

#[test]
fn pointwise_unary_matches_cpu_reference_on_metal() {
    let Some((device, queue)) = bench_device("逐点一元对拍") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-pointwise");

    for entry in ENTRIES {
        run_entry(&device, &queue, &catalogue, &dirs, entry);
    }
}

/// 单个条目：构样本 → 降 MSL → 编译 → 上传输入 → 派发 → 回读对拍。
fn run_entry(
    device: &MetalDevice,
    queue: &MetalCommandQueue,
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    entry: &Entry,
) {
    let input = (entry.inputs)(LEN);
    let reference: Vec<f32> = input.iter().map(|&x| (entry.apply)(x)).collect();

    let sample = Sample {
        m: LEN,
        n: LEN,
        scalars: vec![("N", LEN as i64)],
        bufs: vec![
            Buf { name: "a", values: Values::F32(input.clone()) },
            Buf { name: "y", values: Values::F32(vec![0.0; LEN]) },
        ],
    };

    let msl = program_sample(
        catalogue,
        dirs,
        entry.op,
        Metal,
        Mode::pointwise(),
        &sample,
    )
    .and_then(|stages| stages.one("降源"))
    .unwrap_or_else(|error| panic!("{} MSL 应能生成：{error}", entry.op));
    let pipeline = device
        .compile(&msl)
        .unwrap_or_else(|| panic!("{} MSL 应能编译", entry.op));

    let a_buf = upload(device, &input, "a");
    let mut y = output_buf(device, LEN, "y");
    assert!(
        queue.execute(&pipeline, &[&a_buf, &y], [LEN as u32, 1, 1], [1, 1, 1]),
        "{} 派发应成功",
        entry.op
    );

    let got = read_back(y.as_mut_bytes());
    let label = format!("{} Metal 输出", entry.op);
    if entry.tol <= 0.0 {
        for (index, (got, want)) in got.iter().zip(&reference).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label} 元素 {index}：设备 {got:.1e}、参考 {want:.1e}"
            );
        }
    } else {
        testkit::assert_rel_with(testkit::Scale::Unit, &got, &reference, entry.tol, &label);
    }
}
