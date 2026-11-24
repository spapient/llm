//! 职责：Metal 真机逐点族（一元 / 二元）端到端对拍——目录 lowering 出 MSL → 真机编译 → 派发 → 回读 → CPU 参考。
//!
//! 表驱动：装配 / 编译 / 派发 / 回读 / 断言只写一次（[`run_entry`]），每个算子一行配置（[`ENTRIES`]）。
//! 参考闭包与设备 `.k` 表达式同式；域内取值由条目自带生成器。无设备时 early return，不算失败。

use super::*;
use nn::model::{Buf, Sample, Values};

/// 逐点算子的输入档：一元一个输入，二元两个等长输入。
enum Op {
    Unary(fn(f32) -> f32, fn(usize) -> Vec<f32>),
    Binary(fn(f32, f32) -> f32, fn(usize) -> (Vec<f32>, Vec<f32>)),
}

/// 一条逐点算子对拍配置：目录 key、输入档与参考、容差（`0.0` 表示位级相等）。
struct Entry {
    op: &'static str,
    tol: f64,
    kind: Op,
}

/// 对拍长度：96 个元素，单线程组可覆盖；末组多出来的线程由逐点网格的长度守卫跳过。
const LEN: usize = 96;

/// 线性夹具：`off + step·i`，条目按各自取值域选 `step`/`off`。
fn linear(len: usize, step: f32, off: f32) -> Vec<f32> {
    (0..len).map(|i| off + step * i as f32).collect()
}

/// 二元线性夹具：两条独立线性序列。
fn pair(len: usize, step_a: f32, off_a: f32, step_b: f32, off_b: f32) -> (Vec<f32>, Vec<f32>) {
    (linear(len, step_a, off_a), linear(len, step_b, off_b))
}

/// 把 `-0.0` 归一成 `+0.0`：取整族两端可能只差零号位，数值相等不算分歧。
fn zero_clean(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

/// 设备 `.k` 用 exp 组合的激活族，参考按同一 exp 式给出，避免与另一条数学库实现比对齐。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

/// 逐点族条目：纯操作 / 取整族位级对齐（`tol = 0.0`）；库函数 / exp 组合 / 除法族给相对容差。
const ENTRIES: &[Entry] = &[
    // ── 一元：纯算术与三态 ──────────────────────────────────────────────
    Entry {
        op: "abs_f32",
        tol: 0.0,
        kind: Op::Unary(f32::abs, |len| linear(len, 0.125, -6.0)),
    },
    Entry {
        op: "neg_f32",
        tol: 0.0,
        kind: Op::Unary(|x| -x, |len| linear(len, 0.125, -6.0)),
    },
    Entry {
        op: "square_f32",
        tol: 0.0,
        kind: Op::Unary(|x| x * x, |len| linear(len, 0.125, -6.0)),
    },
    Entry {
        op: "sign_f32",
        tol: 0.0,
        kind: Op::Unary(
            |x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            },
            |len| linear(len, 0.125, -6.0),
        ),
    },
    Entry {
        op: "relu_f32",
        tol: 0.0,
        kind: Op::Unary(|x| x.max(0.0), |len| linear(len, 0.125, -6.0)),
    },
    Entry {
        op: "leaky_relu_f32",
        tol: 0.0,
        kind: Op::Unary(
            |x| if x >= 0.0 { x } else { 0.01 * x },
            |len| linear(len, 0.125, -6.0),
        ),
    },
    Entry {
        op: "hard_sigmoid_f32",
        tol: 1e-5,
        kind: Op::Unary(
            |x| (x / 6.0 + 0.5).clamp(0.0, 1.0),
            |len| linear(len, 0.25, -6.0),
        ),
    },
    Entry {
        op: "scale_f32",
        tol: 0.0,
        kind: Op::Unary(|x| 0.5 * x, |len| linear(len, 0.25, -2.0)),
    },
    // GELU：两侧同一份 A&S 7.1.26 系数，系数字面与 exp 实现略有出入，给相对容差。
    Entry {
        op: "gelu_f32",
        tol: 1e-5,
        kind: Op::Unary(
            |x| 0.5 * x * (1.0 + nn::gelu_f32::erf(x * std::f32::consts::FRAC_1_SQRT_2)),
            |len| linear(len, 0.125, -3.0),
        ),
    },
    // ── 一元：exp 组合激活 ───────────────────────────────────────────────
    Entry {
        op: "silu_f32",
        tol: 1e-5,
        kind: Op::Unary(|x| x * sigmoid(x), |len| linear(len, 0.125, -6.0)),
    },
    Entry {
        op: "sigmoid_f32",
        tol: 1e-5,
        kind: Op::Unary(sigmoid, |len| linear(len, 0.125, -6.0)),
    },
    Entry {
        op: "tanh_f32",
        tol: 1e-5,
        kind: Op::Unary(
            |x| 2.0 / (1.0 + f32::exp(-2.0 * x)) - 1.0,
            |len| linear(len, 0.125, -6.0),
        ),
    },
    // 指数：输入压在不溢出 f32 的区间，参考取同一标量核。
    Entry {
        op: "exp_f32",
        tol: 1e-5,
        kind: Op::Unary(f32::exp, |len| linear(len, 0.08, -6.0)),
    },
    Entry {
        op: "exp2_f32",
        tol: 1e-5,
        kind: Op::Unary(f32::exp2, |len| linear(len, 0.08, -6.0)),
    },
    // 正域：log / sqrt / rsqrt / recip 只取正值，避开 NaN 与除零。
    Entry {
        op: "log_f32",
        tol: 1e-5,
        kind: Op::Unary(f32::ln, |len| linear(len, 0.125, 0.25)),
    },
    Entry {
        op: "sqrt_f32",
        tol: 1e-5,
        kind: Op::Unary(f32::sqrt, |len| linear(len, 0.125, 0.25)),
    },
    Entry {
        op: "rsqrt_f32",
        tol: 1e-5,
        kind: Op::Unary(|x| 1.0 / x.sqrt(), |len| linear(len, 0.125, 0.25)),
    },
    Entry {
        op: "recip_f32",
        tol: 1e-5,
        kind: Op::Unary(|x| 1.0 / x, |len| linear(len, 0.125, 0.25)),
    },
    Entry {
        op: "log1p_f32",
        tol: 1e-5,
        kind: Op::Unary(f32::ln_1p, |len| linear(len, 0.0625, -0.5)),
    },
    // 三角：输入限在主值区间，避开大参数下库实现分歧。
    Entry {
        op: "sin_f32",
        tol: 1e-5,
        kind: Op::Unary(f32::sin, |len| linear(len, 0.04, -1.57)),
    },
    Entry {
        op: "cos_f32",
        tol: 1e-5,
        kind: Op::Unary(f32::cos, |len| linear(len, 0.04, -1.57)),
    },
    // 取整：结果是整数，设备与 Rust 取整规则一致（round 半值远离零），位级对齐。
    Entry {
        op: "ceil_f32",
        tol: 0.0,
        kind: Op::Unary(f32::ceil, |len| linear(len, 0.125, -3.0)),
    },
    Entry {
        op: "floor_f32",
        tol: 0.0,
        kind: Op::Unary(f32::floor, |len| linear(len, 0.125, -3.0)),
    },
    Entry {
        op: "round_f32",
        tol: 0.0,
        kind: Op::Unary(f32::round, |len| linear(len, 0.125, -3.0)),
    },
    Entry {
        op: "trunc_f32",
        tol: 0.0,
        kind: Op::Unary(f32::trunc, |len| linear(len, 0.125, -3.0)),
    },
    // ── 二元：算术与比较 ────────────────────────────────────────────────
    Entry {
        op: "add_f32",
        tol: 0.0,
        kind: Op::Binary(|x, y| x + y, |len| pair(len, 0.125, -3.0, 0.125, -3.0)),
    },
    Entry {
        op: "sub_f32",
        tol: 0.0,
        kind: Op::Binary(|x, y| x - y, |len| pair(len, 0.125, -3.0, 0.125, -3.0)),
    },
    Entry {
        op: "mul_f32",
        tol: 0.0,
        kind: Op::Binary(|x, y| x * y, |len| pair(len, 0.125, -3.0, 0.125, -3.0)),
    },
    Entry {
        op: "max_f32",
        tol: 0.0,
        kind: Op::Binary(|x, y| x.max(y), |len| pair(len, 0.125, -3.0, 0.125, -3.0)),
    },
    Entry {
        op: "min_f32",
        tol: 0.0,
        kind: Op::Binary(|x, y| x.min(y), |len| pair(len, 0.125, -3.0, 0.125, -3.0)),
    },
    Entry {
        op: "div_f32",
        tol: 1e-5,
        // 除数取恒正非零序列，避开除零与 NaN；除法给相对容差。
        kind: Op::Binary(
            |x, y| x / y,
            |len| {
                let a = linear(len, 0.125, -3.0);
                let b = (0..len).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
                (a, b)
            },
        ),
    },
];

#[test]
fn pointwise_elementwise_matches_cpu_reference_on_metal() {
    let Some((device, queue)) = bench_device("逐点族对拍") else {
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
    // 按输入档取出输入与 CPU 参考，并按签名顺序拼出样本缓冲与派发缓冲表。
    let (a, b, reference, bufs) = match entry.kind {
        Op::Unary(apply, inputs) => {
            let a = (inputs)(LEN);
            let reference: Vec<f32> = a.iter().map(|&x| apply(x)).collect();
            let bufs = vec![
                Buf {
                    name: "a",
                    values: Values::F32(a.clone()),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; LEN]),
                },
            ];
            (a, None, reference, bufs)
        }
        Op::Binary(apply, inputs) => {
            let (a, b) = (inputs)(LEN);
            let reference: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| apply(x, y)).collect();
            let bufs = vec![
                Buf {
                    name: "a",
                    values: Values::F32(a.clone()),
                },
                Buf {
                    name: "b",
                    values: Values::F32(b.clone()),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; LEN]),
                },
            ];
            (a, Some(b), reference, bufs)
        }
    };

    let sample = Sample {
        m: LEN,
        n: LEN,
        scalars: vec![("N", LEN as i64)],
        bufs,
    };

    let msl = program_sample(catalogue, dirs, entry.op, Metal, Mode::pointwise(), &sample)
        .and_then(|stages| stages.one("降源"))
        .unwrap_or_else(|error| panic!("{} MSL 应能生成：{error}", entry.op));
    let pipeline = device
        .compile(&msl)
        .unwrap_or_else(|| panic!("{} MSL 应能编译", entry.op));

    let a_buf = upload(device, &a, "a");
    let b_buf = b.map(|values| upload(device, &values, "b"));
    let mut y = output_buf(device, LEN, "y");

    let mut bufs: Vec<&MetalBuf> = vec![&a_buf];
    if let Some(b_buf) = &b_buf {
        bufs.push(b_buf);
    }
    bufs.push(&y);

    assert!(
        queue.execute(&pipeline, &bufs, [LEN as u32, 1, 1], [1, 1, 1]),
        "{} 派发应成功",
        entry.op
    );

    let got = read_back(y.as_mut_bytes());
    let label = format!("{} Metal 输出", entry.op);
    if entry.tol <= 0.0 {
        for (idx, (got, want)) in got.iter().zip(&reference).enumerate() {
            // 取整族可能给出 +0.0 / -0.0：数值相等只是零号位不同，断言前把 ±0 归一成 +0。
            let (got, want) = (zero_clean(*got), zero_clean(*want));
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label} 元素 {idx}：设备 {got:.1e}、参考 {want:.1e}"
            );
        }
    } else {
        testkit::assert_rel_with(testkit::Scale::Unit, &got, &reference, entry.tol, &label);
    }
}
