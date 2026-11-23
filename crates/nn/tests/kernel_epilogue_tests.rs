// `epilogue::apply` 的执行面：几何步、激活步、量化步、顺序与错误。

use super::apply;
use prof::epilogue::parse;

/// 空步骤是恒等：`out` 不变。
#[test]
fn empty_steps_are_identity() {
    let mut out = [1.0, 2.0, 3.0];
    assert!(apply(&[], &mut out, 3, None, None).is_ok());
    assert_eq!(out, [1.0, 2.0, 3.0]);
}

/// `bias` 按列广播：`out[r][c] += bias[c]`。
#[test]
fn bias_broadcasts_across_rows() {
    let steps = parse("bias").unwrap();
    let mut out = [1.0, 2.0, 3.0, 4.0];
    assert!(apply(&steps, &mut out, 2, Some(&[10.0, 20.0]), None).is_ok());
    assert_eq!(out, [11.0, 22.0, 13.0, 24.0]);
}

/// `bias` 列宽不符：返回错误而不是静默截断。
#[test]
fn bias_wrong_width_is_rejected() {
    let steps = parse("bias").unwrap();
    let mut out = [0.0; 4];
    assert!(apply(&steps, &mut out, 2, Some(&[10.0, 20.0, 30.0]), None).is_err());
}

/// `residual` 逐元素加：`out[i] += residual[i]`。
#[test]
fn residual_adds_elementwise() {
    let steps = parse("residual").unwrap();
    let mut out = [1.0, 2.0, 3.0];
    assert!(apply(&steps, &mut out, 3, None, Some(&[10.0, 20.0, 30.0])).is_ok());
    assert_eq!(out, [11.0, 22.0, 33.0]);
}

/// `residual` 长度不符：返回错误。
#[test]
fn residual_wrong_len_is_rejected() {
    let steps = parse("residual").unwrap();
    let mut out = [0.0; 4];
    assert!(apply(&steps, &mut out, 4, None, Some(&[1.0, 2.0])).is_err());
}

/// `scale:<α>` 逐元素乘。
#[test]
fn scale_multiplies() {
    let steps = parse("scale:0.5").unwrap();
    let mut out = [2.0, 4.0, 6.0];
    assert!(apply(&steps, &mut out, 3, None, None).is_ok());
    assert_eq!(out, [1.0, 2.0, 3.0]);
}

/// `act:relu` 负区归零、正区恒等。
#[test]
fn relu_applies() {
    let steps = parse("act:relu").unwrap();
    let mut out = [-2.0, 0.0, 2.0];
    assert!(apply(&steps, &mut out, 3, None, None).is_ok());
    assert_eq!(out, [0.0, 0.0, 2.0]);
}

/// `act:gelu` 与 [`crate::gelu_f32::forward`] 逐位一致：执行层只转发、不写第二份算术。
#[test]
fn gelu_delegates_to_unary_kernel() {
    let steps = parse("act:gelu").unwrap();
    let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut out = input;
    apply(&steps, &mut out, 5, None, None).unwrap();
    let mut reference = input;
    crate::gelu_f32::forward(tensor::View::one(&input), &mut reference).unwrap();
    assert_eq!(out, reference);
}

/// `quant:i8` 对称 round-trip：步长 0.25，最大误差 0.125，越界饱和。
#[test]
fn quant_i8_rounds_to_grid() {
    let steps = parse("quant:i8").unwrap();
    let mut out = [0.4, 0.13, -0.4, 100.0];
    apply(&steps, &mut out, 4, None, None).unwrap();
    assert!((out[0] - 0.5).abs() < 1e-6, "0.4 -> {}", out[0]);
    assert!((out[1] - 0.25).abs() < 1e-6, "0.13 -> {}", out[1]);
    assert!((out[2] + 0.5).abs() < 1e-6, "-0.4 -> {}", out[2]);
    assert!((out[3] - 31.75).abs() < 1e-6, "饱和 -> {}", out[3]);
}

/// `store` 是无副作用标记：`out` 不变。
#[test]
fn store_is_noop() {
    let steps = parse("store").unwrap();
    let mut out = [1.0, 2.0];
    assert!(apply(&steps, &mut out, 2, None, None).is_ok());
    assert_eq!(out, [1.0, 2.0]);
}

/// 顺序语义：`bias; scale:2; quant:i8` 严格按声明顺序执行。
#[test]
fn pipeline_runs_in_declared_order() {
    let steps = parse("bias; scale:2; quant:i8").unwrap();
    let mut out = [0.5, 1.5, 2.5, 3.5];
    assert!(apply(&steps, &mut out, 2, Some(&[1.0, 2.0]), None).is_ok());
    // bias → [1.5, 3.5, 3.5, 5.5]；scale:2 → [3, 7, 7, 11]；quant:i8 已在 0.25 网格上，恒等。
    assert_eq!(out, [3.0, 7.0, 7.0, 11.0]);
}

// 写回尾块的设备对拍：把 `epilogue.<ext>` 里的真实段落按声明顺序装配、编译、执行，再与
// `crate::epilogue::apply` 逐元素比。
// 
// 这是「写回真的接到了设备」的可执行证据：被测的不是另一份 C 算术，而是装配器（`cutile::substitute::epilogue`）
// 对每台设备吐出的那段文本本身。六台设备共用同一组场景，语义漂移会直接落在某一台上。

use cutile::substitute::{EpiArgs, epilogue};
use prof::atom::{ActKind, DType};
use prof::epilogue::Step;

/// 场景数组的最大元素数：见证 ABI 的写出上限 [`testkit::witness::MAX_OUT`] 就是这个数，
/// 用例不可能比它更大，所以数组按上限定长、按形状切片。
const MAX_N: usize = testkit::witness::MAX_OUT;

/// 场景数组的最大列数：`bias` 的长度等于列数，比元素数更紧。
const MAX_COLS: usize = 5;

/// 一个瓦片形状：`bias` 按列广播，其余步按 `rows * cols` 个元素走。
#[derive(Clone, Copy)]
struct Shape {
    /// 行数：`bias` 的外层，也是行内基址 `行号 * 列数` 的来源。
    rows: usize,
    /// 列数：`bias` 的内层宽度。
    cols: usize,
}

impl Shape {
    /// 逐元素步看到的元素数。
    fn count(self) -> usize {
        self.rows * self.cols
    }
}

/// 三个形状合起来把 4-wide 的向量主环与标量尾环都真的跑到，并守住行内基址：
/// `2×3` 让逐元素步落在 6 = 4+2（向量 + 尾）、`2×4` 让 `bias` 落在 8 = 4+4（两行各一整块，
/// 行基址必须逐行偏移）、`1×5` 让 `bias` 与逐元素步都落在 5 = 4+1（向量 + 尾）。
const SHAPES: &[Shape] = &[
    Shape { rows: 2, cols: 3 },
    Shape { rows: 2, cols: 4 },
    Shape { rows: 1, cols: 5 },
];

/// 相对误差上限，与 `cutile::reference` 的 f32 口径一致。
const TOL: f64 = 1e-5;

/// 只有 `bias`。
const ONLY_BIAS: &[Step] = &[Step::Bias, Step::Store];
/// 残差 → 激活 → 缩放：三条都改同一个元素。
const RESID_ACT_SCALE: &[Step] = &[
    Step::Residual,
    Step::Act(ActKind::Relu),
    Step::Scale(0.5),
    Step::Store,
];
/// 四步复合，包含 i4 量化（会在 `[-8, 7]` 上钳位）。
const FUSED: &[Step] = &[
    Step::Bias,
    Step::Residual,
    Step::Act(ActKind::LeakyRelu),
    Step::Quant(DType::I4),
    Step::Store,
];
/// 另一条激活 + i8 量化。
const QUANT8: &[Step] = &[
    Step::Act(ActKind::HardSigmoid),
    Step::Quant(DType::I8),
    Step::Store,
];

/// 一个场景：初始瓦片、两个第二操作数与声明顺序。数组按 [`MAX_N`] / [`MAX_COLS`] 定长，
/// 跑某个 [`Shape`] 时按需切片，同一份场景因此能被多个形状复用。
struct Scenario {
    /// 失败信息里用的名字。
    name: &'static str,
    /// `$0` 的初值（行主序）。
    init: [f32; MAX_N],
    /// `bias` 的列偏置。
    bias: [f32; MAX_COLS],
    /// `residual` 的残差。
    residual: [f32; MAX_N],
    /// 施加顺序。
    steps: &'static [Step],
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "bias",
        init: [0.4, -0.9, 2.6, -3.5, 0.0, 1.25, 0.02, -0.06],
        bias: [0.5, -0.25, 1.0, 0.125, -2.0],
        residual: [0.0; MAX_N],
        steps: ONLY_BIAS,
    },
    Scenario {
        name: "residual-act-scale",
        init: [0.4, -0.9, 2.6, -3.5, 0.0, 1.25, 0.02, -0.06],
        bias: [0.5, -0.25, 1.0, 0.125, -2.0],
        residual: [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
        steps: RESID_ACT_SCALE,
    },
    Scenario {
        name: "fused-i4",
        init: [0.4, -0.9, 2.6, -3.5, 0.0, 1.25, 0.02, -0.06],
        bias: [0.5, -0.25, 1.0, 0.125, -2.0],
        residual: [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
        steps: FUSED,
    },
    Scenario {
        name: "hard-sigmoid-i8",
        init: [0.4, -0.9, 2.6, -3.5, 0.0, 1.25, 0.02, -0.06],
        bias: [0.5, -0.25, 1.0, 0.125, -2.0],
        residual: [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
        steps: QUANT8,
    },
];

/// C 数组初始化：`f32` 的最短往返字面量加 `f` 后缀，值本身逐位不变。
fn literals(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:?}f"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// 装配出的 C 函数：本地瓦片 `y` 承载 `$0`，两个第二操作数各自承载 `$1`。
///
/// 段文本本身是设备写法（`.cu` / `.cl` / `.metal` / `.cpp` 各一份），但都只用宿主 C / C++ 也认的
/// 构造——这正是 `witness.<ext>` 一直守的约定，装配出的尾块沿用它，所以能在宿主上原样执行。
/// 这里补的头文件与设备档案的 `preamble` 对齐：真实装配路径由 `cutile::emit` 前置 `preamble`，
/// 本用例自建翻译单元，就得自己带上同样的 `<math.h>` 与条件 `<arm_neon.h>`。
fn source(device: &prof::Device, scenario: &Scenario, shape: Shape) -> Result<String, String> {
    let n = shape.count();
    let count = n.to_string();
    let cols = shape.cols.to_string();
    let rows = shape.rows.to_string();
    let args = EpiArgs {
        out: "y",
        bias: "BIAS",
        residual: "RES",
        count: &count,
        cols: &cols,
        rows: &rows,
    };
    let body = epilogue(device, scenario.steps, &args).map_err(|error| error.to_string())?;
    Ok(format!(
        "#include <math.h>\n\
         #if defined(__aarch64__)\n\
         #include <arm_neon.h>\n\
         #endif\n\
         static const float INIT[{n}] = {{ {init} }};\n\
         static const float BIAS[{cols}] = {{ {bias} }};\n\
         static const float RES[{n}] = {{ {resid} }};\n\
         int epi(float *out) {{\n\
         float y[{n}];\n\
         for (int i = 0; i < {n}; ++i) y[i] = INIT[i];\n\
         {body}\n\
         for (int i = 0; i < {n}; ++i) out[i] = y[i];\n\
         return {n};\n\
         }}\n",
        n = n,
        cols = shape.cols,
        init = literals(&scenario.init[..n]),
        bias = literals(&scenario.bias[..shape.cols]),
        resid = literals(&scenario.residual[..n]),
        body = body,
    ))
}

/// 参考口径：同一份步骤施加到初始瓦片上。
fn reference(scenario: &Scenario, shape: Shape) -> Vec<f32> {
    let n = shape.count();
    let mut want = scenario.init[..n].to_vec();
    apply(
        scenario.steps,
        &mut want,
        shape.cols,
        Some(&scenario.bias[..shape.cols]),
        Some(&scenario.residual[..n]),
    )
    .expect("场景本身合法");
    want
}

/// 跑一个场景：装配 → 编译 → 执行 → 逐元素比。
fn check_device(device: &prof::Device, scenario: &Scenario, shape: Shape) -> Result<(), String> {
    let text = source(device, scenario, shape)?;
    let want = reference(scenario, shape);
    let calls = [testkit::witness::Call {
        func: "epi",
        expect: &want,
    }];
    let reading = testkit::witness::run(&text, device.ext, &calls)?
        .into_iter()
        .next()
        .ok_or("没有任何读数")?;
    let n = shape.count();
    if reading.count as usize != n {
        return Err(format!(
            "`{}` / `{}` 写出 {} 个元素，应为 {n}",
            device.vendor, scenario.name, reading.count
        ));
    }
    if reading.max_rel_err > TOL {
        return Err(format!(
            "`{}` / `{}` 在 {}×{} 上与 `gemm::epilogue::apply` 不符：最大相对误差 {:.3e}（上限 {TOL:.0e}）\n  \
             got  {:?}\n  want {want:?}",
            device.vendor,
            scenario.name,
            shape.rows,
            shape.cols,
            reading.max_rel_err,
            reading.values
        ));
    }
    Ok(())
}

/// 比半步略小的那个 `f32`（`0.125` 的前一个可表示值）。`QUANT_SCALE` 是 `0.25`，二进制的
/// 幂，所以量化商 `x / 0.25` 就是精确的 `4x`：`4 * 0.124999993 = 0.49999997`，正好落在
/// 「半步下方」——`f32::round` 给 0，而 `(int)(q + 0.5f)` 会先把 `q + 0.5f` 抬成 `1.0f`
/// 再截断成 1。这个值因此能把取整口径的偏差钉成一个确定性的失败，不依赖随机命中。
const JUST_BELOW_EIGHTH: f32 = f32::from_bits(0x3dff_ffff);

/// 只做量化的步骤序列。
const HALFWAY_STEPS: &[Step] = &[Step::Quant(DType::I8), Step::Store];

/// 量化取整口径：半数远离零，与 `gemm::epilogue::apply` 的 `f32::round` 同口径。
///
/// 三步都要守住：向量主环（前四个元素）走 `vcvtaq_s32_f32`、标量尾环（后两个元素）走
/// `roundf`、两者都得与 Rust 侧一致。边界值故意在两条环里各放一份。
///
/// 只跑 `cpu`：其余五台的 `epilogue.*` 里仍是 `(int)(x ± 0.5f)` 的旧写法（同一处偏 1 个码字，
/// 且超出 int 范围时是未定义行为），把这条场景放进共享表会让它们红；等五台一起改到再合表。
#[test]
fn cpu_quant_rounds_halfway_away_from_zero() {
    let edges = Scenario {
        name: "quant-halfway",
        init: [
            JUST_BELOW_EIGHTH,
            -JUST_BELOW_EIGHTH,
            0.125,
            -0.125,
            JUST_BELOW_EIGHTH,
            -JUST_BELOW_EIGHTH,
            0.0,
            0.0,
        ],
        bias: [0.0; MAX_COLS],
        residual: [0.0; MAX_N],
        steps: HALFWAY_STEPS,
    };
    let shape = Shape { rows: 2, cols: 3 };
    let want = reference(&edges, shape);
    assert_eq!(
        want[0], 0.0,
        "构造前提被打破了：`{JUST_BELOW_EIGHTH}` 的量化商应落在半步下方、取整到 0"
    );
    assert!(
        want[2] > 0.0,
        "构造前提被打破了：`0.125` 的量化商正好是半步，半数远离零应进位"
    );
    let device = prof::device("cpu").expect("cpu 在设备表里");
    check_device(device, &edges, shape).expect("cpu 的量化取整口径");
}

/// 六台设备 × 四个场景 × 三个形状：形状这一维专门用来把 4-wide 的向量主环与标量尾环都跑到，
/// 只在其中一个形状上过不算过。
#[test]
fn every_device_runs_the_declared_epilogue() {
    let mut failures = Vec::new();
    for device in prof::devices() {
        for scenario in SCENARIOS {
            for shape in SHAPES {
                if let Err(error) = check_device(device, scenario, *shape) {
                    failures.push(error);
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "写回尾块装配未与设备对齐：\n{}",
        failures.join("\n")
    );
}

#[test]
fn a_wrong_expectation_is_rejected() {
    let device = prof::device("cpu").expect("cpu 在设备表里");
    let shape = SHAPES[0];
    let text = source(device, &SCENARIOS[0], shape).expect("装配");
    let wrong = vec![0.0f32; shape.count()];
    let calls = [testkit::witness::Call {
        func: "epi",
        expect: &wrong,
    }];
    let reading = testkit::witness::run(&text, device.ext, &calls)
        .expect("执行")
        .into_iter()
        .next()
        .expect("一条读数");
    assert!(
        reading.max_rel_err > TOL,
        "错误的期望值不该被判为通过：{reading:?}"
    );
}
