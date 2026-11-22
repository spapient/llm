//! 写回尾块的设备对拍：把 `epilogue.<ext>` 里的真实段落按声明顺序装配、编译、执行，再与
//! `crate::epilogue::apply` 逐元素比。
//!
//! 这是「写回真的接到了设备」的可执行证据：被测的不是另一份 C 算术，而是装配器（`cutile::substitute::epilogue`）
//! 对每台设备吐出的那段文本本身。六台设备共用同一组场景，语义漂移会直接落在某一台上。

use crate::epilogue::apply;
use cutile::substitute::{EpiArgs, epilogue};
use llm_spec::epilogue::Step;
use prof::atom::{ActKind, DType};

/// 瓦片形状：`bias` 的长度是列数，其余步按元素走。
const ROWS: usize = 2;
const COLS: usize = 3;
const N: usize = ROWS * COLS;

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

/// 一个场景：初始瓦片、两个第二操作数与声明顺序。
struct Scenario {
    /// 失败信息里用的名字。
    name: &'static str,
    /// `$0` 的初值（行主序）。
    init: [f32; N],
    /// `bias` 的列偏置。
    bias: [f32; COLS],
    /// `residual` 的残差。
    residual: [f32; N],
    /// 施加顺序。
    steps: &'static [Step],
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "bias",
        init: [0.4, -0.9, 2.6, -3.5, 0.0, 1.25],
        bias: [0.5, -0.25, 1.0],
        residual: [0.0; N],
        steps: ONLY_BIAS,
    },
    Scenario {
        name: "residual-act-scale",
        init: [0.4, -0.9, 2.6, -3.5, 0.0, 1.25],
        bias: [0.5, -0.25, 1.0],
        residual: [0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
        steps: RESID_ACT_SCALE,
    },
    Scenario {
        name: "fused-i4",
        init: [0.4, -0.9, 2.6, -3.5, 0.0, 1.25],
        bias: [0.5, -0.25, 1.0],
        residual: [0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
        steps: FUSED,
    },
    Scenario {
        name: "hard-sigmoid-i8",
        init: [0.4, -0.9, 2.6, -3.5, 0.0, 1.25],
        bias: [0.5, -0.25, 1.0],
        residual: [0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
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
fn source(device: &prof::Device, scenario: &Scenario) -> Result<String, String> {
    let count = N.to_string();
    let rows = ROWS.to_string();
    let args = EpiArgs {
        out: "y",
        bias: "BIAS",
        residual: "RES",
        count: &count,
        rows: &rows,
    };
    let body = epilogue(device, scenario.steps, &args).map_err(|error| error.to_string())?;
    Ok(format!(
        "#include <math.h>\n\
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
        n = N,
        cols = COLS,
        init = literals(&scenario.init),
        bias = literals(&scenario.bias),
        resid = literals(&scenario.residual),
        body = body,
    ))
}

/// 参考口径：同一份步骤施加到初始瓦片上。
fn reference(scenario: &Scenario) -> Vec<f32> {
    let mut want = scenario.init.to_vec();
    apply(
        scenario.steps,
        &mut want,
        COLS,
        Some(&scenario.bias),
        Some(&scenario.residual),
    )
    .expect("场景本身合法");
    want
}

/// 跑一个场景：装配 → 编译 → 执行 → 逐元素比。
fn check(device: &prof::Device, scenario: &Scenario) -> Result<(), String> {
    let text = source(device, scenario)?;
    let want = reference(scenario);
    let calls = [testkit::witness::Call {
        func: "epi",
        expect: &want,
    }];
    let reading = testkit::witness::run(&text, device.ext, &calls)?
        .into_iter()
        .next()
        .ok_or("没有任何读数")?;
    if reading.count as usize != N {
        return Err(format!(
            "`{}` / `{}` 写出 {} 个元素，应为 {N}",
            device.vendor, scenario.name, reading.count
        ));
    }
    if reading.max_rel_err > TOL {
        return Err(format!(
            "`{}` / `{}` 与 `gemm::epilogue::apply` 不符：最大相对误差 {:.3e}（上限 {TOL:.0e}）\n  \
             got  {:?}\n  want {want:?}",
            device.vendor, scenario.name, reading.max_rel_err, reading.values
        ));
    }
    Ok(())
}

#[test]
fn every_device_runs_the_declared_epilogue() {
    let mut failures = Vec::new();
    for device in prim::devices() {
        for scenario in SCENARIOS {
            if let Err(error) = check(device, scenario) {
                failures.push(error);
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
    let device = prim::device("cpu").expect("cpu 在设备表里");
    let text = source(device, &SCENARIOS[0]).expect("装配");
    let wrong = vec![0.0f32; N];
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
