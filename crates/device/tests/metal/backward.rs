//! 职责：Metal 真机反向梯度对拍——设备反向核经 [`MetalExecutor`] 登记 → 绑梯度操作数 → 图提交 →
//! 同步回读，与 CPU 中心差分逐元素对拍。
//!
//! 前向设备核只对前向输出负责；本用例补反向这条边：对每个逐点一元算子，喂一组上游梯度 `dout`，
//! 设备核算 `dx = dout · f′(x)`，参考用标量前向函数在 `x±h` 上做中心差分 `dL/dx = dout·[f(x+h)−f(x−h)]/(2h)`。
//! 无设备时 early return，不算失败。

use super::*;
use crate::backward::lookup;
use engine::exec::Executor;
use engine::exec::{Graph, Launch, Node, Stream};
use shard::execution::Backend;

/// 对拍长度：96 个元素，单线程组可覆盖；末组越界线程由内核长度守卫跳过。
const LEN: usize = 96;

/// 中心差分步长：参考在 f64 里求值，步长取 f64 量级，避免 f32 相减把有效位吃掉。
const STEP: f64 = 1e-6;

/// 一条反向对拍配置：算子名、f64 标量前向函数（只供中心差分参考）、输入夹具生成器。
struct Entry {
    op: &'static str,
    f: fn(f64) -> f64,
    inputs: fn(usize) -> Vec<f32>,
}

/// 线性夹具：`off + step·i`，条目按取值域选 `step`/`off`。
fn linear(len: usize, step: f32, off: f32) -> Vec<f32> {
    (0..len).map(|i| off + step * i as f32).collect()
}

/// σ(x)：与前向参考同式，避免对齐到另一条数学库实现。
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

/// tanh(x)：exp 组合口径，与前向夹具一致。
fn tanh(x: f64) -> f64 {
    2.0 / (1.0 + f64::exp(-2.0 * x)) - 1.0
}

/// A&S 7.1.26 误差函数（|误差| < 1.5e-7），与 `nn::gelu_f32::erf` 同式，f64 档供中心差分参考。
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let p = 0.327_591_1;
    let t = 1.0 / (1.0 + p * x);
    let series = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    let y = 1.0 - series * (-x * x).exp();
    sign * y
}

/// GELU 前向：`0.5·x·(1 + erf(x/√2))`。
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// 反向逐点族条目：域内取值避开溢出 / 除零；导数由 [`crate::backward`] 描述表给出。
const ENTRIES: &[Entry] = &[
    Entry {
        op: "relu_f32",
        f: |x| x.max(0.0),
        // 错开半步，避开 x=0 折点：折点处中心差分取到单侧斜率，与次梯度约定不可比。
        inputs: |len| linear(len, 0.125, -6.0625),
    },
    Entry {
        op: "leaky_relu_f32",
        f: |x| if x >= 0.0 { x } else { 0.01 * x },
        inputs: |len| linear(len, 0.125, -6.0625),
    },
    Entry {
        op: "sigmoid_f32",
        f: sigmoid,
        inputs: |len| linear(len, 0.125, -6.0),
    },
    Entry {
        op: "tanh_f32",
        f: tanh,
        inputs: |len| linear(len, 0.125, -6.0),
    },
    Entry {
        op: "exp_f32",
        f: f64::exp,
        inputs: |len| linear(len, 0.08, -6.0),
    },
    Entry {
        op: "silu_f32",
        f: |x| x * sigmoid(x),
        inputs: |len| linear(len, 0.0625, -3.0),
    },
    Entry {
        op: "gelu_f32",
        f: gelu,
        inputs: |len| linear(len, 0.0625, -3.0),
    },
];

#[test]
fn backward_gradient_matches_central_difference_on_metal() {
    for entry in ENTRIES {
        run_entry(entry);
    }
}

/// 单条目：造输入 / 上游梯度 → 生成反向 MSL 并登记进执行器 → 绑四路缓冲 → 提交同步 → 回读对拍。
fn run_entry(entry: &Entry) {
    let Some(descriptor) = lookup(entry.op) else {
        panic!("{} 应在设备反向描述表内", entry.op);
    };
    let Some(mut executor) = MetalExecutor::new("metal", "arm64") else {
        return;
    };

    let x = (entry.inputs)(LEN);
    let y: Vec<f32> = x.iter().map(|&v| (entry.f)(v as f64) as f32).collect();
    // 非恒一的上游梯度：同时验出「逐元素乘 dout」这一步，而不只是偏导本身。
    let dout: Vec<f32> = (0..LEN).map(|i| 0.5 + 0.1 * i as f32).collect();

    let msl = crate::backward::source(Backend::Metal, descriptor, LEN);
    let name = format!("{}_grad", entry.op);
    executor
        .register(&name, &msl)
        .unwrap_or_else(|error| panic!("{} 反向管线登记失败：{error}", entry.op));

    bind_f32(&mut executor, 0, &x);
    bind_f32(&mut executor, 1, &y);
    bind_f32(&mut executor, 2, &dout);
    // 输出梯度槽留白：内核必须把每一项写满，不允许残留旧值。
    executor
        .bind_bytes(3, LEN * 4)
        .unwrap_or_else(|error| panic!("{} 梯度输出缓冲分配失败：{error}", entry.op));

    let mut node = Node::new(&name, Launch::new([LEN as u32, 1, 1], [1, 1, 1]));
    node.bind_input(0);
    node.bind_input(1);
    node.bind_input(2);
    node.bind_output(3);
    let mut graph = Graph::new();
    graph.push(node);

    executor
        .submit(&graph, Stream::new(0))
        .unwrap_or_else(|error| panic!("{} 反向图提交失败：{error}", entry.op));
    executor
        .sync(Stream::new(0))
        .unwrap_or_else(|error| panic!("{} 反向同步失败：{error}", entry.op));

    let dx = executor
        .buf_mut(3)
        .map(|buf| read_back(buf.as_mut_bytes()))
        .unwrap_or_else(|| panic!("{} 梯度输出缓冲不在场", entry.op));

    // 参考在 f64 里做中心差分：`dL/dx = dout·[f(x+h)−f(x−h)]/(2h)`，再压回 f32 与设备输出比。
    let want: Vec<f32> = x
        .iter()
        .zip(&dout)
        .map(|(&value, &upstream)| {
            let value = value as f64;
            let central = ((entry.f)(value + STEP) - (entry.f)(value - STEP)) / (2.0 * STEP);
            (upstream as f64 * central) as f32
        })
        .collect();

    // 容差对齐 train gradcheck 口径：逐元素相对 5%，Unit 地板同时兜住近零梯度的绝对误差。
    testkit::assert_rel_with(
        testkit::Scale::Unit,
        &dx,
        &want,
        5e-2,
        &format!("{} Metal 反向梯度与中心差分", entry.op),
    );
}

/// 按编号分配一路 f32 缓冲并写入给定值；返回被替换的旧块（这里不关心）。
fn bind_f32(executor: &mut MetalExecutor, id: u32, values: &[f32]) {
    executor
        .bind_bytes(id, values.len() * 4)
        .unwrap_or_else(|error| panic!("缓冲 {id} 分配失败：{error}"));
    let Some(buf) = executor.buf_mut(id) else {
        panic!("缓冲 {id} 分配后应在场");
    };
    buf.as_mut_bytes().copy_from_slice(&as_bytes(values));
}
