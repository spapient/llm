//! 反向数值执行器：按 [`Schedule`] 的逆拓扑序把 loss 梯度回传到参数终点。

use autograd::{Node, Schedule, Tape, Var};
use diagnostic::{Code, Error, Result};

use crate::geometry::Geometry;
use crate::kernel::{Kernel, same_str};
use crate::pool::Buffers;

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::backprop";

/// 按已规划的 [`Schedule`] 回传梯度：起点 loss 梯度为全 `1`，逐节点分配到图上输入。
///
/// 梯度累加进 `buffers` 的梯度槽；图外（需求为无梯度）输入不收账。
///
/// # Errors
///
/// 前向值/上游梯度不在场、账本顺序不符，或遇到尚未接反向公式的算子时返回对应 [`Code`]。
pub(crate) fn backprop<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    schedule: &Schedule,
) -> Result<()> {
    // 暂存从缓冲区里借出：`Context` 持只读快照、`apply_formula` 要 `&mut buffers`，
    // 同一结构无法同时借，故把暂存放在独立结构里逐节点复用——稳态不再向分配器要内存。
    let mut scratch = std::mem::take(&mut buffers.scratch);
    seed_loss(buffers, schedule.loss(), &mut scratch)?;
    let mut account = schedule.account();
    for step in schedule.steps() {
        reverse_step(buffers, tape, step, &mut scratch)?;
        account.settle(step).map_err(|error| error.with(SITE))?;
    }
    buffers.scratch = scratch;
    Ok(())
}

/// 单节点反向：取上游梯度、装配公式上下文，查表执行并把逐输入贡献喂回梯度槽。
fn reverse_step<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    step: &Node,
    scratch: &mut Scratch,
) -> Result<()> {
    copy_into(&mut scratch.incoming, incoming(buffers, step)?);
    collect_inputs(buffers, step, &mut scratch.inputs)?;
    copy_into(&mut scratch.output, values(buffers, step.output())?);
    let context = Context {
        incoming: &scratch.incoming,
        inputs: &scratch.inputs,
        output: &scratch.output,
        geometry: buffers.geometry(step.output()),
    };
    apply_formula(buffers, tape, step, context)
}

/// 反向逐节点复用的暂存：容量只增不减，热路径上不再逐节点向分配器要内存。
///
/// 三者都是本节点前向快照的只读副本，故与 `Buffers` 分开持有：一份借不冲突的读视图，
/// 一份可写缓冲区，36 条公式因此仍然只吃 `Context`、不碰 `Buffers`。
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    incoming: Vec<f32>,
    inputs: Vec<Vec<f32>>,
    output: Vec<f32>,
    ones: Vec<f32>,
}

impl Scratch {
    /// 暂存的实际占用（**容量**之和）：容量只增不减，故容量不变即"没有重新分配"。
    #[cfg(test)]
    pub(crate) fn footprint(&self) -> usize {
        self.incoming.capacity()
            + self.output.capacity()
            + self.ones.capacity()
            + self.inputs.capacity()
            + self.inputs.iter().map(Vec::capacity).sum::<usize>()
    }
}

/// 整块覆盖写：保留已有容量，只改变长度与内容。
fn copy_into(dst: &mut Vec<f32>, src: &[f32]) {
    dst.clear();
    dst.extend_from_slice(src);
}

/// 把各输入的前向值整块拷出只读快照；任一前向值不在场即回缺失。
fn collect_inputs<K: Kernel>(
    buffers: &Buffers<K>,
    step: &Node,
    out: &mut Vec<Vec<f32>>,
) -> Result<()> {
    let inputs = step.inputs();
    out.resize_with(inputs.len(), Vec::new);
    for (slot, &input) in out.iter_mut().zip(inputs) {
        copy_into(slot, values(buffers, input)?);
    }
    Ok(())
}

/// 按算子名查公式目录执行，把返回的逐输入贡献喂回对应梯度槽；未登记公式即拒绝。
fn apply_formula<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    step: &Node,
    context: Context<'_>,
) -> Result<()> {
    let formula = lookup(step.op())?;
    let contributions = formula(context)?;
    for (&input, contrib) in step.inputs().iter().zip(contributions) {
        feed(buffers, tape, input, contrib)?;
    }
    Ok(())
}

/// 记一份梯度贡献；图外输入没有梯度缓冲，直接丢弃。
fn feed<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    var: Var,
    contrib: Vec<f32>,
) -> Result<()> {
    let receives = tape
        .requirement(var)
        .is_some_and(|requirement| !requirement.is_none());
    if receives {
        buffers
            .accumulate_gradient(var, &contrib)
            .map_err(|error| error.with(SITE))?;
    }
    Ok(())
}

/// loss 起点梯度：每个前向元素都从上游 `1` 出发。
fn seed_loss<K: Kernel>(buffers: &mut Buffers<K>, loss: Var, scratch: &mut Scratch) -> Result<()> {
    let width = values(buffers, loss)?.len();
    scratch.ones.clear();
    scratch.ones.resize(width, 1.0);
    buffers
        .load_gradient(loss, &scratch.ones)
        .map_err(|error| error.with(SITE))?;
    Ok(())
}

/// 读变量在场的前向 f32 值；缺失是重放/装配口径问题。
fn values<K: Kernel>(buffers: &Buffers<K>, var: Var) -> Result<&[f32]> {
    buffers
        .value(var)
        .and_then(|view| view.f32())
        .ok_or_else(|| {
            Error::new(
                Code::Missing,
                format!("变量 {} 的前向值不在场，无法反传", var.id()),
            )
            .with(SITE)
        })
}

/// 读节点输出已就位的上游梯度；缺失说明拓扑序上有前驱没回传到。
fn incoming<'a, K: Kernel>(buffers: &'a Buffers<K>, step: &Node) -> Result<&'a [f32]> {
    buffers.gradient(step.output()).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("节点 {} 的上游梯度未就位", step.output().id()),
        )
        .with(SITE)
    })
}

fn unsupported(op: &str) -> Error {
    Error::new(Code::Unsupported, format!("算子 `{op}` 还没有解析反向公式")).with(SITE)
}

/// 公式只读上下文：上游梯度、各输入前向值、输出前向值（均为本节点前向快照）。
struct Context<'a> {
    incoming: &'a [f32],
    inputs: &'a [Vec<f32>],
    output: &'a [f32],
    /// 本节点的实例几何；`None` 即该节点按声明档实例执行。
    geometry: Option<Geometry>,
}

impl Context<'_> {
    /// 本节点的实例几何：显式登记过就用它，否则取该算子的声明档实例（对拍基准）。
    fn instance(&self, op: &str) -> Result<Geometry> {
        match self.geometry {
            Some(geometry) => Ok(geometry),
            None => Geometry::declared(op),
        }
    }
}

/// 反向公式：按上下文算出与输入等长的逐元素贡献，顺序与节点输入对齐。
type Formula = fn(Context<'_>) -> Result<Vec<Vec<f32>>>;

/// 一条公式目录项：算子名与其解析反向公式。
struct Entry {
    op: &'static str,
    formula: Formula,
}

/// 已接反向公式的可微算子目录；其余算子（量化权重 / 反量化）在 [`apply_formula`] 处回
/// [`Code::Unsupported`]。与 [`op::REGISTRY`] 的 `differentiable` 标记一一对应，由下方 `const`
/// 断言在编译期守住。
/// 展开公式目录：`op => formula` 一行一条，数组长度由下方 [`FORMULAS`] 的声明给定。
macro_rules! formulas {
    ($($op:literal => $formula:ident),* $(,)?) => {
        [$(Entry { op: $op, formula: $formula }),*]
    };
}

const FORMULAS: [Entry; 36] = formulas![
    "adaptive_pool_f32" => adaptive_pool,
    "add_f32" => passthrough,
    "residual_add_f32" => passthrough,
    "mul_f32" => mul,
    "scale_f32" => scale,
    "sub_f32" => sub,
    "div_f32" => div,
    "exp_f32" => exp,
    "log_f32" => log,
    "min_f32" => min,
    "max_f32" => max,
    "max_pool1d_f32" => max_pool1d,
    "max_pool2d_f32" => max_pool2d,
    "silu_f32" => silu,
    "gelu_f32" => gelu,
    "hard_sigmoid_f32" => hard_sigmoid,
    "instance_norm_f32" => instance_norm,
    "interpolate_f32" => interpolate,
    "layer_norm_f32" => layer_norm,
    "leaky_relu_f32" => leaky_relu,
    "reduce_sum_f32" => reduce_sum,
    "reduce_max_f32" => reduce_max,
    "softmax_f32" => softmax,
    "gemv_f32" => gemv,
    "group_norm_f32" => group_norm,
    "gemm_f32" => gemm,
    "embed_f32" => embed,
    "rmsnorm_f32" => rmsnorm,
    "rope_f32" => rope,
    "attention_f32" => attention,
    "avg_pool2d_f32" => avg_pool2d,
    "batch_norm_f32" => batch_norm,
    "conv2d_f32" => conv2d,
    "flash_attention_f32" => attention,
    "decode_attention_f32" => decode_attention,
    "decode_flash_attention_f32" => decode_attention,
];

/// 编译期闭合：公式目录与 [`op::REGISTRY`] 的 `differentiable` 标记一一对应。
const _: () = formulas_cover(&FORMULAS, &op::REGISTRY);

/// 双向互锁：可微算子缺公式、公式指向不可微/未注册算子、或公式重复，都是编译错误。
const fn formulas_cover(formulas: &[Entry], registry: &[op::Registration]) {
    let mut i = 0;
    while i < registry.len() {
        assert!(
            !registry[i].differentiable || has_formula(formulas, registry[i].name),
            "存在标记为可微却缺少反向公式的算子"
        );
        i += 1;
    }
    let mut j = 0;
    while j < formulas.len() {
        assert!(
            differentiable(registry, formulas[j].op),
            "存在反向公式指向不可微或未注册的算子"
        );
        assert!(!duplicate(formulas, j), "存在重复登记的反向公式");
        j += 1;
    }
}

/// 公式目录里是否已有某算子名。
const fn has_formula(formulas: &[Entry], name: &str) -> bool {
    let mut i = 0;
    while i < formulas.len() {
        if same_str(formulas[i].op, name) {
            return true;
        }
        i += 1;
    }
    false
}

/// 算子名是否登记为可微；未注册返回 `false`（不存在"可微但未注册"的算子）。
const fn differentiable(registry: &[op::Registration], name: &str) -> bool {
    let mut i = 0;
    while i < registry.len() {
        if same_str(registry[i].name, name) {
            return registry[i].differentiable;
        }
        i += 1;
    }
    false
}

/// 公式目录第 `at` 条之前是否已登记同名算子。
const fn duplicate(formulas: &[Entry], at: usize) -> bool {
    let mut i = 0;
    while i < at {
        if same_str(formulas[i].op, formulas[at].op) {
            return true;
        }
        i += 1;
    }
    false
}

/// 按算子名查公式；目录未登记即回 [`Code::Unsupported`]。
fn lookup(op: &str) -> Result<Formula> {
    FORMULAS
        .iter()
        .find(|entry| entry.op == op)
        .map(|entry| entry.formula)
        .ok_or_else(|| unsupported(op))
}

mod formulas;
use formulas::*;

#[cfg(test)]
#[path = "../tests/backprop_tests.rs"]
mod tests;
