//! 前向内核：按行（解码）与按批（prefill）两条路径，点积本身分「带标度」与「稠密」两种。
//!
//! 两条路径都把「权重行 × 输入向量」叫成一行计算，差别只在输入是单向量还是批；点积的
//! lane 数（[`LANES`]）按格式的元素宽度挑，保证每轮取满一条向量寄存器。
use diagnostic::{Code, Error};

use super::format::Plain;
use super::shape::{finite_scales, required_weights, scaled_groups};

/// 带标度行骨架：分组、粒度、权重与标度守卫连同行切片统一收口，逐行点积由 `row_dot` 给出。
///
/// `cpb` 是「一个权重单位里有多少个元素」：打包格式给每字节码数（`2` / `4`），不打包的逐元素
/// 格式给 `1`——那时行宽就是 `m` 个元素，粒度倍数守卫恒真。于是打包与不打包只差这一处行宽
/// 折算和一份逐行算术，守卫、切片与遍历是同一份代码，按元素类型单态化。
pub(super) fn forward_rows<E>(
    x: &[f32],
    weights: &[E],
    scales: &[f32],
    out: &mut [f32],
    cpb: usize,
    row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
) -> diagnostic::Result<()> {
    let ctx = scaled_row_context(x.len(), out.len(), scales, weights.len(), cpb)?;
    forward_rows_scaled(x, weights, scales, out, ctx, row_dot);
    Ok(())
}

/// 逐行遍历一遍，算术由 `row_dot` 给：守卫已经过了（`ctx` 就是证据），这里只剩切片与调用。
///
/// 单独一支是因为叶有时要把好几行算在一起（`gemv::simd` 的四列累加器），那时 `row_dot`
/// 这一行一个值的接口不合适；叶拿着同一个 `ctx` 自己走遍历，且**不必重跑守卫**——
/// [`scaled_row_context`] 里的标度有限性判定是 `O(n·groups)`，在 4096² 上重复一次就是内核
/// 算术的两成。
pub(crate) fn forward_rows_scaled<E>(
    x: &[f32],
    weights: &[E],
    scales: &[f32],
    out: &mut [f32],
    ctx: ScaledRow,
    row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
) {
    for (index, value) in out.iter_mut().enumerate() {
        let w_row = &weights[index * ctx.row..][..ctx.row];
        let s_row = &scales[index * ctx.groups..][..ctx.groups];
        *value = row_dot(w_row, x, s_row, ctx.group);
    }
}

/// 行批骨架：输出行在外、输入行在里，一行权重只读一遍。
///
/// 与 [`forward_rows`] 是同一个遍历的两种嵌套：那里输入是向量（一层），这里输入是
/// `rows × m` 的行批（两层）。输入行放内层就是"权重复用"的全部内容——权重行只从内存读一遍、
/// 解包一遍，而不是按行数重复；逐行调用单行入口会把同一份权重读 `rows` 遍。`cpb` 见
/// [`forward_rows`]：打包与不打包共用这一副嵌套。
fn forward_batch_rows<E>(
    x: &[f32],
    weights: &[E],
    scales: &[f32],
    geo: Geometry,
    out: &mut [f32],
    cpb: usize,
    row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
) -> diagnostic::Result<()> {
    batch_geometry(x.len(), out.len(), geo)?;
    let ctx = scaled_row_context(geo.m, geo.n, scales, weights.len(), cpb)?;
    forward_batch_rows_scaled(x, weights, scales, geo, out, ctx, row_dot);
    Ok(())
}

/// 行批的逐行遍历，算术由 `row_dot` 给：与 [`forward_rows_scaled`] 同一个"守卫已过"约定。
pub(crate) fn forward_batch_rows_scaled<E>(
    x: &[f32],
    weights: &[E],
    scales: &[f32],
    geo: Geometry,
    out: &mut [f32],
    ctx: ScaledRow,
    row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
) {
    for j in 0..geo.n {
        let w_row = &weights[j * ctx.row..][..ctx.row];
        let s_row = &scales[j * ctx.groups..][..ctx.groups];
        for r in 0..geo.rows {
            out[r * geo.n + j] = row_dot(w_row, &x[r * geo.m..][..geo.m], s_row, ctx.group);
        }
    }
}

/// 带标度格式的行守卫：分组、粒度倍数、权重长度与标度有限四项一次判齐，返回 `(groups, group)`。
///
/// 单行与行批两条入口共用这一处：批入口只是先判几何。先判结构（分组、倍数、长度）再判取值
/// （标度有限），于是报错指向第一处真正不对的地方；`cpb = 1` 时不打包格式的倍数守卫恒真。
pub(super) fn scaled_row_context(
    m: usize,
    n: usize,
    scales: &[f32],
    weights: usize,
    cpb: usize,
) -> diagnostic::Result<ScaledRow> {
    let (groups, group) = scaled_groups(m, n, scales.len())?;
    if !group.is_multiple_of(cpb) {
        return Err(Error::new(
            Code::Invalid,
            format!("分组 {group} 必须是 {cpb} 的倍数"),
        ));
    }
    required_weights(weights, n, m / cpb)?;
    finite_scales(scales, groups)?;
    Ok(ScaledRow {
        groups,
        group,
        row: m / cpb,
    })
}

/// 已过守卫的带标度行上下文：每行多少组、每组多少元素、每行多少个权重单位。
///
/// 这是"守卫已经过了"的证据而不是三个裸 `usize`——裸数传进叶里，"它们一定对"就成了一句注释；
/// 收成一个值之后，重跑守卫（尤其 `O(n·groups)` 的标度判定）就不再有理由，叶只需要读权重与
/// 标度。构造只有 [`scaled_row_context`] 一条路。
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaledRow {
    /// 每行多少组。
    pub groups: usize,
    /// 每组多少元素（必须是 `cpb` 的倍数）。
    pub group: usize,
    /// 每行多少个权重单位（打包格式是 `m / cpb` 个字节）。
    pub row: usize,
}

/// 行批逐元素入口：按配置选择带标度 / 无标度两条骨架，与 [`Format::forward`] 同一分叉。
pub fn forward_plain_batch<C: Plain>(
    x: &[f32],
    weights: &[C::Elem],
    scales: &[f32],
    geo: Geometry,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    if C::SCALES.is_some() {
        forward_batch_rows(x, weights, scales, geo, out, 1, scaled_dot::<C>)
    } else {
        if !scales.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                format!("稠密格式没有标度，收到 {} 个", scales.len()),
            ));
        }
        forward_batch_rows(x, weights, scales, geo, out, 1, |w, x, _, _| {
            dense_dot::<C>(w, x)
        })
    }
}

/// 行批几何：`rows` 行输入、输入行宽 `m`、输出行宽 `n`。
///
/// 三个数必须同时自洽（`x.len() == rows·m`、`out.len() == rows·n`），且都只对行批有意义；
/// 分开传三个同型 `usize` 既容易在调用点换位，也让每个入口都多出两个参数。收成一个值后，
/// 几何的校只在入口守卫一处，量化与逐元素两条骨架传同一个类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// 输入行数。
    pub rows: usize,
    /// 输入行宽（权重行宽）。
    pub m: usize,
    /// 输出行宽。
    pub n: usize,
}

impl Geometry {
    /// 由三个数直接构造；合法性留给入口守卫，构造本身不报错。
    pub const fn new(rows: usize, m: usize, n: usize) -> Self {
        Self { rows, m, n }
    }
}

/// 行批几何：行数非零，输入与输出的长度分别对上 `rows × m` 与 `rows × n`。
///
/// 三处一起判而不分判：批次调用时三个数必须同时自洽，单独对上一个没有意义。
pub(super) fn batch_geometry(x: usize, out: usize, geo: Geometry) -> diagnostic::Result<()> {
    if geo.rows == 0 {
        return Err(Error::new(Code::Invalid, "行批至少一行"));
    }
    let (want_x, want_out) = (
        geo.rows.saturating_mul(geo.m),
        geo.rows.saturating_mul(geo.n),
    );
    if x != want_x || out != want_out {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "行批尺寸不符：x={x}，out={out}，期望 {}×{}={want_x}、{}×{}={want_out}",
                geo.rows, geo.m, geo.rows, geo.n
            ),
        ));
    }
    Ok(())
}

/// 无标度逐元素格式的行遍历：换错算子（给了标度）必须报错，而不是静默忽略。
pub(super) fn forward_dense<C: Plain>(
    x: &[f32],
    weights: &[C::Elem],
    scales: &[f32],
    out: &mut [f32],
) -> diagnostic::Result<()> {
    if !scales.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!("稠密格式没有标度，收到 {} 个", scales.len()),
        ));
    }
    let (m, n) = (x.len(), out.len());
    required_weights(weights.len(), n, m)?;
    for (row, value) in out.iter_mut().enumerate() {
        let w_row = &weights[row * m..][..m];
        *value = dense_dot::<C>(w_row, x);
    }
    Ok(())
}

/// 一行的带标度点积：按组推进，每组解包点积一次乘标度（[`dot`] 承担单元素反量化）。
pub(super) fn scaled_dot<C: Plain>(
    w_row: &[C::Elem],
    x: &[f32],
    s_row: &[f32],
    group: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for ((w_chunk, scale), x_chunk) in w_row.chunks(group).zip(s_row).zip(x.chunks(group)) {
        acc += dot::<C>(w_chunk, x_chunk) * scale;
    }
    acc
}

/// 累加链数：16 条 = 4 条 NEON 寄存器（x86 的 AVX2 上是 2 条 `ymm`）。
///
/// 这个数不是凑出来的：`crates/benchmarks/src/kernel.rs` 的 `gemv_f32/decode_1x4096x4096`
/// （1×4096×4096 decode 步，权重一遍 64 MiB）换链数实测 4 条链 3.8 ms、8 条 1.3 ms、12 条
/// 3.7 ms、16 条 0.89 ms、24 条 2.4 ms、32 条 0.91 ms——
/// 只有 2 的幂的条数能被展开成整条向量指令，16 条在多一个寄存器轮转之后就不再获益。
const LANES: usize = 16;

/// 一行的无标度点积：`LANES` 条独立累加链，内层编译成整条向量乘加。
///
/// 与 [`dot`] 的 `sum` 差在归约的形状：`sum` 是严格的左折叠，f32 没有 fast-math，LLVM 只能一次
/// 加一个——上面那组实测里它就是 2.85 ms 那一档，decode 的带宽全被它挡住。`LANES` 条链彼此独立，
/// 同样的权重读一遍算出整条向量的宽度（实测 3.2×）；链内顺序与左折叠不同，只带来末位舍入差，
/// 确定性样本的取值落在 2 的幂网格上，两者逐位相同。
fn dense_dot<C: Plain>(w_row: &[C::Elem], x: &[f32]) -> f32 {
    let (w_chunks, w_tail) = w_row.as_chunks::<LANES>();
    let (x_chunks, x_tail) = x.as_chunks::<LANES>();
    let mut lanes = [0.0f32; LANES];
    for (w, x) in w_chunks.iter().zip(x_chunks) {
        let mut lane = 0;
        while lane < LANES {
            lanes[lane] += C::to_f32(w[lane]) * x[lane];
            lane += 1;
        }
    }
    let mut acc = lanes.iter().sum::<f32>();
    for (w, x) in w_tail.iter().zip(x_tail) {
        acc += C::to_f32(*w) * *x;
    }
    acc
}

/// 一段逐元素点积：`sum(to_f32(w) · x)`，按配置的单元素反量化。
fn dot<C: Plain>(w: &[C::Elem], x: &[f32]) -> f32 {
    w.iter()
        .zip(x)
        .map(|(elem, xv)| C::to_f32(*elem) * xv)
        .sum()
}
