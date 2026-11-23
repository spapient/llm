//! 注意力两路：flash 在线 softmax 流式扫描，与 threadgroup 协作分块整行归约。

use super::geometry::{attention_bufs, attention_dims};
use crate::dialect::fill;
use diagnostic::{Code, Error, Result};
use prof::model::Sample;
use prof::sig::Signature;

/// 生产态 flash attention（Metal）：一个线程组负责一个查询行，K/V 按 `block` 个键分块
/// 流式扫描，在线 softmax（running max/sum）把输出累加在每线程寄存器，不保留完整分数向量。
///
/// 语义与 `O = softmax(QKᵀ/√D)·V` 逐元素相同，只是把「先算完整行、再归一化」改成
/// 「逐块修正」：每块先求块内 max，与 running max 合并得到新的 max，指数与输出按
/// `exp(m_old - m_new)` 缩放后并入。内存从 `O(S)` 降到 `O(block)`，因而可流式处理长序列。
///
/// 派发几何：`grid = [S·threads]`、`block = [threads]`；每组的 `gid / threads` 是查询行。
///
/// # Errors
///
/// 签名不足 4 缓冲（q/k/v/y）或 2 整数标量（S/D）、样本缺标量、线程/块几何非法、
/// 或头宽 `D` 不在 `1..=64` 时返回说明。
pub(crate) fn metal_flash_attention(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    threads: usize,
    block: usize,
) -> Result<String> {
    if !flash_geometry_ok(threads, block) {
        return Err(Error::new(
            Code::Invalid,
            "flash attention 线程数须为 2 的幂、块为线程数倍数",
        )
        .with("engine::tiled::metal_flash_attention"));
    }
    let (q, k, v, y) = attention_bufs(signature)?;
    let (s, d) = attention_dims(signature, sample)?;
    if !(1..=64).contains(&d) {
        return Err(
            Error::new(Code::Invalid, "flash attention 要求头宽 D 在 1..=64")
                .with("engine::tiled::metal_flash_attention"),
        );
    }
    Ok(flash_attention_msl(
        q,
        k,
        v,
        y,
        Flash {
            seq: s,
            dim: d,
            threads,
            block,
        },
    ))
}

/// 线程/块几何：线程数为 2 的幂且至少 2，块非零且是线程数倍数。
fn flash_geometry_ok(threads: usize, block: usize) -> bool {
    threads.is_power_of_two() && threads >= 2 && block != 0 && block.is_multiple_of(threads)
}

/// 派发几何：序列长、头宽、线程数与键块。四个数在正文里只当常量用，收在一处以免每个
/// 生成函数都重算一遍 `div_ceil` 与“除不除得尽”。
#[derive(Clone, Copy)]
struct Flash {
    seq: usize,
    dim: usize,
    threads: usize,
    block: usize,
}

impl Flash {
    /// 每线程承担的寄存器输出维数。
    fn lanes(self) -> usize {
        self.dim.div_ceil(self.threads)
    }

    /// 键块是否除不尽序列：除不尽才需要尾块守卫。
    fn tail(self) -> bool {
        !self.seq.is_multiple_of(self.block)
    }
}

/// flash attention 的 MSL 正文：在线 softmax 的块内归约 + 寄存器输出累加。
fn flash_attention_msl(q: &str, k: &str, v: &str, y: &str, shape: Flash) -> String {
    let nd = shape.lanes();
    let tail = shape.tail();
    let (s, d) = (shape.seq.to_string(), shape.dim.to_string());
    let (threads, block) = (shape.threads.to_string(), shape.block.to_string());
    let nds = nd.to_string();
    let decls: String = (0..nd)
        .map(|n| format!("    float acc_{n} = 0.0f;\n"))
        .collect();
    let updates: String = (0..nd)
        .map(|n| flash_acc_update(v, n, shape, tail))
        .collect();
    let writes: String = (0..nd).map(|n| flash_output(y, n, shape)).collect();
    fill(
        FLASH_KERNEL,
        &[
            ("q", q),
            ("k", k),
            ("v", v),
            ("y", y),
            ("seq", &s),
            ("dim", &d),
            ("threads", &threads),
            ("block", &block),
            ("nd", &nds),
            ("score", &flash_score_body(q, k, tail)),
            ("weight", flash_weight(tail)),
            ("decls", &decls),
            ("updates", &updates),
            ("writes", &writes),
        ],
    )
}

/// 一个寄存器输出维的回写：除以 `l` 后只写本线程负责的那一维。
fn flash_output(y: &str, n: usize, shape: Flash) -> String {
    if shape.dim.is_multiple_of(shape.threads) {
        return format!(
            "    {{ const uint h = t + {}; d_{y}[i * D + h] = acc_{n} / l; }}\n",
            n * shape.threads
        );
    }
    format!(
        "    {{ const uint h = t + {}; if (h < D) d_{y}[i * D + h] = acc_{n} / l; }}\n",
        n * shape.threads
    )
}

/// flash 内核模板：`@name` 占位由 [`crate::dialect::fill`] 替换。
const FLASH_KERNEL: &str = "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device const float* d_@q [[buffer(0)]],\n\
         \x20    device const float* d_@k [[buffer(1)]],\n\
         \x20    device const float* d_@v [[buffer(2)]],\n\
         \x20    device float* d_@y [[buffer(3)]],\n\
         \x20    uint3 gid [[thread_position_in_grid]],\n\
         \x20    uint3 lid [[thread_position_in_threadgroup]]) {\n\
         \x20    const uint S = @seq, D = @dim, T = @threads, BK = @block, ND = @nd;\n\
         \x20    threadgroup float scores[BK];\n\
         \x20    threadgroup float red[T];\n\
         \x20    threadgroup float red_sum[T];\n\
         \x20    const uint i = gid.x / T, t = lid.x;\n\
         \x20    const float scale = sqrt((float)D);\n\
         \x20    float m = -3.402823466e38f, l = 0.0f;\n\
         @decls\
         \x20    for (uint kb = 0; kb < S; kb += BK) {\n\
         \x20        float tmax = -3.402823466e38f;\n\
         \x20        for (uint j = t; j < BK; j += T) {\n\
         \x20            float sc = -3.402823466e38f;\n\
         @score\
         \x20            scores[j] = sc;\n\
         \x20            tmax = max(tmax, sc);\n\
         \x20        }\n\
         \x20        red[t] = tmax;\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint st = T / 2; st > 0; st >>= 1) {\n\
         \x20            if (t < st) red[t] = max(red[t], red[t + st]);\n\
         \x20            threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        }\n\
         \x20        const float mn = max(m, red[0]);\n\
         \x20        const float alpha = exp(m - mn);\n\
         \x20        float tsum = 0.0f;\n\
         \x20        for (uint j = t; j < BK; j += T) {\n\
         \x20            float e = @weight;\n\
         \x20            scores[j] = e;\n\
         \x20            tsum += e;\n\
         \x20        }\n\
         \x20        red_sum[t] = tsum;\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint st = T / 2; st > 0; st >>= 1) {\n\
         \x20            if (t < st) red_sum[t] += red_sum[t + st];\n\
         \x20            threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        }\n\
         \x20        l = l * alpha + red_sum[0];\n\
         @updates\
         \x20        m = mn;\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }\n\
         @writes\
         }";

/// 分数块的单键计算体：`S` 不是块大小整数倍时带尾块守卫。
fn flash_score_body(q: &str, k: &str, tail: bool) -> String {
    let dot = format!(
        "float dot = 0.0f;\n\
         \x20               for (uint p = 0; p < D; ++p) dot += d_{q}[i * D + p] * d_{k}[(kb + j) * D + p];\n\
         \x20               sc = dot / scale;\n"
    );
    if tail {
        format!(
            "            if (kb + j < S) {{\n\
             \x20               {dot}\
             \x20           }}\n"
        )
    } else {
        format!("            {dot}")
    }
}

/// 指数权重：无尾块守卫时省掉逐键分支。
fn flash_weight(tail: bool) -> &'static str {
    if tail {
        "(kb + j < S) ? exp(scores[j] - mn) : 0.0f"
    } else {
        "exp(scores[j] - mn)"
    }
}

/// 一个寄存器输出维 `acc_n` 的在线更新：先按 `alpha` 缩放旧累加，再加当前键块的 `P·V`。
fn flash_acc_update(v: &str, n: usize, shape: Flash, tail: bool) -> String {
    let h = n * shape.threads;
    let add = format!("a += scores[j] * d_{v}[(kb + j) * D + h];");
    let value = if tail {
        format!("if (kb + j < S) {add}")
    } else {
        add
    };
    if shape.dim.is_multiple_of(shape.threads) {
        return format!(
            "        {{ const uint h = t + {h};\n\
             \x20           float a = acc_{n} * alpha;\n\
             \x20           for (uint j = 0; j < BK; ++j) {{ {value} }}\n\
             \x20           acc_{n} = a; }}\n"
        );
    }
    format!(
        "        {{ const uint h = t + {h};\n\
         \x20           if (h < D) {{\n\
         \x20               float a = acc_{n} * alpha;\n\
         \x20               for (uint j = 0; j < BK; ++j) {{ {value} }}\n\
         \x20               acc_{n} = a;\n\
         \x20           }} }}\n"
    )
}

/// 生产态 threadgroup 协作分块注意力（Metal）：一个线程组负责一个查询行。
///
/// `threads` 个线程分摊 S 个分数与 D 个输出维；行内 max 与指数和走线程组树归约，完整分数块留在
/// threadgroup 内存，避免每个查询行被单线程串行扫完整条序列。语义与
/// `O = softmax(QKᵀ/√D)·V` 逐元素相同。前置条件由调用方保证（线程数为 `32..=1024` 内的 2 的幂、
/// `S·4` 字节的分数块不超过线程组内存上限、`D ≤ 64`）。
///
/// 派发几何：`grid = [S·threads]`、`block = [threads]`；每组的 `gid / threads` 是查询行。
///
/// # Errors
///
/// 签名不足 4 缓冲（q/k/v/y）或 2 整数标量（S/D）、样本缺标量、或线程数非法时返回说明。
pub(crate) fn metal_tiled_attention(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    threads: usize,
) -> Result<String> {
    if !threads.is_power_of_two() || threads < 2 {
        return Err(
            Error::new(Code::Invalid, "注意力线程数必须是 2 的幂且至少为 2")
                .with("engine::tiled::metal_tiled_attention"),
        );
    }
    let (q, k, v, y) = attention_bufs(signature)?;
    let (s, d) = attention_dims(signature, sample)?;
    Ok(tiled_attention_msl(q, k, v, y, s, d, threads))
}

/// 分块注意力的 MSL 正文：只拼字面与已解出的维度，不含错误处理。
fn tiled_attention_msl(
    q: &str,
    k: &str,
    v: &str,
    y: &str,
    s: usize,
    d: usize,
    threads: usize,
) -> String {
    format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device float* d_{q} [[buffer(0)]],\n\
         \x20    device float* d_{k} [[buffer(1)]],\n\
         \x20    device float* d_{v} [[buffer(2)]],\n\
         \x20    device float* d_{y} [[buffer(3)]],\n\
         \x20    uint gid [[thread_position_in_grid]],\n\
         \x20    uint lid [[thread_position_in_threadgroup]]) {{\n\
         \x20    const uint S = {s}, D = {d}, T = {threads};\n\
         \x20    threadgroup float scores[{s}];\n\
         \x20    threadgroup float red[{threads}];\n\
         \x20    threadgroup float stats[2];\n\
         \x20    const uint i = gid / T, t = lid;\n\
         \x20    const float scale = sqrt((float)D);\n\
         \x20    for (uint j = t; j < S; j += T) {{\n\
         \x20        float dot = 0.0f;\n\
         \x20        for (uint p = 0; p < D; ++p) dot += d_{q}[i * D + p] * d_{k}[j * D + p];\n\
         \x20        scores[j] = dot / scale;\n\
         \x20    }}\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    float lmax = -3.402823466e38f;\n\
         \x20    for (uint j = t; j < S; j += T) lmax = max(lmax, scores[j]);\n\
         \x20    red[t] = lmax;\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    for (uint step = T / 2; step > 0; step >>= 1) {{\n\
         \x20        if (t < step) red[t] = max(red[t], red[t + step]);\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    if (t == 0) stats[0] = red[0];\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    const float row_max = stats[0];\n\
         \x20    float lsum = 0.0f;\n\
         \x20    for (uint j = t; j < S; j += T) {{\n\
         \x20        float e = exp(scores[j] - row_max);\n\
         \x20        scores[j] = e;\n\
         \x20        lsum += e;\n\
         \x20    }}\n\
         \x20    red[t] = lsum;\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    for (uint step = T / 2; step > 0; step >>= 1) {{\n\
         \x20        if (t < step) red[t] += red[t + step];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    if (t == 0) stats[1] = red[0];\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    const float prob_sum = stats[1];\n\
         \x20    for (uint h = t; h < D; h += T) {{\n\
         \x20        float acc = 0.0f;\n\
         \x20        for (uint j = 0; j < S; ++j) acc += scores[j] * d_{v}[j * D + h];\n\
         \x20        d_{y}[i * D + h] = acc / prob_sum;\n\
         \x20    }}\n\
         }}"
    )
}
