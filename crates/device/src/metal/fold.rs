//! Metal 多段并行程序：分块族（归约 / 归一化 / 扫描）的「多段发射 + 主机折叠」。

use crate::dialect::{
    EXPSUM, NORM, PARTIAL, RMS_EPS, SCAN_CORRECT, SCAN_LOCAL, carry_fold, length_scalar,
};
use crate::metal::{
    MetalStage, metal_chunk_reduce, metal_chunk_rmsnorm, metal_chunk_scan, metal_chunk_softmax,
    metal_multistage_driver,
};
use cutile::harness::output_name;
use diagnostic::{Code, Error, Result};
use llm_optim::backend::Source;
use nn::model::Sample;
use nn::sig::Signature;

/// 折叠的一步：与 `.k` 模板的 `reduce.op` 同口径，`sum` / `max` 两个闭集表达。
const SUM_STEP: &str = "acc += parts[g];";
const MAX_STEP: &str = "if (parts[g] > acc) { acc = parts[g]; }";

/// 一段程序的公共几何：长度、块数、输出缓冲名、全部缓冲名。
struct Prep<'a> {
    len: usize,
    blocks: usize,
    output: &'a str,
    names: Vec<&'a str>,
}

impl<'a> Prep<'a> {
    fn of(signature: &'a Signature, sample: &Sample, chunk: usize) -> Result<Self> {
        let len = length(signature, sample)?;
        Ok(Self {
            len,
            blocks: len.div_ceil(chunk.max(1)),
            output: output_name(signature)?,
            names: buf_names(signature),
        })
    }

    /// 输出缓冲的容量：它同时当「部分结果数组」与「整行结果」用，取大者开满，否则局部扫描 /
    /// 归一化写回整行时越界（Metal 不做边界检查，错的是内存不是报告）。
    fn span(&self) -> usize {
        self.blocks.max(self.len)
    }
}

/// 分块归约程序：一次部分归约段 + 主机折叠（`sum` / `max` 由 `reduce.op` 派生）。
///
/// # Errors
///
/// 长度标量缺失、签名与样本不一致，或设备段生成失败。
pub fn reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    func: &str,
) -> Result<Source> {
    let prep = Prep::of(signature, sample, chunk)?;
    let (device, host) = reduce_parts(kernel_source, signature, sample, chunk, func, &prep)?;
    Ok(Source::Pair { device, host })
}

/// 归约的段与主机折叠：`sum` / `max` 取哪一步由 `func` 派生，折叠本身只有一份。
fn reduce_parts(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    func: &str,
    prep: &Prep<'_>,
) -> Result<(String, String)> {
    let device = metal_chunk_reduce(kernel_source, signature, sample, chunk, func)?;
    let step = if func.contains("max") {
        MAX_STEP
    } else {
        SUM_STEP
    };
    let fold = fold_parts(prep.output, prep.blocks, step);
    let host = metal_multistage_driver(
        signature,
        sample,
        prep.span(),
        1,
        &[],
        &[MetalStage {
            entry: "kernel_main",
            bufs: &prep.names,
            grid: [prep.blocks, 1, 1],
            fold: &fold,
        }],
    )?;
    Ok((device, host))
}

/// 分块 RMSNorm 程序：部分平方和段 → `rms` 折叠 → 归一化段。
///
/// # Errors
///
/// 长度标量缺失、签名不是「两个只读缓冲 + 输出 + 标量」的归一化形态，或设备段生成失败。
pub fn rmsnorm(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<Source> {
    let prep = Prep::of(signature, sample, chunk)?;
    let (device, host) = rmsnorm_parts(kernel_source, signature, sample, chunk, &prep)?;
    Ok(Source::Pair { device, host })
}

/// RMSNorm 的段与主机折叠：平方和折成 `rms` 只走一份算式（[`rms_fold`]）。
fn rmsnorm_parts(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    prep: &Prep<'_>,
) -> Result<(String, String)> {
    let (partial, norm) = metal_chunk_rmsnorm(kernel_source, signature, sample, chunk)?;
    let device = concat(&partial, &[(norm.as_str(), NORM)])?;
    let (x, gamma) = normalization_reads(signature)?;
    let fold = rms_fold(prep.output, prep.blocks, prep.len);
    let map = ["rms", x, gamma, prep.output];
    let host = metal_multistage_driver(
        signature,
        sample,
        prep.span(),
        prep.len,
        &[("rms", 1)],
        &[
            MetalStage {
                entry: PARTIAL,
                bufs: &prep.names,
                grid: [prep.blocks, 1, 1],
                fold: &fold,
            },
            MetalStage {
                entry: NORM,
                bufs: &map,
                grid: [prep.blocks, 1, 1],
                fold: "",
            },
        ],
    )?;
    Ok((device, host))
}

/// 分块 Softmax 程序：部分最大段 → `max` 折叠 → 部分指数和段 → `sum` 折叠 → 归一化段。
///
/// # Errors
///
/// 长度标量缺失、签名不是「一个只读缓冲 + 输出 + 标量」的 softmax 形态，或设备段生成失败。
pub fn softmax(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<Source> {
    let prep = Prep::of(signature, sample, chunk)?;
    let (device, host) = softmax_parts(kernel_source, signature, sample, chunk, &prep)?;
    Ok(Source::Pair { device, host })
}

/// Softmax 的段与主机折叠：`max` 与 `sum` 各落一个单独缓冲，留给下一段当入口参数。
fn softmax_parts(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    prep: &Prep<'_>,
) -> Result<(String, String)> {
    let (max, expsum, norm) = metal_chunk_softmax(kernel_source, signature, sample, chunk)?;
    let device = concat(&max, &[(expsum.as_str(), EXPSUM), (norm.as_str(), NORM)])?;
    let x = only_read(signature)?;
    let parts = format!("b_{}", prep.output);
    let fold_max = fold_into(&parts, "b_max", prep.blocks, MAX_STEP);
    let fold_sum = fold_into(&parts, "b_sum", prep.blocks, SUM_STEP);
    let exp_bufs = ["max", x, prep.output];
    let norm_bufs = ["max", "sum", x, prep.output];
    let host = metal_multistage_driver(
        signature,
        sample,
        prep.span(),
        prep.len,
        &[("max", 1), ("sum", 1)],
        &[
            MetalStage {
                entry: PARTIAL,
                bufs: &prep.names,
                grid: [prep.blocks, 1, 1],
                fold: &fold_max,
            },
            MetalStage {
                entry: EXPSUM,
                bufs: &exp_bufs,
                grid: [prep.blocks, 1, 1],
                fold: &fold_sum,
            },
            MetalStage {
                entry: NORM,
                bufs: &norm_bufs,
                grid: [prep.blocks, 1, 1],
                fold: "",
            },
        ],
    )?;
    Ok((device, host))
}

/// 分块扫描程序：局部扫描段 → 携带前缀折叠（f64）→ 校正段。
///
/// # Errors
///
/// 长度标量缺失、模板里找不到以长度标量为上界的循环，或设备段生成失败。
pub fn scan(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
) -> Result<Source> {
    let prep = Prep::of(signature, sample, chunk)?;
    let (device, host) = scan_parts(kernel_source, signature, sample, chunk, decay, &prep)?;
    Ok(Source::Pair { device, host })
}

/// 扫描的段与携带前缀折叠：`a^len` 用平方乘，块长最多 `chunk` 次乘法。
fn scan_parts(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
    prep: &Prep<'_>,
) -> Result<(String, String)> {
    let (local, correct) = metal_chunk_scan(kernel_source, signature, sample, chunk, decay)?;
    let device = concat(&local, &[(correct.as_str(), SCAN_CORRECT)])?;
    let fold = fold_carry(prep.output, chunk, prep.len, decay);
    let correct_bufs = ["c", prep.output];
    let host = metal_multistage_driver(
        signature,
        sample,
        prep.span(),
        prep.len,
        &[("c", prep.blocks)],
        &[
            MetalStage {
                entry: SCAN_LOCAL,
                bufs: &prep.names,
                grid: [prep.blocks, 1, 1],
                fold: &fold,
            },
            MetalStage {
                entry: SCAN_CORRECT,
                bufs: &correct_bufs,
                grid: [prep.blocks, 1, 1],
                fold: "",
            },
        ],
    )?;
    Ok((device, host))
}

/// 长度标量（`N` / `n` / `len`）的样本取值。
fn length(signature: &Signature, sample: &Sample) -> Result<usize> {
    usize::try_from(length_scalar(signature, sample)?).map_err(|_| {
        Error::new(Code::Invalid, "长度标量不是非负整数".to_owned())
            .with("engine::metal::fold::length")
    })
}

/// 签名里全部缓冲的名字（按 `buf(i)` 顺序）。
fn buf_names(signature: &Signature) -> Vec<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
        .map(|param| param.name.as_str())
        .collect()
}

/// 签名里的只读缓冲名字（输入），顺序与设备段的 `d_*` 一致。
fn readonly_names(signature: &Signature) -> Vec<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf() && !param.mutable)
        .map(|param| param.name.as_str())
        .collect()
}

/// 归一化的两个只读缓冲（`x` 与 `gamma`）：少一个就做不了 RMSNorm，在入口拒绝。
fn normalization_reads(signature: &Signature) -> Result<(&str, &str)> {
    let reads = readonly_names(signature);
    reads
        .first()
        .zip(reads.get(1))
        .map(|(first, second)| (*first, *second))
        .ok_or_else(|| {
            Error::new(
                Code::Invalid,
                "RMSNorm 需要两个只读缓冲（x, gamma）".to_owned(),
            )
            .with("engine::metal::fold::rmsnorm")
        })
}

/// Softmax 的唯一只读缓冲（`x`）：分布定义在整行上，多给一个就说不清算谁。
fn only_read(signature: &Signature) -> Result<&str> {
    readonly_names(signature).first().copied().ok_or_else(|| {
        Error::new(Code::Invalid, "Softmax 需要一个只读缓冲（x）".to_owned())
            .with("engine::metal::fold::softmax")
    })
}

/// `sqrt(Σx²/N + ε)` 折叠：f64 求和、写回前收窄 f32（与 `.k` 模板同口径）。
fn rms_fold(output: &str, blocks: usize, len: usize) -> String {
    format!(
        "        {{\n            const float *parts = (const float *)b_{output}.contents;\n            double acc = 0.0;\n            for (int g = 0; g < {blocks}; ++g) {{ acc += (double)parts[g]; }}\n            ((float *)b_rms.contents)[0] = (float)sqrt(acc / (double){len} + (double){eps}f);\n        }}\n",
        eps = RMS_EPS
    )
}

/// 多段设备源拼接：第一段带整份 preamble，后续段只取自己的 `kernel void <entry>` 之后——
/// 同一翻译单元里 preamble 出现一次即可，重复会撞重定义。
///
/// # Errors
///
/// 生成的段里找不到该入口函数（生成器与入口名漂移）时返回 `Internal`。
fn concat(first: &str, more: &[(&str, &str)]) -> Result<String> {
    let mut text = first.to_string();
    for (segment, entry) in more {
        let marker = format!("kernel void {entry}(");
        let at = segment.find(&marker).ok_or_else(|| {
            Error::new(
                Code::Internal,
                format!("段 `{entry}` 不在生成的设备源里（漂移）"),
            )
            .with("engine::metal::fold::concat")
        })?;
        text.push_str("\n\n");
        text.push_str(&segment[at..]);
    }
    Ok(text)
}

/// 部分结果折叠：把 `parts[0..blocks]` 折成一个值写回 `parts[0]`。
fn fold_parts(output: &str, blocks: usize, combine: &str) -> String {
    format!(
        "        {{\n            float *parts = (float *)b_{output}.contents;\n            float acc = parts[0];\n            for (int g = 1; g < {blocks}; ++g) {{ {combine} }}\n            parts[0] = acc;\n        }}\n"
    )
}

/// 部分结果折叠到另一个缓冲：Softmax 的 `max` / `sum` 要留给下一段当独立入口参数。
fn fold_into(parts: &str, target: &str, blocks: usize, step: &str) -> String {
    format!(
        "        {{\n            const float *parts = (const float *)({parts}).contents;\n            float acc = parts[0];\n            for (int g = 1; g < {blocks}; ++g) {{ {step} }}\n            ((float *)({target}).contents)[0] = acc;\n        }}\n"
    )
}

/// 携带前缀折叠（f64）：`C_0 = 0`、`C_{g+1} = a^len·C_g + y[start+len-1]`，`a^len` 用平方乘。
fn fold_carry(output: &str, chunk: usize, len: usize, decay: f32) -> String {
    let blocks = len.div_ceil(chunk.max(1));
    format!(
        "        {{\n            const float *y = (const float *)b_{output}.contents;\n            float *c = (float *)b_c.contents;\n{fold}        }}\n",
        fold = carry_fold(
            "y",
            "c",
            &blocks.to_string(),
            chunk,
            len as i64,
            decay,
            "            "
        ),
    )
}
