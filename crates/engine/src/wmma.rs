//! 张量核特化的 f32 GEMM（WMMA）CUDA lowering。
//!
//! 每个 CTA 负责 `64×64` 的输出块：`BK=8` 一层、4 个 `32×32×8` 累积片段（`i`×`j` 四块）、
//! 每层 4 次 `mma_sync`，线程固定 128（S4000 的 QY2 是 wave128 共持片段）。
//!
//! 布局与 S4000 实测口径一致：A 片按 `as[row][k]`（行主、k 连续）配 `row_major`/`ldm=BK`；
//! B 片转置存 `bs[n][k]` 配 `col_major`/`ldm=BK`。
//!
//! **形状在生成期已知**：`M`/`N`/`K` 取自签名标量的样本值，直接写进生成文本。`M`/`N` 是 64 的
//! 倍数时载入与写出都不带守卫（编译器把分支整体消掉）；否则载入按行/列守卫、写出先落共享内存
//! 再带守卫拷回（`store_matrix_sync` 会写满整块 32×32，不能直接落在非对齐输出上）。
//! `K % 8 != 0` 报 `Unsupported`，由调用方回落到循环注入的通用路径。

use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::Signature;

use cutile::harness::output_name;

use crate::codegen::{cuda_host_buffers_grid, cuda_readback};

/// 输出块边长。
const TILE: usize = 64;
/// 单个累积片段边长。
const FRAG: usize = 32;
/// 收缩轴每层宽度。
const BK: usize = 8;
/// CTA 线程数（wave128）。
const THREADS: usize = 128;
/// 每边片段数（2×2）。
const PER_SIDE: usize = TILE / FRAG;

/// WMMA GEMM 的设备源（`kernel_main` 即入口，网格 `ceil(M/64) × ceil(N/64)`、128 线程）。
///
/// # Errors
///
/// 签名不是「两个只读 f32 缓冲 + 一个输出 + 三个标量」、标量缺样本值、或 `K % 8 != 0` 时返回说明。
pub fn gemm_kernel(signature: &Signature, sample: &Sample) -> Result<String> {
    let (a, b) = input_pair(signature)?;
    let y = output_name(signature)?;
    let (m, n, k) = shape(signature, sample)?;
    if k == 0 || k % BK != 0 {
        return Err(Error::new(
            Code::Unsupported,
            format!("WMMA GEMM 要求 K 是 {BK} 的正整数倍，得到 {k}"),
        )
        .with("engine::wmma::gemm_kernel"));
    }
    let exact = m % TILE == 0 && n % TILE == 0;
    let mut text = String::from(
        "#define __MUSA_QY2_MMA__ 1\n#include <musa_runtime.h>\n#include <cuda_runtime.h>\n#include <mma.h>\n\nusing namespace mtmusa::wmma;\n",
    );
    text.push_str(&format!(
        "\n// 形状特化：M={m} N={n} K={k}，网格 {}×{}，128 线程\n",
        m.div_ceil(TILE),
        n.div_ceil(TILE)
    ));
    text.push_str(&format!(
        "__global__ void __launch_bounds__({THREADS}, 8) kernel_main(const float* __restrict__ d_{a}, const float* __restrict__ d_{b}, float* __restrict__ d_{y}) {{\n"
    ));
    text.push_str(&shared_decls(exact));
    text.push_str(&accumulators());
    text.push_str(&main_loop(a, b, m, n, k, exact));
    text.push_str(&write_out(y, m, n, exact));
    text.push_str("}\n");
    Ok(text)
}

/// 共享内存：两个输入片；非整除时加输出暂存（4 块 32×32）。
fn shared_decls(exact: bool) -> String {
    let mut text =
        format!("    __shared__ float as[{TILE}][{BK}];\n    __shared__ float bs[{TILE}][{BK}];\n");
    if !exact {
        text.push_str(&format!(
            "    __shared__ float cs[{}][{FRAG}][{FRAG}];\n",
            PER_SIDE * PER_SIDE
        ));
    }
    text
}

/// 4 个累积片段并清零。
fn accumulators() -> String {
    let count = PER_SIDE * PER_SIDE;
    format!(
        "    fragment<accumulator, 32, 32, 8, float> acc[{count}];\n    #pragma unroll\n    for (int u = 0; u < {count}; ++u) fill_fragment(acc[u], 0.0f);\n"
    )
}

/// 主循环：载入一片 A/B → 同步 → 4 次 `mma_sync` → 同步。
fn main_loop(a: &str, b: &str, m: usize, n: usize, k: usize, exact: bool) -> String {
    format!(
        "    for (int k0 = 0; k0 < {k}; k0 += {BK}) {{\n{}{}        __syncthreads();\n{}        __syncthreads();\n    }}\n",
        load_a(a, m, k, exact),
        load_b(b, n, exact),
        mma(),
    )
}

/// A 片载入：每线程一个 `float4`（`64×8` 元素 = 128 个），非整除时按行守卫。
fn load_a(a: &str, m: usize, k: usize, exact: bool) -> String {
    let base = format!("((int)blockIdx.y * {TILE} + row) * {k} + k0 + v * 4");
    let fetch = if exact {
        format!("*(const float4*)(d_{a} + {base})")
    } else {
        format!(
            "(((int)blockIdx.y * {TILE} + row) < {m}) ? *(const float4*)(d_{a} + {base}) : make_float4(0.0f, 0.0f, 0.0f, 0.0f)"
        )
    };
    format!(
        "        {{ int row = threadIdx.x >> 1, v = threadIdx.x & 1; float4 val = {fetch};\n\
         \x20         as[row][v*4+0] = val.x; as[row][v*4+1] = val.y; as[row][v*4+2] = val.z; as[row][v*4+3] = val.w; }}\n"
    )
}

/// B 片载入：转置存 `bs[n][k]`；整除时 `float4` 读，否则逐元素按列守卫。
fn load_b(b: &str, n: usize, exact: bool) -> String {
    let body = if exact {
        format!(
            "float4 val = *(const float4*)(d_{b} + (k0 + kk) * {n} + (int)blockIdx.x * {TILE} + n0);\n\
             \x20           bs[n0+0][kk] = val.x; bs[n0+1][kk] = val.y; bs[n0+2][kk] = val.z; bs[n0+3][kk] = val.w;"
        )
    } else {
        format!(
            "#pragma unroll\n            for (int t = 0; t < 4; ++t) {{\n\
             \x20               int col = (int)blockIdx.x * {TILE} + n0 + t;\n\
             \x20               bs[n0+t][kk] = (col < {n}) ? d_{b}[(k0 + kk) * {n} + col] : 0.0f;\n            }}"
        )
    };
    format!(
        "        {{ int kk = threadIdx.x >> 4, n0 = (threadIdx.x & 15) * 4;\n          {body}\n        }}\n"
    )
}

/// 4 次 `mma_sync`：`acc[i*2+j] += fa[i] × fb[j]`。
fn mma() -> String {
    format!(
        "        fragment<matrix_a, 32, 32, 8, precision::tf32, row_major> fa[{PER_SIDE}];\n\
         \x20       fragment<matrix_b, 32, 32, 8, precision::tf32, col_major> fb[{PER_SIDE}];\n\
         \x20       #pragma unroll\n        for (int i = 0; i < {PER_SIDE}; ++i) load_matrix_sync(fa[i], &as[i * {FRAG}][0], {BK});\n\
         \x20       #pragma unroll\n        for (int j = 0; j < {PER_SIDE}; ++j) load_matrix_sync(fb[j], &bs[j * {FRAG}][0], {BK});\n\
         \x20       #pragma unroll\n        for (int i = 0; i < {PER_SIDE}; ++i)\n\
         \x20           #pragma unroll\n            for (int j = 0; j < {PER_SIDE}; ++j)\n\
         \x20               mma_sync(acc[i * {PER_SIDE} + j], fa[i], fb[j], acc[i * {PER_SIDE} + j]);\n"
    )
}

/// 写出：整除时直接 `store_matrix_sync`；否则先落共享内存，再按行列守卫拷回。
fn write_out(y: &str, m: usize, n: usize, exact: bool) -> String {
    let target = if exact {
        format!(
            "d_{y} + ((int)blockIdx.y * {TILE} + i * {FRAG}) * {n} + (int)blockIdx.x * {TILE} + j * {FRAG}"
        )
    } else {
        format!("&cs[i * {PER_SIDE} + j][0][0]")
    };
    let ldm = if exact {
        n.to_string()
    } else {
        FRAG.to_string()
    };
    let stores = format!(
        "    #pragma unroll\n    for (int i = 0; i < {PER_SIDE}; ++i)\n        #pragma unroll\n        for (int j = 0; j < {PER_SIDE}; ++j)\n            store_matrix_sync({target}, acc[i * {PER_SIDE} + j], {ldm}, mem_row_major);\n"
    );
    if exact {
        return stores;
    }
    stores
        + &format!(
            "    __syncthreads();\n    for (int idx = threadIdx.x; idx < {}; idx += {THREADS}) {{\n\
             \x20       int blk = idx / {block};\n\
             \x20       int r = (idx / {FRAG}) % {FRAG}, c = idx % {FRAG};\n\
             \x20       int gr = (int)blockIdx.y * {TILE} + (blk / {PER_SIDE}) * {FRAG} + r;\n\
             \x20       int gc = (int)blockIdx.x * {TILE} + (blk % {PER_SIDE}) * {FRAG} + c;\n\
             \x20       if (gr < {m} && gc < {n}) d_{y}[gr * {n} + gc] = cs[blk][r][c];\n    }}\n",
            PER_SIDE * PER_SIDE * FRAG * FRAG,
            block = FRAG * FRAG,
        )
}

/// 两个只读输入缓冲名（A、B）。
fn input_pair(signature: &Signature) -> Result<(&str, &str)> {
    let reads: Vec<&str> = signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer() && !param.mutable)
        .map(|param| param.name.as_str())
        .collect();
    match reads.as_slice() {
        [a, b] => Ok((a, b)),
        _ => Err(Error::new(
            Code::Invalid,
            format!("WMMA GEMM 需要两个只读缓冲（A, B），得到 {}", reads.len()),
        )
        .with("engine::wmma::input_pair")),
    }
}

/// 形状 `(M, N, K)`：三个标量的样本值（`gemm` 签名的 `M`/`N`/`K`）。
fn shape(signature: &Signature, sample: &Sample) -> Result<(usize, usize, usize)> {
    let mut values = Vec::with_capacity(3);
    for param in signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buffer())
    {
        let value = sample.scalar(&param.name).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("WMMA GEMM 的标量 `{}` 没有样本取值", param.name),
            )
            .with("engine::wmma::shape")
        })?;
        values.push(usize::try_from(value).map_err(|_| {
            Error::new(
                Code::Invalid,
                format!("WMMA GEMM 的标量 `{}` = {value} 不是非负 usize", param.name),
            )
            .with("engine::wmma::shape")
        })?);
    }
    match values.as_slice() {
        [m, n, k] => Ok((*m, *n, *k)),
        _ => Err(Error::new(
            Code::Invalid,
            format!("WMMA GEMM 需要三个标量（M, N, K），得到 {}", values.len()),
        )
        .with("engine::wmma::shape")),
    }
}

/// WMMA GEMM 的 host 驱动：`dim3(ceil(M/64), ceil(N/64))`、128 线程，分配/回读与
/// [`cuda_grid_driver`](crate::codegen::cuda_grid_driver) 同口径（输出长度取样本值）。
///
/// # Errors
///
/// 缺输出或形状标量时返回说明。
pub fn gemm_driver(device_source: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let y = output_name(signature)?;
    let (m, n, _) = shape(signature, sample)?;
    let (output, values) = cutile::harness::output_of(signature, sample)?;
    debug_assert_eq!(output, y);
    let out_len = values.len();
    let mut text = String::from("#include <cstdio>\n");
    text.push_str(device_source);
    if !device_source.ends_with('\n') {
        text.push('\n');
    }
    text.push_str("\nint main(void) {\n");
    text.push_str(&cuda_host_buffers_grid(
        sample,
        signature.params(),
        output,
        out_len,
    )?);
    text.push_str(&format!(
        "    kernel_main<<<dim3({}, {}), {THREADS}>>>({});\n    cudaError_t error = cudaDeviceSynchronize();\n    if (error != cudaSuccess) {{ fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; }}\n",
        n.div_ceil(TILE),
        m.div_ceil(TILE),
        buffer_args(signature),
    ));
    text.push_str(&cuda_readback(output, out_len));
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
    {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// 启动实参：只传缓冲（形状是生成期常量，不进 kernel 形参）。
fn buffer_args(signature: &Signature) -> String {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .map(|param| format!("d_{}", param.name))
        .collect::<Vec<_>>()
        .join(", ")
}
