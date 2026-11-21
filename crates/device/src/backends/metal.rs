//! Metal 后端：`Codegen` 出 MSL、`Driver` 跑 `xcrun metal` + `metallib`。

use crate::metal::host::run_metal;
pub use crate::metal::host::{compile_run_metal, metal_driver, metal_kernel};
use crate::toolchain::detect_metal;

use super::*;

/// MSL 的形参必须带地址空间：MSL 没有默认地址空间，裸 `float *` 在那里是语法错误。
const MSL_PARAMS: &str = "thread float *";

fn metal_bench_geometry(
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<([usize; 3], [usize; 3])> {
    match kind {
        Lower::Pointwise => Ok((
            [benchmark_output_len(signature, sample)?, 1, 1],
            [256, 1, 1],
        )),
        Lower::Loop(grid) => Ok((benchmark_grid(grid, sample)?, [1, 1, 1])),
        _ => Err(Error::new(
            Code::Unsupported,
            "该 lowering 尚无单 kernel 设备计时封装",
        )),
    }
}

pub(crate) struct MetalBackend;

impl Codegen for MetalBackend {
    fn emit(&self) -> Emit {
        Emit::Metal
    }
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let _ = scope; // Metal 只走设备执行，host 自检由 C 后端承担
        metal_program(profile, dirs, selected)
    }
    fn lower(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
    ) -> Result<Vec<String>> {
        metal_lower(kernel, signature, sample, kind)
    }
    fn harness(&self, kernel: &str, signature: &Signature, sample: &Sample) -> Result<Source> {
        let (device, host) = crate::metal::metal_harness(kernel, signature, sample)?;
        Ok(Source::Pair { device, host })
    }
    fn parallel(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        launch: &Launch,
    ) -> Result<Source> {
        crate::program::metal(kernel, signature, sample, launch)
    }
    fn benchmark_harness(
        &self,
        _profile: &Profile,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
        repeat: usize,
    ) -> Result<Source> {
        let lowered = metal_lower(kernel, signature, sample, kind)?;
        let [device] = lowered.as_slice() else {
            return Err(Error::new(
                Code::Unsupported,
                "Metal 基准当前只支持单段生产态 kernel",
            ));
        };
        let (grid, threads) = metal_bench_geometry(signature, sample, kind)?;
        let host = crate::metal::metal_benchmark_driver(signature, sample, grid, threads, repeat)?;
        Ok(Source::Pair {
            device: device.clone(),
            host,
        })
    }
}

impl Driver for MetalBackend {
    fn available(&self, _: &Profile) -> bool {
        detect_metal().is_some()
    }
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        run_metal(program, count, &profile.flags, dirs)
    }
    fn toolchain(&self) -> Option<Toolchain> {
        detect_metal()
    }
    fn compile_run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        source: &Source,
        label: &str,
    ) -> Result<(String, String)> {
        match source {
            Source::Pair { device, host } => compile_run_metal(profile, dirs, device, host, label),
            Source::Single(_) => Err(Error::new(
                Code::Unsupported,
                "Metal 后端需要双源（MSL + host 驱动）".to_string(),
            )),
        }
    }
}

impl Backend for MetalBackend {}

/// 把中性的 [`Lower`] 描述降成 MSL：逐点/循环走网格或串行，分块族（归约/扫描/归一化）走分块内核。
fn metal_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        Pointwise => crate::metal::metal_grid_kernel(kernel, signature, sample).map(|s| vec![s]),
        QuantizedGemvI4 { .. } => Err(Error::new(
            Code::Unsupported,
            "合作式 int4 GEMV 当前仅落地在 CUDA/MUSA",
        )),
        Loop(grid) => {
            crate::metal::metal_loop_kernel(kernel, signature, sample, grid).map(|s| vec![s])
        }
        ChunkReduce { .. } | ChunkScan { .. } | ChunkRmsnorm { .. } | ChunkSoftmax { .. } => {
            metal_chunk_lower(kernel, signature, sample, kind)
        }
        TiledGemm { .. }
        | SimdGroupGemm { .. }
        | TensorGemm { .. }
        | Int8Gemm { .. }
        | TiledAttention { .. }
        | FlashAttention { .. }
        | WmmaGemm => metal_special(kernel, signature, sample, kind),
    }
}

/// 特殊 lowering 家族（分块 / simdgroup GEMM、分块 / flash 注意力、WMMA）：单独一个 match 让 [`metal_lower`] 的
/// 骨架分派保持在门禁臂数内。
fn metal_special(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        TiledGemm { .. }
        | SimdGroupGemm { .. }
        | TensorGemm { .. }
        | Int8Gemm { .. }
        | WmmaGemm => metal_gemm_lower(kernel, signature, sample, kind),
        TiledAttention { .. } | FlashAttention { .. } => {
            metal_attention_lower(kernel, signature, sample, kind)
        }
        _ => Err(Error::new(
            Code::Internal,
            "非特殊 lowering 不应进入 metal_special".to_string(),
        )),
    }
}

/// GEMM 家族（共享内存分块 / simdgroup 矩阵指令 / WMMA 张量核）的 Metal 落地。
fn metal_gemm_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        TiledGemm { tile, per_thread } => {
            crate::tiled::metal_tiled_gemm(kernel, signature, sample, *tile, *per_thread)
                .map(|s| vec![s])
        }
        SimdGroupGemm {
            row_frags,
            col_frags,
        } => crate::tiled::metal_simdgroup_gemm(kernel, signature, sample, *row_frags, *col_frags)
            .map(|s| vec![s]),
        TensorGemm { geometry } => {
            crate::tiled::metal_tensor_gemm(kernel, signature, sample, *geometry).map(|s| vec![s])
        }
        Int8Gemm { tile } => {
            crate::tiled::metal_int8_gemm(kernel, signature, sample, *tile).map(|s| vec![s])
        }
        other => Err(gemm_unsupported(other)),
    }
}

/// GEMM lowering 不在 Metal 生产态内的错误：WMMA 属未覆盖，其余是调用方走错入口。
fn gemm_unsupported(kind: &Lower) -> Error {
    match kind {
        WmmaGemm => Error::new(
            Code::Unsupported,
            "Metal 生产态 lowering 未覆盖 WMMA 张量核特化".to_string(),
        ),
        _ => Error::new(
            Code::Internal,
            "非 GEMM lowering 不应进入 metal_gemm_lower".to_string(),
        ),
    }
}

/// 注意力家族（共享内存分块 / flash 在线 softmax）的 Metal 落地。
fn metal_attention_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        TiledAttention { threads } => {
            crate::tiled::metal_tiled_attention(kernel, signature, sample, *threads)
                .map(|s| vec![s])
        }
        FlashAttention { threads, block } => {
            crate::tiled::metal_flash_attention(kernel, signature, sample, *threads, *block)
                .map(|s| vec![s])
        }
        _ => Err(Error::new(
            Code::Internal,
            "非注意力 lowering 不应进入 metal_attention_lower".to_string(),
        )),
    }
}

/// 分块归约/扫描/逐行规约的 Metal lowering：4 种分块骨架各自降成 1–3 段设备源。
/// 只应由 [`metal_lower`] 在已确认分块后调用；非分块算子返回内部错误而非 panic。
fn metal_chunk_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        ChunkReduce { chunk, func } => {
            crate::metal::metal_chunk_reduce(kernel, signature, sample, *chunk, func)
                .map(|s| vec![s])
        }
        ChunkScan { chunk, decay } => {
            crate::metal::metal_chunk_scan(kernel, signature, sample, *chunk, *decay)
                .map(|(local, correct)| vec![local, correct])
        }
        ChunkRmsnorm { chunk } => {
            crate::metal::metal_chunk_rmsnorm(kernel, signature, sample, *chunk)
                .map(|(partial, map)| vec![partial, map])
        }
        ChunkSoftmax { chunk } => {
            crate::metal::metal_chunk_softmax(kernel, signature, sample, *chunk)
                .map(|(max, expsum, map)| vec![max, expsum, map])
        }
        Pointwise
        | Loop(_)
        | QuantizedGemvI4 { .. }
        | TiledGemm { .. }
        | SimdGroupGemm { .. }
        | TensorGemm { .. }
        | Int8Gemm { .. }
        | TiledAttention { .. }
        | FlashAttention { .. }
        | WmmaGemm => Err(Error::new(
            Code::Internal,
            "非分块算子不应进入 Metal 分块 lowering",
        )),
    }
}

fn metal_program(profile: &Profile, dirs: &Dirs, selected: &[Selected<'_>]) -> Result<String> {
    let mut program = metal_head(profile);
    driver::push_sources(
        &mut program,
        &driver::witness_files(selected, dirs, MSL_PARAMS)?,
    );
    push_metal_kernel(&mut program, MAX_OUT, selected);
    Ok(program)
}

fn metal_head(profile: &Profile) -> String {
    let mut head = profile.preamble.clone();
    driver::normalize_newline(&mut head);
    head
}

fn push_metal_kernel(program: &mut String, max: usize, selected: &[Selected<'_>]) {
    program.push_str(&format!(
        "kernel void cutile_witness(device float *out [[buffer(0)]],\n\
         \x20                          device int *counts [[buffer(1)]],\n\
         \x20                          uint tid [[thread_position_in_grid]]) {{\n\
         \x20   thread float local[{max}];\n"
    ));
    driver::push_dispatch(program, max, selected, "}\n");
}

// Metal 主机侧胶水在 [`crate::metal::host`]：设备源 → `metallib` → host 驱动 → 回读。
// 与 CUDA 那条链并列留在本模块会把单文件撑过行数上限，两套厂商工具链之间也没有共享状态，
// 因此单独成模块；对外入口仍从这里 re-export，原路径不变。
