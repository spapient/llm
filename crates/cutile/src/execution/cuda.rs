//! CUDA 内核源的生成：把算子的产物包成 `kernel_main`。
//!
//! 与执行通道（[`super`]）分开是因为两者变更理由不同：这里产出的是一段文本，
//! 那里决定这段文本交给谁编译、在哪里跑。改内核写法不需要动执行。

use super::*;

/// 生产态 CUDA 逐点网格内核源：走完整条流水线后把逐点内核包成按网格派发的 `kernel_main`。
/// 本入口只覆盖「纯逐点」签名（[`OpDecl::pointwise`]）；gemm/attention 的循环注入与串行兜底的
/// CUDA 镜像后续单独落地，不在此入口静默生成错的内核。
///
/// # Errors
///
/// `nvidia` 档案缺失、设备缺执行证据、协商失败、形状非法或算子非逐点形时返回说明。
pub fn cuda_grid_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let op = run.op;
    if !op.pointwise() {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "`{}` 不是纯逐点形，CUDA 逐点网格未覆盖（循环注入/串行兜底后续落地）",
                op.name
            ),
        )
        .with("cutile::execution::cuda_grid_program"));
    }
    let sample = (run.model.sample)(m, n);
    single(
        backend(Emit::Cuda)?.lower(
            &prepared.source.text,
            &op.signature,
            &sample,
            &Lower::Pointwise,
        )?,
        "cuda 逐点",
    )
}

/// 生产态 CUDA 分块归约内核源：走完整条流水线后把长轴切成块，每线程归约一块写部分结果 `y[g]`，
/// 主机折叠部分结果得最终标量。与 [`metal_grid_program`] 的归约骨架同语义，仅后端方言差异。
///
/// # Errors
///
/// 算子没声明 `[chunk]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_reduce_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    single(
        lower_sources(Emit::Cuda, &reduce_kind(run.op)?, &run, &prepared, m, n)?,
        "cuda 归约",
    )
}

/// 生产态 CUDA 分块 RMSNorm 内核源：两段（部分平方和 + 归一化），主机在两次派发间折叠出
/// `rms` 再跑第二段。与 [`metal_rmsnorm_program`] 同语义，仅后端方言差异。
///
/// # Errors
///
/// 算子没声明 `[chunk]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_rmsnorm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let chunk = chunk_size(run.op, "cutile::execution::cuda_rmsnorm_program")?;
    pair(
        lower_sources(
            Emit::Cuda,
            &Lower::ChunkRmsnorm { chunk },
            &run,
            &prepared,
            m,
            n,
        )?,
        "cuda rmsnorm",
    )
}

/// 生产态 CUDA 分块 Softmax 内核源：三段（部分最大 + 部分指数和 + 归一化），主机在派发间折叠
/// `max` 与 `sum`。与 [`metal_softmax_program`] 同语义，仅后端方言差异。
///
/// # Errors
///
/// 算子没声明 `[chunk]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_softmax_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String, String)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let chunk = chunk_size(run.op, "cutile::execution::cuda_softmax_program")?;
    triple(
        lower_sources(
            Emit::Cuda,
            &Lower::ChunkSoftmax { chunk },
            &run,
            &prepared,
            m,
            n,
        )?,
        "cuda softmax",
    )
}

/// 生产态 CUDA 循环注入网格内核（`Gemm` 骨架：gemv 一维 / gemm 二维共用一条 lowering）。
///
/// 返回 `(设备源, 各维网格大小)`：网格大小按 `[grid]` 声明的标量从样本取值（`dims[i]` 对应
/// `kernel_main<<<dim3(dims[0], dims[1], dims[2]), 1>>>` 的 x/y/z）；host 驱动由调用方用
/// `device::cuda::cuda_loop_driver` 拼——设备段与驱动分属两层，机制层不生成 host C 文本。
///
/// # Errors
///
/// 算子没声明 `[grid]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_gemm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, Vec<usize>)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let grid = grid_of(run.op, op_name)?;
    let dims = grid_dims(grid, &(run.model.sample)(m, n))?;
    Ok((cuda_loop_source(grid, &run, &prepared, m, n)?, dims))
}

/// 算子的 `[grid]` 声明：没有声明就没有网格，循环注入无从谈起。
fn grid_of<'a>(op: &'a OpDecl, op_name: &str) -> diagnostic::Result<&'a Grid> {
    op.grid.as_ref().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!("`{op_name}` 没有声明 `[grid]`，CUDA 循环注入不适用"),
        )
        .with("cutile::execution::cuda_gemm_program")
    })
}

/// CUDA 循环注入的设备源：`Loop` 降级后只取那一段源码（单文件即单翻译单元）。
fn cuda_loop_source(
    grid: &Grid,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    single(
        lower_sources(Emit::Cuda, &Lower::Loop(grid.clone()), run, prepared, m, n)?,
        "cuda gemm",
    )
}

/// 网格每一维的大小：把 `[grid]` 里的标量名按样本求值成 `usize`。
///
/// 抽出来的理由是可测性：`cuda_gemm_program` 本来就是编排（解析 → 求维 → 降级 → 取单源），
/// 让它在循环里再摊三层错误分支，读的人得同时跟两条线。
pub(super) fn grid_dims(grid: &Grid, sample: &Sample) -> diagnostic::Result<Vec<usize>> {
    grid.dims
        .iter()
        .map(|(_, scalar)| dim_size(scalar, sample))
        .collect()
}

/// 一个网格标量的大小：样本没有取值、或取值不是非负 `usize`，都算声明与样本对不上。
pub(super) fn dim_size(scalar: &str, sample: &Sample) -> diagnostic::Result<usize> {
    let size = sample.scalar(scalar).ok_or_else(|| {
        Error::new(Code::Invalid, format!("网格标量 `{scalar}` 没有样本取值"))
            .with("cutile::execution::cuda_gemm_program")
    })?;
    usize::try_from(size).map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("网格标量 `{scalar}` = {size} 不是非负 usize"),
        )
        .with("cutile::execution::cuda_gemm_program")
    })
}

/// 生产态 CUDA 张量核（WMMA）GEMM 内核源：`64×64` 输出块、`BK=8`、4 个 `32×32×8` 片段。
///
/// 形状在生成期写死（每形状一份产物）：`M`/`N` 整除 `64` 时无守卫，否则载入/写出带行列守卫。
/// host 驱动由调用方用 `device::wmma::gemm_driver` 生成；`K % 8 != 0` 时回落
/// [`cuda_gemm_program`] 的循环注入路径。
///
/// # Errors
///
/// 算子不是 `Gemm` 骨架、`nvidia` 档案缺失、设备缺执行证据、协商失败、形状非法或 `K % 8 != 0`。
pub fn cuda_wmma_gemm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    single(
        lower_sources(Emit::Cuda, &Lower::WmmaGemm, &run, &prepared, m, n)?,
        "cuda wmma gemm",
    )
}

/// 生产态 CUDA 分块扫描内核源：两段（局部扫描 + 校正），主机在两次派发间折叠携带前缀
/// `C_{g+1} = a^len·C_g + 尾态`。与 [`metal_scan_program`] 同语义，仅后端方言差异。
///
/// # Errors
///
/// 算子没声明 `[chunk]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_scan_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String, f32)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let decay = scan_decay(run.op)?;
    let (local, correct) = scan_pair(Emit::Cuda, &run, &prepared, m, n, "cuda scan")?;
    Ok((local, correct, decay))
}

/// 扫描的两段源码：块大小与衰减 → `Lower` → 降源 → `(局部扫描, 校正)`。
///
/// 衰减读两遍是有意的：它既要进 `Lower`（分块几何），又要单独回给主机折叠携带前缀。
fn scan_pair(
    emit: Emit,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
    role: &str,
) -> diagnostic::Result<(String, String)> {
    let kind = scan_kind(run.op)?;
    pair(lower_sources(emit, &kind, run, prepared, m, n)?, role)
}
