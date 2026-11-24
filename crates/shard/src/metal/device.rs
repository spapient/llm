//! Metal 设备源单段族：语义门 harness、逐点网格、循环注入网格、`cutile_launch` 内核装配。

use cutile::harness::output_of;
use diagnostic::Result;
use prof::decl::Grid;
use prof::model::Sample;
use prof::sig::Signature;

use crate::dialect::{Dialect, Metal, launch_args, length_scalar};
use crate::metal::host::metal_harness_driver;
/// 拼出可在本机 GPU 上真实执行的 Metal 程序：`.k` 内核补 `device` 地址空间后，`cutile_launch`
/// 把它包成 compute kernel（`device` 缓冲 ↔ 线程局部数组），host 驱动负责把样本发射到 GPU、
/// 回读输出。返回 `(msl, driver)` 两段文本：`msl` 交给 `xcrun metal`，`driver` 交给 clang。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致。
pub fn metal_harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<(String, String)> {
    let (output, _) = output_of(signature, sample)?;
    let msl = format!(
        "{}\n{}",
        Metal::address_text(kernel_source),
        metal_launch(&signature.kernel(), signature, sample)?
    );
    Ok((msl, metal_harness_driver(signature, sample, output)?))
}

/// 生产态 Metal 逐点网格内核（`Dialect::grid_kernel` 的 Metal 方言）。
///
/// 带长度守卫：`dispatchThreads` 会把总线程数向上取整到线程组的整数倍，末组多出来的线程必须
/// 自己跳过——否则越界线程会往缓冲尾部之外写，把别的缓冲改掉而结果看起来「只是有点错」。
///
/// # Errors：不是逐点签名（标量数 ≠ 1）、模板缺内核函数，或样本缺长度标量。
pub fn metal_grid_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    Metal::grid_kernel(
        kernel_source,
        signature,
        Some(length_scalar(signature, sample)?),
    )
}

/// 生产态循环注入网格内核：把模板里的串行循环头绑定到线程下标（N 维网格，`tid.x` / `tid.y` …）。
///
/// 空网格（`dims` 为空）退化为**串行**：不追加 `tid`、不绑循环、缓冲原样透传、标量取样本值——
/// 串行就是循环注入的零维特例，不再单列一条 lowering。
///
/// # Errors：网格维超过 3、模板里找不到声明的循环、标量缺样本取值时返回说明。
pub fn metal_loop_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    if grid.dims.is_empty() {
        return Metal::serial_kernel(kernel_source, signature, sample);
    }
    Metal::injected_kernel(kernel_source, signature, sample, grid)
}

/// `cutile_launch` compute kernel：按签名顺序声明 `device` 缓冲并原样透传给内核；标量直接取样本值。
fn metal_launch(kernel: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let call = launch_args(sample, signature.params())?;
    let bufs = Metal::bufs(signature);
    Ok(format!(
        "kernel void cutile_launch({bufs},\n                          uint tid [[thread_position_in_grid]]) {{\n    if (tid != 0) return;\n    {kernel}({call});\n}}"
    ))
}
