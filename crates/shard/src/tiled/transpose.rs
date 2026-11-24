//! 张量核 B 布局前置：行主序 f32 → 列主序 half 的转置内核。

use diagnostic::{Code, Error, Result};

/// 张量核 GEMM 的 B 布局前置：把行主序 `rows × cols` 的 f32 转成 `cols × rows` 的 half。
///
/// 张量核要的 B 是 `N×K`（规约轴连续）且按 half 存，两者都不是模板里 `K×N` f32 的形状，所以
/// 权重上传后先跑一遍这个内核，之后每次 GEMM 只读转好的缓冲。32×32 分块走线程组内存，
/// 全局读、全局写都按行连续，与标量逐元素转置相比不牺牲合并访存。
///
/// 派发几何：线程组 `[⌈cols/32⌉, ⌈rows/32⌉]`，组内 `32×8` 个线程。
///
/// # Errors
///
/// `rows` 或 `cols` 为 0 时返回说明。
pub fn metal_transpose_half(rows: usize, cols: usize) -> Result<String> {
    if rows == 0 || cols == 0 {
        return Err(Error::new(Code::Invalid, "转置要求行列均非零")
            .with("shard::tiled::metal_transpose_half"));
    }
    Ok(format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         \n\
         kernel void kernel_main(\n\
         \x20    device const float* d_src [[buffer(0)]],\n\
         \x20    device half* d_dst [[buffer(1)]],\n\
         \x20    uint3 tgp [[threadgroup_position_in_grid]],\n\
         \x20    uint3 tid [[thread_position_in_threadgroup]]) {{\n\
         \x20    constexpr int R = {rows}, C = {cols};\n\
         \x20    threadgroup float tile[32][33];\n\
         \x20    const int sx = int(tgp.x) * 32 + int(tid.x);\n\
         \x20    const int sy = int(tgp.y) * 32 + int(tid.y);\n\
         \x20    for (int j = 0; j < 32; j += 8) {{\n\
         \x20        const int y = sy + j;\n\
         \x20        tile[int(tid.y) + j][int(tid.x)] =\n\
         \x20            (y < R && sx < C) ? d_src[(size_t)y * C + sx] : 0.0f;\n\
         \x20    }}\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    const int dx = int(tgp.y) * 32 + int(tid.x);\n\
         \x20    const int dy = int(tgp.x) * 32 + int(tid.y);\n\
         \x20    for (int j = 0; j < 32; j += 8) {{\n\
         \x20        const int y = dy + j;\n\
         \x20        if (y < C && dx < R)\n\
         \x20            d_dst[(size_t)y * R + dx] = half(tile[int(tid.x)][int(tid.y) + j]);\n\
         \x20    }}\n\
         }}"
    ))
}
