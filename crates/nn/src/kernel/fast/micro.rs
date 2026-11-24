//! 职责：微内核整行块扫描的只读侧（[`Sweep`]）与标量残块兜底（[`edge`]）。
//!
//! 8×8 寄存器块本体（[`Tile`](cutile::native::gemm8x8::Tile)、`micro_packed`、NEON/AVX2/scalar
//! 三架构实现与端口探针）已上提 `cutile::native::gemm8x8`；本文件只留 GEMM 骨架职责：面板扫描坐标、
//! 权重布局寻址、残行残列标量点积。nn 不持有任何 arch/feature 门与汇编片段。

/// 一次整行块扫描的只读侧：指针、面板与三个尺寸一次带走（实参不散成一长串）。
#[derive(Clone, Copy)]
pub(super) struct Sweep<'a> {
    pub(super) a: &'a [f32],
    pub(super) apack: &'a [f32],
    pub(super) bpack: &'a [f32],
    pub(super) w: &'a [f32],
    pub(super) full: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) layout: Layout,
}

/// 一个待算的标量尾块：左上角 `(i, j)`、覆盖 `mr × nr`，矩阵列宽 `n`、内积长 `k`。
#[derive(Clone, Copy)]
pub(super) struct Span {
    pub(super) i: usize,
    pub(super) j: usize,
    pub(super) mr: usize,
    pub(super) nr: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    /// `a` 的整行宽：内积分块时 `k` 只是当前块，跨块偏移要按整行宽算。
    pub(super) stride: usize,
}

/// 权重布局：`Nt` 是收缩轴在前的 `b[k][n]`，`Tn` 是按输出行常驻的 `w[n][k]`。
#[derive(Clone, Copy)]
pub(super) enum Layout {
    Nt,
    Tn,
}

/// 权重在 `w` 里的寻址三元组：`(基址, 内积步长, 列步长)`——两种布局在这一处收敛。
pub(super) fn weight_stride(span: Span, k0: usize, layout: Layout) -> (usize, usize, usize) {
    match layout {
        Layout::Nt => (k0 * span.n + span.j, span.n, 1),
        Layout::Tn => (span.j * span.stride + k0, 1, span.stride),
    }
}

/// 标量尾块：`out[i..i+mr, j..j+nr]` 逐元素点积（与微内核同一组累加口径，容差内一致）。
pub(super) fn edge(
    a: &[f32],
    w: &[f32],
    out: &mut [f32],
    span: Span,
    k0: usize,
    first: bool,
    layout: Layout,
) {
    let (base, dp, dc) = weight_stride(span, k0, layout);
    for r in 0..span.mr {
        for c in 0..span.nr {
            let slot = (span.i + r) * span.n + span.j + c;
            let mut acc = if first { 0.0f32 } else { out[slot] };
            let wc = base + c * dc;
            for p in 0..span.k {
                acc += a[(span.i + r) * span.stride + k0 + p] * w[wc + p * dp];
            }
            out[slot] = acc;
        }
    }
}
