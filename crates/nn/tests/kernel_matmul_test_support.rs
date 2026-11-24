// matmul 两档精度（`f32` / `i8`）共用的对拍装配：视图构造 + 标量参考与 tile 两条骨架各跑一遍。
//
// 采样公式与最终断言两档不同（`f32` 按 IEEE 容差、`i8` 逐位相等），留在各自测试文件；
// 两条骨架入口由调用方传入，`i8` 档因此仍能对拍本地 4×16 寄存器分块实现。这里只收拢两档
// 逐字同形的装配段，避免两份样板各自漂移。

use crate::kernel::matmul::{pair_views, Precision};
use tensor::View;

/// 跑通标量参考与 tile 两条骨架，返回 `(参考输出, tile 输出)` 供调用方按本档口径断言。
fn run_both<P: Precision, S, T>(
    a: &[P],
    b: &[P],
    m: usize,
    n: usize,
    k: usize,
    scalar: S,
    tiled: T,
) -> (Vec<P::Acc>, Vec<P::Acc>)
where
    S: FnOnce(View<'_, P>, View<'_, P>, &mut [P::Acc]) -> diagnostic::Result<()>,
    T: FnOnce(View<'_, P>, View<'_, P>, &mut [P::Acc]) -> diagnostic::Result<()>,
{
    let mut expected = vec![P::Acc::default(); m * n];
    let mut actual = vec![P::Acc::default(); m * n];
    let (av, bv) = pair_views(m, n, k, a, b);
    scalar(av.clone(), bv.clone(), &mut expected).expect("参考可计算");
    tiled(av, bv, &mut actual).expect("tile 可计算");
    (expected, actual)
}
