//! 职责：卷积共享支撑：打包、切片与线程分派。
use std::thread;

use diagnostic::{Code, Error, Result};

use crate::Scalar;
use llm_gemm::fast;

use super::geom::{Conv2d, ConvGeom};

pub(crate) trait Fold<S: Scalar> {
    /// 吸收一个窗口内的有效元素（越界补零位不调用）。
    fn feed(&mut self, value: S);

    /// 收尾成该窗口的输出值。
    fn finish(self) -> S;
}

/// 取最大：起点 `-∞`，空窗口也返回 `-∞`。
pub(crate) struct MaxFold<S>(pub(crate) S);

impl<S: Scalar> Fold<S> for MaxFold<S> {
    fn feed(&mut self, value: S) {
        self.0 = self.0.max(value);
    }

    fn finish(self) -> S {
        self.0
    }
}

/// 取平均：分母按 `include_pad` 取窗口面积或有效格数（下限 1，空窗口不除零）。
pub(crate) struct AvgFold<S> {
    pub(crate) sum: S,
    pub(crate) count: usize,
    pub(crate) area: usize,
    pub(crate) include_pad: bool,
}

impl<S: Scalar> Fold<S> for AvgFold<S> {
    fn feed(&mut self, value: S) {
        self.sum = self.sum + value;
        self.count += 1;
    }

    fn finish(self) -> S {
        let denom = if self.include_pad {
            self.area
        } else {
            self.count.max(1)
        };
        self.sum / S::from_usize(denom)
    }
}

/// 池化输出形状：两个池化类型的 `shape` 共用。
pub(crate) fn out_shape(geom: &ConvGeom, h: usize, w: usize) -> Result<(usize, usize)> {
    Ok((geom.oh(h)?, geom.ow(w)?))
}

/// 池化前向骨架：形状推导、长度校验、遍历顺序共用，窗口归约由 `fold` 注入。
pub(crate) fn run<S: Scalar, F: Fold<S>>(
    geom: &ConvGeom,
    x: &[S],
    n: usize,
    (h, w): (usize, usize),
    c: usize,
    out: &mut [S],
    fold: impl FnMut() -> F,
) -> Result<()> {
    let (oh, ow) = out_shape(geom, h, w)?;
    check_shapes(&[
        (x.len(), n * h * w * c, "输入"),
        (out.len(), n * oh * ow * c, "输出"),
    ])?;
    scatter(geom, x, n, (h, w), c, out, (oh, ow), fold);
    Ok(())
}

/// 按 `(n, oh, ow, c)` 四重遍历写满输出；窗口归约仍由 `fold` 注入。
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter<S: Scalar, F: Fold<S>>(
    geom: &ConvGeom,
    x: &[S],
    n: usize,
    (h, w): (usize, usize),
    c: usize,
    out: &mut [S],
    (oh, ow): (usize, usize),
    mut fold: impl FnMut() -> F,
) {
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                for ci in 0..c {
                    out[((ni * oh + oi) * ow + oj) * c + ci] =
                        window(geom, x, (ni, oi, oj, ci), (h, w), c, fold());
                }
            }
        }
    }
}

/// 单个输出格的窗口归约：按 `(kh, kw)` 扫描，越界（补零）位置直接跳过。
pub(crate) fn window<S: Scalar, F: Fold<S>>(
    geom: &ConvGeom,
    x: &[S],
    (ni, oi, oj, ci): (usize, usize, usize, usize),
    (h, w): (usize, usize),
    c: usize,
    mut fold: F,
) -> S {
    for ki in 0..geom.kh {
        let ii = (oi * geom.sh + ki) as isize - geom.ph as isize;
        if !in_bounds_1d(ii, h) {
            continue;
        }
        for kj in 0..geom.kw {
            let jj = (oj * geom.sw + kj) as isize - geom.pw as isize;
            if !in_bounds_1d(jj, w) {
                continue;
            }
            fold.feed(x[((ni * h + ii as usize) * w + jj as usize) * c + ci]);
        }
    }
    fold.finish()
}

/// 一维边界判定。
pub(crate) fn in_bounds_1d(idx: isize, len: usize) -> bool {
    idx >= 0 && (idx as usize) < len
}

/// 二维边界判定。
pub(crate) fn in_bounds(ii: isize, jj: isize, h: usize, w: usize) -> bool {
    in_bounds_1d(ii, h) && in_bounds_1d(jj, w)
}

/// f32/i8 共用的 im2col：行 = `(n, oh, ow)`，列 = `(kh, kw, cin)`。
pub(crate) fn im2col2d<T: Copy + Send + Sync + Default>(
    x: &[T],
    col: &mut [T],
    n: usize,
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let patch = conv.geom.patch(conv.cin);
    let rows = n * oh * ow;
    parallel_rows(x, col, rows, patch, |src, row, r| {
        fill_row(src, row, r, (h, w), conv, (oh, ow));
    });
}

/// 按可用并行度决定单线程还是线程池分片；小张量退化为顺序，避免并行负加速。
pub(crate) fn thread_count(rows: usize, patch: usize) -> usize {
    if rows * patch < (1 << 16) {
        1
    } else {
        thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
            .min(rows)
    }
}

/// 把 `rows` 行按线程分片后逐行调用 `fill`。
pub(crate) fn parallel_rows<T: Copy + Send + Sync>(
    x: &[T],
    col: &mut [T],
    rows: usize,
    patch: usize,
    fill: impl Fn(&[T], &mut [T], usize) + Sync + Send + Copy,
) {
    let threads = thread_count(rows, patch);
    if threads <= 1 {
        for r in 0..rows {
            fill(x, &mut col[r * patch..(r + 1) * patch], r);
        }
        return;
    }
    let per = rows.div_ceil(threads);
    thread::scope(|s| {
        for (bi, chunk) in col.chunks_mut(per * patch).enumerate() {
            let r0 = bi * per;
            s.spawn(move || fill_rows(x, chunk, patch, r0, fill));
        }
    });
}

/// 一个分片上把行逐一交给 `fill`：`chunk` 覆盖 `r0` 起的连续行。
pub(crate) fn fill_rows<T: Copy>(
    x: &[T],
    chunk: &mut [T],
    patch: usize,
    r0: usize,
    fill: impl Fn(&[T], &mut [T], usize) + Sync + Copy,
) {
    for base in 0..chunk.len() / patch {
        fill(x, &mut chunk[base * patch..(base + 1) * patch], r0 + base);
    }
}

/// 展开一行：`row_idx = (ni, oi, oj)` 对应的一整块 `(kh, kw, cin)` 拷贝，越界补零。
pub(crate) fn fill_row<T: Copy + Default>(
    x: &[T],
    row: &mut [T],
    row_idx: usize,
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let cin = conv.cin;
    let g = conv.geom;
    let ni = row_idx / (oh * ow);
    let rem = row_idx % (oh * ow);
    let oi = rem / ow;
    let oj = rem % ow;
    for ki in 0..g.kh {
        let ii = (oi * g.sh + ki) as isize - g.ph as isize;
        for kj in 0..g.kw {
            let jj = (oj * g.sw + kj) as isize - g.pw as isize;
            let tail = (ki * g.kw + kj) * cin;
            if in_bounds(ii, jj, h, w) {
                let src = ((ni * h + ii as usize) * w + jj as usize) * cin;
                row[tail..tail + cin].copy_from_slice(&x[src..src + cin]);
            } else {
                row[tail..tail + cin].fill(T::default());
            }
        }
    }
}

/// `im2col` 的逆向：把每行的收缩梯度按同一窗口位置累加回输入（`stride < k` 的重叠位也要累加）。
pub(crate) fn col2im2d(
    col: &[f32],
    dx: &mut [f32],
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let patch = conv.geom.patch(conv.cin);
    let rows = col.len() / patch;
    for row in 0..rows {
        accumulate_col_row(col, dx, row, (h, w), conv, (oh, ow));
    }
}

/// 散射一行（`row` 对应 `(ni, oi, oj)`）的列梯度。
pub(crate) fn accumulate_col_row(
    col: &[f32],
    dx: &mut [f32],
    row: usize,
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let patch = conv.geom.patch(conv.cin);
    let cin = conv.cin;
    let g = conv.geom;
    let ni = row / (oh * ow);
    let rem = row % (oh * ow);
    let oi = rem / ow;
    let oj = rem % ow;
    for ki in 0..g.kh {
        let ii = (oi * g.sh + ki) as isize - g.ph as isize;
        if !in_bounds_1d(ii, h) {
            continue;
        }
        for kj in 0..g.kw {
            let jj = (oj * g.sw + kj) as isize - g.pw as isize;
            if !in_bounds_1d(jj, w) {
                continue;
            }
            let tail = (ki * g.kw + kj) * cin;
            let dst = ((ni * h + ii as usize) * w + jj as usize) * cin;
            accumulate_cell(dx, col, dst, row * patch + tail, cin);
        }
    }
}

/// 把一个列片段的 `cin` 个值累加进 `dx`。
pub(crate) fn accumulate_cell(dx: &mut [f32], col: &[f32], dst: usize, src: usize, cin: usize) {
    for ci in 0..cin {
        dx[dst + ci] += col[src + ci];
    }
}

/// 一维 im2col：行 = `(n, ol)`，列 = `(kw, cin)`。
pub(crate) fn im2col1d(
    x: &[f32],
    col: &mut [f32],
    n: usize,
    l: usize,
    ol: usize,
    g: ConvGeom,
    cin: usize,
) {
    let patch = g.patch(cin);
    for ni in 0..n {
        for oi in 0..ol {
            fill_row_1d(
                x,
                &mut col[(ni * ol + oi) * patch..(ni * ol + oi + 1) * patch],
                ni,
                oi,
                l,
                g,
                cin,
            );
        }
    }
}

/// 展开一维卷积的一行感受野。
pub(crate) fn fill_row_1d(
    x: &[f32],
    row: &mut [f32],
    ni: usize,
    oi: usize,
    l: usize,
    g: ConvGeom,
    cin: usize,
) {
    for ki in 0..g.kw {
        let src = (oi * g.sw + ki) as isize - g.pw as isize;
        for ci in 0..cin {
            row[ki * cin + ci] = if in_bounds_1d(src, l) {
                x[(ni * l + src as usize) * cin + ci]
            } else {
                0.0
            };
        }
    }
}

/// f32 GEMM 转调：`out[m,n] = a[m,k] · b[k,n]`，走 `fast` 的多线程内核。
pub(crate) fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    fast::gemm_f32(a, b, m, n, k, out);
}

/// 每行加同一个偏置向量，f32/i32 共用。
pub(crate) fn add_bias<T: Copy + core::ops::AddAssign>(out: &mut [T], bias: &[T], cout: usize) {
    for row in out.chunks_mut(cout) {
        for (slot, b) in row.iter_mut().zip(bias.iter()) {
            *slot += *b;
        }
    }
}

/// 批量缓冲长度校验：任一缓冲小于所需即报错，错误信息里带缓冲名。
pub(crate) fn check_shapes(items: &[(usize, usize, &str)]) -> Result<()> {
    for &(actual, required, what) in items {
        if actual < required {
            return Err(Error::new(
                Code::Invalid,
                format!("{what}: 缓冲 {actual}，需要 {required}"),
            ));
        }
    }
    Ok(())
}

/// 线程本地复用的 i8 im2col 缓冲：大形状下省掉每次的零页分配与缺页。
pub(crate) fn with_scratch_col(f: impl FnOnce(&mut Vec<i8>) -> Result<()>) -> Result<()> {
    thread_local! {
        /// 大形状下 im2col 缓冲达几十 MB，跨调用复用可以省掉每次的零页分配与缺页。
        static COL: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    COL.with(|cell| {
        let mut loan = cell.try_borrow_mut();
        match loan.as_mut() {
            Ok(col) => f(col),
            // 同线程嵌套调用（借出中）时才退回一次性缓冲。
            Err(_) => f(&mut Vec::new()),
        }
    })
}

/// 权重梯度累计：`dweight += col^T · dy`，`dbias += Σ dy`。
pub(crate) fn accumulate_weight_grad(
    dy: &[f32],
    col: &[f32],
    dbias: &mut [f32],
    dweight: &mut [f32],
    rows: usize,
    cout: usize,
    patch: usize,
) {
    for r in 0..rows {
        add_row_bias(dbias, &dy[r * cout..(r + 1) * cout]);
        accumulate_dweight_row(
            &col[r * patch..(r + 1) * patch],
            &dy[r * cout..(r + 1) * cout],
            dweight,
            cout,
        );
    }
}

pub(crate) fn add_row_bias(dbias: &mut [f32], dy_row: &[f32]) {
    for (slot, value) in dbias.iter_mut().zip(dy_row) {
        *slot += *value;
    }
}

pub(crate) fn accumulate_dweight_row(
    col_row: &[f32],
    dy_row: &[f32],
    dweight: &mut [f32],
    cout: usize,
) {
    for (k, &value) in col_row.iter().enumerate() {
        if value == 0.0 {
            continue;
        }
        for (slot, upstream) in dweight[k * cout..(k + 1) * cout].iter_mut().zip(dy_row) {
            *slot += value * *upstream;
        }
    }
}

/// `dcol = dy · weight^T`，供 col2im 散射回 `dx`。
pub(crate) fn compute_dcol(
    dy: &[f32],
    weight: &[f32],
    dcol: &mut [f32],
    rows: usize,
    cout: usize,
    patch: usize,
) {
    for r in 0..rows {
        let dy_row = &dy[r * cout..(r + 1) * cout];
        let dcol_row = &mut dcol[r * patch..(r + 1) * patch];
        for (k, slot) in dcol_row.iter_mut().enumerate() {
            *slot = dot(dy_row, &weight[k * cout..(k + 1) * cout]);
        }
    }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}
