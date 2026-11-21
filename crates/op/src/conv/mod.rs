//! 经典卷积与池化：NHWC 布局的 `Conv1d`/`Conv2d`、`MaxPool2d`/`AvgPool2d`。
//!
//! 卷积不写双重循环：先 `im2col` 把每个输出位置展开成一行，再交给 [`crate::fast`] 的
//! GEMM 微核。语义与逐点直算逐位对齐 f32 容差，性能走 SIMD 通道（i8 量化路径见
//! [`Conv2d::forward_i8`]，复用 `fast::gemm_i8` 的多槽池）。
//!
//! **拥有**：本模块的四个层结构。**不拥有**：GEMM 微核（`fast`）、张量元数据（`tensor`）。

use std::thread;

use diagnostic::{Code, Error, Result};

use crate::Scalar;
use crate::fast;

/// 卷积的公共几何：核、步长、补零。通道数不在这里——1d 与 2d 共用同一套核参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvGeom {
    /// 卷积核高（1d 恒为 1）。
    pub kh: usize,
    /// 卷积核宽。
    pub kw: usize,
    /// 步长高（1d 恒为 1）。
    pub sh: usize,
    /// 步长宽。
    pub sw: usize,
    /// 补零高。
    pub ph: usize,
    /// 补零宽。
    pub pw: usize,
}

impl ConvGeom {
    /// 输出高：`floor((h + 2·ph - kh) / sh) + 1`，下溢即形状不合法。
    ///
    /// # Errors
    ///
    /// `h + 2·ph < kh` 或 `sh == 0` 时返回错误。
    pub fn oh(&self, h: usize) -> Result<usize> {
        self.out_len(h + 2 * self.ph, self.kh, self.sh, "h")
    }

    /// 输出宽；口径同 [`ConvGeom::oh`]。
    pub fn ow(&self, w: usize) -> Result<usize> {
        self.out_len(w + 2 * self.pw, self.kw, self.sw, "w")
    }

    fn out_len(&self, padded: usize, k: usize, s: usize, axis: &str) -> Result<usize> {
        if s == 0 || k == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("{axis}: 核与步长必须非零"),
            ));
        }
        if padded < k {
            return Err(Error::new(
                Code::Invalid,
                format!("{axis}: 补零后 {padded} 小于核 {k}，没有合法输出"),
            ));
        }
        Ok((padded - k) / s + 1)
    }

    /// 一行 im2col 的长度：`kh·kw·cin`。
    pub fn patch(&self, cin: usize) -> usize {
        self.kh * self.kw * cin
    }
}

/// 一维卷积：输入 `[n, l, cin]`、权重 `[kw, cin, cout]`、输出 `[n, ol, cout]`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1d {
    /// 输入/输出通道与核宽、步长、补零。
    pub cin: usize,
    /// 输出通道。
    pub cout: usize,
    /// 核、步长、补零（`kh`/`sh`/`ph` 恒为 1）。
    pub geom: ConvGeom,
}

impl Conv1d {
    /// 按核宽与步长构造；补零默认 0。
    pub fn new(cin: usize, cout: usize, kw: usize, stride: usize, pad: usize) -> Self {
        Self {
            cin,
            cout,
            geom: ConvGeom {
                kh: 1,
                kw,
                sh: 1,
                sw: stride,
                ph: 0,
                pw: pad,
            },
        }
    }

    /// 输出长度。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn len(&self, l: usize) -> Result<usize> {
        self.geom.ow(l)
    }

    /// 输出长度是否为 0（clippy 要求与 `len` 成对出现）。
    pub fn is_empty(&self, l: usize) -> Result<bool> {
        Ok(self.len(l)? == 0)
    }

    /// f32 前向：im2col → GEMM，再按需加偏置。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
    pub fn forward_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n: usize,
        l: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let ol = self.len(l)?;
        let cin = self.cin;
        let cout = self.cout;
        let patch = self.geom.patch(cin);
        check_shapes(&[
            (x.len(), n * l * cin, "输入"),
            (weight.len(), patch * cout, "权重"),
            (bias.len(), cout, "偏置"),
            (out.len(), n * ol * cout, "输出"),
        ])?;

        let mut col = vec![0.0f32; n * ol * patch];
        im2col1d(x, &mut col, n, l, ol, self.geom, cin);
        gemm(&col, weight, n * ol, cout, patch, out);
        add_bias(out, bias, cout);
        Ok(())
    }
}

/// 二维卷积：输入 `[n, h, w, cin]`、权重 `[kh, kw, cin, cout]`、输出 `[n, oh, ow, cout]`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2d {
    /// 输入通道。
    pub cin: usize,
    /// 输出通道。
    pub cout: usize,
    /// 核、步长、补零。
    pub geom: ConvGeom,
}

impl Conv2d {
    /// 按核、步长、补零构造。
    pub fn new(cin: usize, cout: usize, geom: ConvGeom) -> Self {
        Self { cin, cout, geom }
    }

    /// 输出形状 `(oh, ow)`。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        Ok((self.geom.oh(h)?, self.geom.ow(w)?))
    }

    /// f32 前向：im2col → GEMM → 偏置。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
    pub fn forward_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n: usize,
        (h, w): (usize, usize),
        out: &mut [f32],
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let cin = self.cin;
        let cout = self.cout;
        let patch = self.geom.patch(cin);
        check_shapes(&[
            (x.len(), n * h * w * cin, "输入"),
            (weight.len(), patch * cout, "权重"),
            (bias.len(), cout, "偏置"),
            (out.len(), n * oh * ow * cout, "输出"),
        ])?;

        let rows = n * oh * ow;
        let mut col = vec![0.0f32; rows * patch];
        im2col2d(x, &mut col, n, (h, w), self, (oh, ow));
        gemm(&col, weight, rows, cout, patch, out);
        add_bias(out, bias, cout);
        Ok(())
    }

    /// f32 反向：上游梯度 `dy` → `dx / dweight / dbias`。
    ///
    /// 与前向共用同一份 `im2col` 与卷积几何，三个输出缓冲**先清零再累加**（多上游可叠加）。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn backward_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        dy: &[f32],
        n: usize,
        h: usize,
        w: usize,
        dx: &mut [f32],
        dweight: &mut [f32],
        dbias: &mut [f32],
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let (cin, cout) = (self.cin, self.cout);
        let patch = self.geom.patch(cin);
        let rows = n * oh * ow;
        check_shapes(&[
            (x.len(), n * h * w * cin, "输入"),
            (weight.len(), patch * cout, "权重"),
            (dy.len(), rows * cout, "上游梯度"),
            (dx.len(), n * h * w * cin, "输入梯度"),
            (dweight.len(), patch * cout, "权重梯度"),
            (dbias.len(), cout, "偏置梯度"),
        ])?;

        let mut col = vec![0.0f32; rows * patch];
        im2col2d(x, &mut col, n, (h, w), self, (oh, ow));
        dx.fill(0.0);
        dweight.fill(0.0);
        dbias.fill(0.0);
        accumulate_weight_grad(dy, &col, dbias, dweight, rows, cout, patch);
        let mut dcol = vec![0.0f32; rows * patch];
        compute_dcol(dy, weight, &mut dcol, rows, cout, patch);
        col2im2d(&dcol, dx, (h, w), self, (oh, ow));
        Ok(())
    }

    /// i8 前向（对称量化）：输入与权重已量化，输出 i32 累加，`bias` 为同尺度整数偏置。
    ///
    /// 量化卷积的全部算力都落在 `fast::gemm_i8` 的多槽池上；`bias` 传全 0 可省。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
    /// 复用外部 `col` 缓冲的 i8 卷积：大形状下省掉每轮 37.7MB 的零页分配与缺页。
    ///
    /// `col` 只需足够长（`rows * patch`）；不足时按需扩容，内容由 im2col 全量覆盖。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_i8_with(
        &self,
        x: &[i8],
        weight: &[i8],
        bias: &[i32],
        n: usize,
        (h, w): (usize, usize),
        out: &mut [i32],
        col: &mut Vec<i8>,
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let cout = self.cout;
        let patch = self.geom.patch(self.cin);
        check_shapes(&[
            (x.len(), n * h * w * self.cin, "输入"),
            (weight.len(), patch * cout, "权重"),
            (bias.len(), cout, "偏置"),
            (out.len(), n * oh * ow * cout, "输出"),
        ])?;

        let rows = n * oh * ow;
        col.resize(rows * patch, 0);
        im2col2d(x, col, n, (h, w), self, (oh, ow));
        fast::gemm_i8(col, weight, rows, cout, patch, out);
        add_bias(out, bias, cout);
        Ok(())
    }

    /// i8 卷积：`col` 内部现分配，单次调用；批量场景用 [`Self::forward_i8_with`] 复用缓冲。
    pub fn forward_i8(
        &self,
        x: &[i8],
        weight: &[i8],
        bias: &[i32],
        n: usize,
        (h, w): (usize, usize),
        out: &mut [i32],
    ) -> Result<()> {
        with_scratch_col(|col| self.forward_i8_with(x, weight, bias, n, (h, w), out, col))
    }
}

/// 二维最大池化：`[n, h, w, c]` → `[n, oh, ow, c]`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPool2d {
    /// 池化核、步长、补零（补零位置不参与比较）。
    pub geom: ConvGeom,
}

/// 二维平均池化：口径同 [`MaxPool2d`]，`count_include_pad` 决定补零是否计入分母。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvgPool2d {
    /// 池化核、步长、补零。
    pub geom: ConvGeom,
    /// 是否把补零位置计入平均的分母。
    pub count_include_pad: bool,
}

impl MaxPool2d {
    /// 输出形状。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        out_shape(&self.geom, h, w)
    }

    /// 前向：逐窗口取最大，补零位不参与。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        run(&self.geom, x, n, (h, w), c, out, || {
            MaxFold(S::NEG_INFINITY)
        })
    }
}

impl AvgPool2d {
    /// 输出形状。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        out_shape(&self.geom, h, w)
    }

    /// 前向：窗口求和后按口径取分母；`count_include_pad=false` 时只除有效位置。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        let fold = || AvgFold {
            sum: S::ZERO,
            count: 0,
            area: self.geom.kh * self.geom.kw,
            include_pad: self.count_include_pad,
        };
        run(&self.geom, x, n, (h, w), c, out, fold)
    }
}

/// 池化窗口的归约口径：最大 / 平均。窗口遍历（补零裁剪、下标推导）全仓只有一份，
/// 两条口径只在累加器与收尾不同；泛型化后各自单态展开，内层循环里没有分支。
trait Fold<S: Scalar> {
    /// 吸收一个窗口内的有效元素（越界补零位不调用）。
    fn feed(&mut self, value: S);

    /// 收尾成该窗口的输出值。
    fn finish(self) -> S;
}

/// 取最大：起点 `-∞`，空窗口也返回 `-∞`。
struct MaxFold<S>(S);

impl<S: Scalar> Fold<S> for MaxFold<S> {
    fn feed(&mut self, value: S) {
        self.0 = self.0.max(value);
    }

    fn finish(self) -> S {
        self.0
    }
}

/// 取平均：分母按 `include_pad` 取窗口面积或有效格数（下限 1，空窗口不除零）。
struct AvgFold<S> {
    sum: S,
    count: usize,
    area: usize,
    include_pad: bool,
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
fn out_shape(geom: &ConvGeom, h: usize, w: usize) -> Result<(usize, usize)> {
    Ok((geom.oh(h)?, geom.ow(w)?))
}

/// 池化前向骨架：形状推导、长度校验、遍历顺序共用，窗口归约由 `fold` 注入。
fn run<S: Scalar, F: Fold<S>>(
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
fn scatter<S: Scalar, F: Fold<S>>(
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
fn window<S: Scalar, F: Fold<S>>(
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
fn in_bounds_1d(idx: isize, len: usize) -> bool {
    idx >= 0 && (idx as usize) < len
}

/// 二维边界判定。
fn in_bounds(ii: isize, jj: isize, h: usize, w: usize) -> bool {
    in_bounds_1d(ii, h) && in_bounds_1d(jj, w)
}

/// f32/i8 共用的 im2col：行 = `(n, oh, ow)`，列 = `(kh, kw, cin)`。
///
/// 行块之间互不重叠，直接按行切给线程；小矩阵不并行（一次 `scope` 的固定开销会超过拷贝本身）。
fn im2col2d<T: Copy + Send + Sync + Default>(
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
fn thread_count(rows: usize, patch: usize) -> usize {
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
fn parallel_rows<T: Copy + Send + Sync>(
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
fn fill_rows<T: Copy>(
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
fn fill_row<T: Copy + Default>(
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
fn col2im2d(
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
fn accumulate_col_row(
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
fn accumulate_cell(dx: &mut [f32], col: &[f32], dst: usize, src: usize, cin: usize) {
    for ci in 0..cin {
        dx[dst + ci] += col[src + ci];
    }
}

/// 一维 im2col：行 = `(n, ol)`，列 = `(kw, cin)`。
fn im2col1d(x: &[f32], col: &mut [f32], n: usize, l: usize, ol: usize, g: ConvGeom, cin: usize) {
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
fn fill_row_1d(
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
fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    fast::gemm_f32(a, b, m, n, k, out);
}

/// 每行加同一个偏置向量，f32/i32 共用。
fn add_bias<T: Copy + core::ops::AddAssign>(out: &mut [T], bias: &[T], cout: usize) {
    for row in out.chunks_mut(cout) {
        for (slot, b) in row.iter_mut().zip(bias.iter()) {
            *slot += *b;
        }
    }
}

/// 批量缓冲长度校验：任一缓冲小于所需即报错，错误信息里带缓冲名。
fn check_shapes(items: &[(usize, usize, &str)]) -> Result<()> {
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
fn with_scratch_col(f: impl FnOnce(&mut Vec<i8>) -> Result<()>) -> Result<()> {
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
fn accumulate_weight_grad(
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

fn add_row_bias(dbias: &mut [f32], dy_row: &[f32]) {
    for (slot, value) in dbias.iter_mut().zip(dy_row) {
        *slot += *value;
    }
}

fn accumulate_dweight_row(col_row: &[f32], dy_row: &[f32], dweight: &mut [f32], cout: usize) {
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
fn compute_dcol(
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

/// 一维最大池化：NHWC，`(k, s, p)` 与二维同口径，补零位不参与比较。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPool1d {
    /// 窗口长。
    pub k: usize,
    /// 步长。
    pub s: usize,
    /// 两端补零。
    pub p: usize,
}

impl MaxPool1d {
    /// 输出长度 `(l + 2p − k) / s + 1`。
    ///
    /// # Errors
    ///
    /// 窗口/步长为 0，或窗口比补零后的输入还长时返回说明。
    pub fn len(&self, l: usize) -> Result<usize> {
        if self.k == 0 || self.s == 0 {
            return Err(Error::new(Code::Invalid, "池化窗口与步长必须大于 0"));
        }
        if l + 2 * self.p < self.k {
            return Err(Error::new(Code::Invalid, "池化窗口大于补零后的输入"));
        }
        Ok((l + 2 * self.p - self.k) / self.s + 1)
    }

    /// 前向：逐窗口取最大。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        l: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        let ol = self.len(l)?;
        check_shapes(&[
            (x.len(), n * l * c, "输入"),
            (out.len(), n * ol * c, "输出"),
        ])?;
        self.pool_max(x, out, n, l, c, ol);
        Ok(())
    }

    fn pool_max<S: Scalar>(&self, x: &[S], out: &mut [S], n: usize, l: usize, c: usize, ol: usize) {
        for ni in 0..n {
            for oi in 0..ol {
                for ci in 0..c {
                    out[(ni * ol + oi) * c + ci] = self.window_max(x, ni, oi, ci, l, c);
                }
            }
        }
    }

    fn window_max<S: Scalar>(
        &self,
        x: &[S],
        ni: usize,
        oi: usize,
        ci: usize,
        l: usize,
        c: usize,
    ) -> S {
        let base = oi * self.s;
        let mut best = S::NEG_INFINITY;
        for ki in 0..self.k {
            let src = base + ki;
            if src < self.p || src - self.p >= l {
                continue;
            }
            best = best.max(x[(ni * l + src - self.p) * c + ci]);
        }
        // 整窗都落在补零上（只会在极端几何下出现）按 0 处理。
        if best == S::NEG_INFINITY {
            S::ZERO
        } else {
            best
        }
    }
}

/// 二维自适应平均池化：任意输入尺寸都压到 `(oh, ow)`。
///
/// 第 `i` 格的窗口是 `[floor(i·h/oh), ceil((i+1)·h/oh))`，与 PyTorch 同口径。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveAvgPool2d {
    /// 输出高。
    pub oh: usize,
    /// 输出宽。
    pub ow: usize,
}

impl AdaptiveAvgPool2d {
    /// 前向：每个输出格取覆盖窗口的平均。
    ///
    /// # Errors
    ///
    /// 输出尺寸为 0 或缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        if self.oh == 0 || self.ow == 0 {
            return Err(Error::new(Code::Invalid, "自适应池化输出尺寸必须大于 0"));
        }
        check_shapes(&[
            (x.len(), n * h * w * c, "输入"),
            (out.len(), n * self.oh * self.ow * c, "输出"),
        ])?;
        self.pool_avg(x, out, n, (h, w), c);
        Ok(())
    }

    fn pool_avg<S: Scalar>(
        &self,
        x: &[S],
        out: &mut [S],
        n: usize,
        (h, w): (usize, usize),
        c: usize,
    ) {
        for ni in 0..n {
            for oi in 0..self.oh {
                let (h0, h1) = adaptive_span(oi, h, self.oh);
                for oj in 0..self.ow {
                    let (w0, w1) = adaptive_span(oj, w, self.ow);
                    let area = S::from_usize((h1 - h0) * (w1 - w0));
                    for ci in 0..c {
                        out[((ni * self.oh + oi) * self.ow + oj) * c + ci] =
                            cell_sum(x, (ni, ci), (h0, h1), (w0, w1), (h, w), c) / area;
                    }
                }
            }
        }
    }
}

/// 自适应池化的连续输出区间 `[lo, hi)`。
fn adaptive_span(i: usize, len: usize, out_len: usize) -> (usize, usize) {
    let lo = i * len / out_len;
    let hi = ((i + 1) * len).div_ceil(out_len).min(len);
    (lo, hi)
}

/// 求和 `x` 上矩形区域 `[h0,h1)×[w0,w1)` 的通道 `ci`。
fn cell_sum<S: Scalar>(
    x: &[S],
    (ni, ci): (usize, usize),
    (h0, h1): (usize, usize),
    (w0, w1): (usize, usize),
    (h, w): (usize, usize),
    c: usize,
) -> S {
    let mut sum = S::ZERO;
    for ii in h0..h1 {
        for jj in w0..w1 {
            sum = sum + x[((ni * h + ii) * w + jj) * c + ci];
        }
    }
    sum
}

/// 插值取样方式：最近邻（`align_corners=False` 的 floor 口径）或双线性。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// 取左上角样本，源坐标 `floor(i·h/oh)`。
    Nearest,
    /// 双线性，源坐标 `(i+0.5)·h/oh − 0.5` 后夹到边界。
    Linear,
}

/// 二维上/下采样，NHWC。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interpolate {
    /// 取样方式。
    pub sampler: Sampler,
    /// 目标高。
    pub oh: usize,
    /// 目标宽。
    pub ow: usize,
}

impl Interpolate {
    /// 前向。
    ///
    /// # Errors
    ///
    /// 目标尺寸为 0 或缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        self.ensure_nonzero(h, w)?;
        check_shapes(&[
            (x.len(), n * h * w * c, "输入"),
            (out.len(), n * self.oh * self.ow * c, "输出"),
        ])?;
        match self.sampler {
            Sampler::Nearest => self.forward_nearest(x, n, h, w, c, out),
            Sampler::Linear => self.forward_linear(x, n, h, w, c, out),
        }
        Ok(())
    }

    fn ensure_nonzero(&self, h: usize, w: usize) -> Result<()> {
        if self.oh == 0 || self.ow == 0 {
            return Err(Error::new(Code::Invalid, "插值目标尺寸必须大于 0"));
        }
        if h == 0 || w == 0 {
            return Err(Error::new(Code::Invalid, "插值输入尺寸必须大于 0"));
        }
        Ok(())
    }

    fn forward_nearest<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) {
        for ni in 0..n {
            for oi in 0..self.oh {
                let ii = (oi * h / self.oh).min(h - 1);
                for oj in 0..self.ow {
                    let jj = (oj * w / self.ow).min(w - 1);
                    let dst = ((ni * self.oh + oi) * self.ow + oj) * c;
                    let src = ((ni * h + ii) * w + jj) * c;
                    out[dst..dst + c].copy_from_slice(&x[src..src + c]);
                }
            }
        }
    }

    fn forward_linear<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) {
        for ni in 0..n {
            for oi in 0..self.oh {
                let (y0, y1, ly) = axis_samples::<S>(oi, h, self.oh);
                for oj in 0..self.ow {
                    let (x0, x1, lx) = axis_samples::<S>(oj, w, self.ow);
                    let dst = ((ni * self.oh + oi) * self.ow + oj) * c;
                    for ci in 0..c {
                        let a = x[((ni * h + y0) * w + x0) * c + ci];
                        let b = x[((ni * h + y0) * w + x1) * c + ci];
                        let d = x[((ni * h + y1) * w + x0) * c + ci];
                        let e = x[((ni * h + y1) * w + x1) * c + ci];
                        let top = a + (b - a) * lx;
                        let bot = d + (e - d) * lx;
                        out[dst + ci] = top + (bot - top) * ly;
                    }
                }
            }
        }
    }
}

/// 插值单轴采样点：`(下界, 上界, 插值权重)`。
fn axis_samples<S: Scalar>(i: usize, len: usize, out_len: usize) -> (usize, usize, S) {
    let f = ((S::from_usize(i) + S::from_f64(0.5)) * S::from_usize(len) / S::from_usize(out_len)
        - S::from_f64(0.5))
    .max(S::ZERO);
    let lo = f.floor().min(S::from_usize(len - 1)).to_usize();
    let hi = (lo + 1).min(len - 1);
    let lam = if lo == hi {
        S::ZERO
    } else {
        f - S::from_usize(lo)
    };
    (lo, hi, lam)
}

#[cfg(test)]
mod tests;
