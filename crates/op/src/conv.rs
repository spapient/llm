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
        check(x.len(), n * l * cin, "输入")?;
        check(weight.len(), patch * cout, "权重")?;
        check(bias.len(), cout, "偏置")?;
        check(out.len(), n * ol * cout, "输出")?;

        let mut col = vec![0.0f32; n * ol * patch];
        // im2col：行 = (n, ol)，列 = (kw, cin)。
        for ni in 0..n {
            for oi in 0..ol {
                let row = &mut col[(ni * ol + oi) * patch..(ni * ol + oi + 1) * patch];
                for ki in 0..self.geom.kw {
                    let src = (oi * self.geom.sw + ki) as isize - self.geom.pw as isize;
                    for ci in 0..cin {
                        let slot = &mut row[(ki * cin + ci)..][..1];
                        slot[0] = if src < 0 || src as usize >= l {
                            0.0
                        } else {
                            x[(ni * l + src as usize) * cin + ci]
                        };
                    }
                }
            }
        }
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
        h: usize,
        w: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let cin = self.cin;
        let cout = self.cout;
        let patch = self.geom.patch(cin);
        check(x.len(), n * h * w * cin, "输入")?;
        check(weight.len(), patch * cout, "权重")?;
        check(bias.len(), cout, "偏置")?;
        check(out.len(), n * oh * ow * cout, "输出")?;

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
        check(x.len(), n * h * w * cin, "输入")?;
        check(weight.len(), patch * cout, "权重")?;
        check(dy.len(), rows * cout, "上游梯度")?;
        check(dx.len(), n * h * w * cin, "输入梯度")?;
        check(dweight.len(), patch * cout, "权重梯度")?;
        check(dbias.len(), cout, "偏置梯度")?;

        let mut col = vec![0.0f32; rows * patch];
        im2col2d(x, &mut col, n, (h, w), self, (oh, ow));
        dx.fill(0.0);
        dweight.fill(0.0);
        dbias.fill(0.0);
        for r in 0..rows {
            let dy_row = &dy[r * cout..(r + 1) * cout];
            for (slot, value) in dbias.iter_mut().zip(dy_row) {
                *slot += *value;
            }
            // 权重梯度按行累加：`dweight[k, co] += col[r, k] · dy[r, co]`。
            let col_row = &col[r * patch..(r + 1) * patch];
            for (k, &value) in col_row.iter().enumerate() {
                if value == 0.0 {
                    continue;
                }
                let dw = &mut dweight[k * cout..(k + 1) * cout];
                for (slot, upstream) in dw.iter_mut().zip(dy_row) {
                    *slot += value * *upstream;
                }
            }
        }
        // 输入梯度走同一份 col 的转置乘：`dcol[r, k] = Σ_co dy[r, co] · weight[k, co]`。
        let mut dcol = vec![0.0f32; rows * patch];
        for r in 0..rows {
            let dy_row = &dy[r * cout..(r + 1) * cout];
            let dcol_row = &mut dcol[r * patch..(r + 1) * patch];
            for (k, slot) in dcol_row.iter_mut().enumerate() {
                let w_row = &weight[k * cout..(k + 1) * cout];
                *slot = dy_row.iter().zip(w_row).map(|(a, b)| a * b).sum();
            }
        }
        col2im2d(&dcol, dx, n, (h, w), self, (oh, ow));
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
        h: usize,
        w: usize,
        out: &mut [i32],
        col: &mut Vec<i8>,
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let cout = self.cout;
        let patch = self.geom.patch(self.cin);
        check(x.len(), n * h * w * self.cin, "输入")?;
        check(weight.len(), patch * cout, "权重")?;
        check(bias.len(), cout, "偏置")?;
        check(out.len(), n * oh * ow * cout, "输出")?;

        let rows = n * oh * ow;
        col.resize(rows * patch, 0);
        im2col2d_i8(x, col, n, (h, w), self, (oh, ow));
        fast::gemm_i8(col, weight, rows, cout, patch, out);
        for r in 0..rows {
            for c in 0..cout {
                out[r * cout + c] += bias[c];
            }
        }
        Ok(())
    }

    /// i8 卷积：`col` 内部现分配，单次调用；批量场景用 [`Self::forward_i8_with`] 复用缓冲。
    pub fn forward_i8(
        &self,
        x: &[i8],
        weight: &[i8],
        bias: &[i32],
        n: usize,
        h: usize,
        w: usize,
        out: &mut [i32],
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let cin = self.cin;
        let cout = self.cout;
        let patch = self.geom.patch(cin);
        check(x.len(), n * h * w * cin, "输入")?;
        check(weight.len(), patch * cout, "权重")?;
        check(bias.len(), cout, "偏置")?;
        check(out.len(), n * oh * ow * cout, "输出")?;

        thread_local! {
            /// 大形状下 im2col 缓冲达几十 MB，跨调用复用可以省掉每次的零页分配与缺页。
            static COL: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        COL.with(|cell| {
            let mut loan = cell.try_borrow_mut();
            match loan.as_mut() {
                Ok(col) => self.forward_i8_with(x, weight, bias, n, h, w, out, col),
                // 同线程嵌套调用（借出中）时才退回一次性缓冲。
                Err(_) => self.forward_i8_with(x, weight, bias, n, h, w, out, &mut Vec::new()),
            }
        })
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
        Ok((self.geom.oh(h)?, self.geom.ow(w)?))
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
        let (oh, ow) = self.shape(h, w)?;
        check(x.len(), n * h * w * c, "输入")?;
        check(out.len(), n * oh * ow * c, "输出")?;
        for ni in 0..n {
            for oi in 0..oh {
                for oj in 0..ow {
                    for ci in 0..c {
                        let mut best = S::NEG_INFINITY;
                        for ki in 0..self.geom.kh {
                            let ii = (oi * self.geom.sh + ki) as isize - self.geom.ph as isize;
                            if ii < 0 || ii as usize >= h {
                                continue;
                            }
                            for kj in 0..self.geom.kw {
                                let jj = (oj * self.geom.sw + kj) as isize - self.geom.pw as isize;
                                if jj < 0 || jj as usize >= w {
                                    continue;
                                }
                                let v = x[((ni * h + ii as usize) * w + jj as usize) * c + ci];
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        out[((ni * oh + oi) * ow + oj) * c + ci] = best;
                    }
                }
            }
        }
        Ok(())
    }
}

impl AvgPool2d {
    /// 输出形状。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        Ok((self.geom.oh(h)?, self.geom.ow(w)?))
    }

    /// 前向：窗口求和除窗口面积；`count_include_pad=false` 时只除有效位置。
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
        let (oh, ow) = self.shape(h, w)?;
        check(x.len(), n * h * w * c, "输入")?;
        check(out.len(), n * oh * ow * c, "输出")?;
        for ni in 0..n {
            for oi in 0..oh {
                for oj in 0..ow {
                    for ci in 0..c {
                        let mut sum = S::ZERO;
                        let mut count = 0usize;
                        for ki in 0..self.geom.kh {
                            let ii = (oi * self.geom.sh + ki) as isize - self.geom.ph as isize;
                            if ii < 0 || ii as usize >= h {
                                continue;
                            }
                            for kj in 0..self.geom.kw {
                                let jj = (oj * self.geom.sw + kj) as isize - self.geom.pw as isize;
                                if jj < 0 || jj as usize >= w {
                                    continue;
                                }
                                sum = sum + x[((ni * h + ii as usize) * w + jj as usize) * c + ci];
                                count += 1;
                            }
                        }
                        let denom = if self.count_include_pad {
                            S::from_usize(self.geom.kh * self.geom.kw)
                        } else {
                            S::from_usize(count.max(1))
                        };
                        out[((ni * oh + oi) * ow + oj) * c + ci] = sum / denom;
                    }
                }
            }
        }
        Ok(())
    }
}

/// f32 im2col：行 = `(n, oh, ow)`，列 = `(kh, kw, cin)`。
///
/// 行块之间互不重叠，直接按行切给线程；小矩阵不并行（一次 `scope` 的固定开销会超过拷贝本身）。
fn im2col2d(
    x: &[f32],
    col: &mut [f32],
    n: usize,
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let patch = conv.geom.patch(conv.cin);
    let rows = n * oh * ow;
    let threads = if rows * patch < (1 << 16) {
        1
    } else {
        thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
            .min(rows)
    };
    if threads <= 1 {
        for r in 0..rows {
            fill_row(
                x,
                &mut col[r * patch..(r + 1) * patch],
                r,
                n,
                (h, w),
                conv,
                (oh, ow),
            );
        }
        return;
    }
    let per = rows.div_ceil(threads);
    thread::scope(|s| {
        for (bi, chunk) in col.chunks_mut(per * patch).enumerate() {
            let r0 = bi * per;
            s.spawn(move || {
                let cnt = chunk.len() / patch;
                for base in 0..cnt {
                    let row_idx = r0 + base;
                    fill_row(
                        x,
                        &mut chunk[base * patch..(base + 1) * patch],
                        row_idx,
                        n,
                        (h, w),
                        conv,
                        (oh, ow),
                    );
                }
            });
        }
    });
}

/// 展开一行：`row_idx = (ni, oi, oj)` 对应的一整块 `(kh, kw, cin)` 拷贝。
#[allow(clippy::too_many_arguments)]
fn fill_row(
    x: &[f32],
    row: &mut [f32],
    row_idx: usize,
    n: usize,
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
    let _ = n;
    for ki in 0..g.kh {
        let ii = (oi * g.sh + ki) as isize - g.ph as isize;
        for kj in 0..g.kw {
            let jj = (oj * g.sw + kj) as isize - g.pw as isize;
            let tail = (ki * g.kw + kj) * cin;
            if ii < 0 || ii as usize >= h || jj < 0 || jj as usize >= w {
                row[tail..tail + cin].fill(0.0);
            } else {
                let src = ((ni * h + ii as usize) * w + jj as usize) * cin;
                row[tail..tail + cin].copy_from_slice(&x[src..src + cin]);
            }
        }
    }
}

/// i8 im2col：口径与 [`im2col2d`] 一致，补零位置为 `i8` 的 0，同样按行块并行。
/// `im2col` 的逆向：把每行的收缩梯度按同一窗口位置累加回输入（`stride < k` 的重叠位也要累加）。
fn col2im2d(
    col: &[f32],
    dx: &mut [f32],
    n: usize,
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let patch = conv.geom.patch(conv.cin);
    let cin = conv.cin;
    let g = conv.geom;
    let _ = n;
    for row in 0..oh * ow * (col.len() / (oh * ow * patch)) {
        let ni = row / (oh * ow);
        let rem = row % (oh * ow);
        let oi = rem / ow;
        let oj = rem % ow;
        for ki in 0..g.kh {
            let ii = (oi * g.sh + ki) as isize - g.ph as isize;
            if ii < 0 || ii as usize >= h {
                continue;
            }
            for kj in 0..g.kw {
                let jj = (oj * g.sw + kj) as isize - g.pw as isize;
                if jj < 0 || jj as usize >= w {
                    continue;
                }
                let tail = (ki * g.kw + kj) * cin;
                let dst = ((ni * h + ii as usize) * w + jj as usize) * cin;
                let src = row * patch + tail;
                for ci in 0..cin {
                    dx[dst + ci] += col[src + ci];
                }
            }
        }
    }
}

fn im2col2d_i8(
    x: &[i8],
    col: &mut [i8],
    n: usize,
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let patch = conv.geom.patch(conv.cin);
    let rows = n * oh * ow;
    let threads = if rows * patch < (1 << 16) {
        1
    } else {
        thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
            .min(rows)
    };
    if threads <= 1 {
        for r in 0..rows {
            fill_row_i8(
                x,
                &mut col[r * patch..(r + 1) * patch],
                r,
                (h, w),
                conv,
                (oh, ow),
            );
        }
        return;
    }
    let per = rows.div_ceil(threads);
    thread::scope(|s| {
        for (bi, chunk) in col.chunks_mut(per * patch).enumerate() {
            let r0 = bi * per;
            s.spawn(move || {
                let cnt = chunk.len() / patch;
                for base in 0..cnt {
                    let row_idx = r0 + base;
                    fill_row_i8(
                        x,
                        &mut chunk[base * patch..(base + 1) * patch],
                        row_idx,
                        (h, w),
                        conv,
                        (oh, ow),
                    );
                }
            });
        }
    });
}

/// i8 版单行展开。
fn fill_row_i8(
    x: &[i8],
    row: &mut [i8],
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
            if ii < 0 || ii as usize >= h || jj < 0 || jj as usize >= w {
                row[tail..tail + cin].fill(0);
            } else {
                let src = ((ni * h + ii as usize) * w + jj as usize) * cin;
                row[tail..tail + cin].copy_from_slice(&x[src..src + cin]);
            }
        }
    }
}

/// f32 GEMM 转调：`out[m,n] = a[m,k] · b[k,n]`，走 `fast` 的多线程内核。
fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    fast::gemm_f32(a, b, m, n, k, out);
}

/// 每行加同一个偏置向量。
fn add_bias(out: &mut [f32], bias: &[f32], cout: usize) {
    for row in out.chunks_mut(cout) {
        for (slot, b) in row.iter_mut().zip(bias.iter()) {
            *slot += *b;
        }
    }
}

/// 缓冲长度校验。
fn check(actual: usize, required: usize, what: &str) -> Result<()> {
    if actual < required {
        return Err(Error::new(
            Code::Invalid,
            format!("{what}: 缓冲 {actual}，需要 {required}"),
        ));
    }
    Ok(())
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
        check(x.len(), n * l * c, "输入")?;
        check(out.len(), n * ol * c, "输出")?;
        for ni in 0..n {
            for oi in 0..ol {
                let base = oi * self.s;
                for ci in 0..c {
                    let mut best = S::NEG_INFINITY;
                    for ki in 0..self.k {
                        let src = base + ki;
                        if src < self.p || src - self.p >= l {
                            continue;
                        }
                        let v = x[(ni * l + src - self.p) * c + ci];
                        if v > best {
                            best = v;
                        }
                    }
                    // 整窗都落在补零上（只会在极端几何下出现）按 0 处理。
                    out[(ni * ol + oi) * c + ci] = if best == S::NEG_INFINITY {
                        S::ZERO
                    } else {
                        best
                    };
                }
            }
        }
        Ok(())
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
        check(x.len(), n * h * w * c, "输入")?;
        check(out.len(), n * self.oh * self.ow * c, "输出")?;
        for ni in 0..n {
            for oi in 0..self.oh {
                let h0 = oi * h / self.oh;
                let h1 = ((oi + 1) * h).div_ceil(self.oh).min(h);
                for oj in 0..self.ow {
                    let w0 = oj * w / self.ow;
                    let w1 = ((oj + 1) * w).div_ceil(self.ow).min(w);
                    let area = S::from_usize((h1 - h0) * (w1 - w0));
                    for ci in 0..c {
                        let mut sum = S::ZERO;
                        for ii in h0..h1 {
                            for jj in w0..w1 {
                                sum = sum + x[((ni * h + ii) * w + jj) * c + ci];
                            }
                        }
                        out[((ni * self.oh + oi) * self.ow + oj) * c + ci] = sum / area;
                    }
                }
            }
        }
        Ok(())
    }
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
        if self.oh == 0 || self.ow == 0 {
            return Err(Error::new(Code::Invalid, "插值目标尺寸必须大于 0"));
        }
        if h == 0 || w == 0 {
            return Err(Error::new(Code::Invalid, "插值输入尺寸必须大于 0"));
        }
        check(x.len(), n * h * w * c, "输入")?;
        check(out.len(), n * self.oh * self.ow * c, "输出")?;
        match self.sampler {
            Sampler::Nearest => {
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
            Sampler::Linear => {
                for ni in 0..n {
                    for oi in 0..self.oh {
                        let fy = ((S::from_usize(oi) + S::from_f64(0.5)) * S::from_usize(h)
                            / S::from_usize(self.oh)
                            - S::from_f64(0.5))
                        .max(S::ZERO);
                        let y0 = fy.floor().min(S::from_usize(h - 1)).to_usize();
                        let y1 = (y0 + 1).min(h - 1);
                        let ly = if y0 == y1 {
                            S::ZERO
                        } else {
                            fy - S::from_usize(y0)
                        };
                        for oj in 0..self.ow {
                            let fx = ((S::from_usize(oj) + S::from_f64(0.5)) * S::from_usize(w)
                                / S::from_usize(self.ow)
                                - S::from_f64(0.5))
                            .max(S::ZERO);
                            let x0 = fx.floor().min(S::from_usize(w - 1)).to_usize();
                            let x1 = (x0 + 1).min(w - 1);
                            let lx = if x0 == x1 {
                                S::ZERO
                            } else {
                                fx - S::from_usize(x0)
                            };
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x >> 9) as f32) / 512.0 - 4.0
            })
            .collect()
    }

    fn sample_i8(len: usize, salt: u32) -> Vec<i8> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x % 31) as i32 - 15) as i8
            })
            .collect()
    }

    /// 朴素二维卷积：四重循环直算，作为唯一参考口径。
    fn naive2d(
        x: &[f32],
        wgt: &[f32],
        bias: &[f32],
        conv: &Conv2d,
        n: usize,
        h: usize,
        ww: usize,
        out: &mut [f32],
    ) {
        let (oh, ow) = conv.shape(h, ww).unwrap();
        let cin = conv.cin;
        let cout = conv.cout;
        let g = conv.geom;
        for ni in 0..n {
            for oi in 0..oh {
                for oj in 0..ow {
                    for co in 0..cout {
                        let mut acc = bias[co];
                        for ki in 0..g.kh {
                            let ii = (oi * g.sh + ki) as isize - g.ph as isize;
                            if ii < 0 || ii as usize >= h {
                                continue;
                            }
                            for kj in 0..g.kw {
                                let jj = (oj * g.sw + kj) as isize - g.pw as isize;
                                if jj < 0 || jj as usize >= ww {
                                    continue;
                                }
                                for ci in 0..cin {
                                    let xv =
                                        x[((ni * h + ii as usize) * ww + jj as usize) * cin + ci];
                                    let wv = wgt[((ki * g.kw + kj) * cin + ci) * cout + co];
                                    acc += xv * wv;
                                }
                            }
                        }
                        out[((ni * oh + oi) * ow + oj) * cout + co] = acc;
                    }
                }
            }
        }
    }

    fn close(a: &[f32], b: &[f32]) {
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let tol = 1e-4 + 1e-3 * y.abs();
            assert!((x - y).abs() <= tol, "第 {i} 个元素 {x} vs {y}");
        }
    }

    #[test]
    fn conv2d_matches_naive() {
        let cases = [
            (
                (1, 8, 8, 3, 4),
                ConvGeom {
                    kh: 3,
                    kw: 3,
                    sh: 1,
                    sw: 1,
                    ph: 1,
                    pw: 1,
                },
            ),
            (
                (2, 7, 9, 2, 5),
                ConvGeom {
                    kh: 3,
                    kw: 3,
                    sh: 2,
                    sw: 2,
                    ph: 0,
                    pw: 0,
                },
            ),
            (
                (1, 5, 5, 1, 1),
                ConvGeom {
                    kh: 5,
                    kw: 5,
                    sh: 1,
                    sw: 1,
                    ph: 0,
                    pw: 0,
                },
            ),
            (
                (1, 6, 6, 4, 3),
                ConvGeom {
                    kh: 2,
                    kw: 4,
                    sh: 3,
                    sw: 1,
                    ph: 2,
                    pw: 0,
                },
            ),
        ];
        for ((n, h, w, cin, cout), geom) in cases {
            let conv = Conv2d::new(cin, cout, geom);
            let (oh, ow) = conv.shape(h, w).unwrap();
            let x = sample(n * h * w * cin, 1);
            let wgt = sample(geom.patch(cin) * cout, 2);
            let bias = sample(cout, 3);
            let mut got = vec![0.0f32; n * oh * ow * cout];
            let mut want = vec![0.0f32; n * oh * ow * cout];
            conv.forward_f32(&x, &wgt, &bias, n, h, w, &mut got)
                .unwrap();
            naive2d(&x, &wgt, &bias, &conv, n, h, w, &mut want);
            close(&got, &want);
        }
    }

    #[test]
    fn conv1d_matches_naive() {
        let (n, l, cin, cout, kw, stride, pad) = (2, 9, 3, 4, 3, 2, 1);
        let conv = Conv1d::new(cin, cout, kw, stride, pad);
        let ol = conv.len(l).unwrap();
        let x = sample(n * l * cin, 1);
        let wgt = sample(kw * cin * cout, 2);
        let bias = sample(cout, 3);
        let mut got = vec![0.0f32; n * ol * cout];
        conv.forward_f32(&x, &wgt, &bias, n, l, &mut got).unwrap();

        // 参考：1d 就是 `kh=1` 的 2d。
        let conv2 = Conv2d::new(cin, cout, conv.geom);
        let mut want = vec![0.0f32; n * ol * cout];
        naive2d(&x, &wgt, &bias, &conv2, n, 1, l, &mut want);
        close(&got, &want);
    }

    #[test]
    fn conv2d_i8_matches_exact() {
        let (n, h, w, cin, cout) = (1, 6, 6, 3, 5);
        let geom = ConvGeom {
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
        };
        let conv = Conv2d::new(cin, cout, geom);
        let (oh, ow) = conv.shape(h, w).unwrap();
        let x = sample_i8(n * h * w * cin, 1);
        let wgt = sample_i8(geom.patch(cin) * cout, 2);
        let bias = vec![0i32; cout];
        let mut got = vec![0i32; n * oh * ow * cout];
        conv.forward_i8(&x, &wgt, &bias, n, h, w, &mut got).unwrap();

        let mut want = vec![0i32; n * oh * ow * cout];
        for ni in 0..n {
            for oi in 0..oh {
                for oj in 0..ow {
                    for co in 0..cout {
                        let mut acc = 0i32;
                        for ki in 0..geom.kh {
                            let ii = (oi * geom.sh + ki) as isize - geom.ph as isize;
                            if ii < 0 || ii as usize >= h {
                                continue;
                            }
                            for kj in 0..geom.kw {
                                let jj = (oj * geom.sw + kj) as isize - geom.pw as isize;
                                if jj < 0 || jj as usize >= w {
                                    continue;
                                }
                                for ci in 0..cin {
                                    let xv =
                                        x[((ni * h + ii as usize) * w + jj as usize) * cin + ci];
                                    let wv = wgt[((ki * geom.kw + kj) * cin + ci) * cout + co];
                                    acc += i32::from(xv) * i32::from(wv);
                                }
                            }
                        }
                        want[((ni * oh + oi) * ow + oj) * cout + co] = acc;
                    }
                }
            }
        }
        assert_eq!(got, want, "量化卷积与整数直算逐位相同");
    }

    #[test]
    fn max_pool_matches_naive() {
        let pool = MaxPool2d {
            geom: ConvGeom {
                kh: 2,
                kw: 2,
                sh: 2,
                sw: 2,
                ph: 0,
                pw: 0,
            },
        };
        let (n, h, w, c) = (1, 4, 4, 2);
        let (oh, ow) = pool.shape(h, w).unwrap();
        let x = sample(n * h * w * c, 1);
        let mut got = vec![0.0f32; n * oh * ow * c];
        pool.forward(&x, n, h, w, c, &mut got).unwrap();
        let mut want = vec![0.0f32; n * oh * ow * c];
        for ni in 0..n {
            for oi in 0..oh {
                for oj in 0..ow {
                    for ci in 0..c {
                        let mut best = f32::NEG_INFINITY;
                        for ki in 0..2 {
                            for kj in 0..2 {
                                let v = x[((ni * h + oi * 2 + ki) * w + oj * 2 + kj) * c + ci];
                                best = best.max(v);
                            }
                        }
                        want[((ni * oh + oi) * ow + oj) * c + ci] = best;
                    }
                }
            }
        }
        close(&got, &want);
    }

    #[test]
    fn avg_pool_counts_only_real_cells() {
        // 3×3 输入、2×2..1 补零、步长 2 → 输出 2×2；左上窗口只有 (0,0) 一个真实单元。
        let pool = AvgPool2d {
            geom: ConvGeom {
                kh: 2,
                kw: 2,
                sh: 2,
                sw: 2,
                ph: 1,
                pw: 1,
            },
            count_include_pad: false,
        };
        let (n, h, w, c) = (1, 3, 3, 1);
        let (oh, ow) = pool.shape(h, w).unwrap();
        let x = sample(n * h * w * c, 1);
        let mut got = vec![0.0f32; n * oh * ow * c];
        pool.forward(&x, n, h, w, c, &mut got).unwrap();
        assert_eq!((oh, ow), (2, 2));
        assert!((got[0] - x[0]).abs() < 1e-6, "剔除补零后只剩左上单元");
    }

    /// 卷积端到端吞吐（含 im2col + 分配）：`cargo test --release -p llm-op --lib -- --ignored --nocapture bench_conv`
    #[test]
    #[ignore = "卷积吞吐基线，手动跑"]
    fn bench_conv() {
        use std::time::Instant;
        let shapes = [
            (1usize, 128usize, 128usize, 256usize, 256usize, 3usize),
            (1, 64, 64, 256, 256, 3),
            (1, 32, 32, 512, 512, 3),
        ];
        for (n, h, w, cin, cout, k) in shapes {
            let geom = ConvGeom {
                kh: k,
                kw: k,
                sh: 1,
                sw: 1,
                ph: k / 2,
                pw: k / 2,
            };
            let conv = Conv2d::new(cin, cout, geom);
            let (oh, ow) = conv.shape(h, w).unwrap();
            let rows = n * oh * ow;
            let patch = geom.patch(cin);
            let flops = 2.0 * rows as f64 * cout as f64 * patch as f64;

            let x8 = sample_i8(n * h * w * cin, 1);
            let w8 = sample_i8(patch * cout, 2);
            let b8 = vec![0i32; cout];
            let mut o8 = vec![0i32; rows * cout];
            conv.forward_i8(&x8, &w8, &b8, n, h, w, &mut o8).unwrap();
            let mut best = f64::INFINITY;
            let mut bestcol = f64::INFINITY;
            let mut buf = Vec::new();
            for _ in 0..5 {
                let t = Instant::now();
                conv.forward_i8(&x8, &w8, &b8, n, h, w, &mut o8).unwrap();
                best = best.min(t.elapsed().as_nanos() as f64);
                let t = Instant::now();
                conv.forward_i8_with(&x8, &w8, &b8, n, h, w, &mut o8, &mut buf)
                    .unwrap();
                bestcol = bestcol.min(t.elapsed().as_nanos() as f64);
            }
            // 拆两段看谁在吃时间。
            let mut tcol = f64::INFINITY;
            let mut tgemm = f64::INFINITY;
            for col in 0..5 {
                let _ = col;
                let t = Instant::now();
                im2col2d_i8(&x8, &mut buf, n, (h, w), &conv, (oh, ow));
                tcol = tcol.min(t.elapsed().as_nanos() as f64);
                let t = Instant::now();
                fast::gemm_i8(&buf, &w8, rows, cout, patch, &mut o8);
                tgemm = tgemm.min(t.elapsed().as_nanos() as f64);
            }
            println!(
                "  conv2d_i8 {n}x{h}x{w} cin{cin} cout{cout} k{k}: 新分配 {:.1} / 复用 {:.1} GF/s \
                 (im2col {:.2} ms, gemm {:.2} ms)",
                flops / best,
                flops / bestcol,
                tcol / 1e6,
                tgemm / 1e6
            );

            let x32 = sample(n * h * w * cin, 1);
            let w32 = sample(patch * cout, 2);
            let b32 = sample(cout, 3);
            let mut o32 = vec![0.0f32; rows * cout];
            conv.forward_f32(&x32, &w32, &b32, n, h, w, &mut o32)
                .unwrap();
            let mut best32 = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                conv.forward_f32(&x32, &w32, &b32, n, h, w, &mut o32)
                    .unwrap();
                best32 = best32.min(t.elapsed().as_nanos() as f64);
            }
            println!(
                "  conv2d_f32 {n}x{h}x{w} cin{cin} cout{cout} k{k}: {:.1} GF/s",
                flops / best32
            );
        }
    }

    #[test]
    fn maxpool1d_matches_naive() {
        // NHWC，l=6、c=1，窗口 3 步长 2 补零 1 → 输出长 4。
        let x = [1.0f32, 5.0, 2.0, 8.0, 3.0, 0.5];
        let pool = MaxPool1d { k: 3, s: 2, p: 1 };
        assert_eq!(pool.len(6).unwrap(), 3);
        let mut out = [0.0f32; 3];
        pool.forward(&x, 1, 6, 1, &mut out).unwrap();
        // 窗口（输入坐标）：[-1,0,1]→5；[1,2,3]→8；[3,4,5]→8。
        assert_eq!(out, [5.0, 8.0, 8.0]);
        // 多通道：各通道独立，不串味。
        let x = [1.0f32, 9.0, 3.0, 2.0, 0.0, 4.0];
        let pool = MaxPool1d { k: 2, s: 1, p: 0 };
        let mut out = [0.0f32; 4];
        pool.forward(&x, 1, 3, 2, &mut out).unwrap();
        assert_eq!(out, [3.0, 9.0, 3.0, 4.0]);
        assert!(MaxPool1d { k: 0, s: 1, p: 0 }.len(4).is_err());
    }

    #[test]
    fn adaptive_avg_pool_matches_block_means() {
        // 4×4 单通道压到 2×2：每格正好 2×2 块。
        let x: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let pool = AdaptiveAvgPool2d { oh: 2, ow: 2 };
        let mut out = [0.0f32; 4];
        pool.forward(&x, 1, 4, 4, 1, &mut out).unwrap();
        assert_eq!(out, [2.5, 4.5, 10.5, 12.5]);
        // 5×5 压到 2×2：窗口按 floor/ceil 划分，末格覆盖 [2,5)×[2,5)。
        let x25: Vec<f32> = (0..25).map(|v| v as f32).collect();
        let pool = AdaptiveAvgPool2d { oh: 2, ow: 2 };
        let mut out = [0.0f32; 4];
        pool.forward(&x25, 1, 5, 5, 1, &mut out).unwrap();
        let mut want = [0.0f32; 4];
        for oi in 0usize..2 {
            let h0 = oi * 5 / 2;
            let h1 = ((oi + 1) * 5).div_ceil(2);
            for oj in 0usize..2 {
                let w0 = oj * 5 / 2;
                let w1 = ((oj + 1) * 5).div_ceil(2);
                let area = (h1 - h0) * (w1 - w0);
                let mut sum = 0.0f32;
                for ii in h0..h1 {
                    for jj in w0..w1 {
                        sum += (ii * 5 + jj) as f32;
                    }
                }
                want[oi * 2 + oj] = sum / area as f32;
            }
        }
        for (got, exp) in out.iter().zip(want.iter()) {
            assert!((got - exp).abs() < 1e-5, "{got} vs {exp}");
        }
        assert!(
            AdaptiveAvgPool2d { oh: 0, ow: 1 }
                .forward(&x, 1, 4, 4, 1, &mut out)
                .is_err()
        );
    }

    #[test]
    fn interpolate_nearest_and_bilinear() {
        // 2×2 单通道升到 4×4：最近邻严格复制左上角。
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let up = Interpolate {
            sampler: Sampler::Nearest,
            oh: 4,
            ow: 4,
        };
        let mut out = [0.0f32; 16];
        up.forward(&x, 1, 2, 2, 1, &mut out).unwrap();
        for oi in 0..4 {
            for oj in 0..4 {
                assert_eq!(out[oi * 4 + oj], x[(oi * 2 / 4) * 2 + (oj * 2 / 4)]);
            }
        }
        // 双线性在 4×4 上：中心附近的点应是四邻的加权，边界被夹住。
        let up = Interpolate {
            sampler: Sampler::Linear,
            oh: 4,
            ow: 4,
        };
        up.forward(&x, 1, 2, 2, 1, &mut out).unwrap();
        // 左上角 (0,0)：源坐标 (−0.25,−0.25) → 夹到 (0,0) → 值 1.0。
        assert!((out[0] - 1.0).abs() < 1e-6);
        // 右下角 (3,3)：源坐标 (1.25,1.25) → 夹到 (1,1) → 值 4.0。
        assert!((out[15] - 4.0).abs() < 1e-6);
        // 内部 (1,1)：源 (0.25,0.25) → 0.25/0.25 加权四邻 = 1.75。
        assert!((out[5] - 1.75).abs() < 1e-6, "{}", out[5]);
        // 恒等（同尺寸）时两种取样都必须逐元素一致。
        let id = Interpolate {
            sampler: Sampler::Linear,
            oh: 2,
            ow: 2,
        };
        let mut same = [0.0f32; 4];
        id.forward(&x, 1, 2, 2, 1, &mut same).unwrap();
        assert_eq!(same, x);
        assert!(
            Interpolate {
                sampler: Sampler::Nearest,
                oh: 0,
                ow: 2
            }
            .forward(&x, 1, 2, 2, 1, &mut out)
            .is_err()
        );
    }
}
