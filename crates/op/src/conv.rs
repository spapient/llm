//! 经典卷积与池化：NHWC 布局的 `Conv1d`/`Conv2d`、`MaxPool2d`/`AvgPool2d`。
//!
//! 卷积不写双重循环：先 `im2col` 把每个输出位置展开成一行，再交给 [`crate::fast`] 的
//! GEMM 微核。语义与逐点直算逐位对齐 f32 容差，性能走 SIMD 通道（i8 量化路径见
//! [`Conv2d::forward_i8`]，复用 `fast::gemm_i8` 的多槽池）。
//!
//! **拥有**：本模块的四个层结构。**不拥有**：GEMM 微核（`fast`）、张量元数据（`tensor`）。

use diagnostic::{Code, Error, Result};

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
            return Err(Error::new(Code::Invalid, format!("{axis}: 核与步长必须非零")));
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

    /// i8 前向（对称量化）：输入与权重已量化，输出 i32 累加，`bias` 为同尺度整数偏置。
    ///
    /// 量化卷积的全部算力都落在 `fast::gemm_i8` 的多槽池上；`bias` 传全 0 可省。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
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

        let rows = n * oh * ow;
        let mut col = vec![0i8; rows * patch];
        im2col2d_i8(x, &mut col, n, (h, w), self, (oh, ow));
        fast::gemm_i8(&col, weight, rows, cout, patch, out);
        for r in 0..rows {
            for c in 0..cout {
                out[r * cout + c] += bias[c];
            }
        }
        Ok(())
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
    pub fn forward(
        &self,
        x: &[f32],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        check(x.len(), n * h * w * c, "输入")?;
        check(out.len(), n * oh * ow * c, "输出")?;
        for ni in 0..n {
            for oi in 0..oh {
                for oj in 0..ow {
                    for ci in 0..c {
                        let mut best = f32::NEG_INFINITY;
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
    pub fn forward(
        &self,
        x: &[f32],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        check(x.len(), n * h * w * c, "输入")?;
        check(out.len(), n * oh * ow * c, "输出")?;
        for ni in 0..n {
            for oi in 0..oh {
                for oj in 0..ow {
                    for ci in 0..c {
                        let mut sum = 0.0f32;
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
                                sum += x[((ni * h + ii as usize) * w + jj as usize) * c + ci];
                                count += 1;
                            }
                        }
                        let denom = if self.count_include_pad {
                            (self.geom.kh * self.geom.kw) as f32
                        } else {
                            count.max(1) as f32
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
fn im2col2d(
    x: &[f32],
    col: &mut [f32],
    n: usize,
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let cin = conv.cin;
    let patch = conv.geom.patch(cin);
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                let row = &mut col[((ni * oh + oi) * ow + oj) * patch..][..patch];
                for ki in 0..conv.geom.kh {
                    let ii = (oi * conv.geom.sh + ki) as isize - conv.geom.ph as isize;
                    for kj in 0..conv.geom.kw {
                        let jj = (oj * conv.geom.sw + kj) as isize - conv.geom.pw as isize;
                        let tail = (ki * conv.geom.kw + kj) * cin;
                        if ii < 0 || ii as usize >= h || jj < 0 || jj as usize >= w {
                            row[tail..tail + cin].fill(0.0);
                        } else {
                            let src = ((ni * h + ii as usize) * w + jj as usize) * cin;
                            row[tail..tail + cin].copy_from_slice(&x[src..src + cin]);
                        }
                    }
                }
            }
        }
    }
}

/// i8 im2col：口径与 [`im2col2d`] 一致，补零位置为 `i8` 的 0。
fn im2col2d_i8(
    x: &[i8],
    col: &mut [i8],
    n: usize,
    (h, w): (usize, usize),
    conv: &Conv2d,
    (oh, ow): (usize, usize),
) {
    let cin = conv.cin;
    let patch = conv.geom.patch(cin);
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                let row = &mut col[((ni * oh + oi) * ow + oj) * patch..][..patch];
                for ki in 0..conv.geom.kh {
                    let ii = (oi * conv.geom.sh + ki) as isize - conv.geom.ph as isize;
                    for kj in 0..conv.geom.kw {
                        let jj = (oj * conv.geom.sw + kj) as isize - conv.geom.pw as isize;
                        let tail = (ki * conv.geom.kw + kj) * cin;
                        if ii < 0 || ii as usize >= h || jj < 0 || jj as usize >= w {
                            row[tail..tail + cin].fill(0);
                        } else {
                            let src = ((ni * h + ii as usize) * w + jj as usize) * cin;
                            row[tail..tail + cin].copy_from_slice(&x[src..src + cin]);
                        }
                    }
                }
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
                                    let xv = x[((ni * h + ii as usize) * ww + jj as usize) * cin + ci];
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
            ((1, 8, 8, 3, 4), ConvGeom { kh: 3, kw: 3, sh: 1, sw: 1, ph: 1, pw: 1 }),
            ((2, 7, 9, 2, 5), ConvGeom { kh: 3, kw: 3, sh: 2, sw: 2, ph: 0, pw: 0 }),
            ((1, 5, 5, 1, 1), ConvGeom { kh: 5, kw: 5, sh: 1, sw: 1, ph: 0, pw: 0 }),
            ((1, 6, 6, 4, 3), ConvGeom { kh: 2, kw: 4, sh: 3, sw: 1, ph: 2, pw: 0 }),
        ];
        for ((n, h, w, cin, cout), geom) in cases {
            let conv = Conv2d::new(cin, cout, geom);
            let (oh, ow) = conv.shape(h, w).unwrap();
            let x = sample(n * h * w * cin, 1);
            let wgt = sample(geom.patch(cin) * cout, 2);
            let bias = sample(cout, 3);
            let mut got = vec![0.0f32; n * oh * ow * cout];
            let mut want = vec![0.0f32; n * oh * ow * cout];
            conv.forward_f32(&x, &wgt, &bias, n, h, w, &mut got).unwrap();
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
        let geom = ConvGeom { kh: 3, kw: 3, sh: 1, sw: 1, ph: 1, pw: 1 };
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
                                    let xv = x[((ni * h + ii as usize) * w + jj as usize) * cin + ci];
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
        let pool = MaxPool2d { geom: ConvGeom { kh: 2, kw: 2, sh: 2, sw: 2, ph: 0, pw: 0 } };
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
    fn avg_pool_counts_only_real_cells() {        // 3×3 输入、2×2..1 补零、步长 2 → 输出 2×2；左上窗口只有 (0,0) 一个真实单元。
        let pool = AvgPool2d {
            geom: ConvGeom { kh: 2, kw: 2, sh: 2, sw: 2, ph: 1, pw: 1 },
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
            let geom = ConvGeom { kh: k, kw: k, sh: 1, sw: 1, ph: k / 2, pw: k / 2 };
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
            for _ in 0..5 {
                let t = Instant::now();
                conv.forward_i8(&x8, &w8, &b8, n, h, w, &mut o8).unwrap();
                best = best.min(t.elapsed().as_nanos() as f64);
            }
            println!(
                "  conv2d_i8 {n}x{h}x{w} cin{cin} cout{cout} k{k}: {:.1} GF/s",
                flops / best
            );

            let x32 = sample(n * h * w * cin, 1);
            let w32 = sample(patch * cout, 2);
            let b32 = sample(cout, 3);
            let mut o32 = vec![0.0f32; rows * cout];
            conv.forward_f32(&x32, &w32, &b32, n, h, w, &mut o32).unwrap();
            let mut best32 = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                conv.forward_f32(&x32, &w32, &b32, n, h, w, &mut o32).unwrap();
                best32 = best32.min(t.elapsed().as_nanos() as f64);
            }
            println!(
                "  conv2d_f32 {n}x{h}x{w} cin{cin} cout{cout} k{k}: {:.1} GF/s",
                flops / best32
            );
        }
    }
}
