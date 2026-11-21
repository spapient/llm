
use diagnostic::{Code, Error, Result};

use crate::Scalar;
use crate::fast;

use super::support::*;

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
