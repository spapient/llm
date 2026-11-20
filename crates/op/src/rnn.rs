//! 经典循环层：`Rnn` / `Lstm` / `Gru` 的单步与整序列前向（f32，batch 优先布局）。
//!
//! 三个层共用同一条路径：`x·Wᵢₕ + h·Wₕₕ + b` 走 [`crate::fast::gemm_f32`]，再按门控激活。
//! 权重布局统一为 `[in, gates·hidden]` / `[hidden, gates·hidden]`，门顺序取 PyTorch 口径
//! （LSTM 为 `i,f,g,o`；GRU 为 `r,z,n`），`bias` 为已合并的 `bᵢₕ + bₕₕ`。

use diagnostic::{Code, Error, Result};

use crate::fast;

/// 单层 RNN：`h' = tanh(x·Wᵢₕ + h·Wₕₕ + b)`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rnn {
    /// 输入宽度。
    pub input: usize,
    /// 隐状态宽度。
    pub hidden: usize,
}

/// 单层 LSTM：四门 `i,f,g,o`，`c' = f⊙c + i⊙g`、`h' = o⊙tanh(c')`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lstm {
    /// 输入宽度。
    pub input: usize,
    /// 隐状态宽度。
    pub hidden: usize,
}

/// 单层 GRU：`r,z,n` 三门，`h' = (1-z)⊙n + z⊙h`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gru {
    /// 输入宽度。
    pub input: usize,
    /// 隐状态宽度。
    pub hidden: usize,
}

impl Rnn {
    /// 单步：`x` 为 `[batch, input]`，`h` 为 `[batch, hidden]`，写回 `out` 同形。
    ///
    /// # Errors
    ///
    /// 缓冲长度与 `(batch, input, hidden)` 不符时返回说明。
    pub fn step(
        &self,
        x: &[f32],
        w_ih: &[f32],
        w_hh: &[f32],
        b: &[f32],
        h: &[f32],
        batch: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let (i, hd) = (self.input, self.hidden);
        check_all(&[
            (x.len(), batch * i, "x"),
            (w_ih.len(), i * hd, "w_ih"),
            (w_hh.len(), hd * hd, "w_hh"),
            (b.len(), hd, "b"),
            (h.len(), batch * hd, "h"),
            (out.len(), batch * hd, "out"),
        ])?;
        linear(x, w_ih, h, w_hh, b, batch, i, hd, hd, out);
        tanh_in_place(out);
        Ok(())
    }

    /// 整序列：逐时间步推进，返回 `[seq, batch, hidden]`。
    ///
    /// # Errors
    ///
    /// 形状与缓冲不符时返回说明。
    pub fn run(
        &self,
        x: &[f32],
        w_ih: &[f32],
        w_hh: &[f32],
        b: &[f32],
        h0: &[f32],
        seq: usize,
        batch: usize,
    ) -> Result<Vec<f32>> {
        let hd = self.hidden;
        let mut h = h0.to_vec();
        let mut trace = vec![0.0f32; seq * batch * hd];
        let mut next = vec![0.0f32; batch * hd];
        for t in 0..seq {
            let xt = &x[t * batch * self.input..(t + 1) * batch * self.input];
            self.step(xt, w_ih, w_hh, b, &h, batch, &mut next)?;
            trace[t * batch * hd..(t + 1) * batch * hd].copy_from_slice(&next);
            h.copy_from_slice(&next);
        }
        Ok(trace)
    }
}

impl Lstm {
    /// 单步：`h`/`c` 均为 `[batch, hidden]`，`out_h`/`out_c` 写回。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        x: &[f32],
        w_ih: &[f32],
        w_hh: &[f32],
        b: &[f32],
        h: &[f32],
        c: &[f32],
        batch: usize,
        out_h: &mut [f32],
        out_c: &mut [f32],
    ) -> Result<()> {
        let (i, hd) = (self.input, self.hidden);
        let gates = 4 * hd;
        check_all(&[
            (x.len(), batch * i, "x"),
            (w_ih.len(), i * gates, "w_ih"),
            (w_hh.len(), hd * gates, "w_hh"),
            (b.len(), gates, "b"),
            (h.len(), batch * hd, "h"),
            (c.len(), batch * hd, "c"),
            (out_h.len(), batch * hd, "out_h"),
            (out_c.len(), batch * hd, "out_c"),
        ])?;

        let mut pre = vec![0.0f32; batch * gates];
        linear(x, w_ih, h, w_hh, b, batch, i, hd, gates, &mut pre);
        for bi in 0..batch {
            let row = &pre[bi * gates..(bi + 1) * gates];
            for j in 0..hd {
                let ig = sigmoid(row[j]);
                let fg = sigmoid(row[hd + j]);
                let gg = row[2 * hd + j].tanh();
                let og = sigmoid(row[3 * hd + j]);
                let cj = fg * c[bi * hd + j] + ig * gg;
                out_c[bi * hd + j] = cj;
                out_h[bi * hd + j] = og * cj.tanh();
            }
        }
        Ok(())
    }
}

impl Gru {
    /// 单步：`r` 作用于 `h·Wₕₙ`，`z` 在旧新状态间插值。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符时返回说明。
    pub fn step(
        &self,
        x: &[f32],
        w_ih: &[f32],
        w_hh: &[f32],
        b: &[f32],
        h: &[f32],
        batch: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let (i, hd) = (self.input, self.hidden);
        let gates = 3 * hd;
        check_all(&[
            (x.len(), batch * i, "x"),
            (w_ih.len(), i * gates, "w_ih"),
            (w_hh.len(), hd * gates, "w_hh"),
            (b.len(), gates, "b"),
            (h.len(), batch * hd, "h"),
            (out.len(), batch * hd, "out"),
        ])?;

        // x 侧三门一次算完；h 侧只需 r、n 两门（z 直接作用于最终插值）。
        let mut pre = vec![0.0f32; batch * gates];
        fast::gemm_f32(x, w_ih, batch, gates, i, &mut pre);
        add_bias(&mut pre, b, batch, gates);
        // h·Wₕₕ，取出 3·hidden 列后再逐列处理。
        let mut hpre = vec![0.0f32; batch * gates];
        fast::gemm_f32(h, w_hh, batch, gates, hd, &mut hpre);

        for bi in 0..batch {
            for j in 0..hd {
                let xr = pre[bi * gates + j];
                let xz = pre[bi * gates + hd + j];
                let xn = pre[bi * gates + 2 * hd + j];
                let hr = hpre[bi * gates + j];
                let hn = hpre[bi * gates + 2 * hd + j];
                let r = sigmoid(xr + hr);
                let z = sigmoid(xz + hpre[bi * gates + hd + j]);
                let n = (xn + r * hn).tanh();
                let hj = h[bi * hd + j];
                out[bi * hd + j] = (1.0 - z) * n + z * hj;
            }
        }
        Ok(())
    }
}

/// `pre[batch, gates] = x·w_ih + h·w_hh + b`，两步 GEMM 后加偏置。
#[allow(clippy::too_many_arguments)]
fn linear(
    x: &[f32],
    w_ih: &[f32],
    h: &[f32],
    w_hh: &[f32],
    b: &[f32],
    batch: usize,
    input: usize,
    hidden: usize,
    gates: usize,
    pre: &mut [f32],
) {
    fast::gemm_f32(x, w_ih, batch, gates, input, pre);
    let mut tmp = vec![0.0f32; batch * gates];
    fast::gemm_f32(h, w_hh, batch, gates, hidden, &mut tmp);
    for bi in 0..batch {
        for j in 0..gates {
            pre[bi * gates + j] += tmp[bi * gates + j] + b[j];
        }
    }
}

/// logistic sigmoid。
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// 就地把 `out` 逐元素过 `tanh`。
fn tanh_in_place(out: &mut [f32]) {
    for v in out.iter_mut() {
        *v = v.tanh();
    }
}

/// 把 `pre` 的每一行加上偏置 `b`。
fn add_bias(pre: &mut [f32], b: &[f32], batch: usize, gates: usize) {
    for bi in 0..batch {
        for j in 0..gates {
            pre[bi * gates + j] += b[j];
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

/// 一组 `(实际, 需要, 名称)` 长度约束，一次失败即短路；把逐条 `?` 收进一条出口。
fn check_all(constraints: &[(usize, usize, &str)]) -> Result<()> {
    for &(actual, required, what) in constraints {
        check(actual, required, what)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let z = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                // 值域压到 [−1, 1]：门控激活不饱和，才能验证"有界"而不是"恒为 1"。
                ((z >> 8) % 2048) as f32 / 1024.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn rnn_step_matches_hand_computation() {
        let rnn = Rnn {
            input: 2,
            hidden: 2,
        };
        // 单位权重、零偏：h' = tanh(x + h)，可手算。
        let w_ih = [1.0, 0.0, 0.0, 1.0];
        let w_hh = [1.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.0];
        let x = [0.5, -0.25];
        let h = [0.1, 0.2];
        let mut out = [0.0f32; 2];
        rnn.step(&x, &w_ih, &w_hh, &b, &h, 1, &mut out).unwrap();
        assert!((out[0] - (0.6f32).tanh()).abs() < 1e-6);
        assert!((out[1] - (-0.05f32).tanh()).abs() < 1e-6);
    }

    #[test]
    fn lstm_cell_keeps_state_and_is_bounded() {
        let lstm = Lstm {
            input: 2,
            hidden: 2,
        };
        let w_ih = sample(2 * 8, 1);
        let w_hh = sample(2 * 8, 2);
        let b = sample(8, 3);
        let x = sample(2, 4);
        let h = vec![0.0f32; 2];
        let c = vec![0.0f32; 2];
        let mut oh = vec![0.0f32; 2];
        let mut oc = vec![0.0f32; 2];
        lstm.step(&x, &w_ih, &w_hh, &b, &h, &c, 1, &mut oh, &mut oc)
            .unwrap();
        assert!(oh.iter().all(|v| v.abs() <= 1.0), "h 经 tanh 有界");
        assert!(oc.iter().all(|v| v.abs() < 1.0), "c = f⊙c+i⊙g 首步有界");
    }

    #[test]
    fn gru_step_is_convex_blend() {
        let gru = Gru {
            input: 2,
            hidden: 2,
        };
        let w_ih = vec![0.0f32; 2 * 6];
        let w_hh = vec![0.0f32; 2 * 6];
        let b = vec![0.0f32; 6];
        let x = vec![0.0f32; 2];
        let h = [0.3, -0.4];
        let mut out = [0.0f32; 2];
        gru.step(&x, &w_ih, &w_hh, &b, &h, 1, &mut out).unwrap();
        // 全零权重：z = 0.5、n = 0 → h' = 0.5·h。
        assert!((out[0] - 0.15).abs() < 1e-6);
        assert!((out[1] + 0.2).abs() < 1e-6);
    }

    #[test]
    fn rnn_run_returns_every_step() {
        let rnn = Rnn {
            input: 3,
            hidden: 4,
        };
        let w_ih = sample(12, 1);
        let w_hh = sample(16, 2);
        let b = sample(4, 3);
        let h0 = vec![0.0f32; 4];
        let x = sample(5 * 3, 4);
        let trace = rnn.run(&x, &w_ih, &w_hh, &b, &h0, 5, 1).unwrap();
        assert_eq!(trace.len(), 5 * 4);
    }
}
