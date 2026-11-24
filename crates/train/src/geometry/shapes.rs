//! 形状面：实例几何的输入/输出长度核对与窗口几何、归一化反向公共骨架。

use super::Geometry;
use crate::geometry::EPS;
use diagnostic::{Code, Error, Result};
use nn::conv::ConvGeom;

// ---- 形状面：核对、窗口几何与共享工具 ----------------------------------------

/// 单次几何分派：每个分支只投影配置，共享函数拥有长度判据。
pub(crate) fn check_geometry(g: Geometry, lens: &[usize], out: usize) -> Result<()> {
    match g {
        Geometry::Pool1d { k, s, p, l, c } => check_pool1d(lens, out, k, s, p, l, c),
        Geometry::Pool2d { k, s, p, h, w, c } => {
            check_pool2d(lens, out, Window { k, s, p }, (h, w, c), "二维池化")
        }
        Geometry::Avg2d { k, s, p, h, w, c, .. } => {
            check_pool2d(lens, out, Window { k, s, p }, (h, w, c), "二维平均池化")
        }
        Geometry::Adaptive { oh, ow, h, w, c } => {
            check_plane(lens, out, (oh, ow), (h, w, c), "自适应池化")
        }
        Geometry::Interp { oh, ow, h, w, c, .. } => {
            check_plane(lens, out, (oh, ow), (h, w, c), "插值")
        }
        Geometry::Norm { n, spatial, channels } => {
            check_normalization(lens, out, n * spatial * channels, channels, "批归一化")
        }
        Geometry::Group { groups, n, spatial, channels } => {
            check_normalization(lens, out, n * spatial * channels, channels, "组归一化")?;
            ensure_shape(groups > 0 && channels.is_multiple_of(groups), lens, out, "组归一化")
        }
        Geometry::Layer { rows, width } => {
            check_normalization(lens, out, rows * width, width, "层归一化")
        }
        Geometry::Conv2d { k, s, p, h, w, cin, cout } => {
            check_conv2d_shape(lens, out, Window { k, s, p }, (h, w, cin, cout))
        }
    }
}

/// 一维池化：输入 `n·l·c`、输出 `n·ol·c`。
fn check_pool1d(
    lens: &[usize], out: usize, k: usize, s: usize, p: usize, l: usize, c: usize,
) -> Result<()> {
    let per = l.checked_mul(c).filter(|&v| v > 0).ok_or_else(|| invalid("池化输入为空"))?;
    let want = lens[0] / per * pool_len(l, k, s, p)? * c;
    ensure_shape(lens[0] == lens[0] / per * per && out == want, lens, out, "一维池化")
}

/// 二维池化：先推输出窗口，再走公共平面判据。
fn check_pool2d(
    lens: &[usize], out: usize, win: Window, plane: (usize, usize, usize), what: &str,
) -> Result<()> {
    let output = pool2d_out(win, plane.0, plane.1)?;
    check_plane(lens, out, output, plane, what)
}

/// 二维空间算子：输入 `n·h·w·c`、输出 `n·oh·ow·c`，拒绝零平面与除零。
fn check_plane(
    lens: &[usize], out: usize, output: (usize, usize), plane: (usize, usize, usize), what: &str,
) -> Result<()> {
    let (oh, ow) = output;
    let (h, w, c) = plane;
    let per = h.checked_mul(w).and_then(|v| v.checked_mul(c)).filter(|&v| v > 0)
        .ok_or_else(|| invalid(format!("{what}输入平面为空")))?;
    let n = lens[0] / per;
    ensure_shape(lens[0] == n * per && out == n * oh * ow * c, lens, out, what)
}

/// 归一化统一契约：`(x, gamma, beta, out) = (total, width, width, total)`。
fn check_normalization(
    lens: &[usize], out: usize, total: usize, width: usize, what: &str,
) -> Result<()> {
    ensure_shape(
        [lens[0], lens[1], lens[2], out] == [total, width, width, total],
        lens, out, what,
    )
}

/// 二维卷积的形状核对：激活/权重/偏置/输出四者长度对齐。
pub(crate) fn check_conv2d_shape(
    lens: &[usize],
    out: usize,
    win: Window,
    shape: (usize, usize, usize, usize),
) -> Result<()> {
    let (h, w, cin, cout) = shape;
    let (oh, ow) = pool2d_out(win, h, w)?;
    let patch = win.k * win.k * cin;
    let io_ok = [lens[0], lens[1], lens[2]] == [h * w * cin, patch * cout, cout];
    ensure_shape(io_ok && out == oh * ow * cout, lens, out, "二维卷积")
}

/// 方形窗口的 `ConvGeom`。
pub(crate) fn square(k: usize, s: usize, p: usize) -> ConvGeom {
    ConvGeom {
        kh: k,
        kw: k,
        sh: s,
        sw: s,
        ph: p,
        pw: p,
    }
}

/// 池化输出长度：`(len + 2p − k)/s + 1`，窗口超过输入即拒绝（不静默出 0）。
pub(crate) fn pool_len(len: usize, k: usize, s: usize, p: usize) -> Result<usize> {
    if k == 0 || s == 0 || len + 2 * p < k {
        return Err(invalid(format!(
            "池化几何不成立：len={len}, k={k}, s={s}, p={p}"
        )));
    }
    Ok((len + 2 * p - k) / s + 1)
}

/// 池化输出边长：`(len + 2p − k)/s + 1`，两轴一起求，供核对与反向复用。
pub(crate) fn pool2d_out(win: Window, h: usize, w: usize) -> Result<(usize, usize)> {
    Ok((
        pool_len(h, win.k, win.s, win.p)?,
        pool_len(w, win.k, win.s, win.p)?,
    ))
}

/// 输出位在第 `axis` 维上的有效窗口（闭区间）。
pub(crate) fn bounds(out: usize, s: usize, p: usize, k: usize, len: usize) -> (usize, usize) {
    let lo = (out * s).saturating_sub(p);
    let hi = (out * s + k - 1 - p).min(len - 1);
    (lo.min(len - 1), hi)
}

/// 方形窗口参数：`k/s/p` 三个数一起走，避免每个调用点再摊成三个实参。
#[derive(Clone, Copy)]
pub(crate) struct Window {
    pub(crate) k: usize,
    pub(crate) s: usize,
    pub(crate) p: usize,
}

/// 一维最大池化窗口里的最大值位置（与前向同一判定：严格更大才替换，平手取先遇到的）。
pub(crate) fn window1d(
    x: &[f32],
    base: usize,
    l: usize,
    oi: usize,
    win: Window,
    ci: usize,
) -> usize {
    let mut best = f32::MIN;
    let mut pick = base + ci;
    for ki in 0..win.k {
        let ii = oi as i64 * win.s as i64 - win.p as i64 + ki as i64;
        if !(0..l as i64).contains(&ii) {
            continue;
        }
        let at = base + ii as usize + ci;
        if x[at] > best {
            best = x[at];
            pick = at;
        }
    }
    pick
}

/// 二维最大池化窗口里的最大值位置；`plane` 是 `(h, w, c)` 三轴，`base` 是本样本起点。
pub(crate) fn window2d(
    x: &[f32],
    base: usize,
    plane: (usize, usize, usize),
    at: (usize, usize),
    win: Window,
    ci: usize,
) -> usize {
    let (h, w, c) = plane;
    let mut best = f32::MIN;
    let mut pick = base + ci;
    for q in 0..win.k * win.k {
        let di = q / win.k;
        let dj = q % win.k;
        let ii = at.0 as i64 * win.s as i64 - win.p as i64 + di as i64;
        if !(0..h as i64).contains(&ii) {
            continue;
        }
        let jj = at.1 as i64 * win.s as i64 - win.p as i64 + dj as i64;
        if !(0..w as i64).contains(&jj) {
            continue;
        }
        let slot = base + (ii as usize * w + jj as usize) * c + ci;
        if x[slot] > best {
            best = x[slot];
            pick = slot;
        }
    }
    pick
}

/// 归一化反向的公共骨架：对一组元素按 `(均值, 方差)` 求 `dx`，并把 `dgamma`/`dbeta` 累加进
/// 对应通道。`members` 给出该组元素下标，`channel` 把下标映射回通道（层归一化映射回列）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn norm_group(
    x: &[f32],
    gamma: &[f32],
    incoming: &[f32],
    members: &[usize],
    channel: impl Fn(usize) -> usize,
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    let count = members.len() as f32;
    let (mut sum, mut sq) = (0.0f32, 0.0f32);
    for &idx in members {
        sum += x[idx];
        sq += x[idx] * x[idx];
    }
    let mean = sum / count;
    let var = (sq / count - mean * mean).max(0.0);
    let inv = 1.0 / (var + EPS).sqrt();

    let (mut sum_dxhat, mut sum_dxhat_xhat) = (0.0f32, 0.0f32);
    for &idx in members {
        let col = channel(idx);
        let xhat = (x[idx] - mean) * inv;
        let dxhat = incoming[idx] * gamma[col];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
        // 对 gamma 的偏导只经 xhat 一条路：dxhat 已乘过 gamma，不能再乘第二次。
        dgamma[col] += incoming[idx] * xhat;
        dbeta[col] += incoming[idx];
    }
    for &idx in members {
        let col = channel(idx);
        let xhat = (x[idx] - mean) * inv;
        let dxhat = incoming[idx] * gamma[col];
        dx[idx] = inv * (dxhat - sum_dxhat / count - xhat * sum_dxhat_xhat / count);
    }
}

/// 几何不符的统一报错。
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into())
}

/// 分派闭环外的兜底：几何变体不该进入这一级路由。
pub(crate) fn internal(message: impl Into<String>) -> Error {
    Error::new(Code::Internal, message.into())
}

/// 形状核对：不符时把实际值带上，避免只报一句"形状不对"。
pub(crate) fn ensure_shape(ok: bool, lens: &[usize], out: usize, what: &str) -> Result<()> {
    if ok {
        return Ok(());
    }
    Err(invalid(format!(
        "{what} 的形状与实例几何不符：输入 {lens:?}、输出 {out} 个元素"
    ))
    .with("train::Geometry::check_shape"))
}
