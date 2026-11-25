//! 形状面：实例几何的长度核对、窗口几何与归一化反向公共骨架。
//! 这里集中处理零维、输入个数、窗口边界与归一化槽长，避免前向与恢复各写一套校验。
//! 校验固定先于任何计算或状态提交，错误同时保留算子类别与具体槽位。

use super::{EPS, Geometry};
use diagnostic::{Code, Error, Result};
use nn::conv::ConvGeom;

type Plane = (usize, usize, usize);
type Shape = (usize, usize, usize, usize);

macro_rules! route {
    ($(#[$meta:meta])* $name:ident, $g:ident, $lens:ident, $out:ident;
     $($pattern:pat => $call:expr),+ $(,)?) => {
        $(#[$meta])*
        pub(crate) fn $name($g: Geometry, $lens: &[usize], $out: usize) -> Result<()> {
            match $g { $($pattern => $call),+ }
        }
    };
    ($name:ident, $message:literal, $g:ident, $lens:ident, $out:ident;
     $($pattern:pat => $call:expr),+ $(,)?) => {
        #[doc = $message]
        pub(crate) fn $name($g: Geometry, $lens: &[usize], $out: usize) -> Result<()> {
            match $g {
                $($pattern => $call),+,
                _ => Err(internal($message)),
            }
        }
    };
}

route!(
    #[doc = "按几何族分派，每个小路由只处理自己的闭集。"]
    check_geometry, g, lens, out;
    Geometry::Pool1d { .. } | Geometry::Pool2d { .. } | Geometry::Avg2d { .. } => check_pool(g, lens, out),
    Geometry::Adaptive { .. } | Geometry::Interp { .. } => check_plane_geometry(g, lens, out),
    Geometry::Norm { .. } | Geometry::Group { .. } | Geometry::Layer { .. } => check_norm(g, lens, out),
    Geometry::Conv2d { k, s, p, h, w, cin, cout } => {
        check_conv2d_shape(lens, out, Window { k, s, p }, (h, w, cin, cout))
    }
);

route!(check_pool, "池化形状核对只接收池化几何", g, lens, out;
    Geometry::Pool1d { k, s, p, l, c } => check_plane(lens, out, (pool_len(l, k, s, p)?, 1), (l, 1, c), "一维池化"),
    Geometry::Pool2d { k, s, p, h, w, c } => check_plane(lens, out, pool2d_out(Window { k, s, p }, h, w)?, (h, w, c), "二维池化"),
    Geometry::Avg2d { k, s, p, h, w, c, .. } => check_plane(lens, out, pool2d_out(Window { k, s, p }, h, w)?, (h, w, c), "二维平均池化")
);

route!(check_plane_geometry, "平面形状核对只接收自适应池化或插值几何", g, lens, out;
    Geometry::Adaptive { oh, ow, h, w, c } => check_plane(lens, out, (oh, ow), (h, w, c), "自适应池化"),
    Geometry::Interp { oh, ow, h, w, c, .. } => check_plane(lens, out, (oh, ow), (h, w, c), "插值")
);

route!(check_norm, "归一化形状核对只接收归一化几何", g, lens, out;
    Geometry::Norm { n, spatial, channels } => check_normalization(lens, out, n * spatial * channels, channels, "批归一化"),
    Geometry::Group { groups, n, spatial, channels } => check_group(lens, out, groups, n * spatial * channels, channels),
    Geometry::Layer { rows, width } => check_normalization(lens, out, rows * width, width, "层归一化")
);

fn check_group(
    lens: &[usize],
    out: usize,
    groups: usize,
    total: usize,
    channels: usize,
) -> Result<()> {
    check_normalization(lens, out, total, channels, "组归一化")?;
    ensure_shape(
        groups > 0 && channels.is_multiple_of(groups),
        lens,
        out,
        "组归一化",
    )
}

/// 二维空间算子：输入 `n·h·w·c`、输出 `n·oh·ow·c`，拒绝零平面与除零。
fn check_plane(
    lens: &[usize],
    out: usize,
    (oh, ow): (usize, usize),
    (h, w, c): Plane,
    what: &str,
) -> Result<()> {
    let per = h
        .checked_mul(w)
        .and_then(|v| v.checked_mul(c))
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid(format!("{what}输入平面为空")))?;
    let n = lens[0] / per;
    ensure_shape(
        lens[0] == n * per && out == n * oh * ow * c,
        lens,
        out,
        what,
    )
}

/// 归一化统一契约：`(x, gamma, beta, out) = (total, width, width, total)`。
fn check_normalization(
    lens: &[usize],
    out: usize,
    total: usize,
    width: usize,
    what: &str,
) -> Result<()> {
    ensure_shape(
        [lens[0], lens[1], lens[2], out] == [total, width, width, total],
        lens,
        out,
        what,
    )
}

/// 二维卷积的形状核对：激活/权重/偏置/输出四者长度对齐。
pub(crate) fn check_conv2d_shape(
    lens: &[usize],
    out: usize,
    win: Window,
    (h, w, cin, cout): Shape,
) -> Result<()> {
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

/// 方形窗口参数：`k/s/p` 三个数一起传递。
#[derive(Clone, Copy)]
pub(crate) struct Window {
    pub(crate) k: usize,
    pub(crate) s: usize,
    pub(crate) p: usize,
}

/// 最大值位置：严格更大才替换，平手保留先遇到的；空窗口返回 `fallback`。
#[must_use]
fn argmax(x: &[f32], fallback: usize, positions: impl Iterator<Item = usize>) -> usize {
    positions.fold(fallback, |pick, at| if x[at] > x[pick] { at } else { pick })
}

/// 一维最大池化窗口里的最大值位置，平手取先遇到的。
#[must_use]
pub(crate) fn window1d(
    x: &[f32],
    base: usize,
    l: usize,
    oi: usize,
    win: Window,
    ci: usize,
) -> usize {
    let positions = (0..win.k).filter_map(|ki| {
        let ii = oi as i64 * win.s as i64 - win.p as i64 + ki as i64;
        (0..l as i64).contains(&ii).then(|| base + ii as usize + ci)
    });
    argmax(x, base + ci, positions)
}

/// 二维最大池化窗口里的最大值位置；`plane` 是 `(h, w, c)` 三轴。
#[must_use]
pub(crate) fn window2d(
    x: &[f32],
    base: usize,
    (h, w, c): Plane,
    at: (usize, usize),
    win: Window,
    ci: usize,
) -> usize {
    let positions = (0..win.k * win.k).filter_map(|q| {
        let ii = at.0 as i64 * win.s as i64 - win.p as i64 + (q / win.k) as i64;
        let jj = at.1 as i64 * win.s as i64 - win.p as i64 + (q % win.k) as i64;
        ((0..h as i64).contains(&ii) && (0..w as i64).contains(&jj))
            .then(|| base + (ii as usize * w + jj as usize) * c + ci)
    });
    argmax(x, base + ci, positions)
}

/// 归一化反向的三个输出切片。
pub(crate) type NormOut<'a> = (&'a mut [f32], &'a mut [f32], &'a mut [f32]);

/// 组成员数、均值与归一化逆标准差；统计顺序与前向参考一致。
fn norm_stats(x: &[f32], members: impl Iterator<Item = usize> + Clone) -> (f32, f32, f32) {
    let count = members.clone().count() as f32;
    let (mut sum, mut sq) = (0.0f32, 0.0f32);
    for idx in members {
        sum += x[idx];
        sq += x[idx] * x[idx];
    }
    let mean = sum / count;
    let var = (sq / count - mean * mean).max(0.0);
    (count, mean, 1.0 / (var + EPS).sqrt())
}

/// 归一化反向：按 `(均值, 方差)` 求 `dx`，并累加 `dgamma`/`dbeta`。
/// 成员下标惰性迭代并克隆游标，保持遍历顺序且不为每个组分配临时向量。
pub(crate) fn norm_group(
    x: &[f32],
    gamma: &[f32],
    incoming: &[f32],
    members: impl Iterator<Item = usize> + Clone,
    channel: impl Fn(usize) -> usize,
    out: NormOut<'_>,
) {
    let (dx, dgamma, dbeta) = out;
    let (count, mean, inv) = norm_stats(x, members.clone());
    let (mut sum_dxhat, mut sum_dxhat_xhat) = (0.0f32, 0.0f32);
    for idx in members.clone() {
        let col = channel(idx);
        let xhat = (x[idx] - mean) * inv;
        let dxhat = incoming[idx] * gamma[col];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
        dgamma[col] += incoming[idx] * xhat;
        dbeta[col] += incoming[idx];
    }
    for idx in members {
        let col = channel(idx);
        let xhat = (x[idx] - mean) * inv;
        let dxhat = incoming[idx] * gamma[col];
        dx[idx] = inv * (dxhat - sum_dxhat / count - xhat * sum_dxhat_xhat / count);
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into())
}

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

impl Geometry {
    /// 输入个数：池化与插值是一元，三种归一化与卷积是 `(x, gamma, beta)` 或三输入。
    pub fn arity(self) -> usize {
        match self {
            Self::Pool1d { .. }
            | Self::Pool2d { .. }
            | Self::Avg2d { .. }
            | Self::Adaptive { .. }
            | Self::Interp { .. } => 1,
            Self::Norm { .. } | Self::Group { .. } | Self::Layer { .. } | Self::Conv2d { .. } => 3,
        }
    }

    /// 校验一次调用的形状：输入个数、各输入长度、输出长度都必须与实例自洽。
    ///
    /// # Errors
    ///
    /// 任一处不符返回 [`Code::Invalid`]。
    pub fn check_shape(self, lens: &[usize], out: usize) -> Result<()> {
        if lens.len() != self.arity() {
            return Err(invalid(format!(
                "输入个数是 {}，实例要 {}",
                lens.len(),
                self.arity()
            )));
        }
        check_geometry(self, lens, out)
    }
}
