//! 算子实例几何：把窗口、步长、分组、批空间这些**运行时参数**从算子名里拆出来。
//!
//! 本文件只留静态规格（`EPS`/`MOMENTUM`/[`Running`]/[`Geometry`]/`DECLARED`）与方法分派；
//! 形状核对与窗口几何在 `shapes`，逐变体前向/反向数值微核在 `kernels`。
use diagnostic::{Code, Error, Result};

// ---- 静态规格与声明档表 ------------------------------------------------------

/// 归一化 eps：与 `nn::conv::classic` 四个声明档模块的 `EPS` 同值（算术来源留在 `nn` 侧）。
pub const EPS: f32 = nn::conv::batch_norm::EPS;

/// BatchNorm 的滑动统计动量：只用于更新 running，不参与输出。
pub(crate) const MOMENTUM: f32 = 0.1;

/// BatchNorm 的滚动统计：训练期按 batch 统计与动量递推，推理期直接用这一份。
#[derive(Debug, Clone, PartialEq)]
pub struct Running {
    /// 每通道滚动均值。
    pub mean: Vec<f32>,
    /// 每通道滚动方差（无偏口径）。
    pub var: Vec<f32>,
}

impl Running {
    /// 初始口径：均值 0、方差 1。
    pub fn fresh(channels: usize) -> Self {
        Self {
            mean: vec![0.0; channels],
            var: vec![1.0; channels],
        }
    }
}

/// 一次算子调用的实例化几何。字段即前向/反向所需的全部形状信息。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    /// 一维最大池化：`(n, l, c)` 布局，窗口 `k`、步长 `s`、补零 `p`。
    Pool1d {
        /// 窗口长度。
        k: usize,
        /// 步长。
        s: usize,
        /// 两侧补零。
        p: usize,
        /// 输入长度。
        l: usize,
        /// 通道数。
        c: usize,
    },
    /// 二维最大池化：`(n, h, w, c)` 布局，方形窗口 `k×k`。
    Pool2d {
        /// 窗口边长。
        k: usize,
        /// 步长。
        s: usize,
        /// 两侧补零。
        p: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 通道数。
        c: usize,
    },
    /// 二维平均池化；`count_include_pad` 决定补零位是否计入分母。
    Avg2d {
        /// 窗口边长。
        k: usize,
        /// 步长。
        s: usize,
        /// 两侧补零。
        p: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 通道数。
        c: usize,
        /// 补零位是否计入分母。
        count_include_pad: bool,
    },
    /// 自适应平均池化：任意 `h×w` 压到 `oh×ow`。
    Adaptive {
        /// 输出高。
        oh: usize,
        /// 输出宽。
        ow: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 通道数。
        c: usize,
    },
    /// 插值：任意 `h×w` 到 `oh×ow`；`linear = false` 走最近邻。
    Interp {
        /// 输出高。
        oh: usize,
        /// 输出宽。
        ow: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 通道数。
        c: usize,
        /// 是否双线性。
        linear: bool,
    },
    /// 批归一化：统计域是"同通道跨批"，共 `n·spatial` 个元素。
    Norm {
        /// 批大小。
        n: usize,
        /// 每样本的空间元素数。
        spatial: usize,
        /// 通道数。
        channels: usize,
    },
    /// 组归一化：统计域是每个 `(样本, 组)` 的 `C/G·spatial` 个元素；
    /// 实例归一化即 `groups = channels`。
    Group {
        /// 组数。
        groups: usize,
        /// 批大小。
        n: usize,
        /// 每样本的空间元素数。
        spatial: usize,
        /// 通道数。
        channels: usize,
    },
    /// 二维卷积：NHWC 激活、`[(ki·k+kj)·cin+ci, cout]` 行主序权重、`same` 式补零（单样本）。
    Conv2d {
        /// 核边长。
        k: usize,
        /// 步长。
        s: usize,
        /// 两侧补零。
        p: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 输入通道。
        cin: usize,
        /// 输出通道。
        cout: usize,
    },
    /// 层归一化：逐行在最后一维统计。
    Layer {
        /// 行数。
        rows: usize,
        /// 行宽。
        width: usize,
    },
}

impl Geometry {
    /// 声明档实例：与 `nn::conv::classic` 里的固定几何逐字段对齐（梯度对拍的基准）。
    ///
    /// # Errors
    ///
    /// `op` 不在十个经典算子之列时返回 [`Code::Unsupported`]。
    pub fn declared(op: &str) -> Result<Self> {
        DECLARED
            .iter()
            .find(|(name, _)| *name == op)
            .map(|(_, geom)| *geom)
            .ok_or_else(|| {
                Error::new(Code::Unsupported, format!("`{op}` 没有声明档实例几何"))
                    .with("train::Geometry::declared")
            })
    }
}

/// 表中大写标识符取自该行 `nn::conv` 模块，数字与布尔字面量原样进入几何。
macro_rules! declaration_value {
    ($module:ident, $value:literal) => {
        $value
    };
    ($module:ident, $value:ident) => {
        nn::conv::$module::$value
    };
}

/// 一行声明同时确定算子名、几何变体与全部字段，避免表项包装和模块路径重复。
macro_rules! declared_geometries {
    ($( $module:ident => $variant:ident { $($field:ident : $value:tt),* $(,)? } ),* $(,)?) => {
        &[$((
            concat!(stringify!($module), "_f32"),
            Geometry::$variant { $($field: declaration_value!($module, $value)),* },
        )),*]
    };
}

/// 十个经典算子的声明档实例：`declared` 的唯一事实来源。
const DECLARED: &[(&str, Geometry)] = declared_geometries! {
    max_pool1d => Pool1d { k: 2, s: 2, p: 0, l: L, c: 1 },
    max_pool2d => Pool2d { k: 3, s: 1, p: 1, h: H, w: W, c: 1 },
    avg_pool2d => Avg2d { k: 3, s: 1, p: 1, h: H, w: W, c: 1, count_include_pad: false },
    adaptive_pool => Adaptive { oh: OH, ow: OW, h: H, w: W, c: 1 },
    interpolate => Interp { oh: OH, ow: OW, h: H, w: W, c: 1, linear: true },
    conv2d => Conv2d { k: K, s: S, p: P, h: H, w: W, cin: CIN, cout: COUT },
    batch_norm => Norm { n: N, spatial: S, channels: C },
    group_norm => Group { groups: G, n: N, spatial: S, channels: C },
    instance_norm => Group { groups: C, n: N, spatial: S, channels: C },
    layer_norm => Layer { rows: R, width: W },
};

#[path = "microkernels.rs"]
mod kernels;
#[path = "shapes.rs"]
mod shapes;
use shapes::pool_len;

// ---- 方法分派：形状核对 + 逐变体路由 -----------------------------------------
impl Geometry {
    /// 输出元素数。前向缓冲长度必须与它相等。
    pub fn outputs(self) -> usize {
        match self {
            Self::Pool1d { .. }
            | Self::Pool2d { .. }
            | Self::Avg2d { .. }
            | Self::Conv2d { .. } => self.pool_output(),
            _ => self.other_output(),
        }
    }

    /// 窗口类几何的输出元素数。
    fn pool_output(self) -> usize {
        match self {
            Self::Pool1d { k, s, p, l, c } => pool_len(l, k, s, p).map_or(0, |ol| ol * c),
            Self::Pool2d { k, s, p, h, w, c }
            | Self::Avg2d {
                k, s, p, h, w, c, ..
            } => plane_out(h, w, k, s, p, c),
            Self::Conv2d {
                k,
                s,
                p,
                h,
                w,
                cout,
                ..
            } => plane_out(h, w, k, s, p, cout),
            _ => 0,
        }
    }

    /// 固定平面与归一化几何的输出元素数。
    fn other_output(self) -> usize {
        match self {
            Self::Adaptive { oh, ow, c, .. } | Self::Interp { oh, ow, c, .. } => oh * ow * c,
            Self::Norm {
                n,
                spatial,
                channels,
            }
            | Self::Group {
                n,
                spatial,
                channels,
                ..
            } => n * spatial * channels,
            Self::Layer { rows, width } => rows * width,
            _ => 0,
        }
    }

    /// 按实例几何前向：算术全部复用 `nn::conv` / `nn::norm` 的真实现。
    ///
    /// # Errors
    ///
    /// 形状与实例不符、或底层算子报错时返回说明。
    pub fn forward(self, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
        self.forward_with(inputs, out, None)
    }

    /// 带滚动统计的前向：只有 [`Geometry::Norm`] 消费 `running`，其余变体忽略。
    ///
    /// 传 `None` 等价于每次前向从"均值 0、方差 1"起算，输出与 [`Geometry::forward`] 逐位一致。
    ///
    /// # Errors
    ///
    /// 形状与实例不符、或底层算子报错时返回说明。
    pub fn forward_with(
        self,
        inputs: &[&[f32]],
        out: &mut [f32],
        running: Option<&mut Running>,
    ) -> Result<()> {
        self.check_shape(&shapes::lengths(inputs), out.len())?;
        kernels::forward(self, inputs, out, running)
    }

    /// 推理前向：`Norm` 用给定的滚动统计归一化、不改状态；其余变体与训练一致。
    ///
    /// # Errors
    ///
    /// 形状与实例不符、统计通道数不匹配、或底层算子报错时返回说明。
    pub fn forward_eval(self, inputs: &[&[f32]], out: &mut [f32], running: &Running) -> Result<()> {
        let Self::Norm {
            n,
            spatial,
            channels,
        } = self
        else {
            return self.forward(inputs, out);
        };
        self.check_shape(&shapes::lengths(inputs), out.len())?;
        shapes::check_running(running, channels)?;
        kernels::forward_batch_norm_eval(inputs, out, running, n, spatial, channels)
    }

    /// 按实例几何反向：贡献顺序与输入对齐（一元给 `dx`，三种归一化给 `dx/dgamma/dbeta`）。
    ///
    /// # Errors
    ///
    /// 形状与实例不符时返回 [`Code::Invalid`]。
    pub fn backward(self, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.check_shape(&shapes::lengths(inputs), incoming.len())?;
        kernels::backward(self, inputs, incoming)
    }
}

/// 二维平面输出元素数：两次窗口展开相乘，几何不成立时退 0。
fn plane_out(h: usize, w: usize, k: usize, s: usize, p: usize, channels: usize) -> usize {
    shapes::pool2d_out(shapes::Window { k, s, p }, h, w).map_or(0, |(oh, ow)| oh * ow * channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostic::Code;

    /// 声明档样本的输入切片：卷积是 `(x, weight, bias)`，其余按一元/三元口径。
    fn declared_inputs(sample: &nn::model::Sample, g: Geometry) -> Vec<&[f32]> {
        match g {
            Geometry::Conv2d { .. } => vec![
                sample.f32("x").expect("x"),
                sample.f32("weight").expect("weight"),
                sample.f32("bias").expect("bias"),
            ],
            _ if g.arity() == 1 => vec![sample.f32("x").expect("x")],
            _ => vec![
                sample.f32("x").expect("x"),
                sample.f32("gamma").expect("gamma"),
                sample.f32("beta").expect("beta"),
            ],
        }
    }

    /// 十种声明档算子及其样本，供形状对齐与前向对拍两个测试共用。
    fn declared_samples() -> [(&'static str, nn::model::Sample); 10] {
        [
            ("max_pool1d_f32", nn::conv::max_pool1d::sample(1, 1)),
            ("max_pool2d_f32", nn::conv::max_pool2d::sample(1, 1)),
            ("avg_pool2d_f32", nn::conv::avg_pool2d::sample(1, 1)),
            ("adaptive_pool_f32", nn::conv::adaptive_pool::sample(1, 1)),
            ("interpolate_f32", nn::conv::interpolate::sample(1, 1)),
            ("batch_norm_f32", nn::conv::batch_norm::sample(1, 1)),
            ("conv2d_f32", nn::conv::conv2d::sample(1, 1)),
            ("group_norm_f32", nn::conv::group_norm::sample(1, 1)),
            ("instance_norm_f32", nn::conv::instance_norm::sample(1, 1)),
            ("layer_norm_f32", nn::conv::layer_norm::sample(1, 1)),
        ]
    }

    testkit::test_cases! {
    /// 声明档实例与 `nn::conv::classic` 的声明档样本长度逐一对齐：前向缓冲与样本同长。
    fn declared_geometries_match_the_sample_lengths() {
        for (op, sample) in declared_samples() {
            let g = Geometry::declared(op).expect("声明档实例");
            let y = sample.f32("y").expect("y");
            let inputs = declared_inputs(&sample, g);
            let lens: Vec<usize> = inputs.iter().map(|input| input.len()).collect();
            g.check_shape(&lens, y.len()).expect("声明档形状自洽");
        }
    }

    /// 前向：实例几何的结果必须与 `nn::conv::classic` 的 reference 逐位一致（同一份算术的两次执行）。
    fn declared_forward_matches_the_reference() {
        for (op, sample) in declared_samples() {
            let g = Geometry::declared(op).expect("声明档实例");
            let mut out = vec![0.0f32; g.outputs()];
            g.forward(&declared_inputs(&sample, g), &mut out)
                .expect("前向");
            let want = (nn::registry::model(op).expect("已注册").reference)(&sample).expect("参考");
            let want = want.as_f32().expect("参考档为 f32");
            assert_eq!(out.as_slice(), want, "{op} 实例前向与声明档参考不一致");
        }
    }

    /// 任意几何也要自洽：非声明档的窗口/分组必须能前向，且形状校验拦住不成立的几何。
    /// 滚动统计按动量递推，同时不改变训练前向的输出（训练语义与数值参考同源）。
    /// 推理用滚动统计归一化：`(x - m) / sqrt(v + eps) * gamma + beta`，与手算逐位一致。
    fn evaluation_normalizes_with_the_given_rolling_statistics() {
        let (n, spatial, channels) = (1usize, 2usize, 2usize);
        let len = n * spatial * channels;
        let geom = Geometry::Norm {
            n,
            spatial,
            channels,
        };
        let x: Vec<f32> = (0..len).map(|i| 0.4 + i as f32 * 0.3).collect();
        let gain = vec![1.5f32, 0.5];
        let shift = vec![0.2f32, -0.1];
        let running = Running {
            mean: vec![0.3, -0.2],
            var: vec![2.0, 0.5],
        };

        let mut out = vec![0.0f32; len];
        geom.forward_eval(&[&x, &gain, &shift], &mut out, &running)
            .expect("推理前向");
        for (idx, got) in out.iter().enumerate() {
            let ci = idx % channels;
            let want =
                (x[idx] - running.mean[ci]) / (running.var[ci] + EPS).sqrt() * gain[ci] + shift[ci];
            assert!(
                (got - want).abs() <= 1e-6,
                "第 {idx} 个输出 {got} vs {want}"
            );
        }
        // 通道数不匹配要在形状层报错，不静默套用错位统计。
        let wrong = Running::fresh(channels + 1);
        let error = geom
            .forward_eval(&[&x, &gain, &shift], &mut out, &wrong)
            .expect_err("通道数不符");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn running_statistics_update_by_momentum_without_changing_the_output() {
        let (n, spatial, channels) = (2usize, 3usize, 2usize);
        let len = n * spatial * channels;
        let geom = Geometry::Norm {
            n,
            spatial,
            channels,
        };
        let gain = vec![1.0f32; channels];
        let shift = vec![0.0f32; channels];
        let first: Vec<f32> = (0..len).map(|i| 0.3 + i as f32 * 0.1).collect();
        let second: Vec<f32> = (0..len).map(|i| 1.4 - i as f32 * 0.06).collect();

        let mut stats = Running::fresh(channels);
        for batch in [&first, &second] {
            let mut out = vec![0.0f32; len];
            geom.forward_with(&[batch, &gain, &shift], &mut out, Some(&mut stats))
                .expect("带统计的前向");
            let mut plain = vec![0.0f32; len];
            geom.forward(&[batch, &gain, &shift], &mut plain)
                .expect("无统计的前向");
            for (idx, (a, b)) in out.iter().zip(&plain).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-6,
                    "第 {idx} 个输出被统计影响：{a} vs {b}"
                );
            }
        }

        // 手动复算两批的滑动口径：`m = 0.9 * m + 0.1 * batch_mean`，方差用无偏值。
        let count = n * spatial;
        let mut mean = vec![0.0f32; channels];
        let mut var = vec![1.0f32; channels];
        for batch in [&first, &second] {
            for ci in 0..channels {
                let values: Vec<f32> = (0..count).map(|p| batch[p * channels + ci]).collect();
                let batch_mean = values.iter().sum::<f32>() / count as f32;
                let batch_var =
                    values.iter().map(|v| (v - batch_mean).powi(2)).sum::<f32>() / count as f32;
                let unbiased = batch_var * count as f32 / (count - 1) as f32;
                mean[ci] = 0.9 * mean[ci] + 0.1 * batch_mean;
                var[ci] = 0.9 * var[ci] + 0.1 * unbiased;
            }
        }
        for ci in 0..channels {
            assert!(
                (stats.mean[ci] - mean[ci]).abs() <= 1e-5,
                "通道 {ci} 均值 {} vs {}",
                stats.mean[ci],
                mean[ci]
            );
            assert!(
                (stats.var[ci] - var[ci]).abs() <= 1e-5,
                "通道 {ci} 方差 {} vs {}",
                stats.var[ci],
                var[ci]
            );
        }
    }

    fn arbitrary_geometries_forward_and_reject_bad_shapes() {
        let pool = Geometry::Pool2d {
            k: 3,
            s: 2,
            p: 0,
            h: 5,
            w: 5,
            c: 2,
        };
        assert_eq!(pool.outputs(), 2 * 2 * 2);
        let mut out = vec![0.0f32; pool.outputs()];
        let x: Vec<f32> = (0..5 * 5 * 2).map(|i| i as f32 - 12.0).collect();
        pool.forward(&[&x], &mut out).expect("前向");
        assert!(out.iter().all(|v| v.is_finite()));

        // 窗口比输入还大：显式拒绝，不出 0 长度。
        let bad = Geometry::Pool1d {
            k: 5,
            s: 1,
            p: 0,
            l: 3,
            c: 1,
        };
        assert!(bad.check_shape(&[3], bad.outputs()).is_err());
    }

    /// 训练前向也不能把错位滚动统计静默替换成局部统计。
    fn training_rejects_mismatched_running_statistics() {
        let geometry = Geometry::Norm {
            n: 1,
            spatial: 2,
            channels: 2,
        };
        let (x, gain, shift) = (vec![0.0; 4], vec![1.0; 2], vec![0.0; 2]);
        let mut out = vec![0.0; 4];
        let mut wrong = Running::fresh(1);
        let error = geometry
            .forward_with(&[&x, &gain, &shift], &mut out, Some(&mut wrong))
            .expect_err("滚动统计通道不符");
        assert_eq!(error.code(), Code::Invalid);
    }
    }
}
