//! 算子实例几何：把窗口、步长、分组、批空间这些**运行时参数**从算子名里拆出来。
//!
//! 本文件只留静态规格（`EPS`/`MOMENTUM`/[`Running`]/[`Geometry`]/`DECLARED`）与方法分派；
//! 形状核对与窗口几何在 `shapes`，逐变体前向/反向数值微核在 `kernels`。
use diagnostic::{Code, Error, Result};
use nn::kernel::norm::BatchNorm2d;

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

mod kernels;
mod shapes;
use shapes::{invalid, norm_outputs, pool_len, spatial_outputs};

// ---- 方法分派：形状核对 + 逐变体路由 -----------------------------------------
impl Geometry {
    /// 输入个数：池化与插值是一元，三种归一化是 `(x, gamma, beta)`。
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

    /// 输出元素数。前向缓冲长度必须与它相等。
    pub fn outputs(self) -> usize {
        let plane = |len, k, s, p, width, channels| {
            pool_len(len, k, s, p)
                .and_then(|out| Ok(out * pool_len(width, k, s, p)? * channels))
                .unwrap_or(0)
        };
        match self {
            Self::Pool1d { k, s, p, l, c } => pool_len(l, k, s, p).map_or(0, |ol| ol * c),
            Self::Pool2d { k, s, p, h, w, c }
            | Self::Avg2d { k, s, p, h, w, c, .. } => plane(h, k, s, p, w, c),
            Self::Adaptive { oh, ow, c, .. } | Self::Interp { oh, ow, c, .. } => oh * ow * c,
            Self::Norm { n, spatial, channels }
            | Self::Group { n, spatial, channels, .. } => n * spatial * channels,
            Self::Conv2d { k, s, p, h, w, cout, .. } => plane(h, k, s, p, w, cout),
            Self::Layer { rows, width } => rows * width,
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
        shapes::check_geometry(self, lens, out)
    }

    /// 按实例几何前向：算术全部复用 `nn::conv` / `nn::kernel::norm` 的真实现。
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
        self.check_shape(
            &inputs.iter().map(|x| x.len()).collect::<Vec<_>>(),
            out.len(),
        )?;
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
        self.check_shape(
            &inputs.iter().map(|x| x.len()).collect::<Vec<_>>(),
            out.len(),
        )?;
        if running.mean.len() != channels || running.var.len() != channels {
            return Err(invalid(format!(
                "滚动统计有 {} 个均值 / {} 个方差，实例几何要 {channels} 个通道",
                running.mean.len(),
                running.var.len()
            )));
        }
        BatchNorm2d {
            channels,
            eps: EPS,
            momentum: MOMENTUM,
        }
        .forward_eval(
            inputs[0],
            inputs[1],
            inputs[2],
            n,
            1,
            spatial,
            &running.mean,
            &running.var,
            out,
        )
    }

    /// 按实例几何反向：贡献顺序与输入对齐（一元给 `dx`，三种归一化给 `dx/dgamma/dbeta`）。
    ///
    /// # Errors
    ///
    /// 形状与实例不符时返回 [`Code::Invalid`]。
    pub fn backward(self, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.check_shape(
            &inputs.iter().map(Vec::len).collect::<Vec<_>>(),
            incoming.len(),
        )?;
        kernels::backward(self, inputs, incoming)
    }
}

#[cfg(test)]
#[path = "../tests/geometry_tests.rs"]
mod tests;
