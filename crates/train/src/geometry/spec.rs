/// 归一化 eps：与 `op::classic` 四个声明档模块的 `EPS` 同值（算术来源留在 op 侧）。
pub const EPS: f32 = op::batch_norm::EPS;

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
