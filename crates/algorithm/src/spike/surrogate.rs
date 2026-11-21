use super::*;

/// 代理梯度（surrogate）：只定义「发放对膜电位」的导数，不参与前向。
///
/// 前向的发放是阶跃——导数在阈值处没有定义、别处为零，照它反传什么也学不到。surrogate 用一个
/// 面积为 1 的窗把这段导数抹平：窗内按形状给值，窗外为零。它**不是**「软化的前向」：前向仍走
/// [`Lif`] 的硬判据，只有反传这一侧看窗，所以同一个 `step` 还是给出同一个脉冲。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surrogate {
    /// 矩形窗：`|v − θ| < width` 内恒为 `1/(2·width)`。
    ///
    /// 窗内一律同值，所以「离阈值多远」不影响梯度——简单，且对电位抖动不敏感。
    Rect {
        /// 窗的半宽。
        width: f32,
    },
    /// 三角形：窗内从阈值处的 `1/width` 线性衰减到边界上的零。
    ///
    /// 峰落在阈值上，于是「刚好压线」的通道拿到的梯度最大——与阶跃真正被跨越的位置一致。
    Triangle {
        /// 窗的半宽。
        width: f32,
    },
}

impl Surrogate {
    /// 窗的半宽。
    #[must_use]
    pub fn width(&self) -> f32 {
        match self {
            Self::Rect { width } | Self::Triangle { width } => *width,
        }
    }

    /// 发放对膜电位的导数：`potential` 与 `threshold` 的距离落在窗内才有值。
    ///
    /// 两种形状的面积都是 1，所以换形状不换「总敏感度」，只换它怎么摊在窗内。
    #[must_use]
    pub fn derivative(&self, potential: f32, threshold: f32) -> f32 {
        let offset = (potential - threshold).abs();
        // 非有限偏移连 `NaN` 一起挡在窗外：窗宽由构造期校验为有限正数，而 `NaN` 与任何值的
        // `>=` 都为假，单写比较会让它走进窗内拿到一个假的导数。
        if !offset.is_finite() || offset >= self.width() {
            return 0.0;
        }
        match self {
            Self::Rect { width } => 1.0 / (2.0 * width),
            Self::Triangle { width } => (width - offset) / (width * width),
        }
    }

    /// 参数判据：半宽必须有限为正（零宽等于没有梯度，负宽会让窗内外颠倒）。
    pub(super) fn check(&self) -> Result<()> {
        let width = self.width();
        if !finite(width) || width <= 0.0 {
            return Err(bad(format!("代理梯度的窗宽必须有限为正，得到 {width:?}")));
        }
        Ok(())
    }
}

/// 一层脉冲线性的梯度，布局与本层参数逐项对齐。
#[derive(Debug, Clone, PartialEq)]
pub struct Gradients {
    pub(super) weights: Vec<f32>,
    pub(super) scale: Vec<f32>,
    pub(super) current: Vec<f32>,
}

impl Gradients {
    /// 对权重的梯度，行主序 `[outputs, inputs]`（与本层的权重同一布局）。
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// 对逐行反量化尺度的梯度，一条对一行。
    #[must_use]
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// 对本层输出**电流**的梯度，可直接交给上一层的反传。
    #[must_use]
    pub fn current(&self) -> &[f32] {
        &self.current
    }

    /// 缓冲判据：三条梯度必须与本层参数同形，长了短了都会静默写错行。
    pub(super) fn check(&self, layer: &Spiking) -> Result<()> {
        self.layout(layer.weights.len(), layer.outputs)
    }

    /// 反传入口判据：缓冲与本层同形，且输出梯度与通道数同长。
    pub(super) fn check_for(&self, layer: &Spiking, d_out: &[f32]) -> Result<()> {
        self.check(layer)?;
        check_gradient(d_out, layer.outputs)
    }

    /// 判据本体：只按「权重格数 + 输出通道数」核对，好让量化前的 [`Master`] 与量化后的
    /// [`Spiking`] 共用同一份判据。
    pub(super) fn layout(&self, weights: usize, outputs: usize) -> Result<()> {
        let shape = [self.weights.len(), self.scale.len(), self.current.len()];
        let want = [weights, outputs, outputs];
        if shape != want {
            return Err(bad(format!("梯度缓冲 {shape:?} 与本层 {want:?} 不符")));
        }
        Ok(())
    }
}
