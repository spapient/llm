//! `DecisionHead`：唯一门控实现（§1.11）。
//!
//! 输入是候选上的分数，输出是该集合上的分布。MoE 专家、题目选项、层间路由都是这个集合；随后的
//! TopK 只调用 L0（不在本模块）。默认校准名 `spherical`；`softmax` / `temperature` / `brier` 只在
//! 同一端口注册后才能被配置点名，未注册名字启动失败。§1.6 的 clip 公式不是第二个模块，而是
//! `head_layers = 0`（逐 token 默认）时的特征组合。

use op::scan::Scalar;

/// `DecisionHead` 的逐 token 默认层深（§1.11：`head_layers = 0`）。
pub const HEAD_LAYERS_TOKEN: usize = 0;

/// 校准名闭集（score → 分布）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Calib {
    /// 球面（默认）：非负截断后 L2 归一化，严格适当评分族。
    Spherical,
    /// 标准 softmax。
    Softmax,
    /// 温度 softmax（`τ` 为温度，须为正）。
    Temperature(f64),
    /// Brier：先 softmax 得分布，再用 [`brier`] 评分（同一端口注册）。
    Brier,
}

impl Calib {
    /// 按名解析；未注册名字返回 `None`（启动失败，§1.11）。
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "spherical" => Some(Self::Spherical),
            "softmax" => Some(Self::Softmax),
            "temperature" => Some(Self::Temperature(1.0)),
            "brier" => Some(Self::Brier),
            _ => None,
        }
    }

    /// 分数 → 候选集合上的分布。
    pub fn to_dist<S: Scalar>(self, scores: &[S]) -> Vec<S> {
        match self {
            Self::Spherical => spherical(scores),
            Self::Softmax => softmax(scores),
            Self::Temperature(t) => {
                debug_assert!(t > 0.0, "温度必须为正");
                let scaled: Vec<S> = scores.iter().map(|&s| s.div(S::from_f64(t))).collect();
                softmax(&scaled)
            }
            Self::Brier => softmax(scores),
        }
    }
}

/// 稳定 softmax（先减最大值防溢出，输出和为 1）。
fn softmax<S: Scalar>(scores: &[S]) -> Vec<S> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores
        .iter()
        .copied()
        .fold(scores[0], |m, x| if x > m { x } else { m });
    let exps: Vec<S> = scores.iter().map(|&s| s.sub(max).exp()).collect();
    let sum = exps.iter().copied().fold(S::ZERO, |a, b| a.add(b));
    exps.into_iter().map(|e| e.div(sum)).collect()
}

/// 球面投影：非负截断后 L2 归一化；零范数退化为均匀分布。
fn spherical<S: Scalar>(scores: &[S]) -> Vec<S> {
    let pos: Vec<S> = scores
        .iter()
        .map(|&s| if s < S::ZERO { S::ZERO } else { s })
        .collect();
    let norm = pos
        .iter()
        .copied()
        .fold(S::ZERO, |a, b| a.add(b.mul(b)))
        .sqrt();
    if norm == S::ZERO {
        let u = S::ONE.div(S::from_f64(pos.len() as f64));
        return vec![u; pos.len()];
    }
    pos.into_iter().map(|p| p.div(norm)).collect()
}

/// Brier 严格适当评分：`B = (1/K) Σ (pᵢ − yᵢ)²`，`y` 为目标 one-hot（`target` 为命中下标）。
///
/// 与 `spherical` 同属严格适当评分族，注册在 `DecisionHead` 的同一端口（§1.11）。
pub fn brier<S: Scalar>(p: &[S], target: usize) -> S {
    debug_assert!(target < p.len());
    if p.is_empty() {
        return S::ZERO;
    }
    let mut sum = S::ZERO;
    for (i, &pi) in p.iter().enumerate() {
        let y = if i == target { S::ONE } else { S::ZERO };
        let d = pi.sub(y);
        sum = sum.add(d.mul(d));
    }
    sum.div(S::from_f64(p.len() as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    /// 已注册名解析成功，未注册名失败。
    #[test]
    fn from_name_accepts_only_registered() {
        assert_eq!(Calib::from_name("spherical"), Some(Calib::Spherical));
        assert_eq!(Calib::from_name("softmax"), Some(Calib::Softmax));
        assert_eq!(
            Calib::from_name("temperature"),
            Some(Calib::Temperature(1.0))
        );
        assert_eq!(Calib::from_name("brier"), Some(Calib::Brier));
        assert_eq!(Calib::from_name("argmax"), None, "未注册名字启动失败");
    }

    /// softmax 输出和为 1。
    #[test]
    fn softmax_sums_to_one() {
        let d = Calib::Softmax.to_dist(&[1.0f64, 2.0, 3.0]);
        assert!((d.iter().sum::<f64>() - 1.0).abs() <= TOL);
    }

    /// 球面校准输出 L2 范数为 1（非负分数）。
    #[test]
    fn spherical_has_unit_l2_norm() {
        let d = Calib::Spherical.to_dist(&[3.0f64, 4.0]);
        let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() <= TOL);
    }

    /// 温度缩放：τ 越小分布越尖锐。
    #[test]
    fn temperature_sharpens_distribution() {
        let cold = Calib::Temperature(0.5).to_dist(&[1.0f64, 2.0]);
        let hot = Calib::Temperature(2.0).to_dist(&[1.0f64, 2.0]);
        assert!(cold[1] / cold[0] > hot[1] / hot[0]);
    }

    /// Brier 评分：命中时为 0（one-hot），未命中为正，值域 `[0,1]`。
    #[test]
    fn brier_scores_perfect_as_zero() {
        let p = vec![0.0f64, 1.0, 0.0];
        assert!((brier(&p, 1)).abs() <= TOL);
        let wrong = brier(&p, 0);
        assert!(wrong > 0.0 && wrong <= 1.0);
    }
}
