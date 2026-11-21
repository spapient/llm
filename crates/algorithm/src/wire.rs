//! 17 族 → 生产层栈的**接线台账**：哪些族已能落成 `nn` 层，哪些缺什么。
//!
//! 这张表存在的理由：族"存在但没接线"过去只是口头事实，靠删文件才被发现。现在每个族在表里
//! 有且只有一行，缺的写明缺什么，[`tests/wire_tests.rs`] 对着 `src/` 核名——表和树对不上就红。
//!
//! 三条阻塞口径都**不接受"用形状相近的层顶替"**：宁可如实缺席，不可冒充已通。

use diagnostic::Result;
use nn::MatMul;

use crate::mhc::Mhc;

/// 一个族接不进生产路径的具体原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blocked {
    /// `op` 注册表没有这个算子，层声明过不了 `ContractSpec`。
    Operator(&'static str),
    /// 算子齐了也表达不出结构：`nn` 缺这层声明。
    Layer(&'static str),
    /// 族自身不暴露这份几何，补宽度等于替它编一个模型。
    Geometry(&'static str),
}

/// 一个族的接线结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wiring {
    /// 能落成 `nn` 层，括号内是层数。
    Wired(usize),
    /// 接不进，原因具名。
    Blocked(Blocked),
}

/// 台账的一行。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Family {
    /// 族的源文件名（`src/<name>.rs`）。
    pub name: &'static str,
    /// 接线结论。
    pub wiring: Wiring,
}

const fn blocked(name: &'static str, why: Blocked) -> Family {
    Family {
        name,
        wiring: Wiring::Blocked(why),
    }
}

/// 17 族的接线台账，顺序即族的声明顺序。
///
/// 唯一已通的是 `mhc`：它的流重组就是 `streams × streams` 的定值矩阵，正是 [`nn::MatMul`]
/// 的形状，两者不是"像"而是同一个运算。其余族缺的是算子或层，不是配置。
#[must_use]
pub fn ledger() -> [Family; 17] {
    [
        blocked("attnres", Blocked::Layer("nn 无按数据加权的残差混合层")),
        blocked("codec", Blocked::Layer("nn 无字节编解码层")),
        blocked("csa", Blocked::Operator("csa")),
        blocked("emit", Blocked::Layer("自回归发射属 serve 的循环，不是层")),
        blocked("hybrid", Blocked::Operator("latent_attention")),
        blocked("tower", Blocked::Operator("kda")),
        blocked("kda", Blocked::Operator("kda")),
        blocked("latent", Blocked::Layer("nn 无潜空间补丁层")),
        blocked("mamba", Blocked::Operator("ssm_scan")),
        Family {
            name: "mhc",
            wiring: Wiring::Wired(1),
        },
        blocked("mix", Blocked::Operator("lif")),
        blocked("mla", Blocked::Operator("latent_attention")),
        blocked("moe", Blocked::Layer("nn 无专家路由层")),
        blocked("patch", Blocked::Layer("nn 无补丁层")),
        blocked("policy", Blocked::Layer("nn 无门控策略层")),
        blocked("pulse", Blocked::Operator("lif")),
        blocked("spike", Blocked::Operator("lif")),
    ]
}

/// `mhc` 的接线：流的双随机重组落成一层定值矩阵乘。
///
/// 重组矩阵是 `streams × streams`，作用在"流"这条轴上，与 [`nn::MatMul`] 的 `rows × cols`
/// 同义；`width` 由调用方的批轴承担，不进层声明。这样 `N` 条流重建残差时**不需要**任何运行时
/// 分支，`serve` / `cli` 的既有执行路径直接可跑。
///
/// # Errors
///
/// 流数为零（[`Blocked::Geometry`] 的口径：没有流就没有重组）。
pub fn mhc_layer(m: &Mhc<f32>) -> Result<MatMul> {
    let streams = m.streams();
    MatMul::new("mhc.mix", streams, streams)
}
