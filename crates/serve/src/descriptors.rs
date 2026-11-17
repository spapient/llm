//! 编号 ↔ 形状/条目名描述表（D10 / ADR 0044）：只留 serve 装配侧。
//!
//! engine [`Bindings`](engine::exec::Bindings) 仍是纯 `u32 → Vec<f32>`，不解释编号含义，
//! 也不加只读形状访问器；形状与条目名只在装配期由层声明推出，写进这张表供装配校验与诊断。

use diagnostic::{Code, Error, Result};

use crate::kernels::SITE;
use crate::params::Params;

/// 编号 ↔ 形状/条目名描述表条目。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BindingDesc {
    /// 不透明绑定编号（与 engine `Bindings` 的键一致）。
    pub(crate) id: u32,
    /// 驻留权重条目名；激活缓冲为 `None`。
    pub(crate) entry: Option<String>,
    /// 声明形状（轴顺序与层视图一致）；激活只记一条轴长度，历史占位记 `[0]`。
    pub(crate) shape: Vec<usize>,
}

/// 装配期自洽：描述表编号与权重/激活台账一一对应，形状积（权重）或单轴长度（激活）相符。
///
/// 描述表是装配侧事实；引擎不读这张表，也不解释编号含义。
pub(crate) fn check_all(
    descriptors: &[BindingDesc],
    params: &Params,
    activations: &[(u32, usize)],
) -> Result<()> {
    for desc in descriptors {
        check_one(desc, params, activations)?;
    }
    Ok(())
}

fn check_one(desc: &BindingDesc, params: &Params, activations: &[(u32, usize)]) -> Result<()> {
    match &desc.entry {
        Some(entry) => check_weight(desc, entry, params),
        None => check_activation(desc, activations),
    }
}

/// 权重条目：参数表须有条目，形状积与权重整块长度相符。
fn check_weight(desc: &BindingDesc, entry: &str, params: &Params) -> Result<()> {
    let Some(block) = params.get(entry) else {
        return Err(Error::new(Code::Missing, format!("参数表缺权重条目 `{entry}`")).with(SITE));
    };
    let product: usize = desc.shape.iter().product();
    if product != block.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "描述表条目 `{entry}` 形状积 {product} 与权重长度 {} 不符",
                block.len()
            ),
        )
        .with(SITE));
    }
    Ok(())
}

/// 激活条目：编号须在激活台账内，单轴长度与台账相符；历史占位 `[0]` 由台账长度 0 对上。
fn check_activation(desc: &BindingDesc, activations: &[(u32, usize)]) -> Result<()> {
    let Some(&len) = activations
        .iter()
        .find_map(|(id, activation_len)| (*id == desc.id).then_some(activation_len))
    else {
        return Err(
            Error::new(Code::Internal, format!("激活编号 {} 没有台账记录", desc.id)).with(SITE),
        );
    };
    if desc.shape.first().copied() != Some(len) {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "激活编号 {} 形状 {:?} 与台账长度 {len} 不符",
                desc.id, desc.shape
            ),
        )
        .with(SITE));
    }
    Ok(())
}
