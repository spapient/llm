//! typed IR（`cutile::kernel::Kernel`）→ [`Lower`]/设备 dispatch 的直通桥。
//!
//! [`Kernel`] 自带 `.k` 正文发射（[`Kernel::emit_k`]）；本模块只做三件事：把 `Write::Map` /
//! `Write::Reduce` 映射成 [`Lower`] 形态、把 `.k` 正文解析成 [`Signature`]，再经
//! [`Codegen::lower`] 交给 vendor 后端。样本由调用方在派发时给出，桥本身与样本无关——它回答的是
//! 「这段 typed IR 该降成什么形态、长成什么 `.k` 文本」，不回答「这一份具体跑多长」。

use crate::backend::Lower;
use cutile::kernel::{Access, Expr, Kernel, ValueType, Write};
use cutile::prof::Emit;
use diagnostic::{Code, Error, Result};
use prof::decl::Grid;
use prof::model::Sample;
use prof::sig::Signature;

/// 分块归约的默认块大小：与逐点线程数同阶，兼顾占用与主机折叠成本。
pub const REDUCE_CHUNK: usize = 256;

/// 归约软件原语名：`_cutile_sum_f32` 是唯一落地的标量累加原语（vendor preamble 里定义）。
pub const REDUCE_FUNC: &str = "_cutile_sum_f32";

/// `Write::Map`/`Write::Reduce` → [`Lower`]：逐点（唯一标量是长度）开网格，纯 `sum` 归约
/// （`out[0] += a[idx]`）分块，其余走串行兜底。
pub fn kind(kernel: &Kernel) -> Result<Lower> {
    Ok(kind_of(&optimize(kernel)?))
}

/// `.k` 正文 + 签名：正文由 [`Kernel::emit_k`] 生成，`preamble` 是 vendor 软件原语前缀
/// （分块归约需要 `_cutile_sum_f32`，逐点传空即可）。
pub fn source(kernel: &Kernel, preamble: &str) -> Result<(String, Signature)> {
    let body = kernel.emit_k().map_err(verify_failed)?;
    let text = format!("{preamble}{body}");
    let signature = Signature::parse(&kernel.name, &text)?;
    Ok((text, signature))
}

/// 直通设备 dispatch：形态 → 正文/签名 → vendor 后端降源。
pub fn lower(emit: Emit, kernel: &Kernel, preamble: &str, sample: &Sample) -> Result<Vec<String>> {
    let kernel = optimize(kernel)?;
    let kind = kind_of(&kernel);
    let (text, signature) = source(&kernel, preamble)?;
    let backend = crate::device::require(emit)?;
    backend.lower(&text, &signature, sample, &kind)
}

/// 校验失败统一换成诊断错误；typed IR 只报「为什么不成形」，诊断层再补桥接上下文。
fn verify_failed(error: cutile::kernel::VerificationError) -> Error {
    Error::new(Code::Invalid, format!("typed IR 校验失败：{error}")).with("shard::ir")
}

fn optimize(kernel: &Kernel) -> Result<Kernel> {
    kernel.optimize().map_err(verify_failed)
}

fn kind_of(kernel: &Kernel) -> Lower {
    match &kernel.write {
        Write::Map(_) => {
            if scalar_count(kernel) == 1 {
                Lower::Pointwise
            } else {
                serial()
            }
        }
        Write::Reduce { at, value } => {
            if *at == 0 && plain_sum(kernel, value) {
                Lower::ChunkReduce {
                    chunk: REDUCE_CHUNK,
                    func: REDUCE_FUNC.to_string(),
                }
            } else {
                serial()
            }
        }
    }
}

fn serial() -> Lower {
    Lower::Loop(Grid { dims: Vec::new() })
}

fn scalar_count(kernel: &Kernel) -> usize {
    kernel
        .params
        .iter()
        .filter(|parameter| matches!(parameter.ty, ValueType::Scalar(_)))
        .count()
}

/// `out[0] += a[idx]` 这类「单输入纯累加」才分块；多输入（点积）或带系数走串行，避免
/// [`Lower::ChunkReduce`] 只归约第一个只读缓冲而丢语义。
fn plain_sum(kernel: &Kernel, value: &Expr) -> bool {
    read_input(kernel).is_some_and(|name| matches!(value, Expr::Load(n) if n == name))
}

/// 唯一的只读缓冲名：恰一个只读缓冲才返回 `Some`，否则分块归约无从定位被累加的输入。
fn read_input(kernel: &Kernel) -> Option<&str> {
    let mut reads = kernel
        .params
        .iter()
        .filter(|parameter| {
            matches!(
                parameter.ty,
                ValueType::Buf {
                    access: Access::Read,
                    ..
                }
            )
        })
        .map(|parameter| parameter.name.as_str());
    let name = reads.next()?;
    reads.next().is_none().then_some(name)
}

#[cfg(test)]
#[path = "../tests/ir_tests.rs"]
mod tests;
