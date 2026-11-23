//! 签名 → 缓冲名与几何：GEMM / 注意力各族共用的查表助手。

use crate::dialect::sample_usize;
use diagnostic::{Code, Error, Result};
use prof::model::Sample;
use prof::sig::{Param, Signature};

/// 签名里全部缓冲参数（按序）。
fn buf_params(signature: &Signature) -> Vec<&Param> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
        .collect()
}

/// 签名里全部标量参数（按序）。
fn scalar_params(signature: &Signature) -> Vec<&Param> {
    signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buf())
        .collect()
}

/// 共享块 GEMM 各族的报错出处。
const TILED_SITE: &str = "engine::tiled::metal_tiled_gemm";

/// 三条缓冲名（按签名序：A、B、Y）。
pub(crate) fn gemm_bufs(signature: &Signature) -> Result<(&str, &str, &str)> {
    match buf_params(signature).as_slice() {
        [a, b, y] => Ok((a.name.as_str(), b.name.as_str(), y.name.as_str())),
        _ => Err(Error::new(Code::Internal, "GEMM 签名需要 3 个缓冲（A/B/Y）").with(TILED_SITE)),
    }
}

/// 三个整数标量 `(M, N, K)` 的样本值。
pub(crate) fn gemm_dims(signature: &Signature, sample: &Sample) -> Result<(usize, usize, usize)> {
    match scalar_params(signature).as_slice() {
        [m, n, k] => Ok((dim(sample, m)?, dim(sample, n)?, dim(sample, k)?)),
        _ => {
            Err(Error::new(Code::Internal, "GEMM 签名需要 3 个整数标量（M/N/K）").with(TILED_SITE))
        }
    }
}

/// 一个整数标量的样本取值（必须存在且非负）。
fn dim(sample: &Sample, param: &Param) -> Result<usize> {
    sample_usize(sample, &param.name, "标量", TILED_SITE)
}

/// 四条缓冲名（按签名序：Q、K、V、Y）。
pub(crate) fn attention_bufs(signature: &Signature) -> Result<(&str, &str, &str, &str)> {
    match buf_params(signature).as_slice() {
        [q, k, v, y] => Ok((
            q.name.as_str(),
            k.name.as_str(),
            v.name.as_str(),
            y.name.as_str(),
        )),
        _ => Err(
            Error::new(Code::Internal, "注意力签名需要 4 个缓冲（Q/K/V/Y）")
                .with("engine::tiled::metal_tiled_attention"),
        ),
    }
}

/// 两个整数标量 `(S, D)` 的样本值。
pub(crate) fn attention_dims(signature: &Signature, sample: &Sample) -> Result<(usize, usize)> {
    match scalar_params(signature).as_slice() {
        [s, d] => Ok((dim(sample, s)?, dim(sample, d)?)),
        _ => Err(
            Error::new(Code::Internal, "注意力签名需要 2 个整数标量（S/D）")
                .with("engine::tiled::metal_tiled_attention"),
        ),
    }
}
