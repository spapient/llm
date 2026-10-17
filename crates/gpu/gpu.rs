#![deny(unsafe_code)]
//! Native GPU execution backends: Metal, CUDA, and MUSA.
//!
//! `gpu` is a Layer 1 crate that owns the three platform backends as modules
//! and exposes each one under its own namespace — `gpu::metal`, `gpu::cuda`,
//! `gpu::musa`. Every module implements the frozen [`api::backend`] and
//! [`api::compute`] contracts for `f32` operands and holds no request state and
//! no model weights.
//!
//! Each backend keeps its own device layer, kernel sources, build pipeline, and
//! `unsafe` island; only platform-independent code is shared here. The shared
//! set is deliberately small: tensor construction, host-side concatenation, and
//! the shape predicates that the three backends must classify identically.
//! Sharing an operator dispatch would require a single trait over three
//! genuinely different resource models (streams, command buffers, slot chains)
//! and is not attempted; see `docs/gpu-backend-consolidation-assessment.md`.
//!
//! # Platform
//!
//! The whole crate compiles to an empty library on targets that own no backend,
//! so `cargo check --workspace --all-targets` stays green everywhere.
//!
//! # Unsafe
//!
//! This crate is one of the carriers of the workspace's restricted `unsafe`
//! exemption (docs/design.md §21.8). The crate root uses
//! `#![deny(unsafe_code)]`; each backend's `ffi` module declares the crate's
//! three allowances — one per backend — and confines every platform call there,
//! each preceded by a `// SAFETY:` comment. The registered site counts are
//! metal = 3, cuda = 2, musa = 2. `scripts/checkunsafe.sh` enforces the exact
//! counts and the SAFETY comments mechanically.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated.

use ::tensor::{shape, tensor, tensorview};
use api::backenderror;

#[cfg(target_os = "linux")]
pub mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(all(target_os = "linux", musa_toolkit))]
pub mod musa;

/// Rebuilds a kernel result into a tensor of `outshape`.
///
/// `backend` names the backend in the failure diagnostic.
pub(crate) fn wrap(
    backend: &'static str,
    outshape: shape,
    data: Vec<f32>,
) -> Result<tensor<f32>, backenderror> {
    tensor::new(outshape, data).map_err(|e| backenderror::executionfailed {
        backend,
        kernel: "construct".into(),
        message: e.to_string(),
    })
}

/// Concatenates two `[_, width]` views into one `[a + b, width]` tensor.
pub(crate) fn concat(
    backend: &'static str,
    a: &tensorview<'_, f32>,
    b: &tensorview<'_, f32>,
) -> Result<tensor<f32>, backenderror> {
    let width = a.getshape().getdims()[1];
    let mut data = a.getdata().to_vec();
    data.extend_from_slice(b.getdata());
    let len = data.len() / width;
    let outshape = shape::new(vec![len, width]).map_err(|_| backenderror::sizeoverflow)?;
    wrap(backend, outshape, data)
}

/// Concatenates two rank-two shapes along their sequence axis.
pub(crate) fn concatshape(a: &shape, b: &shape) -> Result<shape, backenderror> {
    let ad = a.getdims();
    let bd = b.getdims();
    if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[1] {
        return Err(backenderror::shapemismatch { op: "attention" });
    }
    shape::new(vec![ad[0] + bd[0], ad[1]]).map_err(|_| backenderror::sizeoverflow)
}

/// Validates scan ranks, channel alignment, and state-width alignment,
/// returning `(n, d, statewidth)`.
///
/// The spectral precondition on `a` is a separate predicate ([`scanstable`]):
/// the single-operator path of all three backends applies it; `musa`'s
/// recorded path does not read operand data and therefore skips it.
pub(crate) fn scanshape(
    input: &shape,
    init: &shape,
    a: &shape,
    b: &shape,
    c: &shape,
    skip: &shape,
) -> Result<(usize, usize, usize), backenderror> {
    let id = input.getdims();
    let initd = init.getdims();
    let ad = a.getdims();
    let bd = b.getdims();
    let cd = c.getdims();
    let skipd = skip.getdims();
    if id.len() != 2
        || initd.len() != 2
        || ad.len() != 2
        || bd.len() != 2
        || cd.len() != 2
        || skipd.len() != 1
    {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    let d = id[1];
    if ad[0] != d || bd[0] != d || cd[0] != d || skipd[0] != d || initd[0] != d {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    let statewidth = initd[1];
    if ad[1] != statewidth || bd[1] != statewidth || cd[1] != statewidth {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    Ok((id[0], d, statewidth))
}

/// Enforces the scan's spectral precondition: every `a` entry is finite and
/// inside `[-1, 1]`.
pub(crate) fn scanstable(a: &tensorview<'_, f32>) -> Result<(), backenderror> {
    for &value in a.getdata() {
        if value.abs() > 1.0 || value.is_nan() {
            return Err(backenderror::invalidparameter { op: "scan" });
        }
    }
    Ok(())
}

/// Builds the gather output shape: `rows` replaces the table's first dimension.
pub(crate) fn embeddingoutshape(tableshape: &shape, rows: usize) -> Result<shape, backenderror> {
    let mut outdims = Vec::with_capacity(tableshape.getrank());
    outdims.push(rows);
    outdims.extend_from_slice(&tableshape.getdims()[1..]);
    shape::new(outdims).map_err(|_| backenderror::sizeoverflow)
}
