#![allow(unsafe_code)]
//! Zero-copy byte views for storage elements.
//!
//! Hand-written core of `zerocopy`'s `Ref` validation (length divisibility +
//! alignment) with none of its derive machinery. The single `unsafe` site is
//! the `from_raw_parts` reinterpret; every public function is safe Rust, so
//! the rest of the crate never touches the byte reinterpretation.
//!
//! This file is the `tensor` crate's restricted unsafe-exemption island
//! (registered in AGENTS.md §6 and docs/design.md §21.8, enforced by
//! `scripts/checkunsafe.sh`).

use crate::{byteable, tensorerror};

/// Interprets `bytes` as a borrowed slice of `T` without copying.
///
/// Succeeds when `bytes.len()` is a whole multiple of `size_of::<T>()` and the
/// slice start is aligned to `align_of::<T>()`; the empty slice always
/// succeeds. Memory-mapped data sections satisfy the alignment rule for every
/// storage element because page alignment is a multiple of each element
/// alignment.
///
/// # Errors
///
/// Returns [`tensorerror::byteconversion`] when the byte length is not a whole
/// multiple of the element width, and [`tensorerror::viewmisaligned`] when the
/// start address is not aligned to `align_of::<T>()`.
pub fn trybyteview<T: byteable>(bytes: &[u8]) -> Result<&[T], tensorerror> {
    let width = core::mem::size_of::<T>();
    if !bytes.len().is_multiple_of(width) {
        return Err(tensorerror::byteconversion);
    }
    if bytes.is_empty() {
        return Ok(&[]);
    }
    let align = core::mem::align_of::<T>();
    if !(bytes.as_ptr() as usize).is_multiple_of(align) {
        return Err(tensorerror::viewmisaligned { align });
    }
    // SAFETY: every `byteable` element type (f16, bf16, f32, f64, i8, u8,
    // i32, i64) accepts every bit pattern of its width as a valid value, so
    // the source bytes are valid `T`s. Length divisibility (checked above)
    // makes the element count exact. Alignment (checked above) satisfies the
    // reference rule for `*const T`; the empty slice was returned above, so
    // `from_raw_parts` never sees it. The returned slice borrows from
    // `bytes`, so it cannot outlive the underlying bytes.
    unsafe {
        Ok(core::slice::from_raw_parts(
            bytes.as_ptr().cast::<T>(),
            bytes.len() / width,
        ))
    }
}
