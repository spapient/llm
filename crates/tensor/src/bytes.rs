//! Safe byte <-> element conversion.
//!
//! Hand-written replacement for the `bytemuck` `Pod` machinery, restricted to
//! the element types the workspace stores. Unlike `bytemuck`, there is no
//! zero-copy reinterpretation: every conversion copies through the native
//! `from_ne_bytes` / `to_ne_bytes` array constructors, so the module is safe
//! Rust throughout and carries no `unsafe` site.

use crate::halffloat::{bf16, f16};

/// A primitive element that can be read from and written to raw bytes in
/// native-endian storage order.
///
/// Implemented for every [`element`](crate::element) storage type. The byte
/// width of each implementation is `size_of::<Self>()`; callers pass slices of
/// exactly that length, which [`readfrom`](byteable::readfrom) reports as
/// `None` when violated.
pub trait byteable: Copy {
    /// Reads one element from a byte slice of exactly `size_of::<Self>()`
    /// bytes.
    fn readfrom(bytes: &[u8]) -> Option<Self>;

    /// Writes this element into a byte slice of exactly `size_of::<Self>()`
    /// bytes.
    fn writeinto(&self, out: &mut [u8]);
}

impl byteable for f32 {
    fn readfrom(bytes: &[u8]) -> Option<Self> {
        Some(f32::from_ne_bytes(<[u8; 4]>::try_from(bytes).ok()?))
    }

    fn writeinto(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_ne_bytes());
    }
}

impl byteable for f64 {
    fn readfrom(bytes: &[u8]) -> Option<Self> {
        Some(f64::from_ne_bytes(<[u8; 8]>::try_from(bytes).ok()?))
    }

    fn writeinto(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_ne_bytes());
    }
}

impl byteable for i8 {
    fn readfrom(bytes: &[u8]) -> Option<Self> {
        Some(i8::from_ne_bytes(<[u8; 1]>::try_from(bytes).ok()?))
    }

    fn writeinto(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_ne_bytes());
    }
}

impl byteable for u8 {
    fn readfrom(bytes: &[u8]) -> Option<Self> {
        Some(u8::from_ne_bytes(<[u8; 1]>::try_from(bytes).ok()?))
    }

    fn writeinto(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_ne_bytes());
    }
}

impl byteable for i32 {
    fn readfrom(bytes: &[u8]) -> Option<Self> {
        Some(i32::from_ne_bytes(<[u8; 4]>::try_from(bytes).ok()?))
    }

    fn writeinto(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_ne_bytes());
    }
}

impl byteable for i64 {
    fn readfrom(bytes: &[u8]) -> Option<Self> {
        Some(i64::from_ne_bytes(<[u8; 8]>::try_from(bytes).ok()?))
    }

    fn writeinto(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_ne_bytes());
    }
}

impl byteable for f16 {
    fn readfrom(bytes: &[u8]) -> Option<Self> {
        Some(f16::from_bits(u16::from_ne_bytes(
            <[u8; 2]>::try_from(bytes).ok()?,
        )))
    }

    fn writeinto(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_bits().to_ne_bytes());
    }
}

impl byteable for bf16 {
    fn readfrom(bytes: &[u8]) -> Option<Self> {
        Some(bf16::from_bits(u16::from_ne_bytes(
            <[u8; 2]>::try_from(bytes).ok()?,
        )))
    }

    fn writeinto(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_bits().to_ne_bytes());
    }
}

/// Serializes a slice of elements into raw bytes in storage order.
///
/// The output length is `data.len() * size_of::<T>()`. This is a copying
/// conversion (the safe counterpart of `bytemuck::cast_slice`).
#[must_use]
pub fn bytesof<T: byteable>(data: &[T]) -> Vec<u8> {
    let width = core::mem::size_of::<T>();
    let mut out = Vec::with_capacity(core::mem::size_of_val(data));
    let mut scratch = [0u8; 16];
    for value in data {
        let slot = &mut scratch[..width];
        value.writeinto(slot);
        out.extend_from_slice(slot);
    }
    out
}

/// Reads a slice of elements from raw bytes in storage order.
///
/// Returns `None` when the byte length is not a whole multiple of the element
/// width; each element is read with an unaligned, checked read.
#[must_use]
pub fn frombytes<T: byteable>(bytes: &[u8]) -> Option<Vec<T>> {
    let width = core::mem::size_of::<T>();
    if width == 0 || !bytes.len().is_multiple_of(width) {
        return None;
    }
    bytes.chunks_exact(width).map(byteable::readfrom).collect()
}
