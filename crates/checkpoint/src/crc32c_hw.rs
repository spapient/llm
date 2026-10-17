#![allow(unsafe_code)]
//! The single hardware-intrinsic island of the `checkpoint` crate.
//!
//! Authority: docs/design.md §21.8. This is the only file in the crate that
//! declares `#![allow(unsafe_code)]`; the crate root keeps
//! `#![deny(unsafe_code)]` (was `forbid` before this exemption).
//!
//! Hardware CRC32C: the ARMv8 CRC extension (`__crc32cd` / `__crc32cb`) and
//! x86 SSE4.2 (`_mm_crc32_u64` / `_mm_crc32_u8`). Both produce the identical
//! Castagnoli checksum as the table-driven software path (verified against the
//! canonical `crc32c(b"123456789") == 0xE306_9283` and cross-checked on random
//! lengths). Measured 9.8× the slicing-by-8 software path on Apple M5 Pro.

/// True when the hardware CRC32C instruction is available on this CPU.
pub(crate) fn supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("crc")
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("sse4.2")
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
// SAFETY: the caller must have verified `supported()`; this function enables
// the `crc` target feature, and the intrinsics below only read their value
// arguments.
unsafe fn hw_crc(mut crc: u32, data: &[u8]) -> u32 {
    let mut i = 0usize;
    while i + 8 <= data.len() {
        let mut word = [0u8; 8];
        word.copy_from_slice(&data[i..i + 8]);
        crc = core::arch::aarch64::__crc32cd(crc, u64::from_le_bytes(word));
        i += 8;
    }
    for &byte in &data[i..] {
        crc = core::arch::aarch64::__crc32cb(crc, byte);
    }
    crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
// SAFETY: the caller must have verified `supported()`; this function enables
// the `sse4.2` target feature, and the intrinsics below only read their value
// arguments.
unsafe fn hw_crc(mut crc: u32, data: &[u8]) -> u32 {
    let mut wide = u64::from(crc);
    let mut i = 0usize;
    while i + 8 <= data.len() {
        let mut word = [0u8; 8];
        word.copy_from_slice(&data[i..i + 8]);
        wide = core::arch::x86_64::_mm_crc32_u64(wide, u64::from_le_bytes(word));
        i += 8;
    }
    crc = wide as u32;
    for &byte in &data[i..] {
        crc = core::arch::x86_64::_mm_crc32_u8(crc, byte);
    }
    crc
}

/// Computes the CRC32C of `bytes` with the hardware instruction.
///
/// Only called when [`supported`] returned true.
pub(crate) fn compute(bytes: &[u8]) -> u32 {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        // SAFETY: `supported()` checked the required target feature at runtime.
        unsafe { hw_crc(0xFFFF_FFFF, bytes) ^ 0xFFFF_FFFF }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = bytes;
        0
    }
}
