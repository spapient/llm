//! Runtime instruction-tier detection.
//!
//! The detection macros (`is_x86_feature_detected!` / `is_aarch64_feature_detected!`)
//! are safe: they only read a CPUID/auxiliary flag and return a bool. The
//! `unsafe` responsibility begins at the *call* of a `#[target_feature]`
//! kernel, which is confined to the dispatch arms in `lib.rs` and covered by
//! their `SAFETY:` comments.

/// The widest instruction tier available on the current CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// x86-64-v3 tier: AVX2 + FMA, eight `f32` lanes per vector.
    Avx2,
    /// AArch64 NEON tier: four `f32` lanes per vector.
    Neon,
    /// Plain scalar fallback (auto-vectorized by the compiler).
    Scalar,
}

/// Detects the widest available tier. The feature check is paid once per call;
/// callers are expected to call this once per kernel invocation and reuse the
/// returned [`Isa`] across their inner loops.
#[must_use]
pub fn detect() -> Isa {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Isa::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Isa::Neon;
        }
    }
    Isa::Scalar
}
