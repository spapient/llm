//! IEEE 754 binary16 (`f16`) and bfloat16 (`bf16`) storage types.
//!
//! Hand-written replacements for the `half` crate so the workspace carries no
//! third-party half-precision dependency. Both types are storage-only: no
//! backend declares their execution, matching the previous `half` usage. The
//! conversions implement round-to-nearest-even with explicit special-value
//! handling (signed zero, infinities, NaN payloads) and are covered by the
//! boundary vectors in `tests/it/tensorunits.rs`.

/// IEEE 754 binary16: 1 sign bit, 5 exponent bits, 10 mantissa bits.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct f16(u16);

impl f16 {
    /// Reinterprets the raw bit pattern as an `f16`.
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit pattern.
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f32` to `f16` with round-to-nearest-even.
    ///
    /// Signed zero, infinities and NaN payloads are preserved; values that
    /// overflow the binary16 range round to an infinity of the same sign.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let biased = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7f_ffff;

        // Signed zero.
        if bits == 0 || bits == 0x8000_0000 {
            return Self(sign);
        }
        // Infinity and NaN: keep the exponent all-ones; a NaN keeps its
        // payload high bits (forcing a quiet bit so a zero payload stays NaN).
        if biased == 0xff {
            let payload = if mantissa == 0 {
                0
            } else {
                0x0200 | (mantissa >> 13)
            };
            return Self(sign | 0x7c00 | payload.min(0x03ff) as u16);
        }
        // Rebase the exponent: f32 bias 127 -> f16 bias 15.
        let exp = biased - 127 + 15;
        if exp >= 0x1f {
            return Self(sign | 0x7c00); // overflow -> signed infinity
        }
        // Normalized mantissa including the implicit bit (24 bits).
        let full = mantissa | 0x80_0000;
        if exp <= 0 {
            // Subnormal f16: shift right by (1 - exp); keep 10 bits with RNE.
            let shift = (1 - exp) as u32; // 1..=14
            if shift > 24 {
                return Self(sign); // magnitude below the smallest subnormal
            }
            let lsb = (full >> (13 + shift)) & 1;
            let guard = (full >> (12 + shift)) & 1;
            let sticky = full & ((1u32 << (12 + shift)) - 1) != 0;
            let half_mant = full >> (13 + shift);
            let round = if guard == 1 && (sticky || lsb == 1) {
                1
            } else {
                0
            };
            let result = half_mant + round;
            if result >= 0x0400 {
                // Rounded up into the smallest normal.
                Self(sign | 0x0400)
            } else {
                Self(sign | result as u16)
            }
        } else {
            // Normal f16: shift right by 13 with RNE on the dropped bits.
            let lsb = (mantissa >> 13) & 1;
            let guard = (mantissa >> 12) & 1;
            let sticky = mantissa & 0x0fff != 0;
            let half_mant = mantissa >> 13;
            let round = if guard == 1 && (sticky || lsb == 1) {
                1
            } else {
                0
            };
            let result = half_mant + round;
            if result >= 0x0400 {
                // Rounded into the next exponent.
                Self(sign | (((exp + 1) as u16) << 10))
            } else {
                Self(sign | ((exp as u16) << 10) | result as u16)
            }
        }
    }

    /// Converts an `f16` to `f32` exactly.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = ((bits as u32) & 0x8000) << 16;
        let biased = (bits >> 10) & 0x1f;
        let mantissa = (bits & 0x03ff) as u32;
        match biased {
            0 if mantissa == 0 => f32::from_bits(sign), // signed zero
            0 => {
                // Subnormal: value = mantissa * 2^-24. Normalize to the
                // 1.x * 2^exp form: with leading bit at position `leading`
                // (0..=9), value = (1 + rest/2^leading) * 2^(leading - 24).
                let leading = 31 - mantissa.leading_zeros() as i32;
                let exp = 103 + leading;
                let rest = mantissa - (1u32 << leading);
                let frac = rest << (23 - leading);
                f32::from_bits(sign | ((exp as u32) << 23) | frac)
            }
            0x1f if mantissa == 0 => f32::from_bits(sign | 0x7f80_0000), // infinity
            0x1f => f32::from_bits(sign | 0x7fc0_0000 | (mantissa << 13)), // NaN
            _ => {
                let exp = (biased as u32) + 127 - 15;
                f32::from_bits(sign | (exp << 23) | (mantissa << 13))
            }
        }
    }

    /// Converts an `f64` to `f16` through `f32` (binary16 cannot represent
    /// more precision than `f32`).
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self::from_f32(value as f32)
    }

    /// Converts an `f16` to `f64` exactly (through `f32`).
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
}

/// bfloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits (top 16 bits of an
/// `f32`), rounded to nearest even.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct bf16(u16);

impl bf16 {
    /// Reinterprets the raw bit pattern as a `bf16`.
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit pattern.
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f32` to `bf16` with round-to-nearest-even.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            // Keep a NaN a NaN regardless of rounding: force a nonzero payload.
            return Self(((bits >> 16) as u16) | 0x0040);
        }
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + lsb);
        Self((rounded >> 16) as u16)
    }

    /// Converts a `bf16` to `f32` exactly (zero-extended).
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Converts an `f64` to `bf16` through `f32`.
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self::from_f32(value as f32)
    }

    /// Converts a `bf16` to `f64` exactly (through `f32`).
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
}

impl core::fmt::Debug for f16 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_tuple("f16").field(&self.to_f32()).finish()
    }
}

impl core::fmt::Debug for bf16 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_tuple("bf16").field(&self.to_f32()).finish()
    }
}
