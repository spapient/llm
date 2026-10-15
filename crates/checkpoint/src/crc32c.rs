/// The reflected Castagnoli polynomial `0x1EDC6F41`, bit-reversed.
///
/// CRC32C uses the reflected (least-significant-bit-first) form for the
/// standard table-driven computation; this is the CRC32C counterpart of the
/// IEEE CRC32 polynomial `0xEDB88320`.
const CRC32CPOLY: u32 = 0x82F63B78;

/// The 256-entry CRC32C lookup table, built once at compile time.
const CRC32CTABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC32CPOLY
            } else {
                crc >> 1
            };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Computes the CRC32C (Castagnoli) checksum of `bytes`.
///
/// The value is a checksum, not a hash: it detects accidental corruption and
/// bit errors, but it provides no collision resistance and must never be used
/// for deduplication or content identity. The canonical known-answer vector
/// `crc32c(b"123456789") == 0xE306_9283` pins the definition.
#[must_use]
pub fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32CTABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}
