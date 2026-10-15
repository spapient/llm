/// The byte length of a BLAKE3 content identity (128 bits).
const CONTENTHASHLEN: usize = 16;

/// Computes the full BLAKE3-256 digest of `bytes`.
///
/// This is the cryptographic anchor of the `.u` format: the snapshot root hash
/// is `blake3hash` of the canonical manifest, and the 128-bit [`contenthash`]
/// is its leading 16 bytes. BLAKE3-256 has 128-bit collision resistance and is
/// used wherever there is no fallback check (see `docs/design.md` §12.12).
#[must_use]
pub fn blake3hash(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Computes the 128-bit BLAKE3 content identity of `bytes`.
///
/// The value is the leading 16 bytes of [`blake3hash`]; both come from the same
/// BLAKE3 computation, so a content identity costs nothing beyond the digest.
/// It is the persistent identity for deduplication and cross-file block
/// resolution; 128 bits is collision-free at practical model scales but is not
/// an adversarial guarantee.
#[must_use]
pub fn contenthash(bytes: &[u8]) -> [u8; CONTENTHASHLEN] {
    let digest = blake3::hash(bytes);
    let mut out = [0u8; CONTENTHASHLEN];
    out.copy_from_slice(&digest.as_bytes()[..CONTENTHASHLEN]);
    out
}
