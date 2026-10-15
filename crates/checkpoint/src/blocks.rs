use crate::hash::contenthash;

/// The fixed block size for `.u` block addressing, in bytes.
///
/// `256 * 1024 = 2^18`. The power of two keeps block-boundary arithmetic a
/// shift and a mask and keeps blocks 4 KiB page-aligned (see
/// `docs/design.md` §12.12).
pub const BLOCKSIZE: usize = 256 * 1024;

/// A reference to one block of a `.u` data section.
///
/// The hash is the block's BLAKE3-128 content identity over its uncompressed
/// bytes; `len` is its uncompressed byte length. Every block except the last
/// has length [`BLOCKSIZE`]; the last may be shorter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct blockref {
    hash: [u8; 16],
    len: u32,
}

impl blockref {
    /// Constructs a block reference from a content identity and a byte length.
    pub(crate) fn new(hash: [u8; 16], len: u32) -> Self {
        Self { hash, len }
    }

    /// Returns the block's 128-bit content identity.
    #[must_use]
    pub fn gethash(&self) -> [u8; 16] {
        self.hash
    }

    /// Returns the block's uncompressed byte length.
    #[must_use]
    pub fn getlen(&self) -> u32 {
        self.len
    }
}

/// Splits `data` into [`BLOCKSIZE`]-sized blocks and computes each block's
/// content identity.
///
/// Empty input yields no blocks. Block boundaries fall on [`BLOCKSIZE`]
/// multiples, never mid-tensor.
#[must_use]
pub fn chunk(data: &[u8]) -> Vec<blockref> {
    data.chunks(BLOCKSIZE)
        .map(|block| {
            // block.len() <= BLOCKSIZE < u32::MAX, so the cast cannot truncate.
            blockref::new(contenthash(block), block.len() as u32)
        })
        .collect()
}
