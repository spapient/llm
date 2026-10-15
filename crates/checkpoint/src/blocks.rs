use crate::error::checkpointerror;
use crate::hash::contenthash;

/// Codec byte for an uncompressed (raw) block.
pub(crate) const CODECRAW: u8 = 0;

/// Codec byte for an LZ4-compressed block.
pub(crate) const CODECLZ4: u8 = 1;

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

/// Compresses a block, returning its codec byte and the stored bytes.
///
/// The block is stored raw when compression does not shrink it.
pub(crate) fn compressblock(bytes: &[u8]) -> (u8, Vec<u8>) {
    let compressed = lz4_flex::compress_prepend_size(bytes);
    if compressed.len() < bytes.len() {
        (CODECLZ4, compressed)
    } else {
        (CODECRAW, bytes.to_vec())
    }
}

/// Decompresses a block's stored bytes back to its uncompressed form.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidmanifest`] when the codec is unknown or
/// the stored bytes do not decode.
pub(crate) fn decompressblock(codec: u8, stored: &[u8]) -> Result<Vec<u8>, checkpointerror> {
    match codec {
        CODECRAW => Ok(stored.to_vec()),
        CODECLZ4 => lz4_flex::decompress_size_prepended(stored).map_err(|_| {
            checkpointerror::invalidmanifest {
                reason: "lz4 decompression failed".to_string(),
            }
        }),
        _ => Err(checkpointerror::invalidmanifest {
            reason: format!("unknown block codec {codec}"),
        }),
    }
}
