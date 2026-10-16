use std::borrow::Cow;

use crate::error::checkpointerror;
use crate::hash::contenthash;

/// Codec byte for an uncompressed (raw) block.
pub(crate) const CODECRAW: u8 = 0;

/// Codec byte for an LZ4-compressed block.
pub(crate) const CODECLZ4: u8 = 1;

/// Codec byte for a Zstd-compressed block.
pub(crate) const CODECZSTD: u8 = 2;

/// The target average block size for content-defined chunking, in bytes.
///
/// `256 * 1024 = 2^18`; the actual block lengths vary between [`MINBLOCK`] and
/// [`MAXBLOCK`] according to the rolling-hash cut condition.
pub const BLOCKSIZE: usize = 256 * 1024;

/// Minimum block size for content-defined chunking: 64 KiB.
pub const MINBLOCK: usize = 64 * 1024;

/// Maximum block size for content-defined chunking: 1 MiB.
pub const MAXBLOCK: usize = 1024 * 1024;

/// Cut mask: the 18 low bits, for a 256 KiB average block size.
const CUTMASK: u64 = (1 << 18) - 1;

/// The deterministic Gear hash table, built once at compile time.
const GEAR: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut i = 0;
    while i < 256 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        table[i] = state;
        i += 1;
    }
    table
};

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

/// Advances the Gear rolling hash by one byte.
fn gearroll(hash: u64, byte: u8) -> u64 {
    (hash << 1).wrapping_add(GEAR[byte as usize])
}

/// Splits `data` into content-defined blocks and computes each block's content
/// identity.
///
/// Block boundaries are chosen by a Gear rolling hash, so an insertion or
/// deletion shifts only the few boundaries near the change; the blocks after it
/// keep their content and deduplicate. Block lengths stay within
/// [`MINBLOCK`]..=[`MAXBLOCK`], except the final block, which may be shorter.
#[must_use]
pub fn chunk(data: &[u8]) -> Vec<blockref> {
    let mut blocks = Vec::new();
    let mut start = 0usize;
    let mut hash = 0u64;
    for (i, &byte) in data.iter().enumerate() {
        hash = gearroll(hash, byte);
        let len = i + 1 - start;
        if len >= MAXBLOCK || (len >= MINBLOCK && hash & CUTMASK == 0) {
            blocks.push(blockref::new(contenthash(&data[start..i + 1]), len as u32));
            start = i + 1;
            hash = 0;
        }
    }
    if start < data.len() {
        blocks.push(blockref::new(
            contenthash(&data[start..]),
            (data.len() - start) as u32,
        ));
    }
    blocks
}

/// Compresses a block, returning its codec byte and the stored bytes.
///
/// The block is stored raw when compression does not shrink it.
pub(crate) fn compressblock(bytes: &[u8]) -> (u8, Vec<u8>) {
    let lz4 = lz4_flex::compress_prepend_size(bytes);
    let mut best = (CODECRAW, bytes.to_vec());
    if lz4.len() < best.1.len() {
        best = (CODECLZ4, lz4);
    }
    match zstd::stream::encode_all(bytes, 3) {
        Ok(zstd) if zstd.len() < best.1.len() => (CODECZSTD, zstd),
        _ => best,
    }
}

/// Decompresses a block's stored bytes back to its uncompressed form.
///
/// Raw blocks are returned borrowed (zero-copy); compressed blocks are
/// decompressed into an owned buffer.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidmanifest`] when the codec is unknown or
/// the stored bytes do not decode.
pub(crate) fn decompressblock<'a>(
    codec: u8,
    stored: &'a [u8],
) -> Result<Cow<'a, [u8]>, checkpointerror> {
    match codec {
        CODECRAW => Ok(Cow::Borrowed(stored)),
        CODECLZ4 => lz4_flex::decompress_size_prepended(stored)
            .map(Cow::Owned)
            .map_err(|_| checkpointerror::invalidmanifest {
                reason: "lz4 decompression failed".to_string(),
            }),
        CODECZSTD => zstd::stream::decode_all(stored)
            .map(Cow::Owned)
            .map_err(|_| checkpointerror::invalidmanifest {
                reason: "zstd decompression failed".to_string(),
            }),
        _ => Err(checkpointerror::invalidmanifest {
            reason: format!("unknown block codec {codec}"),
        }),
    }
}
