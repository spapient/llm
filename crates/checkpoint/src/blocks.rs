use std::borrow::Cow;

use crate::error::checkpointerror;
use crate::gate;
use crate::hash::contenthash;

/// Codec byte for an uncompressed (raw) block.
pub(crate) const CODECRAW: u8 = 0;

/// Codec byte for a gate rANS-entropy-coded block.
pub(crate) const CODECGATE: u8 = 1;

/// Transform byte for an untransformed block.
pub(crate) const TRANSFORMNONE: u8 = 0;

/// Transform byte for the bit-shuffle + delta pre-pass over 16-bit elements.
pub(crate) const TRANSFORMF16: u8 = 2;

/// Transform byte for the bit-shuffle + delta pre-pass over 32-bit elements.
pub(crate) const TRANSFORMF32: u8 = 4;

/// The target average block size for content-defined chunking, in bytes.
///
/// `256 * 1024 = 2^18`; the actual block lengths vary between [`MINBLOCK`] and
/// [`MAXBLOCK`] according to the rolling-hash cut condition.
pub const BLOCKSIZE: usize = 256 * 1024;

/// Minimum block size for content-defined chunking: 64 KiB.
pub const MINBLOCK: usize = 64 * 1024;

/// Maximum block size for content-defined chunking: 1 MiB.
pub const MAXBLOCK: usize = 1024 * 1024;

/// Segment size for parallel chunking: data larger than this is split into
/// segments that each chunk independently on available cores.
const SEGMENT: usize = 16 * 1024 * 1024;

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
/// [`MINBLOCK`]..=[`MAXBLOCK`], except the final block of the data (and of each
/// parallel segment), which may be shorter.
///
/// Data larger than [`SEGMENT`] is chunked in parallel: each segment is cut at
/// its boundary and chunked independently. Deduplication therefore holds within
/// a segment, and only the few blocks around a segment boundary lose
/// cross-segment identity — a bounded, negligible loss for multi-core chunking.
#[must_use]
pub fn chunk(data: &[u8]) -> Vec<blockref> {
    if data.len() <= SEGMENT {
        return chunksegment(data);
    }
    let segs: Vec<&[u8]> = data.chunks(SEGMENT).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = segs
            .iter()
            .map(|seg| scope.spawn(move || chunksegment(seg)))
            .collect();
        let mut blocks = Vec::new();
        for handle in handles {
            if let Ok(segblocks) = handle.join() {
                blocks.extend(segblocks);
            }
        }
        blocks
    })
}

/// Chunks one segment of data with the Gear rolling hash. See [`chunk`].
fn chunksegment(data: &[u8]) -> Vec<blockref> {
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

/// Transposes the byte planes of `data` and applies a byte-wise delta within
/// each plane, so the codec stage sees the sign/exponent and mantissa bytes
/// grouped and differenced along the element sequence.
///
/// This is a pure, exact byte transform: [`unbitshuffle_delta`] inverts it for
/// any input, and it composes with content-defined chunking because the block
/// identity is computed over the untransformed bytes. The transform covers the
/// largest whole number of `elem`-byte elements (`elem * n` bytes, `n =
/// data.len() / elem`); the trailing 0..`elem` bytes are carried verbatim at
/// the end of the output, so blocks of any length are transformed losslessly.
/// Content-defined chunk boundaries are arbitrary byte offsets, so without the
/// tail clause 70% of chunk bytes would be skipped (measured on a 16 MiB
/// profile, losing 13 percentage points of total savings).
///
/// On correlated floating-point data (realistic weights) the transformed
/// stream compresses far better than the raw byte stream: measured ~42% vs ~0%
/// total savings for gate rANS on a 256 KiB chunk profile (see the codec
/// selection notes in `compressblock`).
#[must_use]
pub fn bitshuffle_delta(data: &[u8], elem: usize) -> Vec<u8> {
    let n = data.len() / elem;
    let mut out = vec![0u8; data.len()];
    for plane in 0..elem {
        let mut prev = 0u8;
        for i in 0..n {
            let byte = data[i * elem + plane];
            out[plane * n + i] = byte.wrapping_sub(prev);
            prev = byte;
        }
    }
    out[elem * n..].copy_from_slice(&data[elem * n..]);
    out
}

/// Inverts [`bitshuffle_delta`]: reverses the intra-plane delta and the plane
/// transposition, restoring the original bytes exactly. The trailing tail
/// bytes are copied back verbatim.
#[must_use]
pub fn unbitshuffle_delta(data: &[u8], elem: usize) -> Vec<u8> {
    let n = data.len() / elem;
    let mut out = vec![0u8; data.len()];
    for plane in 0..elem {
        let mut acc = 0u8;
        for i in 0..n {
            acc = acc.wrapping_add(data[plane * n + i]);
            out[i * elem + plane] = acc;
        }
    }
    out[elem * n..].copy_from_slice(&data[elem * n..]);
    out
}

/// Compresses a block, returning its codec byte, transform byte, and the stored
/// bytes.
///
/// The candidate set is deliberately minimal: the transform pre-pass
/// (`{none, f16, f32}`) is tried with a single codec, gate (rANS entropy
/// coding), and the block is stored raw when no candidate shrinks it. Raw
/// storage protects high-entropy blocks from codec expansion.
pub(crate) fn compressblock(bytes: &[u8]) -> (u8, u8, Vec<u8>) {
    let mut best: (u8, u8, Vec<u8>) = (CODECRAW, TRANSFORMNONE, bytes.to_vec());
    for transform in [TRANSFORMNONE, TRANSFORMF16, TRANSFORMF32] {
        let elem = transform as usize;
        let transformed: Vec<u8> = if transform == TRANSFORMNONE {
            Vec::new()
        } else {
            bitshuffle_delta(bytes, elem)
        };
        let source: &[u8] = if transform == TRANSFORMNONE {
            bytes
        } else {
            &transformed
        };
        let compressed = gate::compress(source, elem);
        if compressed.len() < best.2.len() {
            best = (CODECGATE, transform, compressed);
        }
    }
    best
}

/// Decompresses a block's stored bytes back to its uncompressed form.
///
/// Raw blocks are returned borrowed (zero-copy); compressed blocks are
/// decompressed into an owned buffer, and a transformed block is inverted
/// after decompression.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidmanifest`] when the codec or transform is
/// unknown or the stored bytes do not decode.
pub(crate) fn decompressblock<'a>(
    codec: u8,
    transform: u8,
    stored: &'a [u8],
) -> Result<Cow<'a, [u8]>, checkpointerror> {
    let elem = transform as usize;
    let cow = match codec {
        CODECRAW => Cow::Borrowed(stored),
        CODECGATE => Cow::Owned(gate::decompress(stored, elem)?),
        _ => {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("unknown block codec {codec}"),
            });
        }
    };
    match transform {
        TRANSFORMNONE => Ok(cow),
        TRANSFORMF16 | TRANSFORMF32 => Ok(Cow::Owned(unbitshuffle_delta(&cow, elem))),
        _ => Err(checkpointerror::invalidmanifest {
            reason: format!("unknown block transform {transform}"),
        }),
    }
}
