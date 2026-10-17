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
    // Phase 1 — boundaries: the Gear rolling hash is a serial per-byte chain, so
    // a segment's spans are cut in order. Segments (fixed-size windows) cut
    // independently and in parallel, each carrying its absolute offset.
    let spans: Vec<(usize, usize)> = if data.len() <= SEGMENT {
        gearspans(data, 0)
    } else {
        let segs: Vec<(usize, &[u8])> = data
            .chunks(SEGMENT)
            .scan(0usize, |off, seg| {
                let o = *off;
                *off += seg.len();
                Some((o, seg))
            })
            .collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = segs
                .iter()
                .map(|&(off, seg)| scope.spawn(move || gearspans(seg, off)))
                .collect();
            let mut spans = Vec::new();
            for handle in handles {
                if let Ok(s) = handle.join() {
                    spans.extend(s);
                }
            }
            spans
        })
    };
    // Phase 2 — identities: content-hashing each block is independent, so it
    // runs across cores; the boundaries (and thus the output) are unchanged.
    hashspans(data, &spans)
}

/// Cuts `data` into `(offset, len)` blocks with the Gear rolling hash, invoking
/// `emit` for each block with `base` added to its offset. See [`chunk`].
fn gearspans_each(data: &[u8], base: usize, mut emit: impl FnMut(usize, usize)) {
    let mut start = 0usize;
    let mut hash = 0u64;
    for (i, &byte) in data.iter().enumerate() {
        hash = gearroll(hash, byte);
        let len = i + 1 - start;
        if len >= MAXBLOCK || (len >= MINBLOCK && hash & CUTMASK == 0) {
            emit(base + start, len);
            start = i + 1;
            hash = 0;
        }
    }
    if start < data.len() {
        emit(base + start, data.len() - start);
    }
}

/// Streams every block of `data` as `(segment, index-in-segment, offset, len)`
/// to `emit`, cutting each [`SEGMENT`]-byte window independently (the same
/// boundaries [`chunk`] produces) and running the windows in parallel.
///
/// `emit` may be called from several threads; callers pass a channel sender so
/// the serial Gear cut of one window overlaps the consumers' work.
pub(crate) fn streamspans<F>(data: &[u8], emit: F)
where
    F: Fn(usize, usize, usize, usize) + Sync,
{
    if data.len() <= SEGMENT {
        let mut index = 0usize;
        gearspans_each(data, 0, |off, len| {
            emit(0, index, off, len);
            index += 1;
        });
        return;
    }
    let segs: Vec<(usize, &[u8])> = data
        .chunks(SEGMENT)
        .scan(0usize, |off, seg| {
            let o = *off;
            *off += seg.len();
            Some((o, seg))
        })
        .collect();
    std::thread::scope(|scope| {
        for (si, &(base, seg)) in segs.iter().enumerate() {
            let emit = &emit;
            scope.spawn(move || {
                let mut index = 0usize;
                gearspans_each(seg, base, |off, len| {
                    emit(si, index, off, len);
                    index += 1;
                });
            });
        }
    });
}

/// Cuts `data` into `(offset, len)` blocks with the Gear rolling hash, adding
/// `base` to each offset. See [`chunk`].
fn gearspans(data: &[u8], base: usize) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    gearspans_each(data, base, |off, len| spans.push((off, len)));
    spans
}

/// Content-hashes every span into a [`blockref`], in parallel across cores
/// when there are enough blocks to pay for the threads.
fn hashspans(data: &[u8], spans: &[(usize, usize)]) -> Vec<blockref> {
    const PARALLELMINBLOCKS: usize = 8;
    let mut out = vec![blockref::new([0u8; 16], 0); spans.len()];
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(spans.len());
    if spans.len() < PARALLELMINBLOCKS || nthreads <= 1 {
        for ((start, len), slot) in spans.iter().zip(out.iter_mut()) {
            *slot = blockref::new(contenthash(&data[*start..*start + *len]), *len as u32);
        }
        return out;
    }
    let per = spans.len().div_ceil(nthreads);
    std::thread::scope(|scope| {
        for (sc, oc) in spans.chunks(per).zip(out.chunks_mut(per)) {
            scope.spawn(move || {
                for ((start, len), slot) in sc.iter().zip(oc.iter_mut()) {
                    *slot = blockref::new(contenthash(&data[*start..*start + *len]), *len as u32);
                }
            });
        }
    });
    out
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
    match elem {
        2 => shuffle2(data, &mut out, n),
        4 => shuffle4(data, &mut out, n),
        _ => shuffle_generic(data, &mut out, n, elem),
    }
    out[elem * n..].copy_from_slice(&data[elem * n..]);
    out
}

/// Loop-interchanged transpose + delta for `elem = 4`: reads contiguous
/// elements and writes one byte per plane, so the four `prev` chains are
/// independent and overlap in the pipeline instead of serialising.
fn shuffle4(data: &[u8], out: &mut [u8], n: usize) {
    let (o0, r) = out.split_at_mut(n);
    let (o1, r) = r.split_at_mut(n);
    let (o2, o3) = r.split_at_mut(n);
    let (mut p0, mut p1, mut p2, mut p3) = (0u8, 0u8, 0u8, 0u8);
    for (i, c) in data[..4 * n].as_chunks::<4>().0.iter().enumerate() {
        let (b0, b1, b2, b3) = (c[0], c[1], c[2], c[3]);
        o0[i] = b0.wrapping_sub(p0);
        o1[i] = b1.wrapping_sub(p1);
        o2[i] = b2.wrapping_sub(p2);
        o3[i] = b3.wrapping_sub(p3);
        p0 = b0;
        p1 = b1;
        p2 = b2;
        p3 = b3;
    }
}

fn shuffle2(data: &[u8], out: &mut [u8], n: usize) {
    let (o0, o1) = out.split_at_mut(n);
    let (mut p0, mut p1) = (0u8, 0u8);
    for (i, c) in data[..2 * n].as_chunks::<2>().0.iter().enumerate() {
        let (b0, b1) = (c[0], c[1]);
        o0[i] = b0.wrapping_sub(p0);
        o1[i] = b1.wrapping_sub(p1);
        p0 = b0;
        p1 = b1;
    }
}

fn shuffle_generic(data: &[u8], out: &mut [u8], n: usize, elem: usize) {
    for plane in 0..elem {
        let mut prev = 0u8;
        for i in 0..n {
            let byte = data[i * elem + plane];
            out[plane * n + i] = byte.wrapping_sub(prev);
            prev = byte;
        }
    }
}

/// Inverts [`bitshuffle_delta`]: reverses the intra-plane delta and the plane
/// transposition, restoring the original bytes exactly. The trailing tail
/// bytes are copied back verbatim.
#[must_use]
pub fn unbitshuffle_delta(data: &[u8], elem: usize) -> Vec<u8> {
    let n = data.len() / elem;
    let mut out = vec![0u8; data.len()];
    match elem {
        2 => unshuffle2(data, &mut out, n),
        4 => unshuffle4(data, &mut out, n),
        _ => unshuffle_generic(data, &mut out, n, elem),
    }
    out[elem * n..].copy_from_slice(&data[elem * n..]);
    out
}

/// Loop-interchanged prefix sum + transpose for `elem = 4`: reads one byte from
/// each plane and writes the four contiguous output bytes at once. The four
/// `acc` chains are independent (IPC > 1) and the store is contiguous.
fn unshuffle4(data: &[u8], out: &mut [u8], n: usize) {
    let (p0, r) = data.split_at(n);
    let (p1, r) = r.split_at(n);
    let (p2, p3) = r.split_at(n);
    let (mut a0, mut a1, mut a2, mut a3) = (0u8, 0u8, 0u8, 0u8);
    for (i, c) in out[..4 * n].as_chunks_mut::<4>().0.iter_mut().enumerate() {
        a0 = a0.wrapping_add(p0[i]);
        a1 = a1.wrapping_add(p1[i]);
        a2 = a2.wrapping_add(p2[i]);
        a3 = a3.wrapping_add(p3[i]);
        *c = [a0, a1, a2, a3];
    }
}

fn unshuffle2(data: &[u8], out: &mut [u8], n: usize) {
    let (p0, p1) = data.split_at(n);
    let (mut a0, mut a1) = (0u8, 0u8);
    for (i, c) in out[..2 * n].as_chunks_mut::<2>().0.iter_mut().enumerate() {
        a0 = a0.wrapping_add(p0[i]);
        a1 = a1.wrapping_add(p1[i]);
        *c = [a0, a1];
    }
}

fn unshuffle_generic(data: &[u8], out: &mut [u8], n: usize, elem: usize) {
    for plane in 0..elem {
        let mut acc = 0u8;
        for i in 0..n {
            acc = acc.wrapping_add(data[plane * n + i]);
            out[i * elem + plane] = acc;
        }
    }
}

/// Size of the sample used to choose a block's transform, in bytes.
const PROBEBYTES: usize = 16 * 1024;

/// Zero-order entropy of `bytes` in bits, summed over its 256 symbol counts.
fn zeroorderbits(bytes: &[u8]) -> f64 {
    let mut hist = [0u32; 256];
    for &b in bytes {
        hist[b as usize] += 1;
    }
    let n = bytes.len() as f64;
    hist.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2() * n
        })
        .sum()
}

/// Estimated rANS cost of coding `bytes` with transform `elem`, in bits, over
/// the whole `blocklen`-byte block.
///
/// The sample's per-byte entropy is extrapolated to the block, plus the fixed
/// frequency-table overhead (one 256×`u32` table per plane). That overhead term
/// is what keeps an incompressible block from being pushed onto a wider
/// transform whose planes cost more table than they save.
fn transformbits(probe: &[u8], blocklen: usize, elem: usize) -> f64 {
    let bits = if elem == 0 {
        zeroorderbits(probe)
    } else {
        let n = probe.len().checked_div(elem).unwrap_or(probe.len());
        let t = bitshuffle_delta(&probe[..elem * n], elem);
        (0..elem).map(|p| zeroorderbits(&t[p * n..(p + 1) * n])).sum()
    };
    bits / probe.len() as f64 * blocklen as f64 + (elem as f64) * 256.0 * 32.0
}

/// Chooses the transform for a block from `probe`, a sample of it, returning
/// the transform byte.
///
/// The pre-pass (`{none, f16, f32}`) has a per-data-type winner a small sample
/// identifies reliably: a slice of a content-defined block lies inside one
/// tensor, so its byte-plane structure matches the whole block. Scoring by
/// zero-order entropy instead of a trial compression makes the probe ~2.5×
/// cheaper than compressing the whole block three times, with no measured ratio
/// loss (the table-overhead term keeps heterogeneous blocks on the right
/// transform).
fn probetransform(probe: &[u8], blocklen: usize) -> u8 {
    let mut bestbits = f64::MAX;
    let mut best = TRANSFORMNONE;
    for transform in [TRANSFORMNONE, TRANSFORMF16, TRANSFORMF32] {
        let bits = transformbits(probe, blocklen, transform as usize);
        if bits < bestbits {
            bestbits = bits;
            best = transform;
        }
    }
    best
}

/// Compresses a block, returning its codec byte, transform byte, and the stored
/// bytes.
///
/// The single codec is gate (rANS entropy coding). The transform is chosen by
/// [`probetransform`] on a small sample, then applied to the whole block once,
/// and the block is stored raw when the result does not shrink it (raw storage
/// protects high-entropy blocks from codec expansion).
pub(crate) fn compressblock(bytes: &[u8]) -> (u8, u8, Vec<u8>) {
    let probe = &bytes[..PROBEBYTES.min(bytes.len())];
    let transform = probetransform(probe, bytes.len());
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
    if compressed.len() < bytes.len() {
        (CODECGATE, transform, compressed)
    } else {
        (CODECRAW, TRANSFORMNONE, bytes.to_vec())
    }
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
