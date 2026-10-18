use std::collections::HashMap;
use std::path::Path;

use crate::atomicwrite::writeatomic;
use crate::blocks::{blockref, chunk};
use crate::deltas::delta;
use crate::error::checkpointerror;
use crate::uformat::snapshot;

/// Merges a delta onto its base, producing the full snapshot the delta
/// represents.
///
/// The base's root hash must equal the delta's base identity. Each logical
/// block is resolved from the delta's stored blocks first, then from the base,
/// by content identity; a block that resolves nowhere is a corruption. The
/// result is a full snapshot with the delta's tensors and the reconstructed
/// data, whose root hash equals the delta's root hash.
///
/// # Errors
///
/// Returns [`checkpointerror::basehashmismatch`] when the base does not match
/// the delta's reference, or [`checkpointerror::missingblock`] when a logical
/// block is present in neither the delta nor the base.
pub fn merge(base: &snapshot, delta: &delta) -> Result<snapshot, checkpointerror> {
    let baseroot = base.roothash();
    if baseroot != delta.getbaseid() {
        return Err(checkpointerror::basehashmismatch {
            expected: delta.getbaseid(),
            found: baseroot,
        });
    }

    let mut basemap: HashMap<[u8; 16], Vec<u8>> = HashMap::new();
    let baseblocks = chunk(base.getdata());
    let mut offset = 0usize;
    for block in &baseblocks {
        let len = block.getlen() as usize;
        basemap.insert(
            block.gethash(),
            base.getdata()[offset..offset + len].to_vec(),
        );
        offset += len;
    }

    let mut storedmap: HashMap<[u8; 16], Vec<u8>> = HashMap::new();
    for block in delta.getstored() {
        storedmap.insert(block.gethash(), block.getbytes().to_vec());
    }

    let blocks = delta.getblocks();
    let total: usize = blocks.iter().map(|b| b.getlen() as usize).sum();
    let mut data = vec![0u8; total];

    let nthreads = match std::thread::available_parallelism() {
        Ok(n) => n.get().min(blocks.len()),
        Err(_) => 1,
    };
    let mut spans = Vec::with_capacity(nthreads);
    let mut start = 0usize;
    for t in 0..nthreads {
        let len = blocks.len() / nthreads + usize::from(t < blocks.len() % nthreads);
        let end = start + len;
        spans.push(start..end);
        start = end;
    }

    let blocksref: &[blockref] = blocks;
    let storedmapref: &HashMap<[u8; 16], Vec<u8>> = &storedmap;
    let basemapref: &HashMap<[u8; 16], Vec<u8>> = &basemap;
    std::thread::scope(|scope| {
        let mut cursor: &mut [u8] = &mut data;
        let mut handles = Vec::with_capacity(nthreads);
        for span in spans {
            let seglen: usize = blocksref[span.clone()]
                .iter()
                .map(|b| b.getlen() as usize)
                .sum();
            let (head, tail) = cursor.split_at_mut(seglen);
            cursor = tail;
            handles
                .push(scope.spawn(move || {
                    mergeblockspan(blocksref, storedmapref, basemapref, span, head)
                }));
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(error),
                Err(_) => {
                    return Err(checkpointerror::invalidmanifest {
                        reason: "merge worker panicked".to_string(),
                    });
                }
            }
        }
        Ok(())
    })?;

    Ok(snapshot::fromparts(
        delta.getversion(),
        delta.gettensors().to_vec(),
        data,
    ))
}

/// Resolves one contiguous span of blocks from the delta or base and copies
/// their bytes into `out`, which must hold exactly the span's combined length.
fn mergeblockspan(
    blocks: &[blockref],
    storedmap: &HashMap<[u8; 16], Vec<u8>>,
    basemap: &HashMap<[u8; 16], Vec<u8>>,
    span: std::ops::Range<usize>,
    out: &mut [u8],
) -> Result<(), checkpointerror> {
    let mut written = 0usize;
    for position in span {
        let block = &blocks[position];
        let bytes = if let Some(bytes) = storedmap.get(&block.gethash()) {
            bytes
        } else if let Some(bytes) = basemap.get(&block.gethash()) {
            bytes
        } else {
            return Err(checkpointerror::missingblock {
                hash: block.gethash(),
            });
        };
        let dst = &mut out[written..written + bytes.len()];
        dst.copy_from_slice(bytes);
        written += bytes.len();
    }
    Ok(())
}

/// Merges a delta file onto its base file, writing the full snapshot to
/// `out_path` atomically.
///
/// # Errors
///
/// Returns [`checkpointerror::missingbase`] when the base file is absent,
/// [`checkpointerror::basehashmismatch`] when the base does not match the
/// delta's reference, [`checkpointerror::missingblock`] when a block resolves
/// nowhere, or the decode and I/O failures.
pub fn mergefiles(
    base_path: &Path,
    delta_path: &Path,
    out_path: &Path,
) -> Result<(), checkpointerror> {
    let deltabytes = std::fs::read(delta_path).map_err(|source| checkpointerror::io {
        path: delta_path.to_path_buf(),
        source,
    })?;
    let delta = delta::decode(&deltabytes)?;
    let basebytes = match std::fs::read(base_path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(checkpointerror::missingbase {
                base_id: delta.getbaseid(),
            });
        }
        Err(source) => {
            return Err(checkpointerror::io {
                path: base_path.to_path_buf(),
                source,
            });
        }
    };
    let base = snapshot::decode(&basebytes)?;
    let merged = merge(&base, &delta)?;
    let out = merged.encode()?;
    writeatomic(out_path, &out)
}

/// Merges an encrypted delta file onto its encrypted base file, writing the
/// full encrypted snapshot to `out_path` atomically.
///
/// # Errors
///
/// Returns the same failures as [`mergefiles`], plus
/// [`checkpointerror::missingkey`] or [`checkpointerror::decryptionfailed`]
/// for key or decryption failures.
pub fn mergefiles_encrypted(
    base_path: &Path,
    delta_path: &Path,
    out_path: &Path,
    key: &[u8; 32],
) -> Result<(), checkpointerror> {
    let deltabytes = std::fs::read(delta_path).map_err(|source| checkpointerror::io {
        path: delta_path.to_path_buf(),
        source,
    })?;
    let delta = delta::decode_encrypted(&deltabytes, key)?;
    let basebytes = match std::fs::read(base_path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(checkpointerror::missingbase {
                base_id: delta.getbaseid(),
            });
        }
        Err(source) => {
            return Err(checkpointerror::io {
                path: base_path.to_path_buf(),
                source,
            });
        }
    };
    let base = snapshot::decode_encrypted(&basebytes, key)?;
    let merged = merge(&base, &delta)?;
    let out = merged.encode_encrypted(key)?;
    writeatomic(out_path, &out)
}
