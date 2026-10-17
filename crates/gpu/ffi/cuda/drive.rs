#![allow(unsafe_code)]
//! The CUDA driver-handoff boundary: the only file in this island that
//! invokes cudarc `unsafe` APIs.
//!
//! Authority: docs/design.md §21.8. Exactly two constructs are `unsafe`,
//! one per class of platform interaction rather than one per operator:
//! [`submit`] (the single NVRTC kernel-launch site) and [`gemm`] (the single
//! cuBLAS site). Each is directly preceded by a `// SAFETY:` comment;
//! `scripts/checkunsafe.sh` enforces the count and the comments.
use core::ffi::c_int;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, Gemm, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg};

use super::cache;
use crate::cuda::source;
use crate::engine::gpuerror;
use crate::ffi::gkernel::{buf, launchspec, scalar};

/// Converts a validated extent into the `c_int` a host-side library takes.
pub(super) fn asint(value: u32) -> Result<c_int, gpuerror> {
    c_int::try_from(value).map_err(|_| gpuerror::sizeoverflow)
}

/// Converts a validated extent into the `unsigned` a kernel parameter takes.
pub(super) fn asunsigned(value: usize) -> Result<u32, gpuerror> {
    u32::try_from(value).map_err(|_| gpuerror::sizeoverflow)
}

/// Builds the one-dimensional launch configuration covering `count` elements.
pub(super) fn flatconfig(count: usize) -> Result<LaunchConfig, gpuerror> {
    let tile = source::FLATTILE;
    let grid = count.div_ceil(tile as usize);
    Ok(LaunchConfig {
        grid_dim: (asunsigned(grid)?, 1, 1),
        block_dim: (tile, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Copies a host slice to a fresh device allocation.
pub(super) fn upload<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    src: &[T],
) -> Result<CudaSlice<T>, gpuerror> {
    stream.clone_htod(src).map_err(|error| gpuerror::transfer {
        message: error.to_string(),
    })
}

/// Allocates `len` zeroed device elements.
pub(super) fn alloc(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<f32>, gpuerror> {
    stream
        .alloc_zeros::<f32>(len)
        .map_err(|error| gpuerror::transfer {
            message: error.to_string(),
        })
}

/// Waits for `stream` and copies a device allocation back to the host.
pub(super) fn syncanddownload(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
) -> Result<Vec<f32>, gpuerror> {
    stream.synchronize().map_err(|error| gpuerror::transfer {
        message: error.to_string(),
    })?;
    stream.clone_dtoh(src).map_err(|error| gpuerror::transfer {
        message: error.to_string(),
    })
}

/// Submits a prepared kernel launch to the driver.
///
/// This is the crate's single kernel-launch site, and it is a function rather
/// than an inline block so the registered count does not grow with the number
/// of kernels: callers assemble the argument list with the safe
/// `LaunchArgs::arg` builder and only the submit itself is unchecked.
pub(super) fn submit(args: &mut LaunchArgs<'_>, config: LaunchConfig) -> Result<(), gpuerror> {
    // SAFETY: `args` was built by `CudaStream::launch_builder` from a function
    // resolved out of a module compiled from the very source whose signature
    // the caller's argument list mirrors, and every argument is either a live
    // device allocation or a scalar borrowed from a local that outlives this
    // call. `config` is the caller's launch geometry: for the flat kernels it
    // covers at least the element count (the kernels bounds-test themselves),
    // and for the tiled kernels it matches the shared-memory demand the kernel
    // computes for itself.
    unsafe { args.launch(config) }
        .map(|_| ())
        .map_err(|error| gpuerror::launch {
            message: error.to_string(),
        })
}

/// Runs one cuBLAS strided-batched `gemm`, the crate's single cuBLAS site.
///
/// This is a function rather than an inline block so the registered count stays
/// one: the single-operator [`matmul_cublas`] and the accumulative submission
/// path both call it, and neither adds an `unsafe` site of its own. The batched
/// form serves single matrices too (batch size one), so the one site covers
/// both call shapes.
pub(super) fn gemm(
    blas: &CudaBlas,
    config: StridedBatchedConfig<f32>,
    db: &CudaSlice<f32>,
    da: &CudaSlice<f32>,
    dc: &mut CudaSlice<f32>,
) -> Result<(), gpuerror> {
    // SAFETY: `da`, `db`, and `dc` are live device allocations. `da` holds the
    // `[batch, m, k]` left operand and `db` the `[batch, k, n]` right operand;
    // read as column-major they are the `k x m` and `n x k` matrices the
    // `lda`/`ldb` leading dimensions describe, and `dc` is written as an
    // `n x m` column-major matrix with leading dimension `ldc`, i.e. the
    // row-major `[m, n]` result. The per-slice strides `stride_a`/`stride_b`/
    // `stride_c` are the exact element counts of one slice of each operand, so
    // every batch slice stays inside its allocation. Every dimension is at
    // least 1 and fits `c_int`, so no out-of-bounds access is possible.
    unsafe {
        blas.gemm_strided_batched(config, db, da, dc)
            .map_err(|error| gpuerror::blas {
                message: error.to_string(),
            })?;
    }
    Ok(())
}

/// Converts a shared launch spec into the driver's launch configuration.
pub(super) fn launchspecconfig(spec: launchspec) -> LaunchConfig {
    LaunchConfig {
        grid_dim: spec.grid,
        block_dim: spec.block,
        shared_mem_bytes: spec.shared,
    }
}

/// Runs one kernel through the single driver-handoff block, uploading host
/// operands, launching, synchronising once, and downloading the outputs.
///
/// This is the operator engine the shared [`GENOPS!`] surface expands
/// over: the buffers are bound in order, the scalars follow, and every
/// `out` / `inout` slot is downloaded in buffer order. The lock is held
/// only across kernel resolution (one compile for the whole translation
/// unit); the transfer, launch, and synchronisation all happen outside it,
/// exactly like the hand-written operators. Each call takes a fresh
/// stream, so concurrent calls proceed in parallel.
pub(super) fn drive(
    cache: &cache,
    entry: &'static str,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    spec: launchspec,
) -> Result<Vec<Vec<f32>>, gpuerror> {
    let function = cache.function(entry)?;
    let stream = cache.stream()?;
    // Allocate and upload first, so a failed allocation leaves nothing
    // half-uploaded; the f32 and u64 slices keep their own pools.
    let mut f32s: Vec<CudaSlice<f32>> = Vec::new();
    let mut u64s: Vec<CudaSlice<u64>> = Vec::new();
    let mut outputmask: Vec<bool> = Vec::new();
    for b in buffers {
        match b {
            buf::host(src) => {
                f32s.push(upload(&stream, src)?);
                outputmask.push(false);
            }
            buf::hostu64(src) => {
                u64s.push(upload(&stream, src)?);
                outputmask.push(false);
            }
            buf::inout(src) => {
                f32s.push(upload(&stream, src)?);
                outputmask.push(true);
            }
            buf::out(len) => {
                f32s.push(alloc(&stream, *len)?);
                outputmask.push(true);
            }
        }
    }
    // Assemble the argument list from live allocations and scalars that
    // outlive the launch.
    let mut args = stream.launch_builder(&function);
    let mut fi = 0usize;
    let mut ui = 0usize;
    for b in buffers {
        match b {
            buf::host(_) | buf::inout(_) | buf::out(_) => {
                args.arg(&f32s[fi]);
                fi += 1;
            }
            buf::hostu64(_) => {
                args.arg(&u64s[ui]);
                ui += 1;
            }
        }
    }
    let scalars_owned: Vec<scalar> = scalars.to_vec();
    for s in &scalars_owned {
        match s {
            scalar::u32(v) => args.arg(v),
            scalar::u64(v) => args.arg(v),
            scalar::f32(v) => args.arg(v),
        };
    }
    submit(&mut args, launchspecconfig(spec))?;
    // One synchronisation for every recorded output slot, then download
    // each in buffer order.
    stream.synchronize().map_err(|error| gpuerror::transfer {
        message: error.to_string(),
    })?;
    let mut results = Vec::new();
    let mut fi = 0usize;
    for (i, b) in buffers.iter().enumerate() {
        // `f32s` indexes every f32 slot in buffer order, host inputs
        // included; `hostu64` slots live in the separate `u64s` pool and
        // are never output slots. Advance `fi` on every f32 slot so a
        // download reads the exact allocation launched in that position,
        // rather than the nth output counting only output slots.
        if matches!(b, buf::hostu64(_)) {
            continue;
        }
        if outputmask[i] {
            results.push(
                stream
                    .clone_dtoh(&f32s[fi])
                    .map_err(|error| gpuerror::transfer {
                        message: error.to_string(),
                    })?,
            );
        }
        fi += 1;
    }
    Ok(results)
}
