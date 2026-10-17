// grid/grid.h — GRID kernel-family shared header.
//
// Every GRID kernel source (`grid/cann.c`, `grid/mlu.c`, `grid/musa.c`,
// `sycl/sycl.c`, `metal/metal.c`) pulls in exactly this one header, so the
// 15-operator contract and the cross-dialect constants are written once.
// Dialect-specific tuning (vector block size, buffer depth, WMMA geometry)
// stays in each kernel source: the values and semantics differ per hardware,
// so forcing them into a shared table would be a false unification.
//
// Exactly one dialect macro must be defined by the toolchain:
//   -DGRID_CANN    Ascend C (ccec)        -DGRID_MLU    BANG C (cncc)
//   -DGRID_MUSA    MUSA (mcc)             -DGRID_SYCL   SYCL (icpx)
//   -DGRID_METAL   Metal MSL (spliced by the Rust runtime, see catalog.rs)
//   -DGRID_CUDA    CUDA-family via NVRTC (spliced by the Rust runtime)
//   -DGRID_ROCM    ROCm HIP (hipcc)

#ifndef GRID_H
#define GRID_H

#if defined(GRID_CANN) + defined(GRID_MLU) + defined(GRID_MUSA) + \
    defined(GRID_SYCL) + defined(GRID_METAL) + defined(GRID_CUDA) + \
    defined(GRID_ROCM) != 1
#error "GRID kernels need exactly one of GRID_CANN / GRID_MLU / GRID_MUSA / GRID_SYCL / GRID_METAL / GRID_CUDA / GRID_ROCM"
#endif

// Cross-dialect constants (identical on every target).
#define GRID_NEGLARGE (-3.402823466e+38f)  // -FLT_MAX, softmax sentinel
#define GRID_SCANMAXN 256                  // scan state width cap

// Cross-dialect arithmetic helpers (pure preprocessor, no dialect API):
// GRID_RANGE writes the `id`-th of `cores` contiguous splits of [0, count)
// into `start`/`end`; GRID_TAIL bounds a block offset to the remainder.
#define GRID_RANGE(id, cores, count, start, end)                         \
    const uint64_t gper = ((uint64_t)(count) + (cores) - 1u) / (cores); \
    (start) = (uint64_t)(id) * gper;                                     \
    (end) = ((start) + gper < (count)) ? ((start) + gper) : (count)

#define GRID_TAIL(off, end, block, len)                                         \
    const uint32_t len =                                                        \
        ((end) - (off) < (block)) ? ((end) - (off)) : (uint32_t)(block)

// ---------------------------------------------------------------------------
// The 15-operator contract (single source of truth; per-dialect
// implementations live in the kernel sources listed above).
//
//   add(a, b, c, n)                    c = a + b, flat n
//   mul(a, b, c, n)                    c = a * b, flat n
//   silu(a, c, n)                      c = a / (1 + exp(-a)), flat n
//   embedding(table, indices, out, rows, width)
//                                      gather table rows by indices
//   matmul(a, b, c, m, k, n, batch)    c = a @ b, row-major batched
//   rmsnorm(x, w, y, rows, width, eps) y = x / rms(x) * w
//   rope(x, y, rows, seqaxis, heads, headdim, base)
//                                      rotary position embedding
//   rope_at(x, y, heads, headdim, position, base)
//                                      absolute-position rope of one token
//   softmax(x, y, rows, width)         row softmax
//   attention(q, k, v, out, qseq, kvseq, heads, kvheads, headdim, causal)
//                                      fused scaled-dot-product attention
//   attention_cached(q, ck, cv, nk, out, qseq, kvseq, heads, kvheads,
//                    headdim, causal)  attention with cached k/v ring
//   scan(input, a, b, c, skip, state, out, n, d, N)
//                                      semigroup scan, serial reference
//   scan_agg(input, a, b, c, total, n, d, blocks)
//   scan_prefix(block_total, init, state, N, chunk, blocks)
//   scan_apply(input, a, b, c, state, out, n, d, chunk, blocks)
//                                      three-pass parallel scan helpers
// ---------------------------------------------------------------------------

#endif  // GRID_H
