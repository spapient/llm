// SYCL sources for the Intel backend (Arc discrete GPUs, integrated Xe
// graphics, and the oneAPI stack).
//
// SYCL is standard C++ with a queue-based execution model: kernels are
// submitted as lambdas through `sycl::queue`, and the device code is
// extracted by the DPC++ compiler at build time. Unlike the CUDA-family
// `__global__` functions there is no runtime name lookup, so this unit
// exports one plain-C entry point per operator (the `sycl_*` functions
// below); each receives the `sycl::queue*` from the Rust side (as `void*`),
// the USM device pointers and the shape scalars, and submits the kernel.
//
// Build (Linux or Windows, Intel oneAPI):
//   icpx -fsycl -x c++ -c sycl.c -o sycl.o
// The `-x c++` flag forces C++ mode for the `.c` extension (the repository
// naming convention); SYCL requires C++17, so the `.c` name must not let the
// driver pick C mode. Link with `-fsycl` (pulls in libsycl). The Intel
// module is compiled out when `icpx` is absent, exactly like the other
// toolchain-gated backends (build.rs emits an empty registry).
//
// Memory model: operands are shared-USM allocations owned by the Rust
// side (`sycl_malloc_shared`), so host copies are plain memcpy; kernels
// read/write them directly. The queue is created in-order, so kernels
// execute in submission order; every entry only *submits* and returns.
// Completion is requested explicitly with `sycl_wait(queue)`: the direct
// operator path calls it immediately after the single launch (preserving
// run-to-completion), while the recorded path submits a whole chain and
// pays one wait in `apply`.
//
// oneMKL: when the build finds the oneAPI MKL development files it defines
// GRID_MKL, and matmul is served by oneMKL's row-major SGEMM (including the
// strided-batched form). Otherwise the in-repo tiled SGEMM kernel serves it.
//
// Entries: sycl_attention, sycl_attention_cached, sycl_add, sycl_mul,
//          sycl_silu, sycl_embedding, sycl_matmul, sycl_rmsnorm, sycl_rope,
//          sycl_rope_at, sycl_scan, sycl_scan_agg, sycl_scan_prefix,
//          sycl_scan_apply, sycl_softmax.

#include "grid.h"
#include <sycl/sycl.hpp>

// oneMKL is an optional build-time dependency (build.rs probes the oneAPI
// MKL layout and defines GRID_MKL only when both the headers and the
// libraries are present). Without it the reference FMA kernel below serves
// matmul on every device, so a toolkit without MKL still builds and runs.
#if defined(GRID_MKL)
#include <oneapi/mkl/blas.hpp>

#include <cstdint>
#endif

// Windows ships this unit as `sycl_bridge.dll` (GRID_SYCL_DLL): the oneAPI
// driver owns the final DLL link so the embedded device images are
// registered, and every FFI entry must carry `__declspec(dllexport)`. On
// Linux the bridge is a static archive and the macro is a plain `extern "C"`.
#if defined(_WIN32) && defined(GRID_SYCL_DLL)
#define SYCLAPI extern "C" __declspec(dllexport)
#else
#define SYCLAPI extern "C"
#endif

namespace {
}  // namespace
// Error reporting: C++ exceptions never cross the C ABI (that would be
// undefined behaviour at the extern "C" boundary). Every entry body runs
// under SYCLTRY, which catches and records; the Rust side drains the code
// with sycl_take_error after each call. The slot is per host thread: each
// call/response pair stays on one thread, so concurrent sessions on
// different threads neither race on the counter nor consume each other's
// error.
namespace {
thread_local int sycl_last_error = 0;
}
#define SYCLTRY(code) \
    do { \
        try { code } \
        catch (sycl::exception const&) { sycl_last_error = 1; } \
        catch (...) { sycl_last_error = 2; } \
    } while (0)


// ===========================================================================
// attention
// ===========================================================================
// Fused scaled-dot-product attention, shared semantics with
// grid/cuda.c. One work-group per (query position, head); scores are
// recomputed, not materialized. q is [qseq, heads, headdim], k/v are
// [kvseq, kvheads, headdim], out is [qseq, heads, headdim]. causal selects
// the causal limit. Runs with exactly 256 work-items per group (the
// reduction tree halves the group size).
// Symbol: sycl_attention

SYCLAPI void sycl_attention(void* queue, const float* q, const float* k,
                               const float* v, float* out, unsigned qseq,
                               unsigned kvseq, unsigned heads, unsigned kvheads,
                               unsigned headdim, unsigned causal) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    const size_t groups = (size_t)qseq * heads;
    const size_t local = 256;
    que.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> reduce(sycl::range<1>(local), h);
        sycl::local_accessor<float, 1> weights(sycl::range<1>(local), h);
        sycl::local_accessor<float, 1> accum(
            sycl::range<1>(headdim > 4096 ? headdim : 4096), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * local),
                              sycl::range<1>(local)),
            [q, k, v, out, qseq, kvseq, heads, kvheads, headdim, causal, reduce,
             weights, accum](sycl::nd_item<1> it) {
                const unsigned tid = (unsigned)it.get_local_linear_id();
                const unsigned gid = (unsigned)it.get_group_linear_id();
                const unsigned head = gid % heads;
                const unsigned position = gid / heads;
                const unsigned kvhead = head / (heads / kvheads);
                const unsigned offset = kvseq - qseq;
                const unsigned limit =
                    (causal != 0u && offset + position + 1u < kvseq)
                        ? (offset + position + 1u)
                        : kvseq;
                const float* qrow = q + ((size_t)position * heads + head) * headdim;
                const float scale = 1.0f / sycl::sqrt((float)headdim);

                auto score = [&](unsigned key) {
                    const float* krow =
                        k + ((size_t)key * kvheads + kvhead) * headdim;
                    float dot = 0.0f;
                    for (unsigned d = 0; d < headdim; ++d) {
                        dot += qrow[d] * krow[d];
                    }
                    return dot * scale;
                };

                // Online softmax: one pass computes the running max and the
                // running sum, so each QK score is evaluated twice instead of
                // three times (the PV pass below re-evaluates scores).
                float runningmax = GRID_NEGLARGE;
                float localsum = 0.0f;
                for (unsigned key = tid; key < limit; key += local) {
                    float s = score(key);
                    float nm = (s > runningmax) ? s : runningmax;
                    localsum = localsum * sycl::exp(runningmax - nm) + sycl::exp(s - nm);
                    runningmax = nm;
                }
                it.barrier();
                reduce[tid] = runningmax;
                it.barrier();
                for (size_t s = local >> 1; s > 0; s >>= 1) {
                    if (tid < s && reduce[tid + s] > reduce[tid]) {
                        reduce[tid] = reduce[tid + s];
                    }
                    it.barrier();
                }
                const float maxv = reduce[0];
                // Re-scale each work-item's partial sum to the group max.
                reduce[tid] = localsum * sycl::exp(runningmax - maxv);
                it.barrier();
                for (size_t s = local >> 1; s > 0; s >>= 1) {
                    if (tid < s) {
                        reduce[tid] += reduce[tid + s];
                    }
                    it.barrier();
                }
                const float total = reduce[0];

                for (unsigned d = tid; d < headdim; d += local) {
                    accum[d] = 0.0f;
                }
                it.barrier();

                for (unsigned base = 0; base < limit; base += local) {
                    unsigned span = limit - base;
                    if (span > local) {
                        span = local;
                    }
                    if (tid < span) {
                        weights[tid] = sycl::exp(score(base + tid) - maxv);
                    } else {
                        weights[tid] = 0.0f;
                    }
                    it.barrier();
                    for (unsigned d = tid; d < headdim; d += local) {
                        float partial = 0.0f;
                        for (unsigned j = 0; j < span; ++j) {
                            const float* vrow =
                                v + ((size_t)(base + j) * kvheads + kvhead) * headdim;
                            partial += weights[j] * vrow[d];
                        }
                        accum[d] += partial;
                    }
                    it.barrier();
                }

                float* orow = out + ((size_t)position * heads + head) * headdim;
                for (unsigned d = tid; d < headdim; d += local) {
                    orow[d] = (total > 0.0f) ? (accum[d] / total) : 0.0f;
                }
            });
    });
  ); }

// ===========================================================================
// attention_cached
// ===========================================================================
// Incremental-decode attention, shared semantics with grid/cuda.c. One
// query token attends over a cached key/value buffer followed by one new
// row, without materialising the concatenation. Grid is one group per head.
// Symbol: sycl_attention_cached

SYCLAPI void sycl_attention_cached(void* queue, const float* q,
                                      const float* ck, const float* cv,
                                      const float* nk, const float* nv,
                                      float* out, unsigned kvseq,
                                      unsigned heads, unsigned kvheads,
                                      unsigned headdim, unsigned causal) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    const size_t local = 256;
    que.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> reduce(sycl::range<1>(local), h);
        sycl::local_accessor<float, 1> weights(sycl::range<1>(local), h);
        sycl::local_accessor<float, 1> accum(
            sycl::range<1>(headdim > 4096 ? headdim : 4096), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(heads * local),
                              sycl::range<1>(local)),
            [q, ck, cv, nk, nv, out, kvseq, heads, kvheads, headdim, causal,
             reduce, weights, accum](sycl::nd_item<1> it) {
                const unsigned tid = (unsigned)it.get_local_linear_id();
                const unsigned head = (unsigned)it.get_group_linear_id();
                const unsigned kvhead = head / (heads / kvheads);
                const unsigned pos = kvseq - 1u;
                const unsigned limit = kvseq;
                const float* qrow = q + ((size_t)head) * headdim;
                const float scale = 1.0f / sycl::sqrt((float)headdim);

                auto score = [&](unsigned key) {
                    const float* krow = (key < pos)
                        ? (ck + ((size_t)key * kvheads + kvhead) * headdim)
                        : (nk + ((size_t)kvhead) * headdim);
                    float dot = 0.0f;
                    for (unsigned d = 0; d < headdim; ++d) {
                        dot += qrow[d] * krow[d];
                    }
                    return dot * scale;
                };

                // Online softmax: one pass computes the running max and the
                // running sum, so each QK score is evaluated twice instead of
                // three times (the PV pass below re-evaluates scores).
                float runningmax = GRID_NEGLARGE;
                float localsum = 0.0f;
                for (unsigned key = tid; key < limit; key += local) {
                    float s = score(key);
                    float nm = (s > runningmax) ? s : runningmax;
                    localsum = localsum * sycl::exp(runningmax - nm) + sycl::exp(s - nm);
                    runningmax = nm;
                }
                it.barrier();
                reduce[tid] = runningmax;
                it.barrier();
                for (size_t s = local >> 1; s > 0; s >>= 1) {
                    if (tid < s && reduce[tid + s] > reduce[tid]) {
                        reduce[tid] = reduce[tid + s];
                    }
                    it.barrier();
                }
                const float maxv = reduce[0];
                // Re-scale each work-item's partial sum to the group max.
                reduce[tid] = localsum * sycl::exp(runningmax - maxv);
                it.barrier();
                for (size_t s = local >> 1; s > 0; s >>= 1) {
                    if (tid < s) {
                        reduce[tid] += reduce[tid + s];
                    }
                    it.barrier();
                }
                const float total = reduce[0];

                for (unsigned d = tid; d < headdim; d += local) {
                    accum[d] = 0.0f;
                }
                it.barrier();

                for (unsigned base = 0; base < limit; base += local) {
                    unsigned span = limit - base;
                    if (span > local) {
                        span = local;
                    }
                    if (tid < span) {
                        unsigned key = base + tid;
                        weights[tid] = sycl::exp(score(key) - maxv);
                    } else {
                        weights[tid] = 0.0f;
                    }
                    it.barrier();
                    for (unsigned d = tid; d < headdim; d += local) {
                        float partial = 0.0f;
                        for (unsigned j = 0; j < span; ++j) {
                            unsigned key = base + j;
                            const float* vrow = (key < pos)
                                ? (cv + ((size_t)key * kvheads + kvhead) * headdim)
                                : (nv + ((size_t)kvhead) * headdim);
                            partial += weights[j] * vrow[d];
                        }
                        accum[d] += partial;
                    }
                    it.barrier();
                }

                float* orow = out + ((size_t)head) * headdim;
                for (unsigned d = tid; d < headdim; d += local) {
                    orow[d] = (total > 0.0f) ? (accum[d] / total) : 0.0f;
                }
            });
    });
  ); }

// ===========================================================================
// elementwise
// ===========================================================================
// Elementwise f32 kernels, shared semantics with grid/cuda.c.
// add and mul take two equal-length operands; silu takes one. All stride
// over the flat element count and bounds-test, so any larger grid is safe.
// silu reproduces cpu's x / (1 + exp(-x)) exactly (no fused approximation).
// Symbols: sycl_add, sycl_mul, sycl_silu

SYCLAPI void sycl_add(void* queue, const float* a, const float* b,
                         float* c, unsigned n) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n),
                       [a, b, c, n](sycl::id<1> gid) { c[gid] = a[gid] + b[gid]; });
    });
  ); }

SYCLAPI void sycl_mul(void* queue, const float* a, const float* b,
                         float* c, unsigned n) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n),
                       [a, b, c, n](sycl::id<1> gid) { c[gid] = a[gid] * b[gid]; });
    });
  ); }

SYCLAPI void sycl_silu(void* queue, const float* a, float* c, unsigned n) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [a, c, n](sycl::id<1> gid) {
            float x = a[gid];
            c[gid] = x / (1.0f + sycl::exp(-x));
        });
    });
 ); }

// ===========================================================================
// embedding
// ===========================================================================
// f32 embedding-gather kernel, shared semantics with grid/cuda.c.
// table is [vocab, width], indices holds rows validated on the host, out is
// [rows, width]. One thread per output element; fully coalesced reads.
// Symbol: sycl_embedding

SYCLAPI void sycl_embedding(void* queue, const float* table,
                               const unsigned long long* indices, float* out,
                               unsigned long long rows, unsigned long long width) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    const unsigned long long total = rows * width;
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [table, indices, out, rows, width](sycl::id<1> gid) {
            unsigned long long row = gid / width;
            out[gid] = table[indices[row] * width + (gid % width)];
        });
    });
  ); }

// ===========================================================================
// matmul
// ===========================================================================
// f32 matrix-multiply, shared semantics with grid/cuda.c.
// a is [batch, m, k], b is [batch, k, n], c is [batch, m, n], all row-major.
//
// With oneMKL (GRID_MKL) the row-major SGEMM/SGEMM-batch entry serves the
// call: it dispatches to the Xe matrix engine's own tuned implementation,
// which is the only way to use the XMX units that Arc discrete GPUs and Xe
// integrated GPUs expose through the oneAPI stack. Without MKL the in-repo
// tiled kernel below serves it: each work-group computes a 64x64 output
// tile with 16x16 work-items (4x4 outputs each) and stages 16-deep A/B
// slabs through local memory. One work-item per output would re-read a full
// row and column from global memory for every element; the tile turns that
// into 8 KiB of shared traffic per 16-deep step, which is the difference
// between memory-bound and compute-bound on Xe.
//
// The tile edges are guarded: m, n and k need not be multiples of 64/16, so
// partial tiles fill the shared slabs with zeros and the store-back skips
// out-of-range elements. The 16-byte (float4) staging loads only engage when
// the row stride keeps the address aligned and the whole vector is in
// bounds; otherwise the same slots are filled scalar-wise.
// Symbol: sycl_matmul

#if !defined(GRID_MKL)
namespace {
// 64x64 output tile per work-group, 16x16 work-items, 4x4 outputs per
// work-item, 16-deep shared staging of A and B. This lives outside
// `sycl_matmul` because it carries `#pragma unroll` hints, and a
// preprocessor directive cannot appear inside the `SYCLTRY` macro argument.
void matmul_tiled(sycl::queue& que, const float* a, const float* b, float* c,
                  unsigned m, unsigned k, unsigned n, unsigned batch) {
    const unsigned groups_n = (n + 63) / 64;
    const unsigned groups_m = (m + 63) / 64;
    que.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> as(sycl::range<2>(64, 16), h);
        sycl::local_accessor<float, 2> bs(sycl::range<2>(16, 64), h);
        // The nd_range is in work-items: one 16x16 group per 64x64 tile.
        h.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(groups_n * 16, groups_m * 16, batch),
                              sycl::range<3>(16, 16, 1)),
            [a, b, c, m, k, n, as, bs](sycl::nd_item<3> it) {
                const unsigned lc = (unsigned)it.get_local_id(0);
                const unsigned lr = (unsigned)it.get_local_id(1);
                const unsigned lid = lr * 16 + lc;
                const unsigned col0 = (unsigned)it.get_group(0) * 64;
                const unsigned row0 = (unsigned)it.get_group(1) * 64;
                const unsigned blk = (unsigned)it.get_group(2);
                const float* ab = a + (size_t)blk * m * k;
                const float* bb = b + (size_t)blk * k * n;
                float* cb = c + (size_t)blk * m * n;
                // A vector load needs a 16-byte aligned address. The USM
                // base is page-aligned, so the only thing that can break
                // alignment is a row stride that is not a multiple of four
                // floats (k for A, n for B); the in-bounds check keeps the
                // last vector of a row from reading past the operand.
                const bool a_vec = (k & 3u) == 0;
                const bool b_vec = (n & 3u) == 0;
                float acc[4][4] = {};
                for (unsigned k0 = 0; k0 < k; k0 += 16) {
                    // One 4-float load per work-item stages a 64x16 A slab
                    // (rows 0..63) and a 16x64 B slab (columns 0..63):
                    // 256 work-items x 4 floats covers 1024 slots each.
                    const unsigned ar = lid / 4;
                    const unsigned ac = (lid % 4) * 4;
                    const unsigned br = lid / 16;
                    const unsigned bc = (lid % 16) * 4;
                    const bool a_row = row0 + ar < m;
                    const bool b_row = k0 + br < k;
                    if (a_row && a_vec && k0 + ac + 4 <= k) {
                        const sycl::float4 v =
                            *reinterpret_cast<const sycl::float4*>(
                                ab + (size_t)(row0 + ar) * k + k0 + ac);
                        as[ar][ac + 0] = v.x();
                        as[ar][ac + 1] = v.y();
                        as[ar][ac + 2] = v.z();
                        as[ar][ac + 3] = v.w();
                    } else {
                        for (unsigned j = 0; j < 4; ++j) {
                            const unsigned col = k0 + ac + j;
                            as[ar][ac + j] =
                                (a_row && col < k)
                                    ? ab[(size_t)(row0 + ar) * k + col]
                                    : 0.0f;
                        }
                    }
                    if (b_row && b_vec && col0 + bc + 4 <= n) {
                        const sycl::float4 v =
                            *reinterpret_cast<const sycl::float4*>(
                                bb + (size_t)(k0 + br) * n + col0 + bc);
                        bs[br][bc + 0] = v.x();
                        bs[br][bc + 1] = v.y();
                        bs[br][bc + 2] = v.z();
                        bs[br][bc + 3] = v.w();
                    } else {
                        for (unsigned j = 0; j < 4; ++j) {
                            const unsigned col = col0 + bc + j;
                            bs[br][bc + j] =
                                (b_row && col < n)
                                    ? bb[(size_t)(k0 + br) * n + col]
                                    : 0.0f;
                        }
                    }
                    it.barrier();
#pragma unroll 4
                    for (unsigned kk = 0; kk < 16; ++kk) {
                        float av[4];
                        float bv[4];
#pragma unroll
                        for (unsigned i = 0; i < 4; ++i)
                            av[i] = as[lr * 4 + i][kk];
#pragma unroll
                        for (unsigned j = 0; j < 4; ++j)
                            bv[j] = bs[kk][lc * 4 + j];
#pragma unroll
                        for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                            for (unsigned j = 0; j < 4; ++j)
                                acc[i][j] += av[i] * bv[j];
                    }
                    it.barrier();
                }
                for (unsigned i = 0; i < 4; ++i) {
                    const unsigned row = row0 + lr * 4 + i;
                    if (row >= m) continue;
                    for (unsigned j = 0; j < 4; ++j) {
                        const unsigned col = col0 + lc * 4 + j;
                        if (col < n) cb[(size_t)row * n + col] = acc[i][j];
                    }
                }
            });
    });
}
}  // namespace
#endif

SYCLAPI void sycl_matmul(void* queue, const float* a, const float* b,
                            float* c, unsigned m, unsigned k, unsigned n,
                            unsigned batch) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
#if defined(GRID_MKL)
    const std::int64_t mm = m;
    const std::int64_t nn = n;
    const std::int64_t kk = k;
    if (batch == 1) {
        oneapi::mkl::blas::row_major::gemm(
            que, oneapi::mkl::transpose::nontranspose,
            oneapi::mkl::transpose::nontranspose, mm, nn, kk, 1.0f, a, kk, b,
            nn, 0.0f, c, nn);
    } else {
        oneapi::mkl::blas::row_major::gemm_batch(
            que, oneapi::mkl::transpose::nontranspose,
            oneapi::mkl::transpose::nontranspose, mm, nn, kk, 1.0f, a, kk,
            mm * kk, b, nn, kk * nn, 0.0f, c, nn, mm * nn,
            static_cast<std::int64_t>(batch));
    }
#else
    matmul_tiled(que, a, b, c, m, k, n, batch);
#endif
  ); }

// ===========================================================================
// rmsnorm
// ===========================================================================
// f32 RMS-normalization, shared semantics with grid/cuda.c. One
// work-group per row; x is [rows, width], w is [width], y is [rows, width].
// Runs with exactly 256 work-items per group (a power of two).
// Symbol: sycl_rmsnorm

SYCLAPI void sycl_rmsnorm(void* queue, const float* x, const float* w,
                             float* y, unsigned rows, unsigned width,
                             float eps) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    const size_t local = 256;
    que.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(local), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * local),
                              sycl::range<1>(local)),
            [x, w, y, rows, width, eps, partial](sycl::nd_item<1> it) {
                const unsigned tid = (unsigned)it.get_local_linear_id();
                const unsigned row = (unsigned)it.get_group_linear_id();
                const float* xrow = x + (size_t)row * width;
                float sumsq = 0.0f;
                for (unsigned i = tid; i < width; i += local) {
                    float v = xrow[i];
                    sumsq += v * v;
                }
                partial[tid] = sumsq;
                it.barrier();
                for (size_t s = local >> 1; s > 0; s >>= 1) {
                    if (tid < s) {
                        partial[tid] += partial[tid + s];
                    }
                    it.barrier();
                }
                float inv = 1.0f / sycl::sqrt(partial[0] / (float)width + eps);
                for (unsigned i = tid; i < width; i += local) {
                    y[(size_t)row * width + i] = xrow[i] * inv * w[i];
                }
            });
    });
  ); }

// ===========================================================================
// rope
// ===========================================================================
// f32 rotary-position-embedding, shared semantics with grid/cuda.c. One
// thread per (row, head, pair); x and y are [rows, heads * headdim]. The
// position of a row is row % seqaxis; seqaxis == 0 places every row at 0.
// Symbol: sycl_rope

SYCLAPI void sycl_rope(void* queue, const float* x, float* y, unsigned rows,
                          unsigned seqaxis, unsigned heads, unsigned headdim,
                          float base) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    unsigned half = headdim / 2;
    unsigned long long total = (unsigned long long)rows * heads * half;
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [x, y, rows, seqaxis, heads, headdim, half, total, base](sycl::id<1> gid) {
            unsigned i = (unsigned)(gid % half);
            unsigned hd = (unsigned)((gid / half) % heads);
            unsigned row = (unsigned)(gid / ((unsigned long long)half * heads));
            unsigned position = (seqaxis == 0u) ? 0u : (row % seqaxis);
            float angle = (float)position *
                          sycl::pow(base, -2.0f * (float)i / (float)headdim);
            float s = sycl::sin(angle);
            float c = sycl::cos(angle);
            unsigned long long offset =
                (((unsigned long long)row * heads) + hd) * headdim + 2u * i;
            float even = x[offset];
            float odd = x[offset + 1u];
            y[offset] = even * c - odd * s;
            y[offset + 1u] = even * s + odd * c;
        });
    });
  ); }

// ===========================================================================
// rope_at
// ===========================================================================
// Absolute-position rotary embedding of a single token, shared semantics
// with grid/cuda.c. Position is a scalar parameter, not row % seqaxis.
// Symbol: sycl_rope_at

SYCLAPI void sycl_rope_at(void* queue, const float* x, float* y,
                             unsigned heads, unsigned headdim, unsigned position,
                             float base) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    unsigned half = headdim / 2;
    unsigned long long total = (unsigned long long)heads * half;
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [x, y, heads, headdim, position, half, total, base](sycl::id<1> gid) {
            unsigned i = (unsigned)(gid % half);
            unsigned hd = (unsigned)(gid / half);
            float angle = (float)position *
                          sycl::pow(base, -2.0f * (float)i / (float)headdim);
            float s = sycl::sin(angle);
            float c = sycl::cos(angle);
            unsigned long long offset = ((unsigned long long)hd) * headdim + 2u * i;
            float even = x[offset];
            float odd = x[offset + 1u];
            y[offset] = even * c - odd * s;
            y[offset + 1u] = even * s + odd * c;
        });
    });
  ); }

// ===========================================================================
// scan
// ===========================================================================
// f32 semigroup scan, shared semantics with grid/cuda.c. One thread per
// channel; each thread runs the serial recurrence over the sequence axis for
// its channel's state vector, so the result is bit-for-bit the serial
// reference. The state buffer is initialized to init by the host and updated
// in place. Pointer order: input, a, b, c, skip, state (in/out), out.
// Symbol: sycl_scan

SYCLAPI void sycl_scan(void* queue, const float* input, const float* a,
                          const float* b, const float* c, const float* skip,
                          float* state, float* out, unsigned n, unsigned d,
                          unsigned N) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(d), [input, a, b, c, skip, state, out, n, d, N](sycl::id<1> gid) {
            unsigned i = (unsigned)gid;
            for (unsigned t = 0; t < n; ++t) {
                float x = input[t * d + i];
                float acc = 0.0f;
                for (unsigned j = 0; j < N; ++j) {
                    unsigned idx = i * N + j;
                    float hv = a[idx] * state[idx] + b[idx] * x;
                    state[idx] = hv;
                    acc += c[idx] * hv;
                }
                out[t * d + i] = acc + skip[i] * x;
            }
        });
    });
  ); }

// ===========================================================================
// scan_parallel
// ===========================================================================
// f32 semigroup scan, parallel three-stage family, shared semantics with
// grid/cuda.c. Pass 1 aggregates each block's total, Pass 2 computes
// the inter-block prefix, Pass 3 re-applies each block from its prefix so
// the intra-block recurrence is bit-for-bit the serial one.

// Pass 1: per-block total. Thread id maps block = id / d, channel = id % d.
SYCLAPI void sycl_scan_agg(void* queue, const float* input, const float* a,
                              const float* b, const float* c, const float* skip,
                              const float* init, float* block_total, unsigned n,
                              unsigned d, unsigned N, unsigned chunk,
                              unsigned blocks) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)blocks * d),
                       [input, a, b, c, skip, init, block_total, n, d, N, chunk, blocks](sycl::id<1> gid) {
                           unsigned block = (unsigned)(gid / d);
                           unsigned i = (unsigned)(gid % d);
                           unsigned t0 = block * chunk;
                           unsigned t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
                           unsigned dN = d * N;
                           unsigned base = block * (2u * dN);
                           for (unsigned j = 0; j < N; ++j) {
                               unsigned idx = i * N + j;
                               float A = 1.0f;
                               float B = 0.0f;
                               const float aj = a[idx];
                               const float bj = b[idx];
                               for (unsigned t = t0; t < t1; ++t) {
                                   const float x = input[t * d + i];
                                   B = aj * B + bj * x;
                                   A = aj * A;
                               }
                               block_total[base + idx] = A;
                               block_total[base + dN + idx] = B;
                           }
                       });
    });
  ); }

// Pass 2: inter-block prefix, one thread per component (j in [0, d*N));
// prefix[p] = ⊕_{q<p} total[q] with prefix[0] = identity (1, 0). The
// loop-final accumulator is the whole-sequence total, giving the final
// state: state[j] = A_total * init[j] + B_total. sycl::fma folds the combine
// into one rounding, halving the inter-block error that grows like P·ε·‖B‖.
SYCLAPI void sycl_scan_prefix(void* queue, const float* block_total,
                                 const float* init, float* state,
                                 float* block_prefix, unsigned n, unsigned d,
                                 unsigned N, unsigned chunk, unsigned blocks) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    const unsigned dN = d * N;
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(dN), [block_total, init, state, block_prefix, n, d, N, chunk, blocks, dN](sycl::id<1> gid) {
            unsigned j = (unsigned)gid;
            block_prefix[j] = 1.0f;
            block_prefix[dN + j] = 0.0f;
            float A = 1.0f;
            float B = 0.0f;
            for (unsigned q = 0; q < blocks; ++q) {
                const unsigned base = q * (2u * dN);
                const float aq = block_total[base + j];
                const float bq = block_total[base + dN + j];
                block_prefix[(q + 1u) * (2u * dN) + j] = A * aq;
                block_prefix[(q + 1u) * (2u * dN) + dN + j] = sycl::fma(aq, B, bq);
                B = sycl::fma(aq, B, bq);
                A = A * aq;
            }
            state[j] = A * init[j] + B;
        });
    });
  ); }

// Pass 3: per-block apply. Thread i owns channel i; its thread-local h array
// holds the channel's N state components. The block starts at the inter-block
// prefix state h = A0*init + B0, then runs the same intra-block recurrence as
// the serial reference, writing out[t][i] = skip[i]*x + Σ_j c[i,j]*h[j].
SYCLAPI void sycl_scan_apply(void* queue, const float* input, const float* a,
                                const float* b, const float* c, const float* skip,
                                const float* init, const float* block_prefix,
                                float* out, unsigned n, unsigned d, unsigned N,
                                unsigned chunk, unsigned blocks) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    que.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)blocks * d),
                       [input, a, b, c, skip, init, block_prefix, out, n, d, N, chunk, blocks](sycl::id<1> gid) {
                           unsigned block = (unsigned)(gid / d);
                           unsigned i = (unsigned)(gid % d);
                           unsigned t0 = block * chunk;
                           unsigned t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
                           unsigned dN = d * N;
                           unsigned base = block * (2u * dN);
                           float hv[GRID_SCANMAXN];
                           for (unsigned j = 0; j < N; ++j) {
                               unsigned idx = i * N + j;
                               hv[j] = block_prefix[base + idx] * init[idx]
                                   + block_prefix[base + dN + idx];
                           }
                           for (unsigned t = t0; t < t1; ++t) {
                               const float x = input[t * d + i];
                               float acc = 0.0f;
                               for (unsigned j = 0; j < N; ++j) {
                                   unsigned idx = i * N + j;
                                   hv[j] = a[idx] * hv[j] + b[idx] * x;
                                   acc += c[idx] * hv[j];
                               }
                               out[t * d + i] = acc + skip[i] * x;
                           }
                       });
    });
  ); }

// ===========================================================================
// softmax
// ===========================================================================
// f32 softmax-over-last-axis, shared semantics with grid/cuda.c. One
// work-group per row; x is [rows, width], y is [rows, width]. Runs with
// exactly 256 work-items per group (a power of two). The max-subtracting
// order is the numerically stable one the scalar reference shares.
// Symbol: sycl_softmax

SYCLAPI void sycl_softmax(void* queue, const float* x, float* y,
                             unsigned rows, unsigned width) { SYCLTRY(
    sycl::queue& que = *static_cast<sycl::queue*>(queue);
    const size_t local = 256;
    que.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(local), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * local),
                              sycl::range<1>(local)),
            [x, y, rows, width, scratch](sycl::nd_item<1> it) {
                const unsigned tid = (unsigned)it.get_local_linear_id();
                const unsigned row = (unsigned)it.get_group_linear_id();
                const float* xrow = x + (size_t)row * width;
                float* dst = y + (size_t)row * width;

                float localmax = GRID_NEGLARGE;
                for (unsigned i = tid; i < width; i += local) {
                    float v = xrow[i];
                    if (v > localmax) {
                        localmax = v;
                    }
                }
                it.barrier();
                scratch[tid] = localmax;
                it.barrier();
                for (size_t s = local >> 1; s > 0; s >>= 1) {
                    if (tid < s && scratch[tid + s] > scratch[tid]) {
                        scratch[tid] = scratch[tid + s];
                    }
                    it.barrier();
                }
                float rowmax = scratch[0];

                float localsum = 0.0f;
                for (unsigned i = tid; i < width; i += local) {
                    localsum += sycl::exp(xrow[i] - rowmax);
                }
                it.barrier();
                scratch[tid] = localsum;
                it.barrier();
                for (size_t s = local >> 1; s > 0; s >>= 1) {
                    if (tid < s) {
                        scratch[tid] += scratch[tid + s];
                    }
                    it.barrier();
                }
                float inv = (scratch[0] > 0.0f) ? (1.0f / scratch[0]) : 0.0f;

                for (unsigned i = tid; i < width; i += local) {
                    dst[i] = sycl::exp(xrow[i] - rowmax) * inv;
                }
            });
    });
  ); }

// ===========================================================================
// queue and USM bridge (called by the Rust side)
// ===========================================================================

/// Creates an in-order queue on the `device_ordinal`-th device (-1 selects
/// the default device). Returns null when no matching device exists.
///
/// In-order is what makes the deferred-wait record path legal: kernels are
/// submitted without synchronising, and the queue itself guarantees that each
/// one observes the writes of every kernel submitted before it.
SYCLAPI void* sycl_queue_new(int device_ordinal) {
    try {
        if (device_ordinal < 0) {
            return new sycl::queue(sycl::default_selector_v,
                                   sycl::property::queue::in_order());
        }
        for (auto const& platform : sycl::platform::get_platforms()) {
            auto devices = platform.get_devices(sycl::info::device_type::all);
            if ((int)devices.size() > device_ordinal) {
                return new sycl::queue(devices[(size_t)device_ordinal],
                                       sycl::property::queue::in_order());
            }
            device_ordinal -= (int)devices.size();
        }
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

/// Waits for every kernel submitted to `queue` so far.
///
/// Every operator entry only submits; the direct path calls this immediately
/// after its single launch, and the record path calls it once per `apply`.
/// Failures are recorded for `sycl_take_error`, never thrown across the ABI.
SYCLAPI void sycl_wait(void* queue) { SYCLTRY(
    static_cast<sycl::queue*>(queue)->wait_and_throw();
  ); }

/// Destroys a queue created by `sycl_queue_new`.
SYCLAPI void sycl_queue_free(void* queue) {
    if (queue) {
        delete static_cast<sycl::queue*>(queue);
    }
}

/// Allocates `bytes` of shared USM memory visible to both host and device.
/// Shared USM lets the Rust side upload/download with plain copies instead of
/// an extra queue-copy call.
SYCLAPI void* sycl_malloc_shared(void* queue, size_t bytes) {
    try {
        return sycl::malloc_shared(bytes, *static_cast<sycl::queue*>(queue));
    } catch (...) {
        return nullptr;
    }
}

/// Frees a shared USM allocation created by `sycl_malloc_shared`.
SYCLAPI void sycl_free_shared(void* queue, void* ptr) {
    if (queue && ptr) {
        try {
            sycl::free(ptr, *static_cast<sycl::queue*>(queue));
        } catch (...) {
            // A failed free has no recovery path; never throw across the ABI.
        }
    }
}

/// Returns and clears the last recorded error code (0 = ok).
SYCLAPI int sycl_take_error(void) {
    int code = sycl_last_error;
    sycl_last_error = 0;
    return code;
}

