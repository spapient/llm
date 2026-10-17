// MUSA Tensor-Core matmul family, compiled at build time by mcc.
//
// These kernels need `<mma.h>` (the mtmusa WMMA intrinsics), which NVRTC
// does not provide, so they cannot live in the shared crate root.
// The reference FMA kernel `matmul` is shared and lives in
// `matmul.cu` at the crate root; `rmsinv`/`matmul_wmma`/`matmul_rms` stay here.
//
// Symbols: rmsinv, matmul_wmma, matmul_rms

#include <mma.h>

// Per-row reciprocal RMS: inv[r] = 1 / sqrt(mean(x[r]^2) + eps).
// One threadgroup per row; the block reduction matches rmsnorm.cu.
// Symbol: rmsinv
extern "C" __global__ void rmsinv(const float* x, float* inv, unsigned rows,
                                  unsigned width, float eps) {
    __shared__ float scratch[256];
    const unsigned row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = x[(size_t)row * width + i];
        sum += value * value;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        inv[row] = 1.0f / sqrtf(scratch[0] / (float)width + eps);
    }
}

namespace wmma = mtmusa::wmma;

// One 64x32 output tile per block, computed by two 32x32x8 Tensor-Core
// accumulators (one per 32-row half). The MUSA m32n32k8 tf32 MMA is a
// wave128 instruction: its four 32-lane warps cooperate on one tile, so the
// block is 128 threads and every lane participates in each load/mma/store.
//
// Full tiles prefetch the next 64x16 / 16x32 operand tile into registers
// before the current tile's MMAs and write it into the same shared stage
// afterwards. The load latency overlaps the MMAs without a second stage, so
// shared memory stays at 8.5 KiB and occupancy is not traded for the
// pipeline. Edge tiles keep the simpler guarded path. The result staging
// reuses the operand buffer after the K loop.
//
// `FUSED` folds an RMSNorm weight and the per-row reciprocal RMS into the A
// staging; the template keeps the normal path free of the extra multiply.
//
// Block indices are swizzled in groups of eight row tiles (CTA swizzle), so
// consecutive blocks reuse the same B tile through L2 instead of walking the
// whole column stripe first.
// Symbols: matmul_wmma, matmul_rms
constexpr unsigned MATMULWMMABM = 64;
constexpr unsigned MATMULWMMABN = 32;
constexpr unsigned MATMULWMMABK = 16;
constexpr unsigned MATMULWMMAAS = MATMULWMMABK + 8;
constexpr unsigned MATMULWMMABS = MATMULWMMABN + 8;
constexpr unsigned MATMULWMMACS = MATMULWMMABN;
constexpr unsigned MATMULWMMATHREADS = 128;
constexpr unsigned MATMULWMMAASIZE = MATMULWMMABM * MATMULWMMAAS;
constexpr unsigned MATMULWMMABSIZE = MATMULWMMABK * MATMULWMMABS;
constexpr unsigned MATMULWMMACSIZE = MATMULWMMABM * MATMULWMMACS;
constexpr unsigned MATMULWMMAGROUPM = 8;

// Runs the two 32x32x8 MMAs for one staged K tile.
static __device__ __forceinline__ void matmul_wmma_tile(
    const float* as, const float* bs,
    wmma::fragment<wmma::accumulator, 32, 32, 8, float> (&acc)[2]) {
    for (unsigned kk = 0; kk < MATMULWMMABK; kk += 8) {
        wmma::fragment<wmma::matrix_b, 32, 32, 8, wmma::precision::tf32,
                       wmma::row_major>
            bfrag;
        wmma::load_matrix_sync(bfrag, &bs[kk * MATMULWMMABS], MATMULWMMABS);
        for (int e = 0; e < bfrag.num_elements; ++e) {
            bfrag.x[e] = wmma::__float_to_tf32(bfrag.x[e]);
        }
        for (unsigned half = 0; half < 2; ++half) {
            wmma::fragment<wmma::matrix_a, 32, 32, 8, wmma::precision::tf32,
                           wmma::row_major>
                afrag;
            wmma::load_matrix_sync(afrag,
                                   &as[(half * 32) * MATMULWMMAAS + kk],
                                   MATMULWMMAAS);
            for (int e = 0; e < afrag.num_elements; ++e) {
                afrag.x[e] = wmma::__float_to_tf32(afrag.x[e]);
            }
            wmma::mma_sync(acc[half], afrag, bfrag, acc[half]);
        }
    }
}

// Loads one operand tile into registers (four floats per access).
static __device__ __forceinline__ void matmul_wmma_load(
    const float* ab, const float* bb, unsigned blockrow, unsigned blockcol,
    unsigned k, unsigned n, unsigned k0, unsigned ra0, unsigned ca0,
    unsigned ra1, unsigned ca1, unsigned rb, unsigned cb0, float4& a0,
    float4& a1, float4& b0) {
    a0 = *reinterpret_cast<const float4 *>(
        &ab[(size_t)(blockrow + ra0) * k + k0 + ca0]);
    a1 = *reinterpret_cast<const float4 *>(
        &ab[(size_t)(blockrow + ra1) * k + k0 + ca1]);
    b0 = *reinterpret_cast<const float4 *>(
        &bb[(size_t)(k0 + rb) * n + blockcol + cb0]);
}

// Applies the RMS weight and per-row inverse RMS to one A register tile.
static __device__ __forceinline__ void matmul_wmma_fold(
    float4& value, const float* rmsweight, unsigned k0, unsigned column,
    float invrow) {
    const float4 weight =
        *reinterpret_cast<const float4 *>(&rmsweight[k0 + column]);
    value.x *= invrow * weight.x;
    value.y *= invrow * weight.y;
    value.z *= invrow * weight.z;
    value.w *= invrow * weight.w;
}

// Writes one operand tile from registers into the shared stage.
static __device__ __forceinline__ void matmul_wmma_store(
    float* as, float* bs, unsigned ra0, unsigned ca0, unsigned ra1,
    unsigned ca1, unsigned rb, unsigned cb0, const float4& a0,
    const float4& a1, const float4& b0) {
    *reinterpret_cast<float4 *>(&as[ra0 * MATMULWMMAAS + ca0]) = a0;
    *reinterpret_cast<float4 *>(&as[ra1 * MATMULWMMAAS + ca1]) = a1;
    *reinterpret_cast<float4 *>(&bs[rb * MATMULWMMABS + cb0]) = b0;
}

template <bool FUSED>
static __device__ __forceinline__ void matmul_wmma_body(
    const float* a, const float* rmsweight, const float* inv, const float* b,
    float* c, unsigned m, unsigned k, unsigned n, unsigned batch) {
    // `cs` reuses the operand buffer after the K loop.
    static_assert(MATMULWMMACSIZE <= MATMULWMMAASIZE + MATMULWMMABSIZE,
                  "result staging must fit in the operand buffer");
    __shared__ __align__(32) float smem[MATMULWMMAASIZE + MATMULWMMABSIZE];
    float* as = smem;
    float* bs = smem + MATMULWMMAASIZE;
    float* cs = smem;

    // CTA swizzle: group MATMULWMMAGROUPM row tiles and walk the column
    // stripe inside the group, so a B tile is reused by up to eight blocks.
    const unsigned tilesm = (m + MATMULWMMABM - 1) / MATMULWMMABM;
    const unsigned tilesn = (n + MATMULWMMABN - 1) / MATMULWMMABN;
    const unsigned pid = blockIdx.y * tilesn + blockIdx.x;
    const unsigned groupwidth = MATMULWMMAGROUPM * tilesn;
    const unsigned groupid = pid / groupwidth;
    const unsigned firstm = groupid * MATMULWMMAGROUPM;
    const unsigned remain = tilesm - firstm;
    const unsigned grouprows =
        remain < MATMULWMMAGROUPM ? remain : MATMULWMMAGROUPM;
    const unsigned pidm = firstm + (pid % grouprows);
    const unsigned pidn = (pid % groupwidth) / grouprows;
    const unsigned blockrow = pidm * MATMULWMMABM;
    const unsigned blockcol = pidn * MATMULWMMABN;
    const unsigned invbase = blockIdx.z * m;

    const unsigned tid = threadIdx.x;
    const float* ab = a + (size_t)blockIdx.z * m * k;
    const float* bb = b + (size_t)blockIdx.z * k * n;
    float* cb = c + (size_t)blockIdx.z * m * n;

    wmma::fragment<wmma::accumulator, 32, 32, 8, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    // A full tile whose rows, columns, and contraction extent are all in range
    // can use the vectorized prefetch path; every other tile uses the guarded
    // scalar path. The condition is block-uniform.
    const bool interior = blockrow + MATMULWMMABM <= m &&
                          blockcol + MATMULWMMABN <= n &&
                          k % MATMULWMMABK == 0 && n % 4 == 0;

    const unsigned ktiles = (k + MATMULWMMABK - 1) / MATMULWMMABK;
    if (interior) {
        // Fixed per-thread register mapping: A is 64x16 = 256 float4 loads
        // (two per thread), B is 16x32 = 128 float4 loads (one per thread).
        const unsigned ja0 = tid * 4;
        const unsigned ra0 = ja0 / MATMULWMMABK;
        const unsigned ca0 = ja0 % MATMULWMMABK;
        const unsigned ja1 = (tid + MATMULWMMATHREADS) * 4;
        const unsigned ra1 = ja1 / MATMULWMMABK;
        const unsigned ca1 = ja1 % MATMULWMMABK;
        const unsigned jb = tid * 4;
        const unsigned rb = jb / MATMULWMMABN;
        const unsigned cb0 = jb % MATMULWMMABN;
        const float inv0 = FUSED ? inv[invbase + blockrow + ra0] : 1.0f;
        const float inv1 = FUSED ? inv[invbase + blockrow + ra1] : 1.0f;

        float4 a0;
        float4 a1;
        float4 b0;
        if (ktiles > 0) {
            matmul_wmma_load(ab, bb, blockrow, blockcol, k, n, 0, ra0, ca0,
                             ra1, ca1, rb, cb0, a0, a1, b0);
            if (FUSED) {
                matmul_wmma_fold(a0, rmsweight, 0, ca0, inv0);
                matmul_wmma_fold(a1, rmsweight, 0, ca1, inv1);
            }
            matmul_wmma_store(as, bs, ra0, ca0, ra1, ca1, rb, cb0, a0, a1, b0);
        }
        __syncthreads();

        for (unsigned kt = 0; kt < ktiles; ++kt) {
            const bool more = kt + 1 < ktiles;
            if (more) {
                const unsigned k0 = (kt + 1) * MATMULWMMABK;
                matmul_wmma_load(ab, bb, blockrow, blockcol, k, n, k0, ra0,
                                 ca0, ra1, ca1, rb, cb0, a0, a1, b0);
                if (FUSED) {
                    matmul_wmma_fold(a0, rmsweight, k0, ca0, inv0);
                    matmul_wmma_fold(a1, rmsweight, k0, ca1, inv1);
                }
            }
            matmul_wmma_tile(as, bs, acc);
            __syncthreads();
            if (more) {
                matmul_wmma_store(as, bs, ra0, ca0, ra1, ca1, rb, cb0, a0, a1,
                                  b0);
            }
            __syncthreads();
        }
    } else {
        for (unsigned kt = 0; kt < ktiles; ++kt) {
            const unsigned k0 = kt * MATMULWMMABK;
            for (unsigned index = tid; index < MATMULWMMABM * MATMULWMMABK;
                 index += MATMULWMMATHREADS) {
                const unsigned r = index / MATMULWMMABK;
                const unsigned col = index % MATMULWMMABK;
                const unsigned gr = blockrow + r;
                const unsigned gk = k0 + col;
                float value = 0.0f;
                if (gr < m && gk < k) {
                    value = ab[(size_t)gr * k + gk];
                    if (FUSED) {
                        value *= rmsweight[gk] * inv[invbase + gr];
                    }
                }
                as[r * MATMULWMMAAS + col] = value;
            }
            for (unsigned index = tid; index < MATMULWMMABK * MATMULWMMABN;
                 index += MATMULWMMATHREADS) {
                const unsigned r = index / MATMULWMMABN;
                const unsigned col = index % MATMULWMMABN;
                const unsigned gk = k0 + r;
                const unsigned gc = blockcol + col;
                bs[r * MATMULWMMABS + col] =
                    (gk < k && gc < n) ? bb[(size_t)gk * n + gc] : 0.0f;
            }
            __syncthreads();
            matmul_wmma_tile(as, bs, acc);
            __syncthreads();
        }
    }

    if (interior && n % 8 == 0) {
        for (unsigned half = 0; half < 2; ++half) {
            wmma::store_matrix_sync(
                cb + (size_t)(blockrow + half * 32) * n + blockcol, acc[half],
                n, wmma::mem_row_major);
        }
    } else {
        // The operand tiles are dead; the barrier makes the aliasing safe.
        __syncthreads();
        for (unsigned half = 0; half < 2; ++half) {
            wmma::store_matrix_sync(&cs[(half * 32) * MATMULWMMACS], acc[half],
                                    MATMULWMMACS, wmma::mem_row_major);
        }
        __syncthreads();
        for (unsigned index = tid; index < MATMULWMMABM * MATMULWMMABN;
             index += MATMULWMMATHREADS) {
            const unsigned r = index / MATMULWMMABN;
            const unsigned col = index % MATMULWMMABN;
            const unsigned gr = blockrow + r;
            const unsigned gc = blockcol + col;
            if (gr < m && gc < n) {
                cb[(size_t)gr * n + gc] = cs[r * MATMULWMMACS + col];
            }
        }
    }
}

// Tensor-Core matmul, uncompressed operands.
extern "C" __global__ void matmul_wmma(const float* a, const float* b,
                                       float* c, unsigned m, unsigned k,
                                       unsigned n, unsigned batch) {
    matmul_wmma_body<false>(a, nullptr, nullptr, b, c, m, k, n, batch);
}

// Tensor-Core matmul with RMSNorm weight and per-row reciprocal RMS folded into
// the A staging: out = (x * weight * inv_row) @ b.
extern "C" __global__ void matmul_rms(const float* a, const float* weight,
                                      const float* inv, const float* b,
                                      float* c, unsigned m, unsigned k,
                                      unsigned n, unsigned batch) {
    matmul_wmma_body<true>(a, weight, inv, b, c, m, k, n, batch);
}
