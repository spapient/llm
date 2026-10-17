// MUSA matmul kernels, compiled at build time by mcc.
//
// `matmul` is the reference FMA kernel; `matmul_wmma` is the Tensor-Core
// (TF32) fast path. Both take a is [batch, m, k], b is [batch, k, n], c is
// [batch, m, n], all row-major; blockIdx.z selects the batch slice so one
// dispatch covers every slice. The host guards m == 0, n == 0 and batch == 0;
// k == 0 produces the additive identity.

#include <mma.h>

// Reference f32 matrix-multiply kernel.
// Threads outside the output bounds return early.
// Symbol: matmul
extern "C" __global__ void matmul(const float* a, const float* b, float* c,
                                  unsigned m, unsigned k, unsigned n,
                                  unsigned batch) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    const float* ab = a + (size_t)blockIdx.z * m * k;
    const float* bb = b + (size_t)blockIdx.z * k * n;
    float* cb = c + (size_t)blockIdx.z * m * n;
    float acc = 0.0f;
    for (unsigned i = 0; i < k; ++i) {
        acc += ab[row * k + i] * bb[i * n + col];
    }
    cb[row * n + col] = acc;
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
// Block indices are swizzled in groups of eight row tiles (CTA swizzle), so
// consecutive blocks reuse the same B tile through L2 instead of walking the
// whole column stripe first.
// Symbol: matmul_wmma
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

// Writes one operand tile from registers into the shared stage.
static __device__ __forceinline__ void matmul_wmma_store(
    float* as, float* bs, unsigned ra0, unsigned ca0, unsigned ra1,
    unsigned ca1, unsigned rb, unsigned cb0, const float4& a0,
    const float4& a1, const float4& b0) {
    *reinterpret_cast<float4 *>(&as[ra0 * MATMULWMMAAS + ca0]) = a0;
    *reinterpret_cast<float4 *>(&as[ra1 * MATMULWMMAAS + ca1]) = a1;
    *reinterpret_cast<float4 *>(&bs[rb * MATMULWMMABS + cb0]) = b0;
}

extern "C" __global__ void matmul_wmma(const float* a, const float* b,
                                       float* c, unsigned m, unsigned k,
                                       unsigned n, unsigned batch) {
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

        float4 a0;
        float4 a1;
        float4 b0;
        if (ktiles > 0) {
            matmul_wmma_load(ab, bb, blockrow, blockcol, k, n, 0, ra0, ca0,
                             ra1, ca1, rb, cb0, a0, a1, b0);
            matmul_wmma_store(as, bs, ra0, ca0, ra1, ca1, rb, cb0, a0, a1, b0);
        }
        __syncthreads();

        for (unsigned kt = 0; kt < ktiles; ++kt) {
            const bool more = kt + 1 < ktiles;
            if (more) {
                matmul_wmma_load(ab, bb, blockrow, blockcol, k, n,
                                 (kt + 1) * MATMULWMMABK, ra0, ca0, ra1, ca1,
                                 rb, cb0, a0, a1, b0);
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
                as[r * MATMULWMMAAS + col] =
                    (gr < m && gk < k) ? ab[(size_t)gr * k + gk] : 0.0f;
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
