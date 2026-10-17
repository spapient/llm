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
// Operands are staged through shared memory with guarded, zero-filled loads,
// and the result is staged back through shared memory unless the tile is
// full and the output stride is 32-byte aligned. The result staging reuses
// the operand buffer after the K loop, so the block still fits in 8.5 KiB of
// shared memory. m, n and k need not be multiples of the tile; the leading
// dimension padding keeps every fragment pointer 32-byte aligned.
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

extern "C" __global__ void matmul_wmma(const float* a, const float* b,
                                       float* c, unsigned m, unsigned k,
                                       unsigned n, unsigned batch) {
    // `cs` aliases this buffer after the K loop: the operand tiles are dead
    // once the accumulators are complete, and the two stores are separated by
    // a barrier.
    static_assert(MATMULWMMACSIZE <= MATMULWMMAASIZE + MATMULWMMABSIZE,
                  "result staging must fit in the operand buffer");
    __shared__ __align__(32) float smem[MATMULWMMAASIZE + MATMULWMMABSIZE];
    float* as = smem;
    float* bs = smem + MATMULWMMAASIZE;
    float* cs = smem;

    const unsigned tid = threadIdx.x;
    const unsigned blockrow = blockIdx.y * MATMULWMMABM;
    const unsigned blockcol = blockIdx.x * MATMULWMMABN;

    const float* ab = a + (size_t)blockIdx.z * m * k;
    const float* bb = b + (size_t)blockIdx.z * k * n;
    float* cb = c + (size_t)blockIdx.z * m * n;

    wmma::fragment<wmma::accumulator, 32, 32, 8, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    // A full tile whose rows, columns, and contraction extent are all in range
    // can be staged with 16-byte vector loads; every other tile uses the
    // guarded scalar path. The condition is block-uniform.
    const bool interior = blockrow + MATMULWMMABM <= m &&
                          blockcol + MATMULWMMABN <= n &&
                          k % MATMULWMMABK == 0 && n % 4 == 0;

    const unsigned ktiles = (k + MATMULWMMABK - 1) / MATMULWMMABK;
    for (unsigned kt = 0; kt < ktiles; ++kt) {
        const unsigned k0 = kt * MATMULWMMABK;
        if (interior) {
            // A is 64x16 = 256 float4 loads, two per thread.
            for (unsigned half = 0; half < 2; ++half) {
                const unsigned ja = (tid + half * MATMULWMMATHREADS) * 4;
                const unsigned ra = ja / MATMULWMMABK;
                const unsigned ca = ja % MATMULWMMABK;
                const float4 va = *reinterpret_cast<const float4 *>(
                    &ab[(size_t)(blockrow + ra) * k + k0 + ca]);
                *reinterpret_cast<float4 *>(&as[ra * MATMULWMMAAS + ca]) = va;
            }
            // B is 16x32 = 128 float4 loads, one per thread.
            const unsigned jb = tid * 4;
            const unsigned rb = jb / MATMULWMMABN;
            const unsigned cb0 = jb % MATMULWMMABN;
            const float4 vb = *reinterpret_cast<const float4 *>(
                &bb[(size_t)(k0 + rb) * n + blockcol + cb0]);
            *reinterpret_cast<float4 *>(&bs[rb * MATMULWMMABS + cb0]) = vb;
        } else {
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
        }
        __syncthreads();

        for (unsigned kk = 0; kk < MATMULWMMABK; kk += 8) {
            wmma::fragment<wmma::matrix_b, 32, 32, 8, wmma::precision::tf32,
                           wmma::row_major>
                bfrag;
            wmma::load_matrix_sync(bfrag, &bs[kk * MATMULWMMABS], MATMULWMMABS);
            for (int e = 0; e < bfrag.num_elements; ++e) {
                bfrag.x[e] = wmma::__float_to_tf32(bfrag.x[e]);
            }
            for (unsigned half = 0; half < 2; ++half) {
                wmma::fragment<wmma::matrix_a, 32, 32, 8,
                               wmma::precision::tf32, wmma::row_major>
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
        __syncthreads();
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
