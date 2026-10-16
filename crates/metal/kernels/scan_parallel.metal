// f32 semigroup scan, parallel three-stage kernel family, compiled at run time
// through newLibraryWithSource. The serial per-channel kernel (scan.metal)
// remains the bit-for-bit reference and the fallback for state widths above
// LLM_SCAN_MAXN; this file provides the measured parallel path.
//
// Mathematics (docs/lld.md §5.6, verification 5b): the recurrence is fully
// independent per component j in [0, d*N):
//   h_t[j] = a[j] * h_{t-1}[j] + b[j] * x_t
// The semigroup element is E_t[j] = (a[j], b[j]*x_t) with the associative
// combine (A1,B1) ⊕ (A2,B2) = (A1*A2, A2*B1 + B2). The sequence is split into
// P blocks of CHUNK steps; Pass 1 aggregates each block's total, Pass 2
// computes the inter-block prefix, Pass 3 re-applies each block starting from
// its prefix so the intra-block recurrence is bit-for-bit the serial one. The
// start point comes from the prefix product (a different float association),
// so results are within the §5.10 tolerance gate (tests assert allclose 1e-5).
//
// Grid layout shared by the block kernels: width = P sequence blocks, height =
// ceil(d / LANES) channel groups; each threadgroup holds LANES threads, one per
// channel i = group_y * LANES + thread. Works for any d, and for N ≤
// LLM_SCAN_MAXN (the thread-local state array is compile-time sized).

#include <metal_stdlib>
using namespace metal;

#define LLM_SCAN_MAXN 256
#define LLM_SCAN_LANES 256

// Pass 1: per-block total. Thread i aggregates the block's semigroup product
// for its channel's N components into block_total[p][2][d*N] (A half first).
kernel void scan_agg(device const float* input [[buffer(0)]],
                     device const float* a [[buffer(1)]],
                     device const float* b [[buffer(2)]],
                     device const float* c [[buffer(3)]],
                     device const float* skip [[buffer(4)]],
                     device const float* init [[buffer(5)]],
                     device float* block_total [[buffer(6)]],
                     constant uint* p [[buffer(7)]],
                     uint2 pos [[thread_position_in_grid]]) {
    const uint n = p[0];
    const uint d = p[1];
    const uint N = p[2];
    const uint chunk = p[3];
    const uint blocks = p[4];
    const uint block = pos.x;
    const uint i = pos.y;
    if (i >= d) {
        return;
    }
    const uint t0 = block * chunk;
    const uint t1 = min(t0 + chunk, n);
    const uint dN = d * N;
    const uint base = block * (2u * dN);
    for (uint j = 0; j < N; ++j) {
        const uint idx = i * N + j;
        float A = 1.0f;
        float B = 0.0f;
        const float aj = a[idx];
        const float bj = b[idx];
        for (uint t = t0; t < t1; ++t) {
            const float x = input[t * d + i];
            B = aj * B + bj * x;
            A = aj * A;
        }
        block_total[base + idx] = A;
        block_total[base + dN + idx] = B;
    }
}

// Pass 2: inter-block prefix, one threadgroup. Each thread owns a strided
// subset of the d*N components and accumulates the prefix over the P block
// totals serially; prefix[p] = ⊕_{q<p} total[q] with prefix[0] = identity
// (1, 0). The loop-final accumulator is the whole-sequence total, which yields
// the final state: state[j] = A_total * init[j] + B_total.
kernel void scan_prefix(device const float* block_total [[buffer(0)]],
                        device const float* init [[buffer(1)]],
                        device float* state [[buffer(2)]],
                        device float* block_prefix [[buffer(3)]],
                        constant uint* p [[buffer(4)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]]) {
    const uint d = p[1];
    const uint N = p[2];
    const uint blocks = p[4];
    const uint dN = d * N;
    for (uint j = tid; j < dN; j += threads) {
        // prefix[0] is the identity (1, 0).
        block_prefix[j] = 1.0f;
        block_prefix[dN + j] = 0.0f;
        float A = 1.0f;
        float B = 0.0f;
        for (uint q = 0; q < blocks; ++q) {
            const uint base = q * (2u * dN);
            const float aq = block_total[base + j];
            const float bq = block_total[base + dN + j];
            block_prefix[(q + 1u) * (2u * dN) + j] = A * aq;
            // fma folds the multiply-add into one rounding, halving the
            // inter-block combine error that grows like P·ε·‖B‖.
            block_prefix[(q + 1u) * (2u * dN) + dN + j] = fma(aq, B, bq);
            B = fma(aq, B, bq);
            A = A * aq;
        }
        state[j] = A * init[j] + B;
    }
}

// Pass 3: per-block apply. Thread i owns channel i; its thread-local h array
// holds the channel's N state components. The block starts at the inter-block
// prefix state h = A0*init + B0, then runs the same intra-block recurrence as
// the serial reference, writing out[t][i] = skip[i]*x + Σ_j c[i,j]*h[j].
kernel void scan_apply(device const float* input [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
                       device const float* c [[buffer(3)]],
                       device const float* skip [[buffer(4)]],
                       device const float* init [[buffer(5)]],
                       device const float* block_prefix [[buffer(6)]],
                       device float* out [[buffer(7)]],
                       constant uint* p [[buffer(8)]],
                       uint2 pos [[thread_position_in_grid]]) {
    const uint n = p[0];
    const uint d = p[1];
    const uint N = p[2];
    const uint chunk = p[3];
    const uint blocks = p[4];
    const uint block = pos.x;
    const uint i = pos.y;
    if (i >= d) {
        return;
    }
    const uint t0 = block * chunk;
    const uint t1 = min(t0 + chunk, n);
    const uint dN = d * N;
    const uint base = block * (2u * dN);
    thread float h[LLM_SCAN_MAXN];
    for (uint j = 0; j < N; ++j) {
        const uint idx = i * N + j;
        h[j] = block_prefix[base + idx] * init[idx] + block_prefix[base + dN + idx];
    }
    for (uint t = t0; t < t1; ++t) {
        const float x = input[t * d + i];
        for (uint j = 0; j < N; ++j) {
            const uint idx = i * N + j;
            h[j] = a[idx] * h[j] + b[idx] * x;
        }
        float acc = 0.0f;
        for (uint j = 0; j < N; ++j) {
            acc += c[i * N + j] * h[j];
        }
        out[t * d + i] = acc + skip[i] * x;
    }
}
