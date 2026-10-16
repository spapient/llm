// Three-stage parallel f32 semigroup scan kernels, compiled at run time through
// NVRTC. The design mirrors crates/metal/kernels/scan_parallel.metal: the scan
// semigroup decomposes per component (h_t[j] = a[j]·h_{t-1}[j] + b[j]·x_t, and
// the combine (A1,B1)⊕(A2,B2) = (A1·A2, A2·B1+B2) is associative), so the
// sequence axis is parallelized in three passes:
//
//   scan_agg     one thread per (block, channel) aggregates its block's
//                semigroup product into block_total[p][2][d*N] (A half first);
//   scan_prefix  a single block, one thread per strided component, serially
//                combines the P block totals into block_prefix[q][2][d*N] with
//                prefix[0] = identity (1, 0); the loop-final accumulator is the
//                whole-sequence total, giving the final state
//                state[j] = A_total·init[j] + B_total;
//   scan_apply   one thread per (block, channel) starts from the inter-block
//                prefix state h = A0·init + B0 and runs the same intra-block
//                recurrence as the serial reference (plain multiply-add, no
//                fma, bit-for-bit the serial order), writing
//                out[t][i] = skip[i]·x + Σ_j c[i,j]·h[j].
//
// N > LLM_SCAN_MAXN is rejected at the host by falling back to the serial
// kernel (scan.cu); thread-local h arrays need a compile-time bound.
#define LLM_SCAN_MAXN 256
#define LLM_SCAN_BLOCKDIM 256

// Pass 1: per-block total. Thread (blockIdx.x, channel) aggregates its block's
// semigroup product per component; the intra-block recurrence order matches the
// serial reference so each block total is bit-for-bit the serial prefix at the
// block boundary.
extern "C" __global__ void scan_agg(const float* input, const float* a,
                                    const float* b, const float* c,
                                    const float* skip, const float* init,
                                    float* block_total, unsigned n, unsigned d,
                                    unsigned N, unsigned chunk, unsigned blocks) {
    const unsigned block = blockIdx.x;
    const unsigned i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const unsigned t0 = block * chunk;
    const unsigned t1 = min(t0 + chunk, n);
    const unsigned dN = d * N;
    const unsigned base = block * (2u * dN);
    for (unsigned j = 0; j < N; ++j) {
        const unsigned idx = i * N + j;
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
}

// Pass 2: inter-block prefix, one block. Each thread owns a strided subset of
// the d*N components and accumulates the prefix over the P block totals
// serially; prefix[p] = ⊕_{q<p} total[q] with prefix[0] = identity (1, 0). The
// loop-final accumulator is the whole-sequence total, which yields the final
// state: state[j] = A_total * init[j] + B_total. fma folds the combine into one
// rounding, halving the inter-block error that grows like P·ε·‖B‖.
extern "C" __global__ void scan_prefix(const float* block_total,
                                       const float* init, float* state,
                                       float* block_prefix, unsigned n,
                                       unsigned d, unsigned N, unsigned chunk,
                                       unsigned blocks) {
    const unsigned dN = d * N;
    for (unsigned j = threadIdx.x; j < dN; j += blockDim.x) {
        // prefix[0] is the identity (1, 0).
        block_prefix[j] = 1.0f;
        block_prefix[dN + j] = 0.0f;
        float A = 1.0f;
        float B = 0.0f;
        for (unsigned q = 0; q < blocks; ++q) {
            const unsigned base = q * (2u * dN);
            const float aq = block_total[base + j];
            const float bq = block_total[base + dN + j];
            block_prefix[(q + 1u) * (2u * dN) + j] = A * aq;
            block_prefix[(q + 1u) * (2u * dN) + dN + j] = fmaf(aq, B, bq);
            B = fmaf(aq, B, bq);
            A = A * aq;
        }
        state[j] = A * init[j] + B;
    }
}

// Pass 3: per-block apply. Thread i owns channel i; its thread-local h array
// holds the channel's N state components. The block starts at the inter-block
// prefix state h = A0*init + B0, then runs the same intra-block recurrence as
// the serial reference, writing out[t][i] = skip[i]*x + Σ_j c[i,j]*h[j].
extern "C" __global__ void scan_apply(const float* input, const float* a,
                                      const float* b, const float* c,
                                      const float* skip, const float* init,
                                      const float* block_prefix, float* out,
                                      unsigned n, unsigned d, unsigned N,
                                      unsigned chunk, unsigned blocks) {
    const unsigned block = blockIdx.x;
    const unsigned i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const unsigned t0 = block * chunk;
    const unsigned t1 = min(t0 + chunk, n);
    const unsigned dN = d * N;
    const unsigned base = block * (2u * dN);
    float h[LLM_SCAN_MAXN];
    for (unsigned j = 0; j < N; ++j) {
        const unsigned idx = i * N + j;
        h[j] = block_prefix[base + idx] * init[idx]
            + block_prefix[base + dN + idx];
    }
    for (unsigned t = t0; t < t1; ++t) {
        const float x = input[t * d + i];
        float acc = 0.0f;
        for (unsigned j = 0; j < N; ++j) {
            const unsigned idx = i * N + j;
            h[j] = a[idx] * h[j] + b[idx] * x;
            acc += c[idx] * h[j];
        }
        out[t * d + i] = acc + skip[i] * x;
    }
}
