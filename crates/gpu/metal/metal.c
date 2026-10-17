// Metal Shading Language sources for the Metal backend.
//
// One translation unit holds every kernel entry the backend dispatches; the
// runtime compiles it once through `newLibraryWithSource` and then builds one
// compute pipeline per entry. The sections below were originally separate
// files; they are kept together so a platform-specific kernel change is
// visible in one place.
//
// Entries: attention, attention_cached, add, mul, silu, embedding, matmul,
//          rmsnorm, rope, rope_at, scan, scan_agg, scan_prefix, scan_apply,
//          softmax.
#include <metal_stdlib>
using namespace metal;

#define NEGLARGE (-3.402823466e+38f)

// ===========================================================================
// attention
// ===========================================================================
// Fused scaled-dot-product attention kernel, compiled at run time through
// newLibraryWithSource. One threadgroup per (query position, head); scores are
// recomputed, not materialized. Threadgroup memory is proportional to
// 2 * ATTENTIONLANES + METALMAXHEADDIM, independent of sequence length. NEGLARGE
// is the most negative finite f32 (not -inf). p carries qseq, kvseq, heads,
// kvheads, headdim, causal as six uint values at indices 0..5. Runs with exactly
// ATTENTIONLANES threads per threadgroup; accum is sized by METALMAXHEADDIM, so
// those two bounds MUST stay equal.
// Symbol: attention


float attentiondot(device const float* qrow, device const float* krow, uint headdim) {
    float dot = 0.0f;
    for (uint d = 0; d < headdim; ++d) {
        dot += qrow[d] * krow[d];
    }
    return dot;
}

float reducemax(threadgroup float* scratch, float value, uint tid, uint tg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

float reducesum(threadgroup float* scratch, float value, uint tid, uint tg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

kernel void attention(device const float* q [[buffer(0)]],
                      device const float* k [[buffer(1)]],
                      device const float* v [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      constant uint* p [[buffer(4)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint tg [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    threadgroup float weights[256];
    threadgroup float accum[4096];

    uint qseq = p[0];
    uint kvseq = p[1];
    uint heads = p[2];
    uint kvheads = p[3];
    uint headdim = p[4];
    uint causal = p[5];
    uint offset = kvseq - qseq;
    uint head = tgid % heads;
    uint position = tgid / heads;
    uint kvhead = head / (heads / kvheads);
    uint limit = (causal != 0u && offset + position + 1u < kvseq) ? (offset + position + 1u) : kvseq;
    device const float* qrow = q + ((size_t)position * heads + head) * headdim;
    float scale = 1.0f / sqrt((float)headdim);

    float localmax = NEGLARGE;
    for (uint key = tid; key < limit; key += tg) {
        device const float* krow = k + ((size_t)key * kvheads + kvhead) * headdim;
        float score = attentiondot(qrow, krow, headdim) * scale;
        if (score > localmax) {
            localmax = score;
        }
    }
    float maxv = reducemax(scratch, localmax, tid, tg);

    float localsum = 0.0f;
    for (uint key = tid; key < limit; key += tg) {
        device const float* krow = k + ((size_t)key * kvheads + kvhead) * headdim;
        localsum += exp(attentiondot(qrow, krow, headdim) * scale - maxv);
    }
    float total = reducesum(scratch, localsum, tid, tg);

    for (uint d = tid; d < headdim; d += tg) {
        accum[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base < limit; base += tg) {
        uint span = limit - base;
        if (span > tg) {
            span = tg;
        }
        if (tid < span) {
            device const float* krow = k + ((size_t)(base + tid) * kvheads + kvhead) * headdim;
            weights[tid] = exp(attentiondot(qrow, krow, headdim) * scale - maxv);
        } else {
            weights[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < headdim; d += tg) {
            float partial = 0.0f;
            for (uint j = 0; j < span; ++j) {
                device const float* vrow = v + ((size_t)(base + j) * kvheads + kvhead) * headdim;
                partial += weights[j] * vrow[d];
            }
            accum[d] += partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* orow = out + ((size_t)position * heads + head) * headdim;
    for (uint d = tid; d < headdim; d += tg) {
        orow[d] = (total > 0.0f) ? (accum[d] / total) : 0.0f;
    }
}

// ===========================================================================
// attention_cached
// ===========================================================================
// Incremental-decode attention kernel, compiled at run time through
// newLibraryWithSource. One query token attends over a cached key/value buffer
// followed by one new row, without materialising the concatenation. The query
// has a single position, so the grid is heads. p carries kvseq, heads, kvheads,
// headdim as four uint values at indices 0..3.
// Symbol: attention_cached


float attentioncacheddot(device const float* qrow, device const float* krow, uint headdim) {
    float dot = 0.0f;
    for (uint d = 0; d < headdim; ++d) {
        dot += qrow[d] * krow[d];
    }
    return dot;
}

float cachedreducemax(threadgroup float* scratch, float value, uint tid, uint tg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

float cachedreducesum(threadgroup float* scratch, float value, uint tid, uint tg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

kernel void attention_cached(
    device const float* q [[buffer(0)]],
    device const float* ck [[buffer(1)]],
    device const float* cv [[buffer(2)]],
    device const float* nk [[buffer(3)]],
    device const float* nv [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    threadgroup float weights[256];
    threadgroup float accum[4096];

    uint kvseq = p[0];
    uint heads = p[1];
    uint kvheads = p[2];
    uint headdim = p[3];
    uint pos = kvseq - 1u;
    uint head = tgid % heads;
    uint kvhead = head / (heads / kvheads);
    uint limit = kvseq;
    device const float* qrow = q + ((size_t)head) * headdim;
    float scale = 1.0f / sqrt((float)headdim);

    float localmax = NEGLARGE;
    for (uint key = tid; key < limit; key += tg) {
        device const float* krow = (key < pos)
            ? (ck + ((size_t)key * kvheads + kvhead) * headdim)
            : (nk + ((size_t)kvhead) * headdim);
        float score = attentioncacheddot(qrow, krow, headdim) * scale;
        if (score > localmax) {
            localmax = score;
        }
    }
    scratch[tid] = localmax;
    float maxv = cachedreducemax(scratch, localmax, tid, tg);

    float localsum = 0.0f;
    for (uint key = tid; key < limit; key += tg) {
        device const float* krow = (key < pos)
            ? (ck + ((size_t)key * kvheads + kvhead) * headdim)
            : (nk + ((size_t)kvhead) * headdim);
        localsum += exp(attentioncacheddot(qrow, krow, headdim) * scale - maxv);
    }
    scratch[tid] = localsum;
    float total = cachedreducesum(scratch, localsum, tid, tg);

    for (uint d = tid; d < headdim; d += tg) {
        accum[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base < limit; base += tg) {
        uint span = limit - base;
        if (span > tg) {
            span = tg;
        }
        if (tid < span) {
            uint key = base + tid;
            device const float* krow = (key < pos)
                ? (ck + ((size_t)key * kvheads + kvhead) * headdim)
                : (nk + ((size_t)kvhead) * headdim);
            weights[tid] = exp(attentioncacheddot(qrow, krow, headdim) * scale - maxv);
        } else {
            weights[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < headdim; d += tg) {
            float partial = 0.0f;
            for (uint j = 0; j < span; ++j) {
                uint key = base + j;
                device const float* vrow = (key < pos)
                    ? (cv + ((size_t)key * kvheads + kvhead) * headdim)
                    : (nv + ((size_t)kvhead) * headdim);
                partial += weights[j] * vrow[d];
            }
            accum[d] += partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* orow = out + ((size_t)head) * headdim;
    for (uint d = tid; d < headdim; d += tg) {
        orow[d] = (total > 0.0f) ? (accum[d] / total) : 0.0f;
    }
}

// ===========================================================================
// elementwise
// ===========================================================================
// Elementwise f32 kernels, compiled at run time through newLibraryWithSource.
// add and mul take two equal-length operands; silu takes one. All stride over
// the flat element count and bounds-test. The count is a single uint at p[0].
// silu reproduces cpu's x / (1 + exp(-x)) exactly (no fused approximation).
// Symbols: add, mul, silu

kernel void add(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                constant uint* p [[buffer(3)]],
                uint gid [[thread_position_in_grid]]) {
    if (gid < p[0]) {
        c[gid] = a[gid] + b[gid];
    }
}

kernel void mul(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                constant uint* p [[buffer(3)]],
                uint gid [[thread_position_in_grid]]) {
    if (gid < p[0]) {
        c[gid] = a[gid] * b[gid];
    }
}

kernel void silu(device const float* a [[buffer(0)]],
                 device float* c [[buffer(1)]],
                 constant uint* p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid < p[0]) {
        float x = a[gid];
        c[gid] = x / (1.0f + exp(-x));
    }
}

// ===========================================================================
// embedding
// ===========================================================================
// f32 embedding-gather kernel, compiled at run time through newLibraryWithSource.
// table is [vocab, width], indices holds rows row numbers validated on the host,
// out is [rows, width]. p carries rows and width as two ulong values at indices
// 0 and 1. One thread per output element; fully coalesced reads.
// Symbol: embedding

kernel void embedding(device const float* table [[buffer(0)]],
                      device const ulong* indices [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant ulong* p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    ulong width = p[1];
    ulong total = p[0] * width;
    if ((ulong)gid >= total) {
        return;
    }
    ulong row = (ulong)gid / width;
    out[gid] = table[indices[row] * width + ((ulong)gid % width)];
}

// ===========================================================================
// matmul
// ===========================================================================
// f32 matrix-multiply kernel, compiled at run time through newLibraryWithSource.
// One dispatch carries every batch block: a is [batch, m, k], b is [batch, k, n],
// c is [batch, m, n], all row-major; the third grid dimension is the block index.
// The parameter buffer p carries m, k, n as three uint values at indices 0, 1, 2.
// Threads outside output bounds return early; the host guards m == 0, n == 0, k == 0.
// Symbol: matmul

kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant uint* p [[buffer(3)]],
                   uint3 gid [[thread_position_in_grid]]) {
    uint m = p[0];
    uint k = p[1];
    uint n = p[2];
    if (gid.x >= n || gid.y >= m) {
        return;
    }
    device const float* ab = a + (size_t)gid.z * m * k;
    device const float* bb = b + (size_t)gid.z * k * n;
    device float* cb = c + (size_t)gid.z * m * n;
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) {
        acc += ab[gid.y * k + i] * bb[i * n + gid.x];
    }
    cb[gid.y * n + gid.x] = acc;
}

// ===========================================================================
// rmsnorm
// ===========================================================================
// f32 RMS-normalization kernel, compiled at run time through newLibraryWithSource.
// One threadgroup per row; x is [rows, width], w is [width], y is [rows, width].
// p[0] = width (uint), p[1] = eps (u32 bit pattern of a float). The row extent
// is taken from threads_per_threadgroup, so the caller MUST dispatch exactly
// RMSNORMTILE threads per threadgroup (a power of two).
// Symbol: rmsnorm

kernel void rmsnorm(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device float* y [[buffer(2)]],
                    constant uint* p [[buffer(3)]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tg [[threads_per_threadgroup]]) {
    threadgroup float partial[256];
    uint width = p[0];
    float eps = as_type<float>(p[1]);
    device const float* row = x + (size_t)tgid * width;
    float sumsq = 0.0f;
    for (uint i = tid; i < width; i += tg) {
        float v = row[i];
        sumsq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(partial[0] / (float)width + eps);
    for (uint i = tid; i < width; i += tg) {
        y[(size_t)tgid * width + i] = row[i] * inv * w[i];
    }
}

// ===========================================================================
// rope
// ===========================================================================
// f32 rotary-position-embedding kernel, compiled at run time through
// newLibraryWithSource. One thread per (row, head, pair); x and y are
// [rows, heads * headdim]. p carries rows, seqaxis, heads, headdim as uint at
// indices 0..3 and base as its u32 bit pattern at index 4. seqaxis == 0 places
// every row at position 0.
// Symbol: rope

kernel void rope(device const float* x [[buffer(0)]],
                 device float* y [[buffer(1)]],
                 constant uint* p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    uint pairs = p[3] / 2;
    ulong total = (ulong)p[0] * p[2] * pairs;
    if ((ulong)gid >= total) {
        return;
    }
    uint i = (uint)((ulong)gid % pairs);
    uint h = (uint)(((ulong)gid / pairs) % p[2]);
    uint row = (uint)((ulong)gid / ((ulong)pairs * p[2]));
    uint position = (p[1] == 0u) ? 0u : (row % p[1]);
    float angle = (float)position * pow(as_type<float>(p[4]), -2.0f * (float)i / (float)p[3]);
    float s = sin(angle);
    float c = cos(angle);
    ulong offset = (((ulong)row * p[2]) + h) * p[3] + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}

// ===========================================================================
// rope_at
// ===========================================================================
// Absolute-position rotary embedding of a single token, compiled at run time
// through newLibraryWithSource. Position is a scalar parameter, not row % seqaxis.
// p carries heads, headdim, position, and base's u32 bit pattern at indices 0..3.
// Symbol: rope_at

kernel void rope_at(device const float* x [[buffer(0)]],
                    device float* y [[buffer(1)]],
                    constant uint* p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    uint heads = p[0];
    uint headdim = p[1];
    uint position = p[2];
    float base = as_type<float>(p[3]);
    uint pairs = headdim / 2;
    ulong total = (ulong)heads * pairs;
    if ((ulong)gid >= total) {
        return;
    }
    uint i = (uint)((ulong)gid % pairs);
    uint h = (uint)((ulong)gid / pairs);
    float angle = (float)position * pow(base, -2.0f * (float)i / (float)headdim);
    float s = sin(angle);
    float c = cos(angle);
    ulong offset = ((ulong)h) * headdim + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}

// ===========================================================================
// scan
// ===========================================================================
// Serial per-channel scan kernel, compiled at run time through
// newLibraryWithSource. One thread per channel; each thread runs the serial
// recurrence over the sequence axis for its channel's state vector, so the
// result is bit-for-bit the serial reference (the parallel reorganization is a
// later, measured optimization). The state buffer is initialized to `init` by
// the host and updated in place; `p` carries n, d, N as three uint values at
// indices 0..2. Symbol: scan

kernel void scan(device const float* input [[buffer(0)]],
                 device const float* a [[buffer(1)]],
                 device const float* b [[buffer(2)]],
                 device const float* c [[buffer(3)]],
                 device const float* skip [[buffer(4)]],
                 device float* state [[buffer(5)]],
                 device float* out [[buffer(6)]],
                 constant uint* p [[buffer(7)]],
                 uint gid [[thread_position_in_grid]]) {
    uint n = p[0];
    uint d = p[1];
    uint N = p[2];
    uint i = gid;
    if (i >= d) {
        return;
    }
    for (uint t = 0; t < n; ++t) {
        float x = input[t * d + i];
        float acc = 0.0f;
        for (uint j = 0; j < N; ++j) {
            uint idx = i * N + j;
            float h = a[idx] * state[idx] + b[idx] * x;
            state[idx] = h;
            acc += c[idx] * h;
        }
        out[t * d + i] = acc + skip[i] * x;
    }
}

// ===========================================================================
// scan_parallel
// ===========================================================================
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

// ===========================================================================
// softmax
// ===========================================================================
// f32 last-axis softmax kernel, compiled at run time through newLibraryWithSource.
// One threadgroup per row; subtracts the row max before exponentiating (cpu's
// numerically stable form) and divides only when the sum is positive. p[0] is
// width as a uint. The caller MUST dispatch exactly SOFTMAXTILE threads per
// threadgroup (a power of two). A row of -inf propagates NaN.
// Symbol: softmax

kernel void softmax(device const float* x [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint* p [[buffer(2)]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tg [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    uint width = p[0];
    device const float* row = x + (size_t)tgid * width;
    device float* dst = out + (size_t)tgid * width;

    float localmax = -INFINITY;
    for (uint i = tid; i < width; i += tg) {
        float v = row[i];
        if (v > localmax) {
            localmax = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = localmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s && scratch[tid + s] > scratch[tid]) {
            scratch[tid] = scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowmax = scratch[0];

    float localsum = 0.0f;
    for (uint i = tid; i < width; i += tg) {
        localsum += exp(row[i] - rowmax);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = localsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = scratch[0];

    for (uint i = tid; i < width; i += tg) {
        float e = exp(row[i] - rowmax);
        dst[i] = (total > 0.0f) ? (e / total) : e;
    }
}
