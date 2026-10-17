#include "grid.h"
// BANG C sources for the Cambricon MLU backend (Siyuan / MLU processors, the
// systolic-array architecture).
//
// BANG is the Cambricon kernel dialect. A kernel runs per MLU task (an
// on-chip core with its own NRAM), and parallel work is partitioned
// explicitly with `taskId()` / `taskDim()`. Global memory moves into the
// on-chip `__nram__` buffers through `__memcpy` (GDRAM2NRAM / NRAM2GDRAM),
// vector intrinsics (`__bang_add`, `__bang_mul`, `__bang_exp`,
// `__bang_reduce_max`, `__bang_reduce_sum`, ...) operate in place on NRAM,
// and results are copied back. Scalar global-memory access is legal for the
// correctness-first operators (attention / matmul / rope / scan).
//
// Build (Linux, Cambricon NeuWare):
//   cncc --bang-mlu-arch=mtp_372 -c mlu.c -o mlu.o
// (the exact `--bang-mlu-arch` follows the target MLU; the file is a single
// translation unit so one invocation produces the whole operator set). The
// MLU module is compiled out when `cncc` is absent, exactly like the other
// toolchain-gated backends (build.rs emits an empty registry).
//
// Semantics: every entry below reproduces the shared kernel family
// (grid/cuda.c) numerically for finite inputs. Simple elementwise
// and normalization operators use the NRAM vector pipeline; the rest use
// per-task scalar loops for correctness first.
//
// Entries: attention, attention_cached, add, mul, silu, embedding, matmul,
//          rmsnorm, rope, rope_at, scan, scan_agg, scan_prefix, scan_apply,
//          softmax.

#define NRAMBUF 256  // NRAM 向量块元素数（片上内存有限，按块循环）

// ===========================================================================
// elementwise
// ===========================================================================
// Elementwise f32 kernels: add and mul take two equal-length operands, silu
// takes one. All stride over the flat element count and bounds-test. silu
// reproduces cpu's x / (1 + exp(-x)) exactly (no fused approximation).
// Symbols: add, mul, silu

__global__ void add(const float* a, const float* b, float* c, unsigned n) {
    const unsigned cores = taskDim();
    unsigned start, end;
    GRID_RANGE(taskId(), cores, n, start, end);
    __nram__ float aLocal[NRAMBUF];
    __nram__ float bLocal[NRAMBUF];
    __nram__ float cLocal[NRAMBUF];
    for (unsigned off = start; off < end; off += NRAMBUF) {
        GRID_TAIL(off, end, NRAMBUF, len);
        __memcpy(aLocal, a + off, len * sizeof(float), GDRAM2NRAM);
        __memcpy(bLocal, b + off, len * sizeof(float), GDRAM2NRAM);
        __bang_add(cLocal, aLocal, bLocal, len);
        __memcpy(c + off, cLocal, len * sizeof(float), NRAM2GDRAM);
    }
}

__global__ void mul(const float* a, const float* b, float* c, unsigned n) {
    const unsigned cores = taskDim();
    unsigned start, end;
    GRID_RANGE(taskId(), cores, n, start, end);
    __nram__ float aLocal[NRAMBUF];
    __nram__ float bLocal[NRAMBUF];
    __nram__ float cLocal[NRAMBUF];
    for (unsigned off = start; off < end; off += NRAMBUF) {
        GRID_TAIL(off, end, NRAMBUF, len);
        __memcpy(aLocal, a + off, len * sizeof(float), GDRAM2NRAM);
        __memcpy(bLocal, b + off, len * sizeof(float), GDRAM2NRAM);
        __bang_mul(cLocal, aLocal, bLocal, len);
        __memcpy(c + off, cLocal, len * sizeof(float), NRAM2GDRAM);
    }
}

__global__ void silu(const float* a, float* c, unsigned n) {
    const unsigned cores = taskDim();
    unsigned start, end;
    GRID_RANGE(taskId(), cores, n, start, end);
    __nram__ float aLocal[NRAMBUF];
    __nram__ float cLocal[NRAMBUF];
    __nram__ float tmp[NRAMBUF];
    for (unsigned off = start; off < end; off += NRAMBUF) {
        GRID_TAIL(off, end, NRAMBUF, len);
        __memcpy(aLocal, a + off, len * sizeof(float), GDRAM2NRAM);
        // silu(x) = x / (1 + exp(-x))，分步向量化，与共享内核数值一致。
        __bang_muls(tmp, aLocal, -1.0f, len);
        __bang_exp(tmp, tmp, len);
        __bang_adds(tmp, tmp, 1.0f, len);
        __bang_div(cLocal, aLocal, tmp, len);
        __memcpy(c + off, cLocal, len * sizeof(float), NRAM2GDRAM);
    }
}

// ===========================================================================
// embedding
// ===========================================================================
// f32 embedding-gather kernel: table is [vocab, width], indices holds rows
// validated on the host, out is [rows, width]. Rows are split across tasks;
// each row copies `width` floats from the gathered row.
// Symbol: embedding

__global__ void embedding(const float* table, const unsigned long long* indices,
                          float* out, unsigned long long rows,
                          unsigned long long width) {
    const unsigned cores = taskDim();
    unsigned long long start, end;
    GRID_RANGE(taskId(), cores, rows, start, end);
    __nram__ float rowLocal[NRAMBUF];
    for (unsigned long long row = start; row < end; ++row) {
        const float* src = table + indices[row] * width;
        float* dst = out + row * width;
        unsigned long long off = 0;
        // Width may exceed the NRAM tile, so each row is copied in blocks
        // (the naive single-copy form would overflow the tile).
        while (off < width) {
            const unsigned long long len = (width - off < NRAMBUF) ? (width - off) : NRAMBUF;
            __memcpy(rowLocal, src + off, (size_t)len * sizeof(float), GDRAM2NRAM);
            __memcpy(dst + off, rowLocal, (size_t)len * sizeof(float), NRAM2GDRAM);
            off += NRAMBUF;
        }
    }
}

// ===========================================================================
// matmul
// ===========================================================================
// Reference f32 matrix-multiply: a is [batch, m, k], b is [batch, k, n],
// c is [batch, m, n], all row-major. Output elements are split across tasks;
// each element is a scalar dot product over k, so the result is bit-for-bit
// the shared CUDA-family FMA reference. The MLU matrix-unit fast path is a
// later, measured optimization.
// Symbol: matmul

__global__ void matmul(const float* a, const float* b, float* c, unsigned m,
                       unsigned k, unsigned n, unsigned batch) {
    const unsigned long long total = (unsigned long long)m * n * batch;
    const unsigned cores = taskDim();
    unsigned long long start, end;
    GRID_RANGE(taskId(), cores, total, start, end);
    const unsigned mk = m * k;
    const unsigned kn = k * n;
    const unsigned mn = m * n;
    for (unsigned long long index = start; index < end; ++index) {
        const unsigned blk = (unsigned)(index / mn);
        const unsigned rem = (unsigned)(index % mn);
        const unsigned row = rem / n;
        const unsigned col = rem % n;
        const unsigned long long abase = (unsigned long long)blk * mk;
        const unsigned long long bbase = (unsigned long long)blk * kn;
        float acc = 0.0f;
        for (unsigned i = 0; i < k; ++i) {
            acc += a[abase + (unsigned long long)row * k + i] * b[bbase + (unsigned long long)i * n + col];
        }
        c[(unsigned long long)blk * mn + (unsigned long long)row * n + col] = acc;
    }
}

// ===========================================================================
// rmsnorm
// ===========================================================================
// f32 RMS-normalization: x is [rows, width], w is [width], y is [rows, width].
// Rows are split across tasks; each row reduces sum(x^2), computes the
// reciprocal RMS on the task, and applies it with the weight.
// Symbol: rmsnorm

__global__ void rmsnorm(const float* x, const float* w, float* y, unsigned rows,
                        unsigned width, float eps) {
    const unsigned cores = taskDim();
    unsigned start, end;
    GRID_RANGE(taskId(), cores, rows, start, end);
    __nram__ float xLocal[NRAMBUF];
    __nram__ float wLocal[NRAMBUF];
    __nram__ float yLocal[NRAMBUF];
    __nram__ float tmp[NRAMBUF];
    for (unsigned row = start; row < end; ++row) {
        // 1) 行分块累加平方和。
        float sumsq = 0.0f;
        unsigned off = 0;
        while (off < width) {
            GRID_TAIL(off, width, NRAMBUF, len);
            __memcpy(xLocal, x + (unsigned long long)row * width + off, len * sizeof(float), GDRAM2NRAM);
            __bang_mul(tmp, xLocal, xLocal, len);
            __bang_reduce_sum(tmp, tmp, len);
            sumsq += tmp[0];
            off += NRAMBUF;
        }
        // 2) 标量求倒数 RMS，整行归一化乘权重。
        const float inv = 1.0f / sqrtf(sumsq / (float)width + eps);
        off = 0;
        while (off < width) {
            GRID_TAIL(off, width, NRAMBUF, len);
            __memcpy(xLocal, x + (unsigned long long)row * width + off, len * sizeof(float), GDRAM2NRAM);
            __memcpy(wLocal, w + off, len * sizeof(float), GDRAM2NRAM);
            __bang_muls(yLocal, xLocal, inv, len);
            __bang_mul(yLocal, yLocal, wLocal, len);
            __memcpy(y + (unsigned long long)row * width + off, yLocal, len * sizeof(float), NRAM2GDRAM);
            off += NRAMBUF;
        }
    }
}

// ===========================================================================
// rope
// ===========================================================================
// f32 rotary-position-embedding: x and y are [rows, heads * headdim]. The
// position of a row is row % seqaxis; seqaxis == 0 places every row at 0.
// The angle of pair i is position * base^(-2i / headdim). Scalar per element
// (the transcendental pair math has no vector intrinsic).
// Symbol: rope

__global__ void rope(const float* x, float* y, unsigned rows, unsigned seqaxis,
                     unsigned heads, unsigned headdim, float base) {
    const unsigned half = headdim / 2u;
    const unsigned long long total = (unsigned long long)rows * heads * half;
    const unsigned cores = taskDim();
    unsigned long long start, end;
    GRID_RANGE(taskId(), cores, total, start, end);
    for (unsigned long long index = start; index < end; ++index) {
        const unsigned i = (unsigned)(index % half);
        const unsigned h = (unsigned)((index / half) % heads);
        const unsigned row = (unsigned)(index / ((unsigned long long)half * heads));
        const unsigned position = (seqaxis == 0u) ? 0u : (row % seqaxis);
        const float angle = (float)position * powf(base, -2.0f * (float)i / (float)headdim);
        const float s = sinf(angle);
        const float c = cosf(angle);
        const unsigned long long offset = (((unsigned long long)row * heads) + h) * headdim + 2u * i;
        const float even = x[offset];
        const float odd = x[offset + 1u];
        y[offset] = even * c - odd * s;
        y[offset + 1u] = even * s + odd * c;
    }
}

// ===========================================================================
// rope_at
// ===========================================================================
// Absolute-position rotary embedding of a single token; position is a scalar
// parameter, not row % seqaxis.
// Symbol: rope_at

__global__ void rope_at(const float* x, float* y, unsigned heads, unsigned headdim,
                        unsigned position, float base) {
    const unsigned half = headdim / 2u;
    const unsigned long long total = (unsigned long long)heads * half;
    const unsigned cores = taskDim();
    unsigned long long start, end;
    GRID_RANGE(taskId(), cores, total, start, end);
    for (unsigned long long index = start; index < end; ++index) {
        const unsigned i = (unsigned)(index % half);
        const unsigned h = (unsigned)(index / half);
        const float angle = (float)position * powf(base, -2.0f * (float)i / (float)headdim);
        const float s = sinf(angle);
        const float c = cosf(angle);
        const unsigned long long offset = ((unsigned long long)h) * headdim + 2u * i;
        const float even = x[offset];
        const float odd = x[offset + 1u];
        y[offset] = even * c - odd * s;
        y[offset + 1u] = even * s + odd * c;
    }
}

// ===========================================================================
// scan
// ===========================================================================
// f32 semigroup scan: one task owns a subset of channels; each channel runs
// the serial recurrence over the sequence axis for its state vector, so the
// result is bit-for-bit the serial reference. The state buffer is
// initialized to init by the host and updated in place. Pointer order:
// input, a, b, c, skip, state (in/out), out.
// Symbol: scan

__global__ void scan(const float* input, const float* a, const float* b,
                     const float* c, const float* skip, float* state, float* out,
                     unsigned n, unsigned d, unsigned N) {
    const unsigned cores = taskDim();
    unsigned start, end;
    GRID_RANGE(taskId(), cores, d, start, end);
    for (unsigned i = start; i < end; ++i) {
        for (unsigned t = 0; t < n; ++t) {
            const float x = input[(unsigned long long)t * d + i];
            float acc = 0.0f;
            for (unsigned j = 0; j < N; ++j) {
                const unsigned idx = i * N + j;
                const float h = a[idx] * state[idx] + b[idx] * x;
                state[idx] = h;
                acc += c[idx] * h;
            }
            out[(unsigned long long)t * d + i] = acc + skip[i] * x;
        }
    }
}

// ===========================================================================
// scan_parallel
// ===========================================================================
// f32 semigroup scan, parallel three-stage family, shared semantics with
// grid/cuda.c. Pass 1 aggregates each block's total, Pass 2 computes the
// inter-block prefix, Pass 3 re-applies each block from its prefix so the
// intra-block recurrence is bit-for-bit the serial one.
// Symbols: scan_agg, scan_prefix, scan_apply

// Pass 1: per-block total. Work items (block, channel) split across tasks.
__global__ void scan_agg(const float* input, const float* a, const float* b,
                         const float* c, const float* skip, const float* init,
                         float* block_total, unsigned n, unsigned d, unsigned N,
                         unsigned chunk, unsigned blocks) {
    const unsigned long long total = (unsigned long long)blocks * d;
    const unsigned cores = taskDim();
    unsigned long long start, end;
    GRID_RANGE(taskId(), cores, total, start, end);
    const unsigned dN = d * N;
    for (unsigned long long item = start; item < end; ++item) {
        const unsigned block = (unsigned)(item / d);
        const unsigned i = (unsigned)(item % d);
        const unsigned t0 = block * chunk;
        const unsigned t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
        const unsigned long long base = (unsigned long long)block * 2u * dN;
        for (unsigned j = 0; j < N; ++j) {
            const unsigned idx = i * N + j;
            float A = 1.0f;
            float B = 0.0f;
            const float aj = a[idx];
            const float bj = b[idx];
            for (unsigned t = t0; t < t1; ++t) {
                const float x = input[(unsigned long long)t * d + i];
                B = aj * B + bj * x;
                A = aj * A;
            }
            block_total[base + idx] = A;
            block_total[base + dN + idx] = B;
        }
    }
}

// Pass 2: inter-block prefix; each component (j in [0, d*N)) accumulates the
// prefix over the P block totals serially; prefix[p] = ⊕_{q<p} total[q] with
// prefix[0] = identity (1, 0). The loop-final accumulator is the
// whole-sequence total, giving the final state: state[j] = A_total*init[j]+B_total.
// fmaf folds the combine into one rounding (matches the shared CUDA family).
__global__ void scan_prefix(const float* block_total, const float* init,
                            float* state, float* block_prefix, unsigned n,
                            unsigned d, unsigned N, unsigned chunk, unsigned blocks) {
    const unsigned long long dN = (unsigned long long)d * N;
    const unsigned cores = taskDim();
    unsigned long long start, end;
    GRID_RANGE(taskId(), cores, dN, start, end);
    for (unsigned long long j64 = start; j64 < end; ++j64) {
        const unsigned j = (unsigned)j64;
        block_prefix[j] = 1.0f;
        block_prefix[dN + j] = 0.0f;
        float A = 1.0f;
        float B = 0.0f;
        for (unsigned q = 0; q < blocks; ++q) {
            const unsigned long long base = (unsigned long long)q * 2u * dN;
            const float aq = block_total[base + j];
            const float bq = block_total[base + dN + j];
            block_prefix[(q + 1u) * 2u * dN + j] = A * aq;
            block_prefix[(q + 1u) * 2u * dN + dN + j] = fmaf(aq, B, bq);
            B = fmaf(aq, B, bq);
            A = A * aq;
        }
        state[j] = A * init[j] + B;
    }
}

// Pass 3: per-block apply. Task i owns channel i; the block starts at the
// inter-block prefix state h = A0*init + B0, then runs the same intra-block
// recurrence as the serial reference, writing out[t][i] = skip[i]*x + Σ_j c[i,j]*h[j].
__global__ void scan_apply(const float* input, const float* a, const float* b,
                           const float* c, const float* skip, const float* init,
                           const float* block_prefix, float* out, unsigned n,
                           unsigned d, unsigned N, unsigned chunk, unsigned blocks) {
    const unsigned long long total = (unsigned long long)blocks * d;
    const unsigned cores = taskDim();
    unsigned long long start, end;
    GRID_RANGE(taskId(), cores, total, start, end);
    const unsigned dN = d * N;
    for (unsigned long long item = start; item < end; ++item) {
        const unsigned block = (unsigned)(item / d);
        const unsigned i = (unsigned)(item % d);
        const unsigned t0 = block * chunk;
        const unsigned t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
        const unsigned long long base = (unsigned long long)block * 2u * dN;
        float h[GRID_SCANMAXN];
        for (unsigned j = 0; j < N; ++j) {
            const unsigned idx = i * N + j;
            h[j] = block_prefix[base + idx] * init[idx] + block_prefix[base + dN + idx];
        }
        for (unsigned t = t0; t < t1; ++t) {
            const float x = input[(unsigned long long)t * d + i];
            float acc = 0.0f;
            for (unsigned j = 0; j < N; ++j) {
                const unsigned idx = i * N + j;
                h[j] = a[idx] * h[j] + b[idx] * x;
                acc += c[idx] * h[j];
            }
            out[(unsigned long long)t * d + i] = acc + skip[i] * x;
        }
    }
}

// ===========================================================================
// softmax
// ===========================================================================
// f32 softmax-over-last-axis: one task owns a subset of rows; each row
// subtracts the row max before exponentiating (the numerically stable form)
// and divides only when the sum is positive.
// Symbol: softmax

__global__ void softmax(const float* x, float* y, unsigned rows, unsigned width) {
    const unsigned cores = taskDim();
    unsigned start, end;
    GRID_RANGE(taskId(), cores, rows, start, end);
    __nram__ float xLocal[NRAMBUF];
    __nram__ float yLocal[NRAMBUF];
    __nram__ float tmp[NRAMBUF];
    for (unsigned row = start; row < end; ++row) {
        // 1) 行最大：分块 ReduceMax。
        float rowmax = GRID_NEGLARGE;
        unsigned off = 0;
        while (off < width) {
            GRID_TAIL(off, width, NRAMBUF, len);
            __memcpy(xLocal, x + (unsigned long long)row * width + off, len * sizeof(float), GDRAM2NRAM);
            __bang_reduce_max(tmp, xLocal, len);
            if (tmp[0] > rowmax) {
                rowmax = tmp[0];
            }
            off += NRAMBUF;
        }
        // 2) 移位指数和。
        float total = 0.0f;
        off = 0;
        while (off < width) {
            GRID_TAIL(off, width, NRAMBUF, len);
            __memcpy(xLocal, x + (unsigned long long)row * width + off, len * sizeof(float), GDRAM2NRAM);
            __bang_adds(tmp, xLocal, -rowmax, len);
            __bang_exp(tmp, tmp, len);
            __bang_reduce_sum(tmp, tmp, len);
            total += tmp[0];
            off += NRAMBUF;
        }
        // 3) 归一化写出（total <= 0 时写 0，与共享内核一致）。
        const float inv = (total > 0.0f) ? (1.0f / total) : 0.0f;
        off = 0;
        while (off < width) {
            GRID_TAIL(off, width, NRAMBUF, len);
            __memcpy(xLocal, x + (unsigned long long)row * width + off, len * sizeof(float), GDRAM2NRAM);
            __bang_adds(tmp, xLocal, -rowmax, len);
            __bang_exp(tmp, tmp, len);
            __bang_muls(yLocal, tmp, inv, len);
            __memcpy(y + (unsigned long long)row * width + off, yLocal, len * sizeof(float), NRAM2GDRAM);
            off += NRAMBUF;
        }
    }
}

// ===========================================================================
// attention
// ===========================================================================
// Fused scaled-dot-product attention: one task owns a subset of the
// (query position, head) tiles. Scores are recomputed, not materialized:
// one pass for the row max, one for the shifted-exponential sum, then the
// weighted value sum. Scalar per tile for correctness first.
// Symbol: attention

__global__ void attention(const float* q, const float* k, const float* v,
                          float* out, unsigned qseq, unsigned kvseq,
                          unsigned heads, unsigned kvheads, unsigned headdim,
                          unsigned causal) {
    const unsigned long long total = (unsigned long long)qseq * heads;
    const unsigned cores = taskDim();
    unsigned long long start, end;
    GRID_RANGE(taskId(), cores, total, start, end);
    const float scale = 1.0f / sqrtf((float)headdim);
    for (unsigned long long tile = start; tile < end; ++tile) {
        const unsigned head = (unsigned)(tile % heads);
        const unsigned position = (unsigned)(tile / heads);
        const unsigned kvhead = head / (heads / kvheads);
        const unsigned offset = kvseq - qseq;
        const unsigned limit = (causal != 0u && offset + position + 1u < kvseq)
            ? (offset + position + 1u)
            : kvseq;
        const float* qrow = q + ((unsigned long long)position * heads + head) * headdim;

        // Online softmax: one pass computes the running max and the running
        // sum, so each QK score is evaluated twice instead of three times.
        float runningmax = GRID_NEGLARGE;
        float localsum = 0.0f;
        for (unsigned key = 0; key < limit; ++key) {
            const float* krow = k + ((unsigned long long)key * kvheads + kvhead) * headdim;
            float dot = 0.0f;
            for (unsigned d = 0; d < headdim; ++d) {
                dot += qrow[d] * krow[d];
            }
            const float score = dot * scale;
            const float nm = (score > runningmax) ? score : runningmax;
            localsum = localsum * expf(runningmax - nm) + expf(score - nm);
            runningmax = nm;
        }
        float* orow = out + ((unsigned long long)position * heads + head) * headdim;
        if (localsum > 0.0f) {
            const float inv = 1.0f / localsum;
            // PV reordered: each QK score is evaluated once per key, then
            // accumulated into every d dimension, so this pass is
            // O(limit x headdim) instead of O(limit x headdim^2). acc is
            // sized by GRID_SCANMAXN (headdim is capped like the scan width).
            float acc[GRID_SCANMAXN];
            for (unsigned d = 0; d < headdim; ++d) {
                acc[d] = 0.0f;
            }
            for (unsigned key = 0; key < limit; ++key) {
                const float* krow = k + ((unsigned long long)key * kvheads + kvhead) * headdim;
                float dot = 0.0f;
                for (unsigned d = 0; d < headdim; ++d) {
                    dot += qrow[d] * krow[d];
                }
                const float weight = expf(dot * scale - runningmax) * inv;
                const float* vrow = v + ((unsigned long long)key * kvheads + kvhead) * headdim;
                for (unsigned d = 0; d < headdim; ++d) {
                    acc[d] += weight * vrow[d];
                }
            }
            for (unsigned d = 0; d < headdim; ++d) {
                orow[d] = acc[d];
            }
        } else {
            for (unsigned d = 0; d < headdim; ++d) {
                orow[d] = 0.0f;
            }
        }
    }
}

// ===========================================================================
// attention_cached
// ===========================================================================
// Incremental-decode attention: one query token attends over a cached
// key/value buffer followed by one new row, without materialising the
// concatenation. One task owns a subset of heads.
// Symbol: attention_cached

__global__ void attention_cached(const float* q, const float* ck, const float* cv,
                                 const float* nk, const float* nv, float* out,
                                 unsigned kvseq, unsigned heads, unsigned kvheads,
                                 unsigned headdim, unsigned causal) {
    const unsigned cores = taskDim();
    unsigned start, end;
    GRID_RANGE(taskId(), cores, heads, start, end);
    const float scale = 1.0f / sqrtf((float)headdim);
    const unsigned pos = kvseq - 1u;
    const unsigned limit = kvseq;
    for (unsigned head = start; head < end; ++head) {
        const unsigned kvhead = head / (heads / kvheads);
        const float* qrow = q + (unsigned long long)head * headdim;
        // Online softmax, same as `attention` above: one pass for the
        // running max and running sum (two score evaluations, not three).
        float runningmax = GRID_NEGLARGE;
        float localsum = 0.0f;
        for (unsigned key = 0; key < limit; ++key) {
            const float* krow = (key < pos)
                ? (ck + ((unsigned long long)key * kvheads + kvhead) * headdim)
                : (nk + (unsigned long long)kvhead * headdim);
            float dot = 0.0f;
            for (unsigned d = 0; d < headdim; ++d) {
                dot += qrow[d] * krow[d];
            }
            const float score = dot * scale;
            const float nm = (score > runningmax) ? score : runningmax;
            localsum = localsum * expf(runningmax - nm) + expf(score - nm);
            runningmax = nm;
        }
        float* orow = out + (unsigned long long)head * headdim;
        if (localsum > 0.0f) {
            const float inv = 1.0f / localsum;
            // PV reordered: O(limit x headdim) instead of
            // O(limit x headdim^2); acc sized by GRID_SCANMAXN.
            float acc[GRID_SCANMAXN];
            for (unsigned d = 0; d < headdim; ++d) {
                acc[d] = 0.0f;
            }
            for (unsigned key = 0; key < limit; ++key) {
                const float* krow = (key < pos)
                    ? (ck + ((unsigned long long)key * kvheads + kvhead) * headdim)
                    : (nk + (unsigned long long)kvhead * headdim);
                float dot = 0.0f;
                for (unsigned d = 0; d < headdim; ++d) {
                    dot += qrow[d] * krow[d];
                }
                const float weight = expf(dot * scale - runningmax) * inv;
                const float* vrow = (key < pos)
                    ? (cv + ((unsigned long long)key * kvheads + kvhead) * headdim)
                    : (nv + (unsigned long long)kvhead * headdim);
                for (unsigned d = 0; d < headdim; ++d) {
                    acc[d] += weight * vrow[d];
                }
            }
            for (unsigned d = 0; d < headdim; ++d) {
                orow[d] = acc[d];
            }
        } else {
            for (unsigned d = 0; d < headdim; ++d) {
                orow[d] = 0.0f;
            }
        }
    }
}

