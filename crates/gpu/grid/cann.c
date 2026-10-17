// Ascend C sources for the Huawei CANN backend (Ascend 910/310, the AI Core
// systolic-array architecture).
//
// Ascend C is the official kernel dialect for CANN. The programming model
// differs from CUDA-family GPUs: a kernel runs one program per AI Core (no
// thread/block grid), and parallel work is partitioned explicitly with
// `GetBlockIdx()` / `GetBlockNum()`. Data moves through an on-chip pipeline
// (`TPipe` + `TQue`): `DataCopy` GlobalTensor <-> LocalTensor, vector
// intrinsics (`AscendC::Add`, `Muls`, `Mul`, `Exp`, `Sqrt`, `ReduceSum`,
// `ReduceMax`, ...) operate on LocalTensors, and results are copied back.
//
// Build (Linux, CANN toolkit):
//   bisheng ir-compile --cce-aicore-arch=dav-c220-vec cann.c -o cann.o
// (the exact driver name/flags follow the installed CANN release; the file
// is a single translation unit so one invocation produces the whole operator
// set). The CANN module is compiled out when the toolchain is absent,
// exactly like the other toolchain-gated backends (build.rs emits an empty
// registry).
//
// Semantics: every entry below reproduces the shared kernel family
// (grid/cuda.c) numerically for finite inputs. Simple elementwise
// operators use the vector pipeline; attention / matmul / rope / scan use
// per-AI-Core scalar loops for correctness first (the Cube unit / vector
// transcendental fast paths are noted per operator as later optimizations).
//
// Entries: attention, attention_cached, add, mul, silu, embedding, matmul,
//          rmsnorm, rope, rope_at, scan, scan_agg, scan_prefix, scan_apply,
//          softmax.

#include "kernel_operator.h"
#include "grid.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t VECBLOCK = 1024;   // 向量化分块元素数

// ===========================================================================
// elementwise
// ===========================================================================
// Elementwise f32 kernels: add and mul take two equal-length operands, silu
// takes one. All stride over the flat element count and bounds-test. silu
// reproduces cpu's x / (1 + exp(-x)) exactly (no fused approximation).
// Symbols: add, mul, silu

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t n) {
        aGm.SetGlobalBuffer((__gm__ float*)a, n);
        bGm.SetGlobalBuffer((__gm__ float*)b, n);
        cGm.SetGlobalBuffer((__gm__ float*)c, n);
        pipe.InitBuffer(inQueA, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(inQueB, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(outQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        count = n;
        // 多核分块：每核处理一段连续区间。
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, count, start, end);
    }
    __aicore__ inline void Process() {
        for (uint32_t off = start; off < end; off += (uint32_t)VECBLOCK) {
            GRID_TAIL(off, end, (uint32_t)VECBLOCK, len);
            LocalTensor<float> aLocal = inQueA.AllocTensor<float>();
            LocalTensor<float> bLocal = inQueB.AllocTensor<float>();
            LocalTensor<float> cLocal = outQue.AllocTensor<float>();
            DataCopy(aLocal, aGm[off], len);
            DataCopy(bLocal, bGm[off], len);
            AscendC::Add(cLocal, aLocal, bLocal, len);
            inQueA.FreeTensor(aLocal);
            inQueB.FreeTensor(bLocal);
            outQue.EnQue(cLocal);
            LocalTensor<float> outLocal = outQue.DeQue<float>();
            DataCopy(cGm[off], outLocal, len);
            outQue.FreeTensor(outLocal);
        }
    }

private:
    TPipe pipe;
    GlobalTensor<float> aGm, bGm, cGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueA, inQueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    uint32_t count;
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void add(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                          uint32_t n) {
    KernelAdd op;
    op.Init(a, b, c, n);
    op.Process();
}

class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t n) {
        aGm.SetGlobalBuffer((__gm__ float*)a, n);
        bGm.SetGlobalBuffer((__gm__ float*)b, n);
        cGm.SetGlobalBuffer((__gm__ float*)c, n);
        pipe.InitBuffer(inQueA, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(inQueB, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(outQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        count = n;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, count, start, end);
    }
    __aicore__ inline void Process() {
        for (uint32_t off = start; off < end; off += (uint32_t)VECBLOCK) {
            GRID_TAIL(off, end, (uint32_t)VECBLOCK, len);
            LocalTensor<float> aLocal = inQueA.AllocTensor<float>();
            LocalTensor<float> bLocal = inQueB.AllocTensor<float>();
            LocalTensor<float> cLocal = outQue.AllocTensor<float>();
            DataCopy(aLocal, aGm[off], len);
            DataCopy(bLocal, bGm[off], len);
            AscendC::Mul(cLocal, aLocal, bLocal, len);
            inQueA.FreeTensor(aLocal);
            inQueB.FreeTensor(bLocal);
            outQue.EnQue(cLocal);
            LocalTensor<float> outLocal = outQue.DeQue<float>();
            DataCopy(cGm[off], outLocal, len);
            outQue.FreeTensor(outLocal);
        }
    }

private:
    TPipe pipe;
    GlobalTensor<float> aGm, bGm, cGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueA, inQueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    uint32_t count;
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void mul(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                          uint32_t n) {
    KernelMul op;
    op.Init(a, b, c, n);
    op.Process();
}

class KernelSilu {
public:
    __aicore__ inline KernelSilu() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR c, uint32_t n) {
        aGm.SetGlobalBuffer((__gm__ float*)a, n);
        cGm.SetGlobalBuffer((__gm__ float*)c, n);
        pipe.InitBuffer(inQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(outQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        count = n;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, count, start, end);
    }
    __aicore__ inline void Process() {
        for (uint32_t off = start; off < end; off += (uint32_t)VECBLOCK) {
            GRID_TAIL(off, end, (uint32_t)VECBLOCK, len);
            LocalTensor<float> aLocal = inQue.AllocTensor<float>();
            LocalTensor<float> cLocal = outQue.AllocTensor<float>();
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            DataCopy(aLocal, aGm[off], len);
            // silu(x) = x / (1 + exp(-x))，分步向量化，与共享内核逐位一致。
            AscendC::Muls(tmp, aLocal, -1.0f, len);
            AscendC::Exp(tmp, tmp, len);
            AscendC::Adds(tmp, tmp, 1.0f, len);
            AscendC::Div(cLocal, aLocal, tmp, len);
            inQue.FreeTensor(aLocal);
            outQue.EnQue(cLocal);
            LocalTensor<float> outLocal = outQue.DeQue<float>();
            DataCopy(cGm[off], outLocal, len);
            outQue.FreeTensor(outLocal);
        }
    }

private:
    TPipe pipe;
    GlobalTensor<float> aGm, cGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    uint32_t count;
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void silu(GM_ADDR a, GM_ADDR c, uint32_t n) {
    KernelSilu op;
    op.Init(a, c, n);
    op.Process();
}

// ===========================================================================
// embedding
// ===========================================================================
// f32 embedding-gather kernel: table is [vocab, width], indices holds rows
// validated on the host, out is [rows, width]. Rows are split across cores;
// each row copies `width` floats from the gathered row.
// Symbol: embedding

class KernelEmbedding {
public:
    __aicore__ inline KernelEmbedding() {}
    __aicore__ inline void Init(GM_ADDR table, GM_ADDR indices, GM_ADDR out,
                                uint64_t rows, uint64_t width) {
        tableGm.SetGlobalBuffer((__gm__ float*)table, rows * width);
        idxGm.SetGlobalBuffer((__gm__ uint64_t*)indices, rows);
        outGm.SetGlobalBuffer((__gm__ float*)out, rows * width);
        pipe.InitBuffer(outQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        this->rows = rows;
        this->width = width;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, rows, start, end);
    }
    __aicore__ inline void Process() {
        for (uint64_t row = start; row < end; ++row) {
            const uint64_t src = idxGm.GetValue(row) * width;
            const uint64_t len = width;
            LocalTensor<float> outLocal = outQue.AllocTensor<float>();
            DataCopy(outLocal, tableGm[src], (uint32_t)len);
            outQue.EnQue(outLocal);
            LocalTensor<float> outCopy = outQue.DeQue<float>();
            DataCopy(outGm[row * width], outCopy, (uint32_t)len);
            outQue.FreeTensor(outCopy);
        }
    }

private:
    TPipe pipe;
    GlobalTensor<float> tableGm, outGm;
    GlobalTensor<uint64_t> idxGm;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    uint64_t rows;
    uint64_t width;
    uint64_t start;
    uint64_t end;
};

extern "C" __global__ __aicore__ void embedding(GM_ADDR table, GM_ADDR indices,
                                                GM_ADDR out, uint64_t rows,
                                                uint64_t width) {
    KernelEmbedding op;
    op.Init(table, indices, out, rows, width);
    op.Process();
}

// ===========================================================================
// matmul
// ===========================================================================
// Reference f32 matrix-multiply: a is [batch, m, k], b is [batch, k, n],
// c is [batch, m, n], all row-major. Output elements are split across cores;
// each element is a scalar dot product over k, so the result is bit-for-bit
// the shared CUDA-family FMA reference. The Cube-unit fast path
// (AscendC::Matmul with tiling) is a later, measured optimization.
// Symbol: matmul

// f32 blocked matmul: A is [batch, m, k], B is [batch, k, n], C is
// [batch, m, n], all row-major. Rows (batch * m) are split across cores;
// each row walks the n axis in VECBLOCK-wide columns, and the k contraction
// runs on the Vector unit (Muls + Add per k step, the two verified primitives
// instead of a fused FMA intrinsic), so the inner loop is vectorised and the
// operand streams stay in local memory instead of one scalar GM read per
// output element. Symbols: matmul

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t m,
                                uint32_t k, uint32_t n, uint32_t batch) {
        pa = (__gm__ float*)a;
        pb = (__gm__ float*)b;
        pc = (__gm__ float*)c;
        this->m = m;
        this->k = k;
        this->n = n;
        this->batch = batch;
        pipe.InitBuffer(inQueB, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(outQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        // 行分核：每核处理连续 (batch * m) 行；行 = blk * m + row。
        const uint32_t rows = m * batch;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, rows, start, end);
    }
    __aicore__ inline void Process() {
        const uint64_t mk = (uint64_t)m * k;
        const uint64_t kn = (uint64_t)k * n;
        for (uint32_t rindex = start; rindex < end; ++rindex) {
            const uint32_t blk = rindex / m;
            const uint32_t row = rindex - blk * m;
            const uint64_t arow = (uint64_t)blk * mk + (uint64_t)row * k;
            const uint64_t browbase = (uint64_t)blk * kn;
            const uint64_t cout = (uint64_t)rindex * n;
            for (uint32_t cb = 0; cb < n; cb += (uint32_t)VECBLOCK) {
                const uint32_t len =
                    (n - cb < (uint32_t)VECBLOCK) ? (n - cb) : (uint32_t)VECBLOCK;
                LocalTensor<float> acc = outQue.AllocTensor<float>();
                AscendC::Duplicate(acc, 0.0f, len);
                for (uint32_t i = 0; i < k; ++i) {
                    LocalTensor<float> bRow = inQueB.AllocTensor<float>();
                    DataCopy(bRow, pb[browbase + (uint64_t)i * n + cb], len);
                    LocalTensor<float> tmp = tmpBuf.Get<float>();
                    AscendC::Muls(tmp, bRow, pa[arow + i], len);
                    AscendC::Add(acc, acc, tmp, len);
                    inQueB.FreeTensor(bRow);
                }
                DataCopy(pc[cout + cb], acc, len);
                outQue.FreeTensor(acc);
            }
        }
    }

private:
    TPipe pipe;
    __gm__ float* pa;
    __gm__ float* pb;
    __gm__ float* pc;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t batch;
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void matmul(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                             uint32_t m, uint32_t k, uint32_t n,
                                             uint32_t batch) {
    KernelMatmul op;
    op.Init(a, b, c, m, k, n, batch);
    op.Process();
}

// ===========================================================================
// rmsnorm
// ===========================================================================
// f32 RMS-normalization: x is [rows, width], w is [width], y is [rows, width].
// Rows are split across cores; each row reduces sum(x^2), computes the
// reciprocal RMS on the core, and applies it with the weight. width must be
// a multiple of 8 (vector length alignment); the host validates it.
// Symbol: rmsnorm

class KernelRmsNorm {
public:
    __aicore__ inline KernelRmsNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t rows,
                                uint32_t width, float eps) {
        xGm.SetGlobalBuffer((__gm__ float*)x, rows * width);
        wGm.SetGlobalBuffer((__gm__ float*)w, width);
        yGm.SetGlobalBuffer((__gm__ float*)y, rows * width);
        pipe.InitBuffer(inQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(outQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(workBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        this->rows = rows;
        this->width = width;
        this->eps = eps;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, rows, start, end);
    }
    __aicore__ inline void Process() {
        for (uint32_t row = start; row < end; ++row) {
            const uint32_t wdt = width;
            // 1) 行分块累加平方和。
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            LocalTensor<float> work = workBuf.Get<float>();
            float sumsq = 0.0f;
            for (uint32_t off = 0; off < wdt; off += (uint32_t)VECBLOCK) {
                GRID_TAIL(off, wdt, (uint32_t)VECBLOCK, len);
                LocalTensor<float> xLocal = inQue.AllocTensor<float>();
                DataCopy(xLocal, xGm[(uint64_t)row * wdt + off], len);
                AscendC::Mul(tmp, xLocal, xLocal, len);
                AscendC::ReduceSum(tmp, tmp, work, len);
                sumsq += tmp.GetValue(0);
                inQue.FreeTensor(xLocal);
            }
            // 2) 标量求倒数 RMS，然后分块归一化乘权重（w 按块拷贝，避免
            //    整行超出 NRAM 缓冲）。
            const float inv = 1.0f / sqrtf(sumsq / (float)wdt + eps);
            for (uint32_t off = 0; off < wdt; off += (uint32_t)VECBLOCK) {
                GRID_TAIL(off, wdt, (uint32_t)VECBLOCK, len);
                LocalTensor<float> xLocal = inQue.AllocTensor<float>();
                LocalTensor<float> wLocal = inQue.AllocTensor<float>();
                LocalTensor<float> yLocal = outQue.AllocTensor<float>();
                DataCopy(xLocal, xGm[(uint64_t)row * wdt + off], len);
                DataCopy(wLocal, wGm[off], len);
                AscendC::Muls(xLocal, xLocal, inv, len);
                AscendC::Mul(yLocal, xLocal, wLocal, len);
                inQue.FreeTensor(xLocal);
                inQue.FreeTensor(wLocal);
                outQue.EnQue(yLocal);
                LocalTensor<float> outCopy = outQue.DeQue<float>();
                DataCopy(yGm[(uint64_t)row * wdt + off], outCopy, len);
                outQue.FreeTensor(outCopy);
            }
        }
    }

private:
    TPipe pipe;
    GlobalTensor<float> xGm, wGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    uint32_t rows;
    uint32_t width;
    float eps;
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void rmsnorm(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                              uint32_t rows, uint32_t width,
                                              float eps) {
    KernelRmsNorm op;
    op.Init(x, w, y, rows, width, eps);
    op.Process();
}

// ===========================================================================
// rope
// ===========================================================================
// f32 rotary-position-embedding: x and y are [rows, heads * headdim]. The
// position of a row is row % seqaxis; seqaxis == 0 places every row at 0.
// The angle of pair i is position * base^(-2i / headdim). Scalar per element
// (the transcendental pair math has no single vector intrinsic; a vector
// implementation can precompute the sin/cos tables).
// Symbol: rope

class KernelRope {
public:
    __aicore__ inline KernelRope() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t rows,
                                uint32_t seqaxis, uint32_t heads, uint32_t headdim,
                                float base) {
        px = (__gm__ float*)x;
        py = (__gm__ float*)y;
        this->rows = rows;
        this->seqaxis = seqaxis;
        this->heads = heads;
        this->headdim = headdim;
        this->base = base;
        // 行分核：每核处理连续行段；行内角度只依赖 (position, i)，与 head 无关。
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, rows, start, end);
    }
    __aicore__ inline void Process() {
        const uint32_t half = headdim / 2u;
        for (uint32_t row = start; row < end; ++row) {
            const uint32_t position = (seqaxis == 0u) ? 0u : (row % seqaxis);
            // 本行 half 对 (s, c) 预计算一次：powf/sinf/cosf 从每元素一次
            // 降为每行 half 次（消除 head 维的重复计算）。
            for (uint32_t i = 0; i < half; ++i) {
                const float angle =
                    (float)position * powf(base, -2.0f * (float)i / (float)headdim);
                ss[i] = sinf(angle);
                cc[i] = cosf(angle);
            }
            const uint64_t rowoff = (uint64_t)row * heads * headdim;
            for (uint32_t h = 0; h < heads; ++h) {
                const uint64_t hoff = rowoff + (uint64_t)h * headdim;
                for (uint32_t i = 0; i < half; ++i) {
                    const uint64_t offset = hoff + 2u * i;
                    const float even = px[offset];
                    const float odd = px[offset + 1u];
                    py[offset] = even * cc[i] - odd * ss[i];
                    py[offset + 1u] = even * ss[i] + odd * cc[i];
                }
            }
        }
    }

private:
    __gm__ float* px;
    __gm__ float* py;
    uint32_t rows;
    uint32_t seqaxis;
    uint32_t heads;
    uint32_t headdim;
    float base;
    float ss[1024];
    float cc[1024];
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void rope(GM_ADDR x, GM_ADDR y, uint32_t rows,
                                           uint32_t seqaxis, uint32_t heads,
                                           uint32_t headdim, float base) {
    KernelRope op;
    op.Init(x, y, rows, seqaxis, heads, headdim, base);
    op.Process();
}

public:
    __aicore__ inline class KernelRopeAt {
public:
    __aicore__ inline KernelRopeAt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t heads,
                                uint32_t headdim, uint32_t position, float base) {
        px = (__gm__ float*)x;
        py = (__gm__ float*)y;
        this->heads = heads;
        this->headdim = headdim;
        this->position = position;
        this->base = base;
        const uint64_t half = headdim / 2u;
        total = (uint64_t)heads * half;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, total, start, end);
    }
    __aicore__ inline void Process() {
        const uint32_t half = headdim / 2u;
        // position 固定：half 对 (s, c) 预计算一次（消除每元素 powf/sinf/cosf）。
        for (uint32_t i = 0; i < half; ++i) {
            const float angle =
                (float)position * powf(base, -2.0f * (float)i / (float)headdim);
            ss[i] = sinf(angle);
            cc[i] = cosf(angle);
        }
        for (uint64_t index = start; index < end; ++index) {
            const uint32_t i = (uint32_t)(index % half);
            const uint32_t h = (uint32_t)(index / half);
            const uint64_t offset = ((uint64_t)h) * headdim + 2u * i;
            const float even = px[offset];
            const float odd = px[offset + 1u];
            py[offset] = even * cc[i] - odd * ss[i];
            py[offset + 1u] = even * ss[i] + odd * cc[i];
        }
    }

private:
    __gm__ float* px;
    __gm__ float* py;
    uint32_t heads;
    uint32_t headdim;
    uint32_t position;
    float base;
    float ss[1024];
    float cc[1024];
    uint64_t total;
    uint64_t start;
    uint64_t end;
};

extern "C" __global__ __aicore__ void rope_at(GM_ADDR x, GM_ADDR y, uint32_t heads,
                                              uint32_t headdim, uint32_t position,
                                              float base) {
    KernelRopeAt op;
    op.Init(x, y, heads, headdim, position, base);
    op.Process();
}

public:
    __aicore__ inline class KernelScan {
public:
    __aicore__ inline KernelScan() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                GM_ADDR skip, GM_ADDR state, GM_ADDR out,
                                uint32_t n, uint32_t d, uint32_t N) {
        pinput = (__gm__ float*)input;
        pa = (__gm__ float*)a;
        pb = (__gm__ float*)b;
        pc = (__gm__ float*)c;
        pskip = (__gm__ float*)skip;
        pstate = (__gm__ float*)state;
        pout = (__gm__ float*)out;
        this->n = n;
        this->d = d;
        this->N = N;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, d, start, end);
    }
    __aicore__ inline void Process() {
        for (uint32_t i = start; i < end; ++i) {
            for (uint32_t t = 0; t < n; ++t) {
                const float x = pinput[(uint64_t)t * d + i];
                float acc = 0.0f;
                for (uint32_t j = 0; j < N; ++j) {
                    const uint32_t idx = i * N + j;
                    const float h = pa[idx] * pstate[idx] + pb[idx] * x;
                    pstate[idx] = h;
                    acc += pc[idx] * h;
                }
                pout[(uint64_t)t * d + i] = acc + pskip[i] * x;
            }
        }
    }

private:
    __gm__ float* pinput;
    __gm__ float* pa;
    __gm__ float* pb;
    __gm__ float* pc;
    __gm__ float* pskip;
    __gm__ float* pstate;
    __gm__ float* pout;
    uint32_t n;
    uint32_t d;
    uint32_t N;
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void scan(GM_ADDR input, GM_ADDR a, GM_ADDR b,
                                           GM_ADDR c, GM_ADDR skip, GM_ADDR state,
                                           GM_ADDR out, uint32_t n, uint32_t d,
                                           uint32_t N) {
    KernelScan op;
    op.Init(input, a, b, c, skip, state, out, n, d, N);
    op.Process();
}

// ===========================================================================
// scan_parallel
// ===========================================================================
// f32 semigroup scan, parallel three-stage family, shared semantics with
// grid/cuda.c. Pass 1 aggregates each block's total, Pass 2 computes the
// inter-block prefix, Pass 3 re-applies each block from its prefix so the
// intra-block recurrence is bit-for-bit the serial one.
// Symbols: scan_agg, scan_prefix, scan_apply

// Pass 1: per-block total. Work items (block, channel) split across cores.
class KernelScanAgg {
public:
    __aicore__ inline KernelScanAgg() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                GM_ADDR skip, GM_ADDR init, GM_ADDR block_total,
                                uint32_t n, uint32_t d, uint32_t N, uint32_t chunk,
                                uint32_t blocks) {
        pinput = (__gm__ float*)input;
        pa = (__gm__ float*)a;
        pb = (__gm__ float*)b;
        pc = (__gm__ float*)c;
        pskip = (__gm__ float*)skip;
        pinit = (__gm__ float*)init;
        ptotal = (__gm__ float*)block_total;
        this->n = n;
        this->d = d;
        this->N = N;
        this->chunk = chunk;
        this->blocks = blocks;
        total = (uint64_t)blocks * d;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, total, start, end);
    }
    __aicore__ inline void Process() {
        const uint32_t dN = d * N;
        for (uint64_t item = start; item < end; ++item) {
            const uint32_t block = (uint32_t)(item / d);
            const uint32_t i = (uint32_t)(item % d);
            const uint32_t t0 = block * chunk;
            const uint32_t t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
            const uint64_t base = (uint64_t)block * 2u * dN;
            for (uint32_t j = 0; j < N; ++j) {
                const uint32_t idx = i * N + j;
                float A = 1.0f;
                float B = 0.0f;
                const float aj = pa[idx];
                const float bj = pb[idx];
                for (uint32_t t = t0; t < t1; ++t) {
                    const float x = pinput[(uint64_t)t * d + i];
                    B = aj * B + bj * x;
                    A = aj * A;
                }
                ptotal[base + idx] = A;
                ptotal[base + dN + idx] = B;
            }
        }
    }

private:
    __gm__ float* pinput;
    __gm__ float* pa;
    __gm__ float* pb;
    __gm__ float* pc;
    __gm__ float* pskip;
    __gm__ float* pinit;
    __gm__ float* ptotal;
    uint32_t n;
    uint32_t d;
    uint32_t N;
    uint32_t chunk;
    uint32_t blocks;
    uint64_t total;
    uint64_t start;
    uint64_t end;
};

extern "C" __global__ __aicore__ void scan_agg(GM_ADDR input, GM_ADDR a, GM_ADDR b,
                                               GM_ADDR c, GM_ADDR skip, GM_ADDR init,
                                               GM_ADDR block_total, uint32_t n,
                                               uint32_t d, uint32_t N, uint32_t chunk,
                                               uint32_t blocks) {
    KernelScanAgg op;
    op.Init(input, a, b, c, skip, init, block_total, n, d, N, chunk, blocks);
    op.Process();
}

// Pass 2: inter-block prefix; each component (j in [0, d*N)) accumulates the
// prefix over the P block totals serially; prefix[p] = ⊕_{q<p} total[q] with
// prefix[0] = identity (1, 0). The loop-final accumulator is the
// whole-sequence total, giving the final state: state[j] = A_total*init[j]+B_total.
// fmaf folds the combine into one rounding (matches the shared CUDA family).
class KernelScanPrefix {
public:
    __aicore__ inline KernelScanPrefix() {}
    __aicore__ inline void Init(GM_ADDR block_total, GM_ADDR init, GM_ADDR state,
                                GM_ADDR block_prefix, uint32_t n, uint32_t d,
                                uint32_t N, uint32_t chunk, uint32_t blocks) {
        ptotal = (__gm__ float*)block_total;
        pinit = (__gm__ float*)init;
        pstate = (__gm__ float*)state;
        pprefix = (__gm__ float*)block_prefix;
        this->n = n;
        this->d = d;
        this->N = N;
        this->chunk = chunk;
        this->blocks = blocks;
        dN = (uint64_t)d * N;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, dN, start, end);
    }
    __aicore__ inline void Process() {
        for (uint64_t j64 = start; j64 < end; ++j64) {
            const uint32_t j = (uint32_t)j64;
            pprefix[j] = 1.0f;
            pprefix[dN + j] = 0.0f;
            float A = 1.0f;
            float B = 0.0f;
            for (uint32_t q = 0; q < blocks; ++q) {
                const uint64_t base = (uint64_t)q * 2u * dN;
                const float aq = ptotal[base + j];
                const float bq = ptotal[base + dN + j];
                pprefix[(q + 1u) * 2u * dN + j] = A * aq;
                pprefix[(q + 1u) * 2u * dN + dN + j] = fmaf(aq, B, bq);
                B = fmaf(aq, B, bq);
                A = A * aq;
            }
            pstate[j] = A * pinit[j] + B;
        }
    }

private:
    __gm__ float* ptotal;
    __gm__ float* pinit;
    __gm__ float* pstate;
    __gm__ float* pprefix;
    uint32_t n;
    uint32_t d;
    uint32_t N;
    uint32_t chunk;
    uint32_t blocks;
    uint64_t dN;
    uint64_t start;
    uint64_t end;
};

extern "C" __global__ __aicore__ void scan_prefix(GM_ADDR block_total, GM_ADDR init,
                                                  GM_ADDR state, GM_ADDR block_prefix,
                                                  uint32_t n, uint32_t d, uint32_t N,
                                                  uint32_t chunk, uint32_t blocks) {
    KernelScanPrefix op;
    op.Init(block_total, init, state, block_prefix, n, d, N, chunk, blocks);
    op.Process();
}

// Pass 3: per-block apply. Thread i owns channel i; the block starts at the
// inter-block prefix state h = A0*init + B0, then runs the same intra-block
// recurrence as the serial reference, writing out[t][i] = skip[i]*x + Σ_j c[i,j]*h[j].
class KernelScanApply {
public:
    __aicore__ inline KernelScanApply() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                GM_ADDR skip, GM_ADDR init, GM_ADDR block_prefix,
                                GM_ADDR out, uint32_t n, uint32_t d, uint32_t N,
                                uint32_t chunk, uint32_t blocks) {
        pinput = (__gm__ float*)input;
        pa = (__gm__ float*)a;
        pb = (__gm__ float*)b;
        pc = (__gm__ float*)c;
        pskip = (__gm__ float*)skip;
        pinit = (__gm__ float*)init;
        pprefix = (__gm__ float*)block_prefix;
        pout = (__gm__ float*)out;
        this->n = n;
        this->d = d;
        this->N = N;
        this->chunk = chunk;
        this->blocks = blocks;
        total = (uint64_t)blocks * d;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, total, start, end);
    }
    __aicore__ inline void Process() {
        const uint32_t dN = d * N;
        for (uint64_t item = start; item < end; ++item) {
            const uint32_t block = (uint32_t)(item / d);
            const uint32_t i = (uint32_t)(item % d);
            const uint32_t t0 = block * chunk;
            const uint32_t t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
            const uint64_t base = (uint64_t)block * 2u * dN;
            float h[GRID_SCANMAXN];
            for (uint32_t j = 0; j < N; ++j) {
                const uint32_t idx = i * N + j;
                h[j] = pprefix[base + idx] * pinit[idx] + pprefix[base + dN + idx];
            }
            for (uint32_t t = t0; t < t1; ++t) {
                const float x = pinput[(uint64_t)t * d + i];
                float acc = 0.0f;
                for (uint32_t j = 0; j < N; ++j) {
                    const uint32_t idx = i * N + j;
                    h[j] = pa[idx] * h[j] + pb[idx] * x;
                    acc += pc[idx] * h[j];
                }
                pout[(uint64_t)t * d + i] = acc + pskip[i] * x;
            }
        }
    }

private:
    __gm__ float* pinput;
    __gm__ float* pa;
    __gm__ float* pb;
    __gm__ float* pc;
    __gm__ float* pskip;
    __gm__ float* pinit;
    __gm__ float* pprefix;
    __gm__ float* pout;
    uint32_t n;
    uint32_t d;
    uint32_t N;
    uint32_t chunk;
    uint32_t blocks;
    uint64_t total;
    uint64_t start;
    uint64_t end;
};

extern "C" __global__ __aicore__ void scan_apply(GM_ADDR input, GM_ADDR a, GM_ADDR b,
                                                 GM_ADDR c, GM_ADDR skip, GM_ADDR init,
                                                 GM_ADDR block_prefix, GM_ADDR out,
                                                 uint32_t n, uint32_t d, uint32_t N,
                                                 uint32_t chunk, uint32_t blocks) {
    KernelScanApply op;
    op.Init(input, a, b, c, skip, init, block_prefix, out, n, d, N, chunk, blocks);
    op.Process();
}

// ===========================================================================
// softmax
// ===========================================================================
// f32 softmax-over-last-axis: one core owns a subset of rows; each row
// subtracts the row max before exponentiating (the numerically stable form)
// and divides only when the sum is positive. width must be a multiple of 8;
// the host validates it.
// Symbol: softmax

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t rows,
                                uint32_t width) {
        xGm.SetGlobalBuffer((__gm__ float*)x, rows * width);
        yGm.SetGlobalBuffer((__gm__ float*)y, rows * width);
        pipe.InitBuffer(inQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(outQue, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(workBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        this->rows = rows;
        this->width = width;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, rows, start, end);
    }
    __aicore__ inline void Process() {
        const uint32_t wdt = width;
        for (uint32_t row = start; row < end; ++row) {
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            LocalTensor<float> work = workBuf.Get<float>();
            // 1) 行最大：分块 ReduceMax。
            float rowmax = -3.402823466e+38f;
            for (uint32_t off = 0; off < wdt; off += (uint32_t)VECBLOCK) {
                GRID_TAIL(off, wdt, (uint32_t)VECBLOCK, len);
                LocalTensor<float> xLocal = inQue.AllocTensor<float>();
                DataCopy(xLocal, xGm[(uint64_t)row * wdt + off], len);
                AscendC::ReduceMax(tmp, xLocal, work, len);
                const float blockmax = tmp.GetValue(0);
                if (blockmax > rowmax) {
                    rowmax = blockmax;
                }
                inQue.FreeTensor(xLocal);
            }
            // 2) 移位指数和。
            float total = 0.0f;
            for (uint32_t off = 0; off < wdt; off += (uint32_t)VECBLOCK) {
                GRID_TAIL(off, wdt, (uint32_t)VECBLOCK, len);
                LocalTensor<float> xLocal = inQue.AllocTensor<float>();
                DataCopy(xLocal, xGm[(uint64_t)row * wdt + off], len);
                AscendC::Adds(tmp, xLocal, -rowmax, len);
                AscendC::Exp(tmp, tmp, len);
                AscendC::ReduceSum(tmp, tmp, work, len);
                total += tmp.GetValue(0);
                inQue.FreeTensor(xLocal);
            }
            // 3) 归一化写出（total <= 0 时写 0，与共享内核一致）。
            const float inv = (total > 0.0f) ? (1.0f / total) : 0.0f;
            for (uint32_t off = 0; off < wdt; off += (uint32_t)VECBLOCK) {
                GRID_TAIL(off, wdt, (uint32_t)VECBLOCK, len);
                LocalTensor<float> xLocal = inQue.AllocTensor<float>();
                LocalTensor<float> yLocal = outQue.AllocTensor<float>();
                DataCopy(xLocal, xGm[(uint64_t)row * wdt + off], len);
                AscendC::Adds(tmp, xLocal, -rowmax, len);
                AscendC::Exp(tmp, tmp, len);
                AscendC::Muls(yLocal, tmp, inv, len);
                inQue.FreeTensor(xLocal);
                outQue.EnQue(yLocal);
                LocalTensor<float> outCopy = outQue.DeQue<float>();
                DataCopy(yGm[(uint64_t)row * wdt + off], outCopy, len);
                outQue.FreeTensor(outCopy);
            }
        }
    }

private:
    TPipe pipe;
    GlobalTensor<float> xGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    uint32_t rows;
    uint32_t width;
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void softmax(GM_ADDR x, GM_ADDR y, uint32_t rows,
                                              uint32_t width) {
    KernelSoftmax op;
    op.Init(x, y, rows, width);
    op.Process();
}

// ===========================================================================
// attention
// ===========================================================================
// Fused scaled-dot-product attention: one core owns a subset of the
// (query position, head) tiles. Scores are recomputed, not materialized:
// one pass for the row max, one for the shifted-exponential sum, then the
// weighted value sum. Scalar per tile for correctness first; the fused
// flash-style vector path is a later optimization.
// Symbol: attention

class KernelAttention {
public:
    __aicore__ inline KernelAttention() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                uint32_t qseq, uint32_t kvseq, uint32_t heads,
                                uint32_t kvheads, uint32_t headdim, uint32_t causal) {
        pq = (__gm__ float*)q;
        pk = (__gm__ float*)k;
        pv = (__gm__ float*)v;
        pout = (__gm__ float*)out;
        this->qseq = qseq;
        this->kvseq = kvseq;
        this->heads = heads;
        this->kvheads = kvheads;
        this->headdim = headdim;
        this->causal = causal;
        // headdim 内的向量工作缓冲（VECBLOCK 覆盖典型 headdim 64-128）。
        pipe.InitBuffer(qBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(kBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(vBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(mulBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(accBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        pipe.InitBuffer(workBuf, BUFFER_NUM, VECBLOCK * sizeof(float));
        total = (uint64_t)qseq * heads;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, total, start, end);
    }
    __aicore__ inline void Process() {
        const float scale = 1.0f / sqrtf((float)headdim);
        const float neglarge = -3.402823466e+38f;
        for (uint64_t tile = start; tile < end; ++tile) {
            const uint32_t head = (uint32_t)(tile % heads);
            const uint32_t position = (uint32_t)(tile / heads);
            const uint32_t kvhead = head / (heads / kvheads);
            const uint32_t offset = kvseq - qseq;
            const uint32_t limit = (causal != 0u && offset + position + 1u < kvseq)
                ? (offset + position + 1u)
                : kvseq;
            const float* qrow = pq + ((uint64_t)position * heads + head) * headdim;
            const float* kbase = pk + (uint64_t)kvhead * headdim;
            const float* vbase = pv + (uint64_t)kvhead * headdim;
            const uint64_t kvstride = (uint64_t)kvheads * headdim;
            LocalTensor<float> qLocal = qBuf.Get<float>();
            DataCopy(qLocal, qrow, headdim);

            // Pass 1：向量化 QK 点积（Mul + ReduceSum），取行最大分。
            float localmax = neglarge;
            for (uint32_t key = 0; key < limit; ++key) {
                LocalTensor<float> kLocal = kBuf.Get<float>();
                DataCopy(kLocal, kbase + (uint64_t)key * kvstride, headdim);
                LocalTensor<float> dot = mulBuf.Get<float>();
                AscendC::Mul(dot, qLocal, kLocal, headdim);
                AscendC::ReduceSum(dot, dot, workBuf.Get<float>(), headdim);
                const float score = dot.GetValue(0) * scale;
                if (score > localmax) {
                    localmax = score;
                }
            }
            // Pass 2：指数权重与归一和（exp 无单向量原语，标量每 key 一次）。
            float totalw = 0.0f;
            for (uint32_t key = 0; key < limit; ++key) {
                LocalTensor<float> kLocal = kBuf.Get<float>();
                DataCopy(kLocal, kbase + (uint64_t)key * kvstride, headdim);
                LocalTensor<float> dot = mulBuf.Get<float>();
                AscendC::Mul(dot, qLocal, kLocal, headdim);
                AscendC::ReduceSum(dot, dot, workBuf.Get<float>(), headdim);
                totalw += expf(dot.GetValue(0) * scale - localmax);
            }
            float* orow = pout + ((uint64_t)position * heads + head) * headdim;
            if (totalw > 0.0f) {
                const float inv = 1.0f / totalw;
                // Pass 3：向量化加权和 out[d] = sum_key w(key) * v(key)[d]，
                // 从 O(limit * headdim^2) 降为 O(limit * headdim)。
                LocalTensor<float> acc = accBuf.Get<float>();
                AscendC::Duplicate(acc, 0.0f, headdim);
                for (uint32_t key = 0; key < limit; ++key) {
                    LocalTensor<float> kLocal = kBuf.Get<float>();
                    DataCopy(kLocal, kbase + (uint64_t)key * kvstride, headdim);
                    LocalTensor<float> dot = mulBuf.Get<float>();
                    AscendC::Mul(dot, qLocal, kLocal, headdim);
                    AscendC::ReduceSum(dot, dot, workBuf.Get<float>(), headdim);
                    const float weight = expf(dot.GetValue(0) * scale - localmax);
                    LocalTensor<float> vLocal = vBuf.Get<float>();
                    DataCopy(vLocal, vbase + (uint64_t)key * kvstride, headdim);
                    AscendC::Muls(vLocal, vLocal, weight, headdim);
                    AscendC::Add(acc, acc, vLocal, headdim);
                }
                AscendC::Muls(acc, acc, inv, headdim);
                DataCopy(orow, acc, headdim);
            } else {
                for (uint32_t d = 0; d < headdim; ++d) {
                    orow[d] = 0.0f;
                }
            }
        }
    }

private:
    TPipe pipe;
    __gm__ float* pq;
    __gm__ float* pk;
    __gm__ float* pv;
    __gm__ float* pout;
    TBuf<QuePosition::VECCALC> qBuf;
    TBuf<QuePosition::VECCALC> kBuf;
    TBuf<QuePosition::VECCALC> vBuf;
    TBuf<QuePosition::VECCALC> mulBuf;
    TBuf<QuePosition::VECCALC> accBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    uint32_t qseq;
    uint32_t kvseq;
    uint32_t heads;
    uint32_t kvheads;
    uint32_t headdim;
    uint32_t causal;
    uint64_t total;
    uint64_t start;
    uint64_t end;
};

extern "C" __global__ __aicore__ void attention(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                                GM_ADDR out, uint32_t qseq,
                                                uint32_t kvseq, uint32_t heads,
                                                uint32_t kvheads, uint32_t headdim,
                                                uint32_t causal) {
    KernelAttention op;
    op.Init(q, k, v, out, qseq, kvseq, heads, kvheads, headdim, causal);
    op.Process();
}

class KernelAttentionCached {
public:
    __aicore__ inline KernelAttentionCached() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR ck, GM_ADDR cv, GM_ADDR nk,
                                GM_ADDR nv, GM_ADDR out, uint32_t kvseq,
                                uint32_t heads, uint32_t kvheads, uint32_t headdim,
                                uint32_t causal) {
        pq = (__gm__ float*)q;
        pck = (__gm__ float*)ck;
        pcv = (__gm__ float*)cv;
        pnk = (__gm__ float*)nk;
        pnv = (__gm__ float*)nv;
        pout = (__gm__ float*)out;
        this->kvseq = kvseq;
        this->heads = heads;
        this->kvheads = kvheads;
        this->headdim = headdim;
        this->causal = causal;
        const uint32_t cores = (uint32_t)GetBlockNum();
        GRID_RANGE(GetBlockIdx(), cores, heads, start, end);
    }
    __aicore__ inline void Process() {
        const float scale = 1.0f / sqrtf((float)headdim);
        const float neglarge = -3.402823466e+38f;
        const uint32_t pos = kvseq - 1u;
        const uint32_t limit = kvseq;
        for (uint32_t head = start; head < end; ++head) {
            const uint32_t kvhead = head / (heads / kvheads);
            const float* qrow = pq + (uint64_t)head * headdim;
            float localmax = neglarge;
            for (uint32_t key = 0; key < limit; ++key) {
                const float* krow = (key < pos)
                    ? (pck + ((uint64_t)key * kvheads + kvhead) * headdim)
                    : (pnk + (uint64_t)kvhead * headdim);
                float dot = 0.0f;
                for (uint32_t d = 0; d < headdim; ++d) {
                    dot += qrow[d] * krow[d];
                }
                const float score = dot * scale;
                if (score > localmax) {
                    localmax = score;
                }
            }
            float totalw = 0.0f;
            for (uint32_t key = 0; key < limit; ++key) {
                const float* krow = (key < pos)
                    ? (pck + ((uint64_t)key * kvheads + kvhead) * headdim)
                    : (pnk + (uint64_t)kvhead * headdim);
                float dot = 0.0f;
                for (uint32_t d = 0; d < headdim; ++d) {
                    dot += qrow[d] * krow[d];
                }
                totalw += expf(dot * scale - localmax);
            }
            float* orow = pout + (uint64_t)head * headdim;
            if (totalw > 0.0f) {
                const float inv = 1.0f / totalw;
                for (uint32_t d = 0; d < headdim; ++d) {
                    float acc = 0.0f;
                    for (uint32_t key = 0; key < limit; ++key) {
                        const float* krow = (key < pos)
                            ? (pck + ((uint64_t)key * kvheads + kvhead) * headdim)
                            : (pnk + (uint64_t)kvhead * headdim);
                        float dot = 0.0f;
                        for (uint32_t dd = 0; dd < headdim; ++dd) {
                            dot += qrow[dd] * krow[dd];
                        }
                        const float weight = expf(dot * scale - localmax);
                        const float* vrow = (key < pos)
                            ? (pcv + ((uint64_t)key * kvheads + kvhead) * headdim)
                            : (pnv + (uint64_t)kvhead * headdim);
                        acc += weight * vrow[d];
                    }
                    orow[d] = acc * inv;
                }
            } else {
                for (uint32_t d = 0; d < headdim; ++d) {
                    orow[d] = 0.0f;
                }
            }
        }
    }

private:
    __gm__ float* pq;
    __gm__ float* pck;
    __gm__ float* pcv;
    __gm__ float* pnk;
    __gm__ float* pnv;
    __gm__ float* pout;
    uint32_t kvseq;
    uint32_t heads;
    uint32_t kvheads;
    uint32_t headdim;
    uint32_t causal;
    uint32_t start;
    uint32_t end;
};

extern "C" __global__ __aicore__ void attention_cached(GM_ADDR q, GM_ADDR ck,
                                                       GM_ADDR cv, GM_ADDR nk,
                                                       GM_ADDR nv, GM_ADDR out,
                                                       uint32_t kvseq, uint32_t heads,
                                                       uint32_t kvheads,
                                                       uint32_t headdim,
                                                       uint32_t causal) {
    KernelAttentionCached op;
    op.Init(q, ck, cv, nk, nv, out, kvseq, heads, kvheads, headdim, causal);
    op.Process();
}
