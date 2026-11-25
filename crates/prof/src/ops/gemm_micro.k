// GEMM 主机微内核：生产与后续探针共用 fma_tile / sdot_tile / sme_tile。
// 所有几何来自 ops/kernels.toml；运行时特性探测由 Rust 安全桥负责。
#include <stddef.h>
#include <stdint.h>

#if defined(_M_X64) && !defined(__x86_64__)
#define __x86_64__ 1
#endif
#if defined(_M_ARM64) && !defined(__aarch64__)
#define __aarch64__ 1
#endif
#if defined(__x86_64__) && defined(_MSC_VER)
#define CUTILE_FMA_TARGET
#define CUTILE_FMA_SYMBOL cutile_fma_micro_avx2
#else
#define CUTILE_FMA_TARGET __attribute__((target("avx2,fma")))
#define CUTILE_FMA_SYMBOL cutile_fma_micro
#endif

#define CUTILE_FMA_MR {{FMA_MR}}
#define CUTILE_FMA_NR {{FMA_NR}}
#define CUTILE_FMA_KB {{FMA_KB}}
#define CUTILE_SDOT_MR {{SDOT_MR}}
#define CUTILE_SDOT_NR {{SDOT_NR}}
#define CUTILE_SDOT_CHAINS {{SDOT_CHAINS}}
#define CUTILE_SME_MR {{SME_MR}}
#define CUTILE_SME_NR {{SME_NR}}
#define CUTILE_SME_MIN_ELEMS {{SME_MIN_ELEMS}}

#if !defined(CUTILE_AVX2_ONLY)
static void fma_tile_scalar(const float *a, const float *b, float *out,
                            int row, int col, int n, int kb, int first) {
    for (int r = 0; r < CUTILE_FMA_MR; ++r) {
        for (int c = 0; c < CUTILE_FMA_NR; ++c) {
            const int slot = (row + r) * n + col + c;
            float acc = first ? 0.0f : out[slot];
            /* cutile-unroll: fma_scalar */
            for (int p = 0; p < kb; ++p) acc += a[p * CUTILE_FMA_MR + r] * b[p * CUTILE_FMA_NR + c];
            out[slot] = acc;
        }
    }
}

/* cutile-export: fma_scalar */
void cutile_fma_micro_scalar(const float *a, const float *b, float *out,
                             int row, int col, int n, int kb, int first) {
    fma_tile_scalar(a, b, out, row, col, n, kb, first);
}
#endif

#if defined(__aarch64__)
#include <arm_neon.h>

static void fma_tile(float32x4_t acc[16], const float *a, const float *b, int kb) {
    /* cutile-unroll: fma_neon */
    for (int p = 0; p < kb; ++p) {
        const float32x4_t av0 = vld1q_f32(a + p * CUTILE_FMA_MR);
        const float32x4_t av1 = vld1q_f32(a + p * CUTILE_FMA_MR + 4);
        const float32x4_t bv0 = vld1q_f32(b + p * CUTILE_FMA_NR);
        const float32x4_t bv1 = vld1q_f32(b + p * CUTILE_FMA_NR + 4);
        acc[0] = vfmaq_laneq_f32(acc[0], bv0, av0, 0);
        acc[1] = vfmaq_laneq_f32(acc[1], bv1, av0, 0);
        acc[2] = vfmaq_laneq_f32(acc[2], bv0, av0, 1);
        acc[3] = vfmaq_laneq_f32(acc[3], bv1, av0, 1);
        acc[4] = vfmaq_laneq_f32(acc[4], bv0, av0, 2);
        acc[5] = vfmaq_laneq_f32(acc[5], bv1, av0, 2);
        acc[6] = vfmaq_laneq_f32(acc[6], bv0, av0, 3);
        acc[7] = vfmaq_laneq_f32(acc[7], bv1, av0, 3);
        acc[8] = vfmaq_laneq_f32(acc[8], bv0, av1, 0);
        acc[9] = vfmaq_laneq_f32(acc[9], bv1, av1, 0);
        acc[10] = vfmaq_laneq_f32(acc[10], bv0, av1, 1);
        acc[11] = vfmaq_laneq_f32(acc[11], bv1, av1, 1);
        acc[12] = vfmaq_laneq_f32(acc[12], bv0, av1, 2);
        acc[13] = vfmaq_laneq_f32(acc[13], bv1, av1, 2);
        acc[14] = vfmaq_laneq_f32(acc[14], bv0, av1, 3);
        acc[15] = vfmaq_laneq_f32(acc[15], bv1, av1, 3);
    }
}
#elif defined(__x86_64__) && !defined(CUTILE_SCALAR_ONLY)
#include <immintrin.h>

CUTILE_FMA_TARGET
static void fma_tile(__m256 acc[8], const float *a, const float *b, int kb) {
    /* cutile-unroll: fma_avx2 */
    for (int p = 0; p < kb; ++p) {
        const __m256 bv = _mm256_loadu_ps(b + p * CUTILE_FMA_NR);
        for (int r = 0; r < CUTILE_FMA_MR; ++r) {
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(a[p * CUTILE_FMA_MR + r]), bv, acc[r]);
        }
    }
}
#endif

#if !defined(CUTILE_SCALAR_ONLY) || !defined(__x86_64__)
/* cutile-export: fma */
#if defined(__x86_64__)
CUTILE_FMA_TARGET
#endif
void CUTILE_FMA_SYMBOL(const float *a, const float *b, float *out,
                      int row, int col, int n, int kb, int first) {
#if defined(__aarch64__)
    float32x4_t acc[16];
    for (int r = 0; r < 16; ++r) {
        acc[r] = first ? vdupq_n_f32(0.0f)
                       : vld1q_f32(out + (row + r / 2) * n + col + (r % 2) * 4);
    }
    fma_tile(acc, a, b, kb);
    for (int r = 0; r < 16; ++r) {
        vst1q_f32(out + (row + r / 2) * n + col + (r % 2) * 4, acc[r]);
    }
#elif defined(__x86_64__)
    __m256 acc[8];
    for (int r = 0; r < 8; ++r) {
        acc[r] = first ? _mm256_setzero_ps() : _mm256_loadu_ps(out + (row + r) * n + col);
    }
    fma_tile(acc, a, b, kb);
    for (int r = 0; r < 8; ++r) _mm256_storeu_ps(out + (row + r) * n + col, acc[r]);
#else
    fma_tile_scalar(a, b, out, row, col, n, kb, first);
#endif
}
#endif

#if !defined(CUTILE_AVX2_ONLY)
#if defined(__aarch64__)
__attribute__((target("dotprod"), always_inline))
static inline void sdot_tile(const int8_t *ap, const int8_t *bp, int k, int kvec,
                      int32x4_t acc[CUTILE_SDOT_NR][CUTILE_SDOT_MR]) {
    for (int nb = 0; nb < CUTILE_SDOT_NR; ++nb) {
        for (int r = 0; r < CUTILE_SDOT_MR; ++r) acc[nb][r] = vdupq_n_s32(0);
    }
    /* cutile-unroll: sdot */
    for (int p = 0; p < kvec; p += 16) {
        const int8x16_t av[CUTILE_SDOT_MR] = {
            vld1q_s8(ap + p), vld1q_s8(ap + k + p),
            vld1q_s8(ap + 2 * k + p), vld1q_s8(ap + 3 * k + p),
        };
        for (int nb = 0; nb < CUTILE_SDOT_NR; ++nb) {
            const int8x16_t bv = vld1q_s8(bp + nb * k + p);
            for (int r = 0; r < CUTILE_SDOT_MR; ++r) acc[nb][r] = vdotq_s32(acc[nb][r], av[r], bv);
        }
    }
}
#endif

#if defined(__aarch64__)
/* cutile-probe: fma_pure, core=fma_tile */
void cutile_fma_pure(size_t rounds) {
    float32x4_t acc[16];
    for (int r = 0; r < 16; ++r) acc[r] = vdupq_n_f32(0.0f);
    const float32x4_t av = vdupq_n_f32(1.0000001f);
    for (size_t i = 0; i < rounds; ++i) {
        for (int r = 0; r < 16; ++r) acc[r] = vfmaq_f32(acc[r], av, av);
    }
    for (int r = 0; r < 16; ++r) __asm__ volatile("" : "+w"(acc[r]));
}

/* cutile-probe: fma_block, core=fma_tile */
void cutile_fma_block(size_t rounds, const float *a, const float *b, int k) {
    float32x4_t acc[16];
    for (int r = 0; r < 16; ++r) acc[r] = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < rounds; ++i) fma_tile(acc, a, b, k);
    for (int r = 0; r < 16; ++r) __asm__ volatile("" : "+w"(acc[r]));
}

/* cutile-probe: sdot_pure, core=sdot_tile */
void cutile_sdot_pure(size_t rounds) {
    const int8x16_t a = vdupq_n_s8(3), b = vdupq_n_s8(5);
    int32x4_t acc[CUTILE_SDOT_CHAINS];
    for (int i = 0; i < CUTILE_SDOT_CHAINS; ++i) acc[i] = vdupq_n_s32(0);
    for (size_t i = 0; i < rounds; ++i) {
        for (int chain = 0; chain < CUTILE_SDOT_CHAINS; ++chain) acc[chain] = vdotq_s32(acc[chain], a, b);
    }
    for (int i = 0; i < CUTILE_SDOT_CHAINS; ++i) __asm__ volatile("" : "+w"(acc[i]));
}

/* cutile-probe: sdot_block, core=sdot_tile */
void cutile_sdot_block(size_t rounds, const int8_t *a, const int8_t *b, int k) {
    int32x4_t acc[CUTILE_SDOT_NR][CUTILE_SDOT_MR];
    for (size_t i = 0; i < rounds; ++i) sdot_tile(a, b, k, k, acc);
    for (int nb = 0; nb < CUTILE_SDOT_NR; ++nb) {
        for (int r = 0; r < CUTILE_SDOT_MR; ++r) __asm__ volatile("" : "+w"(acc[nb][r]));
    }
}
#endif

/* cutile-export: sdot */
#if defined(__aarch64__)
__attribute__((target("dotprod")))
#endif
void cutile_sdot_micro(const int8_t *a, const int8_t *bpack, int32_t *out,
                       int row, int blk, int n, int k, int kvec) {
#if defined(__aarch64__)
    const int8_t *ap = a + row * k;
    int32x4_t acc[CUTILE_SDOT_NR][CUTILE_SDOT_MR];
    sdot_tile(ap, bpack + blk * k, k, kvec, acc);
    for (int nb = 0; nb < CUTILE_SDOT_NR; ++nb) {
        const int jj = blk + nb;
        if (jj >= n) continue;
        for (int r = 0; r < CUTILE_SDOT_MR; ++r) {
            int32_t sum = vaddvq_s32(acc[nb][r]);
            for (int p = kvec; p < k; ++p) sum += (int32_t)ap[r * k + p] * bpack[jj * k + p];
            out[(row + r) * n + jj] = sum;
        }
    }
#else
    for (int r = 0; r < CUTILE_SDOT_MR; ++r) {
        for (int nb = 0; nb < CUTILE_SDOT_NR; ++nb) {
            const int jj = blk + nb;
            if (jj >= n) continue;
            int32_t sum = 0;
            for (int p = 0; p < k; ++p) sum += (int32_t)a[(row + r) * k + p] * bpack[jj * k + p];
            out[(row + r) * n + jj] = sum;
        }
    }
#endif
}

#if defined(__aarch64__)
/* 生产外积与两个探针共用这组算术；纯探针环内没有 load／store。 */
#define CUTILE_SME_FMOPA \
    "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n" \
    "fmopa za1.s, p0/m, p0/m, z0.s, z2.s\n" \
    "fmopa za2.s, p0/m, p0/m, z0.s, z3.s\n" \
    "fmopa za3.s, p0/m, p0/m, z0.s, z4.s\n"

__attribute__((target("sme")))
static void sme_enter(void) {
    __asm__ volatile("smstart\nsmstart za\nptrue p0.s" ::: "p0", "memory");
}

__attribute__((target("sme")))
static void sme_exit(void) {
    __asm__ volatile("smstop za\nsmstop" ::: "memory");
}

__attribute__((target("sme")))
static void sme_tile(size_t k, const float *a, const float *b, float *c, size_t n) {
    const size_t b_adv = n * sizeof(float) - 3 * 64;
    float *c16, *c32, *c48;
    __asm__ volatile(
        "zero {za0.s}\nzero {za1.s}\nzero {za2.s}\nzero {za3.s}\n"
        "0:\n"
        "ld1w {z0.s}, p0/z, [%[a]]\nadd %[a], %[a], #64\n"
        "ld1w {z1.s}, p0/z, [%[b]]\nadd %[b], %[b], #64\n"
        "ld1w {z2.s}, p0/z, [%[b]]\nadd %[b], %[b], #64\n"
        "ld1w {z3.s}, p0/z, [%[b]]\nadd %[b], %[b], #64\n"
        "ld1w {z4.s}, p0/z, [%[b]]\nadd %[b], %[b], %[badv]\n"
        CUTILE_SME_FMOPA
        "subs %[k], %[k], #1\nb.ne 0b\n"
        "mov %[c16], %[c]\nadd %[c16], %[c16], #64\n"
        "mov %[c32], %[c]\nadd %[c32], %[c32], #128\n"
        "mov %[c48], %[c]\nadd %[c48], %[c48], #192\n"
        "mov w12, #0\nmov w13, #1\nmov w14, #2\nmov w15, #3\n"
        "1:\n"
        "str za[w12, 0], [%[c]]\nstr za[w13, 0], [%[c16]]\n"
        "str za[w14, 0], [%[c32]]\nstr za[w15, 0], [%[c48]]\n"
        "add %[c], %[c], %[nstride]\nadd %[c16], %[c16], %[nstride]\n"
        "add %[c32], %[c32], %[nstride]\nadd %[c48], %[c48], %[nstride]\n"
        "add w12, w12, #4\nadd w13, w13, #4\nadd w14, w14, #4\nadd w15, w15, #4\n"
        "cmp w12, #64\nb.ne 1b\n"
        : [a] "+r"(a), [b] "+r"(b), [c] "+r"(c), [k] "+r"(k),
          [c16] "=&r"(c16), [c32] "=&r"(c32), [c48] "=&r"(c48)
        : [nstride] "r"(n * sizeof(float)), [badv] "r"(b_adv)
        : "z0", "z1", "z2", "z3", "z4", "w12", "w13", "w14", "w15", "p0", "memory");
}

/* 4×4 转置；两轮交织与原 std::simd 打包布局完全一致。 */
static inline void sme_pack4(const float *src, float *dst, size_t k) {
    const float32x4_t a0 = vld1q_f32(src);
    const float32x4_t a1 = vld1q_f32(src + k);
    const float32x4_t a2 = vld1q_f32(src + 2 * k);
    const float32x4_t a3 = vld1q_f32(src + 3 * k);
    const float32x4_t t0 = vzip1q_f32(a0, a1);
    const float32x4_t t1 = vzip2q_f32(a0, a1);
    const float32x4_t t2 = vzip1q_f32(a2, a3);
    const float32x4_t t3 = vzip2q_f32(a2, a3);
    vst1q_f32(dst, vcombine_f32(vget_low_f32(t0), vget_low_f32(t2)));
    vst1q_f32(dst + CUTILE_SME_MR, vcombine_f32(vget_high_f32(t0), vget_high_f32(t2)));
    vst1q_f32(dst + 2 * CUTILE_SME_MR, vcombine_f32(vget_low_f32(t1), vget_low_f32(t3)));
    vst1q_f32(dst + 3 * CUTILE_SME_MR, vcombine_f32(vget_high_f32(t1), vget_high_f32(t3)));
}
#endif

/* cutile-export: sme_pack */
void cutile_sme_pack(const float *src, float *dst, size_t k, size_t mblk0, size_t blocks) {
    for (size_t bi = 0; bi < blocks; ++bi) {
        const size_t rowbase = (mblk0 + bi) * CUTILE_SME_MR * k;
        float *block = dst + bi * k * CUTILE_SME_MR;
        size_t p = 0;
#if defined(__aarch64__)
        for (; p + 16 <= k; p += 16) {
            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    sme_pack4(src + rowbase + r * 4 * k + p + c * 4,
                              block + (p + c * 4) * CUTILE_SME_MR + r * 4, k);
                }
            }
        }
#endif
        for (; p < k; ++p) {
            for (size_t r = 0; r < CUTILE_SME_MR; ++r) {
                block[p * CUTILE_SME_MR + r] = src[rowbase + r * k + p];
            }
        }
    }
}

/* cutile-export: sme_stage */
#if defined(__aarch64__)
__attribute__((target("sme")))
#endif
void cutile_sme_stage(const float *a, const float *b, float *out,
                      size_t n, size_t k, size_t blocks) {
#if defined(__aarch64__)
    if (blocks == 0) return;
    sme_enter();
    for (size_t bi = 0; bi < blocks; ++bi) {
        for (size_t g = 0; g < n / CUTILE_SME_NR; ++g) {
            sme_tile(k, a + bi * k * CUTILE_SME_MR, b + g * CUTILE_SME_NR,
                     out + bi * CUTILE_SME_MR * n + g * CUTILE_SME_NR, n);
        }
    }
    sme_exit();
#else
    (void)a; (void)b; (void)out; (void)n; (void)k; (void)blocks;
#endif
}

/* cutile-export: sme */
void cutile_sme_micro(const float *a, const float *b, float *out, size_t n, size_t k) {
#if defined(__aarch64__)
    sme_enter();
    sme_tile(k, a, b, out, n);
    sme_exit();
#else
    for (size_t r = 0; r < CUTILE_SME_MR; ++r) {
        for (size_t c = 0; c < CUTILE_SME_NR; ++c) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) sum += a[p * CUTILE_SME_MR + r] * b[p * n + c];
            out[r * n + c] = sum;
        }
    }
#endif
}

#if defined(__aarch64__)
/* cutile-probe: sme_pure, core=sme_tile */
__attribute__((target("sme")))
void cutile_sme_pure(size_t rounds) {
    sme_enter();
    __asm__ volatile(
        "fmov z0.s, #1.0\nfmov z1.s, #1.0\n"
        "zero {za0.s}\nzero {za1.s}\nzero {za2.s}\nzero {za3.s}\n"
        "0:\n"
        CUTILE_SME_FMOPA
        "subs %[k], %[k], #1\nb.ne 0b\n"
        : [k] "+r"(rounds) : : "z0", "z1", "p0", "memory");
    sme_exit();
}
#endif

#if defined(__aarch64__)
/* cutile-probe: sme_mix, core=sme_tile */
__attribute__((target("sme")))
void cutile_sme_mix(size_t rounds, const float *a, const float *b) {
    const float *b1 = b + 16, *b2 = b + 32, *b3 = b + 48;
    sme_enter();
    __asm__ volatile(
        "zero {za0.s}\nzero {za1.s}\nzero {za2.s}\nzero {za3.s}\n"
        "0:\n"
        "ld1w {z0.s}, p0/z, [%[a]]\n"
        "ld1w {z1.s}, p0/z, [%[b]]\n"
        "ld1w {z2.s}, p0/z, [%[b1]]\n"
        "ld1w {z3.s}, p0/z, [%[b2]]\n"
        "ld1w {z4.s}, p0/z, [%[b3]]\n"
        CUTILE_SME_FMOPA
        "subs %[k], %[k], #1\nb.ne 0b\n"
        : [k] "+r"(rounds)
        : [a] "r"(a), [b] "r"(b), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3)
        : "z0", "z1", "z2", "z3", "z4", "p0", "memory");
    sme_exit();
}
#endif
#endif
