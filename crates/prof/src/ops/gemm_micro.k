// GEMM 主机微内核：SME 外积瓦片与两个探针。FMA(f32) 与 sdot(i8 dotprod) 已迁至 Rust intrinsic。
// 所有几何来自 ops/kernels.toml；运行时特性探测由 Rust 安全桥负责。
#include <stddef.h>
#include <stdint.h>

#if defined(_M_X64) && !defined(__x86_64__)
#define __x86_64__ 1
#endif
#if defined(_M_ARM64) && !defined(__aarch64__)
#define __aarch64__ 1
#endif

#define CUTILE_SME_MR {{SME_MR}}
#define CUTILE_SME_NR {{SME_NR}}

#if defined(__aarch64__)
#include <arm_neon.h>
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
