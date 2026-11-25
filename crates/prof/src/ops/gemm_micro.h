/* cutile_sme_* 主机微内核的唯一 C 签名事实源。
 *
 * 实现住在 `gemm_micro.k`；本头只声明原型，几何常量来自 `kernels.toml`，
 * 不在这里抄写数字。Rust 安全桥（`runtime.rs::gemm_micro`）的手写 extern
 * 与本头由构建期对齐门三方核对：头原型 ↔ `.k` 注解 ↔ Rust extern 名集与参数宽类。
 */
#ifndef CUTILE_GEMM_MICRO_H
#define CUTILE_GEMM_MICRO_H

#include <stddef.h>

/* 生产瓦片与打包入口；非 aarch64 时 `.k` 内提供标量兜底，符号在所有目标导出。 */
void cutile_sme_micro(const float *a, const float *b, float *out, size_t n, size_t k);
void cutile_sme_pack(const float *src, float *dst, size_t k, size_t mblk0, size_t blocks);
void cutile_sme_stage(const float *a, const float *b, float *out, size_t n, size_t k,
                      size_t blocks);

/* 测量探针只在 aarch64 编译；Rust 侧另挂 `bench` 特性与架构门。 */
#if defined(__aarch64__)
void cutile_sme_pure(size_t rounds);
void cutile_sme_mix(size_t rounds, const float *a, const float *b);
#endif

#endif
