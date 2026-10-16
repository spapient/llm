# GPU 后端统一架构：消除 JIT 架构债务 + 收敛 unsafe

> **版本**：v1.0（2026-10-16）　**状态**：提议（未实现）；关键前提已真机/源码验证

## 1. 现状审计（JIT 架构债务 + unsafe 面）

| 后端 | 内核编译时机 | JIT 债务 | unsafe 站点 |
| :--- | :--- | :--- | :--- |
| `cuda` | **NVRTC 运行时 JIT**（每内核首启 ~10.8 ms；运行时依赖 nvrtc 库；编译错误运行时才暴露） | 高 | 3（NVRTC launch / cuBLAS gemm / cuDNN softmax） |
| `metal` | **MSL 运行时编译**（`newLibraryWithSource`，每 pipeline 首启编译） | 中 | 4（上传 / 下载 / setBuffer / setBytes，objc2-metal API 面决定） |
| `musa`（规划） | **mcc 构建期预编译**（已无 JIT） | 无 | 2（`unsafe extern` 声明 + 单一驱动交接块） |

**根因同一**：运行时编译是「把工具链塞进运行时」的架构债务；厂商加速库（cuBLAS/cuDNN）是「为两个算子引入两类额外平台交互」的 unsafe 债务。二者都应前移/消除。

## 2. 统一终态命题（一句话）

**三个 GPU 后端的共同终态 = 「构建期预编译 + 运行时零编译 + 最小 unsafe 面（cuda 收敛到 1、musa 收敛到自写 FFI 的 2024 edition 下限 2、metal 收敛到库决定的下限 4）」**。

| 终态 | cuda | musa | metal |
| :--- | :--- | :--- | :--- |
| 内核编译 | **nvcc 构建期预编译** → CUBIN/FatBin 内嵌 | **mcc 构建期预编译** → FatBin 内嵌 | **xcrun metal 构建期预编译** → metallib 内嵌 |
| 运行时加载 | `cuModuleLoadData`（cudarc `PtxKind::Binary`，已验证） | `muModuleLoadData`（已验证） | `newLibraryWithData`（待尖刺） |
| matmul | **WMMA 内核**（Tensor Core） | **WMMA 内核**（Tensor Core） | 现有内核（Metal 无 WMMA 暴露） |
| softmax | 内核（减最大式） | 内核（减最大式） | 现有内核 |
| unsafe | **1**（仅内核启动） | **2**（自写 FFI 2024 下限：`unsafe extern` 声明 + 单一驱动交接块） | **4**（库决定，如实） |
| 运行时工具链依赖 | 无 nvrtc/cublas/cudnn | 无 muBLAS/muDNN | 无 |

## 3. 关键前提（已验证）

| 前提 | 证据 |
| :--- | :--- |
| cudarc 二进制模块加载 | `vendor/cudarc-0.19.10/src/driver/safe/core.rs`：`load_module(PtxKind::Binary)` → `cuModuleLoadData`（源码级确认） |
| MUSA WMMA 指令集 | 3.1.0 `include/crt/mma.h`（156 KB）：`mma_sync` 203 重载 / `load_matrix_sync` 130 / `store_matrix_sync` 21 / `fill_fragment`（真机确认） |
| WMMA 性能可达 | 官方博客：3,640 → 53,572 GFLOPS（~15×，Tensor Core 路径） |
| metal unsafe 下限 | design.md §21.8 实测：objc2-metal 的 4 类 unsafe API 面 |

## 4. 统一构建期内核管线（三个后端同构）

```text
build.rs（构建期，需目标工具链）
  探测工具链（nvcc / mcc / xcrun metal）
    ├─ 存在 → 编译 kernels/*.{cu,metal} → fatbin/cubin/metallib → include_bytes! 内嵌
    │         + 生成 registry.rs（内核名 → 段/entry/shared 预算，编译期枚举，类型安全）
    └─ 缺失 → 编译空库（同现有跨平台空库），workspace 绿
运行时
  加载内嵌二进制一次 → 惰性解析内核函数 → [唯一 unsafe 壳] 内核启动
```

* **零 JIT**：首启零编译延迟；编译错误构建期暴露（错误前移）。
* **类型化注册表**：内核集合编译期锁死，无运行时字符串键（替代 cuda 现行 `HashMap<(&str, &str), CudaFunction>`）。
* **部署自包含**：fatbin/metallib 内嵌，运行时无工具链依赖。

## 5. 分阶段路线（按 价值×风险 排序）

| 阶段 | 内容 | 验收 | 风险 |
| :--- | :--- | :--- | :--- |
| **P1 musa（新后端，按终态直建）** | musa crate：mcc 预编译 + WMMA matmul + 全内核族 → **musa=2** | musacontract 数值 vs 参考 + 真机（MTT S4000）+ checkunsafe.sh 登记 musa=2 | 低（新 crate，不影响既有后端） |
| **P2 cuda 收敛** | 去 cuBLAS/cuDNN；nvcc 预编译 CUBIN 内嵌（`PtxKind::Binary`）；matmul 改 WMMA 内核 → **cuda=1** | cudacontract 数值互证（WMMA vs 保留的 FMA 参考路径）+ 真机 + checkunsafe.sh 登记 cuda=1 | 中（重写 matmul 内核、重验数值、去两依赖） |
| **P3 metal JIT 消除** | xcrun metal 构建期编译 metallib 内嵌 → 运行时 `newLibraryWithData`（待尖刺验证 API） | metalcontract 全量 + 首启延迟对比 | 中（构建期工具链依赖 xcrun；unsafe 4 维持） |

**互斥原则**：P1–P3 各为一个独立可测阶段（AGENTS.md §11），不混提；P2/P3 不阻塞 P1。

## 6. 验证面（与 AGENTS.md §7/§9 对齐）

* 数值：三后端 vs `testkit::reference` 互证；WMMA 路径与 FMA 参考路径互证（构造器区分）。
* 构建期：工具链缺失 → 空库（CI 门禁同现有条件编译）。
* unsafe：checkunsafe.sh 登记 `cuda=1 / musa=2 / metal=4`，机械强制。
* 基准：`gpu/matmul`（WMMA vs FMA，Tensor Core 差距实测）、`gpu/decode`（缓存 vs 重算）、首启延迟对比（JIT 消除收益实测）。
* 真机门禁：Metal（Apple GPU）、CUDA（NVIDIA）、MUSA（MTT S4000）。

## 7. 质量维度对标（ISO/IEC 25010）

| 维度 | 统一架构对应 |
| :--- | :--- |
| 功能性正确性 | 数值互证（三后端 + 双路径）；编译错误构建期暴露（运行时错误面收窄） |
| 性能效率 | WMMA 吃 Tensor Core（~15×）+ submission 单提交（~35×）+ 首启零 JIT |
| 可维护性 | 单文件 unsafe 岛、类型化注册表、三后端同构管线 |
| 可移植性 | 条件编译空库、工具链缺失退化空库 |
| 安全性 | **全仓 unsafe 面收敛到最小可证集合**（cuda=1/musa=2/metal=4）+ SAFETY 注释机械强制 |
| 可测试性 | 构造器区分路径、kernelcount 观测、真机门禁 |

## 8. 状态声明

本计划为提议（未实现，不写已交付）。**已验证**：cudarc 二进制加载（源码）、MUSA WMMA（真机）、WMMA 性能路径（官方）、metal unsafe 下限（§21.8 实测）。**待尖刺**：`newLibraryWithData` API（P3）。P1 不依赖任何未决信息，可立即开始。