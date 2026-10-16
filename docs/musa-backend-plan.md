# MUSA 后端适配计划：国产 GPU 的 CUDA 兼容后端

> **版本**：v1.2（2026-10-16）　**状态**：提议（未实现）；**M0 已收敛 + 5.x/MTRTC 升级评估已完成（结论：不升级，留 3.1.0 走 mcc 预编译 + mu\* driver API）**
> **权威**：`docs/design.md` §21（GPU 后端与 `unsafe` 豁免）、`docs/beta-scan-plan.md`（scan 算子已三后端闭环，本计划复用其内核）
> **目标**：新增 `musa` 后端（摩尔线程 MUSA 架构），复用已验证的 CUDA C 内核，把 host 层从 `cudarc` 替换为 `musart`/`MTRTC`/`muBLAS`/`muDNN` 的 MUSA 绑定，使同一 `api::compute` 算子集在国产 GPU 上跑通且三后端数值一致。

---

## 1. 背景与动机

| 锚点 | 现状 | 结论 |
| :--- | :--- | :--- |
| **E6（跨硬件适配）** | 三后端 = CPU/Metal/CUDA；无国产 GPU | MUSA 是 E6 的国产硬件锚点 |
| **CUDA 源码兼容** | MUSA 提供 **MUSA Mapping**（编译期把 CUDA 源码映射到 MUSA）+ **MUSIFY**（移植工具） | 内核层 `.cu` **已验证可移植**，几乎不改 |
| **host 层差异** | `cudarc` 绑定 `cudart`/`nvrtc`/`cublas`/`cudnn`；MUSA 对应 `musart`/`MTRTC`/`muBLAS`/`muDNN` | host 层必须替换，这是本计划的主体 |
| **scan 内核复用** | `crates/cuda/kernels/*.cu`（含 `scan.cu`、`scan_parallel.cu`）已验证真机 | `musa` crate 直接复用这些内核源码 |

**一句话**：不是「为 MUSA 重写算子」，而是**把 `cuda` crate 的 host 层从 NVIDIA 绑定换成 MUSA 绑定**——内核层、`api::compute` 契约、`scanshape` 校验、`unsafe` 站点登记结构全部复用，只有 FFI 岛换了平台。

## 2. 目标与成功标准

1. **后端落地**：`musa` crate 实现 `api::backend` + `api::compute` 全部算子（含 `scan`/`scan_cached`），三后端数值一致。
2. **host 层替换**：`musart`（context/stream/内存）、`MTRTC`（运行时编译）、`muBLAS`（matmul）、`muDNN`（softmax）四类平台交互各一处 unsafe 站点，`musa = 3` 登记。
3. **内核复用**：`kernels/*.cu` 从 `cuda` 复用，MUSA Mapping 编译期兼容，不改内核逻辑。
4. **文档同步**：design.md §21（musa 站点表 + 后端清单）、lld.md E6、conformance.md 判定行、checkunsafe.sh 登记表。

**成功 = `cudacontract` 的数值用例在 MUSA 真机全绿（等价于 `cudacontract` 48/48 的 MUSA 复刻），且 `musa` 不新增 unsafe 站点类别（仍 3 类）。**

## 3. 技术调研（host 层 API 替换点）

| cudarc（现状） | MUSA 等价物 | 替换范围 |
| :--- | :--- | :--- |
| `cudarc::driver`（`CudaContext`/`CudaStream`/`CudaSlice`/`LaunchArgs`） | `musart`（MUSA Runtime：context/stream/device memory） | context 保留、stream、上传/下载、内核启动 |
| `cudarc::nvrtc::compile_ptx` | **3.1.0 无 MTRTC**（5.x 组件）；可用路径 = **`mcc` 编译期生成**（build.rs 预编译 `kernels/*.cu` → 可执行内嵌，`musaLaunchKernel` 运行时调用，与 `smoke_scan.cu` 验证一致）；5.x 可再切 MTRTC | 内核源码 → 编译产物 |
| `cudarc::cublas::CudaBlas::gemm` | **muBLAS**（cuBLAS 兼容） | `matmul` 快速路径 |
| `cudarc::cudnn::SoftmaxForward` | **muDNN**（cuDNN 兼容） | `softmax` 快速路径 |

**M0 已收敛（2026-10-16 真机实测）**：MUSA host API **非符号级 CUDA 兼容**——`libmusart.so` 导出 185 个 `musa*` 前缀符号（`musaMalloc`/`musaMemcpy`/`musaLaunchKernel`/`musaDeviceGetCount` 等，C ABI）；driver 层 `libmusa.so` 为 MUSA 原生 C++ ABI（`MU*` 类型：`DeviceGetId`/`ContextGetId`/`imuMemcpy*`）。无 `cuInit`/`cuDeviceGetCount` 等 CUDA driver 符号 → `cudarc` 动态加载**不可用** → **自写 MUSA FFI 岛**（本计划原保守规划成立）。

**关键补充（3.1.0 driver API 完整性，2026-10-16 实测）**：`libmusa.so`（driver 层）**已含完整 `mu*` driver API**——`muModuleLoadData(Ex)`（加载二进制产物）、`muLaunchKernel(Ex)`（启动内核）、`muMemAlloc`/`muMemcpyHtoD`/`muMemcpyDtoH`、`muCtxCreate`/`muCtxSynchronize`、`muDeviceGet`（共 24 个内存/上下文/设备符号），且 `include/musa.h` 头齐全。**MUSA driver API 是 5.x MTRTC 加载路径的同一层，3.1.0 就已具备**——缺的只有运行时编译组件（MTRTC）。

### 3.x 5.x/MTRTC 升级评估（2026-10-16，结论：不升级）

| 维度 | 实测/事实 | 判断 |
| :--- | :--- | :--- |
| MTRTC（5.2.0） | `libmtrtc.so` + `mtrtc.h`；与 NVRTC **一一对应**（`mtrtcCreateProgram`/`mtrtcCompileProgram`/`mtrtcGetFatBin`/`mtrtcGetLoweredName`）；输出 FatBin → `muModuleLoadData` → `muLaunchKernel`；编译选项 `--gpu-architecture=mp_22`（S4000 示例即用） | 5.x MTRTC 路径 = 3.1.0 mu\* driver API + 运行时编译；API 形态与 cudarc 的 NVRTC 调用面镜像 |
| 主机前置（sha1） | 内存 1007 GB、内核 5.15.0-105-generic（5.2.0 安装指南指定版本）、**IOMMU 已开**（`intel_iommu=on iommu.passthrough=0`，S4000 精确配置） | 5.x 安装硬件/内核/BIOS 前置**全部已满足** |
| 升级动作与风险 | 需装 5.x 驱动（2.7.0 → 5.x，跨代）+ `musa_toolkits` install.sh（先卸载旧 musa）+ **重启**（云主机，reboot 与驱动 deb 安装有失败/断连风险） | 非零风险 |
| **需求驱动** | 项目 8 个内核 = **固定编译期已知源码**（`kernels/*.cu`），无运行时生成源码/模板实例化需求 | **运行时编译（NVRTC/MTRTC）对项目是多余的**——cudarc 用 NVRTC 只是其实现方式，不是需求 |

**结论：不升级 5.x。** MUSA 后端 host 层 = **`mu*` driver API**（3.1.0 已完整）：build.rs 调 `mcc -mtgpu --offload-arch=mp_22` 预编译 `kernels/*.cu` → FatBin 随包 → 运行时 `muModuleLoadData` + `muModuleGetFunction` + `muLaunchKernel`。与 5.x MTRTC 的唯一差异 = 编译时机（编译期 vs 运行时），对固定内核集无实质收益；零升级风险。**5.x/MTRTC 保留为未来选项**（若出现运行时生成内核、模板实例化、按 arch 多编译等需求再评估）。

## 4. 架构决策

1. **新增 `musa` crate**（Layer 1，`musa -> api`，与 `metal`/`cuda` 并列）。依赖方向：`musa -> api -> tensor -> lib`，无反向边，无 workspace 环。
2. **`#![deny(unsafe_code)]` + `src/ffi.rs` 单文件 `#![allow(unsafe_code)]`**，站点数 `musa = 3`：MTRTC 内核启动（1，闭包注入实参服务全部在仓内核）+ muBLAS `gemm`（1）+ muDNN softmax（1）。与 `cuda` 的「逐类交互恰一处」同构。
3. **内核复用**：`kernels/*.cu` 从 `cuda` crate **复制**（不共享模块，后端实现细节各自持有，AGENTS.md §3.2「共享类型 ≠ 无关职责入 lib」）。内核源码经 MUSA Mapping 编译期兼容，逐文件核对后不改逻辑。
4. **api 累加式扩展**：`devicekind` 加 `musa` 变体（`device::{musa,ismusa}`），`compute`/`submission` 契约**一行不改**。
5. **结构镜像 `cuda`**：`backend`/`error`/`executor`/`ffi`/`source` 五模块 + `kernels/`，`cache` 结构（device context + `(source, entry) -> function`）照搬。

## 5. 阶段拆分（M0–M6，每个提交一个可测阶段）

| 阶段 | 内容 | 验收标准 |
| :--- | :--- | :--- |
| ~~**M0 调研尖刺**~~ | **已完成（2026-10-16，MTT S4000 + MUSA Toolkit 3.1.0 真机）**：① host API = 独立符号（musart 185 个 `musa*`、driver 层 `MU*` C++ ABI）→ 自写 FFI；② **3.1.0 无 MTRTC**（5.x 组件）→ 内核走 `mcc` 编译期（`-mtgpu --offload-arch=mp_22`）+ `musaLaunchKernel`；③ `libmublas.so`/`libmudnn.so` 存在（M4 可行）；④ 三段式并行 scan 内核已真机数值 PASS（`scripts/musa/smoke_scan.cu`）。**M2–M6 实现细节按此收敛** |
| **M1 骨架 + 站点登记** | `musa` crate 骨架（`backend`/`error`/`source`/`ffi` 空壳）；`#![deny]` + `ffi.rs` 单文件 `#![allow]`；`checkunsafe.sh` 加 `musa=3`；api `devicekind::musa` | 骨架编译（非 Linux 空库）；`checkunsafe.sh` 登记表一致；依赖图无环 |
| **M2 mu\* driver FFI** | `mu*` driver API host 层（`muInit`/`muDeviceGet`/`muCtxCreate`/`muMemAlloc`/`muMemcpyHtoD`/`muMemcpyDtoH`/`muModuleLoadData`/`muLaunchKernel`）；`ffi.rs` 首个 unsafe 站点（`muLaunchKernel`） | 单内核 `silu` 经 mcc 预编译 FatBin + mu\* driver 跑通 |
| **M3 内核族（mcc 预编译 + mu\* driver API）** | 复用 `kernels/*.cu`；build.rs 调 `mcc -mtgpu --offload-arch=mp_22` 预编译 FatBin → 运行时 `muModuleLoadData` + `muModuleGetFunction` + `muLaunchKernel`（3.1.0 driver API 已验证存在；与 5.x MTRTC 加载路径同层）；`add`/`mul`/`silu`/`rmsnorm`/`rope`/`embedding`/`attention`/`scan` 全部注册 | 内核数值 vs `testkit::reference` 一致 |
| **M4 muBLAS/muDNN** | `matmul` 走 muBLAS、`softmax` 走 muDNN；第二、三 unsafe 站点 | `matmul`/`softmax` 数值 vs 参考一致；`musa=3` 站点恰等于登记数 |
| **M5 算子全量** | `compute` 全部方法（含 `scan`/`scan_cached`、`submission` 累加式重放）；形状/预算/谱校验镜像 `cuda` | `musacontract` 数值 + 错误分支全绿（等价 `cudacontract` 48 例的复刻） |
| **M6 真机 + 基准 + 文档** | MUSA 真机跑 `musacontract`；`benches/musa.rs` 基准；design.md §21 站点表 + 后端清单、lld.md E6、conformance.md 判定 | 真机数值全绿；基准出数；`checkunsafe.sh`/`checknaming.sh`/`checksrctests.sh` 全绿 |

**依赖**：M0 → M1 → M2 → M3/M4（可并行，依赖 M2）→ M5 → M6。M0 是硬前置（决定 FFI 策略）。

## 6. 验收门（Must 不降）

| 门 | 协议 | 判定 |
| :--- | :--- | :--- |
| 数值正确性 | `musa` vs `testkit::reference`（`rtol/atol` 同 cuda/metal，§21.10） | 全绿 |
| 三后端一致 | 同一算子链经 `compute` 与 `submission` 逐位一致 | 逐位 |
| unsafe | `checkunsafe.sh`：`musa=3` 恰等于 §21.8 登记数，每站 `// SAFETY:` | 机械门禁 |
| 内核复用 | `kernels/*.cu` 与 `cuda` 逐文件一致（仅 MUSA 兼容所需的编译期映射，不改逻辑） | 逐文件 diff 核对 |
| 依赖方向 | `musa -> api`，无反向边、无环 | `cargo metadata` |
| 文档 | design.md §21 / lld.md E6 / conformance.md 同步 | 与实现一致 |

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
| :--- | :--- | :--- |
| MUSA host API 非符号兼容 | 需自写 FFI，工作量翻倍 | M0 尖刺先行决策；保守按自写 FFI 规划 |
| 3.1.0 无 MTRTC，`mcc` 编译期管线（build.rs 预编译 + 产物加载）与 NVRTC 语义差异 | 内核编译/运行失败 | 复用已真机验证的 `smoke_scan` 编译路径（`-mtgpu --offload-arch=mp_22`）；逐内核冒烟；5.x 再评估 MTRTC |
| `muBLAS`/`muDNN` 与 cuBLAS/cuDNN 数值容差 | 数值 vs 参考越界 | 沿用 cuda 的容差协议；若不达标，matmul/softmax 回退 MTRTC 内核 |
| 内核 `.cu` 中 NVIDIA 专用扩展 | 编译失败 | M0 逐文件编译核对；MUSA Mapping 不覆盖的扩展改写为等价 MUSA 形式 |
| unsafe 站点数漂移 | 违反 §21.8 | M2/M4 先评审再实现；期望 `musa=3` 不变 |

## 8. 交付物清单

1. `crates/musa/`：`backend`/`error`/`executor`/`ffi`/`source` + `build.rs`（`mcc` 预编译管线）+ `kernels/*.cu`（复用自 `cuda`）
2. `crates/api/src/devices.rs`：`devicekind::musa` + `device::{musa,ismusa}`
3. `docs/design.md` §21：musa 站点表（`musa=3`）+ 后端清单
4. `scripts/checkunsafe.sh`：musa 登记数
5. `tests/it/musacontract.rs`：数值 + 错误分支（等价 cudacontract 48 例）
6. `benches/musa.rs`：算子基准
7. `docs/musa-backend-plan.md`（本文档）、`docs/lld.md` E6、`docs/conformance.md` 判定行

## 9. 建议提交序列

1. `feat(api): devicekind 加 musa`（M1 一部分，独立可测）
2. `feat(musa): 骨架 + unsafe 站点登记`（M1）
3. `feat(musa): musart host 层 + MTRTC 内核族`（M2+M3）
4. `feat(musa): muBLAS/muDNN + 算子全量`（M4+M5）
5. `test(musa): musacontract 数值 + 错误分支`（M5）
6. `bench(musa): 算子基准 + 文档同步`（M6）

---

**状态声明**：本计划为提议（未实现，不写已交付）。**M0 调研尖刺已于 2026-10-16 在 MTT S4000 真机完成并收敛**：host 层 = 独立符号（自写 FFI 岛，非 cudarc 复用）；**5.x/MTRTC 升级评估完成（结论：不升级）**——3.1.0 的 `mu*` driver API 已完整（`muModuleLoadData`/`muLaunchKernel`/内存/上下文，`musa.h` 齐），host 层 = `mu*` driver API + build.rs `mcc` 预编译 FatBin（与 5.x MTRTC 加载路径同层，编译时机不同，对固定内核集无实质收益）；muBLAS/muDNN 可用。M2–M6 按此收敛实施。


### 4.x 性能路径深度对比（2026-10-16，结论：A1+B1+C2 唯一最优）

**真机/权威证据（本轮新增）**：
| 证据 | 数据 | 来源 |
| :--- | :--- | :--- |
| 单次内核启动稳态开销（MTT S4000） | 异步队列 **18.1 µs**/launch；带同步往返 **42.2 µs**/launch | 本轮 `scripts/musa/launch_probe.cu` 真机实测 |
| S4000 Tensor Core 规格 | FP32 25 T / **TF32 Tensor 50 T** / FP16 Tensor 100 T / INT8 200 TOPS | 官方产品页 |
| GEMM 优化路径 | 基线 FMA **3,640 GFLOPS** → WMMA/Tensor Core 优化 **53,572 GFLOPS**（~15×） | 摩尔线程官方博客（B 预转置 + 128 线程 Wave Tile + 异步双缓冲 + CTA Swizzle） |

**正交维度拆解**（方案非互斥，三个独立性能变量）：

| 维度 | 选项 | 实测/依据 | 是否性能变量 |
| :--- | :--- | :--- | :--- |
| 编译时机 | A1 mcc 编译期 vs A2 MTRTC 运行时 | fatbin 优化程度相同；A2 仅多首启 JIT | **否**（稳态等价） |
| host 层 | A1/A2 `mu*` driver vs A3 `musart` runtime | driver 直通驱动、无 runtime 编组；实测 18µs 为 musart 路径，driver 同量级（µs 级差） | **否**（每算子一次启动） |
| 算子计算 | **B1 muBLAS/muDNN（Tensor Core）** vs B2 自研 FMA 内核 | TF32 50 T vs FMA 3.6 T → **~14×**；muBLAS 已含官方博客全套优化 | **是（最大）** |
| 执行模式 | C1 逐算子同步 vs **C2 submission 单提交** | 35 算子 × 42µs ≈ **1.5 ms 固定开销** → 单提交 ~0.1 ms（~35×）；Metal 端已实证 52ms→2.8ms | **是（次大）** |

**结论（唯一最优）**：**A1（mcc 预编译 + `mu*` driver）+ B1（muBLAS/muDNN）+ C2（submission 批处理）**。
- 综合最优 = A1：零升级风险；稳态性能与 A2 等价；host 层比 A3 薄且为未来 MTRTC 同层接口（A1 ⊃ A3）。
- 性能最优 = B1×C2（Tensor Core 计算差 ~14× × 固定开销差 ~35×）；A1/A2 在性能上无差异，A2 无任何增益。
- 性能预估（外推，非实测；真实差距待 M6 基准）：最差路径（B2+C1）1× → 最优路径（B1+C2）预估 **15–30×**（固定开销 + 计算双杠杆）。
