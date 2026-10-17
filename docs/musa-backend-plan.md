# MUSA 后端适配计划：国产 GPU 的 CUDA 兼容后端

> **版本**：v1.8（2026-10-17）　**状态**：P1b 已验证（提交链 `0b0f7cf`→`6bf912e`）——`crates/musa` 从 171 行类型骨架补成完整后端：`build.rs` mcc 预编译管线（A1，无运行时 JIT）+ `musa=2` unsafe 面（`unsafe extern` 声明岛 + 单一驱动交接 `dispatch`）+ 全部 13 个 `api::compute` 算子 + `musacontract` 数值契约套件 + `benches/musa.rs`（含 scan）；**M0 已收敛 + 5.x 评估 + 性能路径对比 + 消除 unsafe 终极架构（musa=2，2024 下限）已完成；P1 骨架（31ac165）已提交**。**2026-10-17：WMMA（TF32）matmul 已实现并为默认路径，FMA 内核保留为 `newfma` 参考路径；真机 39/39、unsafe 2/2；优化后 1024³ 实测 WMMA 2.1250 ms vs FMA 10.110 ms（4.76×），256³ WMMA 0.9175 ms vs FMA 1.0798 ms；device-resident 8×1024³ 每步 1.062 ms**
> **权威**：`docs/design.md` §21（GPU 后端与 `unsafe` 豁免）、`docs/beta-scan-plan.md`（scan 算子已三后端闭环，本计划复用其内核）
> **目标**：新增 `musa` 后端（摩尔线程 MUSA 架构），复用已验证的 CUDA C 内核，把 host 层从 `cudarc` 替换为自写 `mu*` driver FFI（`mcc` 编译期预编译；不引入 muBLAS/muDNN），使同一 `api::compute` 算子集在国产 GPU 上跑通且三后端数值一致。

---

## 1. 背景与动机

| 锚点 | 现状 | 结论 |
| :--- | :--- | :--- |
| **E6（跨硬件适配）** | 三后端 = CPU/Metal/CUDA；无国产 GPU | MUSA 是 E6 的国产硬件锚点 |
| **CUDA 源码兼容** | MUSA 提供 **MUSA Mapping**（编译期把 CUDA 源码映射到 MUSA）+ **MUSIFY**（移植工具） | 内核层 `.cu` **已验证可移植**，几乎不改 |
| **host 层差异** | `cudarc` 绑定 `cudart`/`nvrtc`/`cublas`/`cudnn`；MUSA 对应 `musart`/`MTRTC`/`muBLAS`/`muDNN`，但终态采用 `mu*` driver + `mcc` 预编译（§5.1） | host 层必须替换，这是本计划的主体 |
| **scan 内核复用** | `crates/cuda/kernels/*.cu`（含 `scan.cu`、`scan_parallel.cu`）已验证真机 | `musa` crate 直接复用这些内核源码 |

**一句话**：不是「为 MUSA 重写算子」，而是**把 `cuda` crate 的 host 层从 NVIDIA 绑定换成 MUSA 绑定**——内核层、`api::compute` 契约、`scanshape` 校验、`unsafe` 站点登记结构全部复用，只有 FFI 岛换了平台。

## 2. 目标与成功标准

1. **后端落地**：`musa` crate 实现 `api::backend` + `api::compute` 全部算子（含 `scan`/`scan_cached`），三后端数值一致。
2. **host 层替换**：自写 `mu*` driver FFI（`mcc` 编译期预编译）+ WMMA `matmul` 内核 + 自研 `softmax` 内核；unsafe 站点 `musa = 2`（`unsafe extern` 声明岛 + 单一驱动交接块），不引入 muBLAS/muDNN。
3. **内核复用**：`kernels/*.cu` 从 `cuda` 复用，MUSA Mapping 编译期兼容，不改内核逻辑。
4. **文档同步**：design.md §21（musa 站点表 + 后端清单）、lld.md E6、conformance.md 判定行、checkunsafe.sh 登记表。

**成功 = `musacontract` 在 MUSA 真机全绿（39/39：13 算子 + WMMA/FMA 双路径 + 边界/提交/device-free 向量 + 设备驻留链），且 `musa` 不新增 unsafe 站点类别（仍 2 类，`musa=2`）。**

## 3. 技术调研（host 层 API 替换点）

| cudarc（现状） | MUSA 等价物 | 替换范围 |
| :--- | :--- | :--- |
| `cudarc::driver`（`CudaContext`/`CudaStream`/`CudaSlice`/`LaunchArgs`） | `musart`（MUSA Runtime：context/stream/device memory） | context 保留、stream、上传/下载、内核启动 |
| `cudarc::nvrtc::compile_ptx` | **3.1.0 无 MTRTC**（5.x 组件）；可用路径 = **`mcc` 编译期生成**（build.rs 预编译 `kernels/*.cu` → 可执行内嵌，`musaLaunchKernel` 运行时调用，与 `smoke_scan.cu` 验证一致）；5.x 可再切 MTRTC | 内核源码 → 编译产物 |
| `cudarc::cublas::CudaBlas::gemm` | **muBLAS**（cuBLAS 兼容） | `matmul` 快速路径 |
| `cudarc::cudnn::SoftmaxForward` | **muDNN**（cuDNN 兼容） | `softmax` 快速路径（该路径最终未采用，§5.1） |

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
2. **`#![deny(unsafe_code)]` + `src/ffi.rs` 单文件 `#![allow(unsafe_code)]`**，站点数 `musa = 2`：`unsafe extern "C"` 声明块（edition 2024 强制）+ 单一驱动交接 `unsafe` 块（全部 `mu*` 调用收于 `drive`）。不引入 muBLAS/muDNN（§5.1）。
3. **内核复用**：`kernels/*.cu` 从 `cuda` crate **复制**（不共享模块，后端实现细节各自持有，AGENTS.md §3.2「共享类型 ≠ 无关职责入 lib」）。内核源码经 MUSA Mapping 编译期兼容，逐文件核对后不改逻辑。
4. **api 累加式扩展**：`devicekind` 加 `musa` 变体（`device::{musa,ismusa}`），`compute`/`submission` 契约**一行不改**。
5. **结构镜像 `cuda`**：`backend`/`error`/`executor`/`ffi`/`source` 五模块 + `kernels/`，`cache` 结构（device context + `(source, entry) -> function`）照搬。

## 5. 阶段拆分（M0–M6，每个提交一个可测阶段）

| 阶段 | 内容 | 验收标准 |
| :--- | :--- | :--- |
| ~~**M0 调研尖刺**~~ | **已完成（2026-10-16，MTT S4000 + MUSA Toolkit 3.1.0 真机）**：① host API = 独立符号（musart 185 个 `musa*`、driver 层 `MU*` C++ ABI）→ 自写 FFI；② **3.1.0 无 MTRTC**（5.x 组件）→ 内核走 `mcc` 编译期（`-mtgpu --offload-arch=mp_22`）+ `musaLaunchKernel`；③ `libmublas.so`/`libmudnn.so` 存在（M4 可行）；④ 三段式并行 scan 内核已真机数值 PASS（`scripts/musa/smoke_scan.cu`）。**M2–M6 实现细节按此收敛** |
| **M1 骨架 + 站点登记** | `musa` crate 骨架（`backend`/`error`/`source`/`ffi` 空壳）；`#![deny]` + `ffi.rs` 单文件 `#![allow]`；`checkunsafe.sh` 加 `musa=2`；api `devicekind::musa` | 骨架编译（非 Linux 空库）；`checkunsafe.sh` 登记表一致；依赖图无环 |
| **M2 mu\* driver FFI** | `mu*` driver API host 层（`muInit`/`muDeviceGet`/`muCtxCreate`/`muMemAlloc`/`muMemcpyHtoD`/`muMemcpyDtoH`/`muModuleLoadData`/`muLaunchKernel`）；`ffi.rs` 的 `unsafe extern` 声明岛 + 单一驱动交接块 `drive`（全部 `mu*` 调用收于一处） | 单内核 `silu` 经 mcc 预编译 FatBin + mu\* driver 跑通 |
| **M3 内核族（mcc 预编译 + mu\* driver API）** | 复用 `kernels/*.cu`；build.rs 调 `mcc -mtgpu --offload-arch=mp_22` 预编译 FatBin → 运行时 `muModuleLoadData` + `muModuleGetFunction` + `muLaunchKernel`（3.1.0 driver API 已验证存在；与 5.x MTRTC 加载路径同层）；`add`/`mul`/`silu`/`rmsnorm`/`rope`/`embedding`/`attention`/`scan` 全部注册 | 内核数值 vs `testkit::reference` 一致 |
| **M4（已实现，2026-10-17）** | 原计划 muBLAS/muDNN；按 §5.1 终态改为 WMMA（TF32）`matmul` 内核 + 自研 `softmax` 内核，不新增站点；WMMA 为默认路径，FMA 内核保留为 `newfma` 参考路径 | `matmul`（两路径）/`softmax` 数值 vs 参考一致；`musacontract` 39/39；站点数保持 `musa=2`；1024³ A/B 实测 4.76×、device-resident 7.99× |
| **M5 算子全量** | `compute` 全部方法（含 `scan`/`scan_cached`、`submission` 累加式重放）；形状/预算/谱校验镜像 `cuda` | `musacontract` 数值 + 错误分支全绿（等价 `cudacontract` 48 例的复刻） |
| **M6 真机 + 基准 + 文档** | MUSA 真机跑 `musacontract`；`benches/musa.rs` 基准；design.md §21 站点表 + 后端清单、lld.md E6、conformance.md 判定 | 真机数值全绿；基准出数；`checkunsafe.sh`/`checknaming.sh`/`checksrctests.sh` 全绿 |

**依赖**：M0 → M1 → M2 → M3/M4（可并行，依赖 M2）→ M5 → M6。M0 是硬前置（决定 FFI 策略）。

## 6. 验收门（Must 不降）

| 门 | 协议 | 判定 |
| :--- | :--- | :--- |
| 数值正确性 | `musa` vs `testkit::reference`（`rtol/atol` 同 cuda/metal，§21.10） | 全绿 |
| 三后端一致 | 同一算子链经 `compute` 与 `submission` 逐位一致 | 逐位 |
| unsafe | `checkunsafe.sh`：`musa=2` 恰等于 §21.8 登记数，每站 `// SAFETY:` | 机械门禁 |
| 内核复用 | 除 `matmul.cu`（保留 FMA `matmul` 并新增 TF32 `matmul_wmma`）外，`kernels/*.cu` 与 `cuda` 逐文件一致（仅 MUSA 兼容所需的编译期映射，不改逻辑） | 逐文件 diff 核对 |
| 依赖方向 | `musa -> api`，无反向边、无环 | `cargo metadata` |
| 文档 | design.md §21 / lld.md E6 / conformance.md 同步 | 与实现一致 |

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
| :--- | :--- | :--- |
| MUSA host API 非符号兼容 | 需自写 FFI，工作量翻倍 | M0 尖刺先行决策；保守按自写 FFI 规划 |
| 3.1.0 无 MTRTC，`mcc` 编译期管线（build.rs 预编译 + 产物加载）与 NVRTC 语义差异 | 内核编译/运行失败 | 复用已真机验证的 `smoke_scan` 编译路径（`-mtgpu --offload-arch=mp_22`）；逐内核冒烟；5.x 再评估 MTRTC |
| ~~`muBLAS`/`muDNN` 与 cuBLAS/cuDNN 数值容差~~ | 已消除（§5.1 不引入两库） | 风险关闭：`matmul`/`softmax` 由自研内核服务，无厂商库精度分叉 |
| 内核 `.cu` 中 NVIDIA 专用扩展 | 编译失败 | M0 逐文件编译核对；MUSA Mapping 不覆盖的扩展改写为等价 MUSA 形式 |
| unsafe 站点数漂移 | 违反 §21.8 | M2 先评审再实现；期望 `musa=2` 不变 |

## 8. 交付物清单

1. `crates/musa/`：`backend`/`error`/`executor`/`ffi`/`source` + `build.rs`（`mcc` 预编译管线）+ `kernels/*.cu`（复用自 `cuda`）
2. `crates/api/src/devices.rs`：`devicekind::musa` + `device::{musa,ismusa}`
3. `docs/design.md` §21：musa 站点表（`musa=2`）+ 后端清单
4. `scripts/checkunsafe.sh`：musa 登记数
5. `tests/it/musacontract.rs`：数值 + 错误分支（等价 cudacontract 48 例）
6. `benches/musa.rs`：算子基准
7. `docs/musa-backend-plan.md`（本文档）、`docs/lld.md` E6、`docs/conformance.md` 判定行

## 9. 建议提交序列

1. `feat(api): devicekind 加 musa`（M1 一部分，独立可测）
2. `feat(musa): 骨架 + unsafe 站点登记`（M1）
3. `feat(musa): mu* driver FFI + mcc 预编译内核族`（M2+M3）
4. `feat(musa): WMMA matmul + 算子全量`（M4+M5；原 muBLAS/muDNN 方案已取消）
5. `test(musa): musacontract 数值 + 错误分支`（M5）
6. `bench(musa): 算子基准 + 文档同步`（M6）

---



### 5. 顶级架构设计（v1.3，2026-10-16）——消除 unsafe

**架构命题（一句话）**：musa 把 `unsafe` 收敛到**自写 FFI 的理论下限 `musa=2`**（`unsafe extern "C"` 声明 + 单一驱动交接块），与 cuda=2 并列全仓最小面；不引入 muBLAS/muDNN，matmul 用 **WMMA 内核**吃 Tensor Core，性能与数值契约双不损。

#### 5.0 「消除 unsafe」的现实边界（先说清事实）

* **字面零 unsafe 不可能**：GPU 编程必须经 C ABI FFI 调用平台库（`mu*` driver / `musart`），Rust 调用 C 函数本身就是 `unsafe extern`；跨进程 IPC 隔离（unsafe 移入守护进程）可做到 crate 内零 unsafe，但引入 IPC 往返、部署复杂化，是过度设计，违反项目极简哲学（design.md §21.2 决策 2 同款论证：外包 ≠ 消除）。
* **可达的顶级形态** = **unsafe 面收敛到最小可证集合**：站点按「交互类别」登记，且类别数降到 1；unsafe 代码体积极小化（单薄壳函数）；理由机械可证（SAFETY 注释 + checkunsafe.sh 强制）。
* 本设计使 **musa=2（自写 FFI 的 2024 edition 理论下限）**：`unsafe extern "C"` 声明块（edition 2024 强制——声明面本身是不可消除的构造）+ 单一驱动交接 `unsafe` 块（`drive`：全部 `mu*` 调用收于一个块，块内不再逐处标记）。与 cuda=2 并列、低于 metal=3（2026-10-17 现值；metal 由 4 站点合并为 3 个交接块；cuda 删除 cuDNN softmax 后由 3 降为 2）；`musa=1` 在 2024 edition 下不可能（extern 声明不可省）。

#### 5.1 关键架构决策：不引入 muBLAS/muDNN（musa=2 的前提）

| 决策 | 依据 |
| :--- | :--- |
| **matmul 用 WMMA 内核，不走 muBLAS** | 真机验证：3.1.0 `include/crt/mma.h`（156 KB）含完整 WMMA 指令集——`mma_sync` 203 重载 / `load_matrix_sync` 130 / `store_matrix_sync` 21 / `fill_fragment` / `bmma_sync`，与 CUDA `nvcuda::wmma` 同形态。**执行模型有实质差异**：MUSA 硬件 `warpSize=128`，`m32n32k8` TF32 MMA 是 wave128 指令（四个 32-lane warp 协作）；真机实测把 128 线程拆成 4 个独立组、或用 32 线程块执行该形状，都只得到零/部分输出。最终内核以 **128 线程 = 一个 wave128 组**、每块 64×32 tile（两个累加器）组织。官方博客的 53,572 GFLOPS 是路径上界，**本项目优化后实测 1024³ 为 4.76×（10.110→2.1250 ms），256³ 为 1.18×（1.0798→0.9175 ms）；device-resident 8×1024³ 为 7.99×**——不把官方数字当作已兑现收益 |
| **softmax 用内核，不走 muDNN** | softmax 是内存带宽受限算子（非计算受限），自研减最大式内核与 muDNN 差距小；且减最大式与 cpu/metal 标量契约**同式**——数值契约设备无关，无厂商库精度分叉 |
| **收益** | unsafe 站点 −2（无 blas/dnn 两类交互）；依赖 −2 库；错误面 −2 类（无 blas/dnn）；数值契约自洽（全内核族、无 fallback 分叉） |
| **代价与缓解** | 内核按 64×32 tile + wave128（128 线程）实现：完整 tile 用 `float4` 向量装载、对齐时直接全局 store，边缘 tile 走 guarded shared staging，结果 staging 复用操作数缓冲（共享内存恒 8.5 KiB）。首版 32-lane/16×16×8 与错误使用 32×32×8 的版本均已在真机废弃；官方博客的 B 预转置 + 异步双缓冲 + CTA Swizzle 尚未移植，列为后续优化 |

#### 5.2 继承的架构内核（cuda/metal 已验证，必须保留）

| 内核 | 依据 |
| :--- | :--- |
| 单文件 `unsafe` 岛（`src/ffi.rs` 唯一 `allow(unsafe_code)`） | design.md §21.8 |
| 条件编译空库（非目标平台零实体，workspace 绿） | design.md §21.2 决策 5 |
| 依赖边 `musa → api/tensor/lib`，api 零反向依赖 | design.md §21.3 |
| `geometry` 校验唯一实现于 api（跨后端一致） | design.md §21.4 |
| 同步 run-to-completion + submission 单 stream 累积、一次 sync 一次下载 | design.md §21.13 |
| cache 单锁、锁不跨传输、每调用新 stream（防并发串行化） | design.md §21.6 |
| 构造器区分路径（快路径 vs 参考路径） | cuda `matmulpath` 同构 |
| 空输入短路 | cuda 全算子 |

#### 5.3 模块结构

```text
crates/musa/
  Cargo.toml              # [target.'cfg(target_os = "linux")'.dependencies] libmusa/libmusart（动态加载）
  build.rs                # mcc 探测 → 预编译 kernels/*.cu → OUT_DIR fatbin → 生成 registry.rs
  kernels/*.cu            # 自 cuda 复制；matmul.cu 同时导出 FMA(matmul) 与 TF32(matmul_wmma)：
                          #   elementwise/rmsnorm/rope/rope_at/embedding/attention/
                          #   attention_cached/matmul/softmax/scan/scan_parallel
  src/lib.rs              # #![deny(unsafe_code)]；cfg(target_os="linux") 条件导出
  src/backend.rs          # musabackend（devicekind::musa）
  src/error.rs            # musaerror（6 类：device/module/transfer/launch/sizeoverflow）
  src/executor.rs         # musaexecutor（api::compute）+ musasubmission（session）
  src/ffi.rs              # 唯一 unsafe 岛：2 站点（unsafe extern 声明块 + drive 驱动交接块）
  src/registry.rs         # 构建期生成：内核 → (fatbin 段, entry, shared 预算)，编译期枚举
```

#### 5.4 unsafe 站点登记（musa=2，2024 edition 自写 FFI 理论下限）

| # | 站点 | 交互类别 | SAFETY 前置 |
| :--- | :--- | :--- | :--- |
| 1 | `submit`（`muLaunchKernel`） | 内核启动（全部在仓内核共用，不随内核数增长） | args 由 `muModuleGetFunction` 解析的内核签名镜像构建、实参为存活设备分配或本调用期标量、launch 几何经 `api::geometry` 校验且覆盖元素数（内核自界检） |

`scripts/checkunsafe.sh` 登记 `musa=2`（与 cuda=2 并列全仓最小，低于 metal=3；2026-10-17 现值，机械强制）。驱动交接函数在安全侧完成 geometry/预算校验后，仅转发给 C ABI。

#### 5.5 执行与内存所有权

* **无请求级状态、无模型权重**（AGENTS.md §5）：cache 只持设备上下文 + 已加载模块 + 惰性解析的内核函数；session 只持 stream + 输出 slot。
* **每调用新 stream**：submission 内单 stream 保序累积；单算子调用取新 stream，并发不串行化。
* **无厂商库 fallback 分叉**：全内核族数值契约与 cpu/metal 同式（softmax 减最大；非有限输入传播 NaN，与标量契约一致）。

#### 5.6 验证面

* `tests/it/musacontract.rs`：37 例全绿（MTT S4000）——WMMA 路径 vs 标量参考 2e-3、FMA 路径 1e-4、两路径互证 2e-3；边界 1×1×1 / 15×16×17 / 16×16×16 / 17×19×13 / 37×51×29 / 64×64×16 / 65×63×17、batch [2,17,13]×[2,13,19]、k=0、m/n=0 空输出；提交模式与 device-free 向量两路径一致。
* `benches/musa.rs`：`musa/matmul`（CPU vs 默认 WMMA）、`musa/matmulbatch`（8×256³）、`musa/matmulpath`（FMA vs WMMA，256³/1024³）、`musa/matmulchain`（device-resident 8×1024³ 链，输入只上传一次）、softmax/scan/decode；criterion 中位数、musa 与 cpu 同组同 run。实测：1024³ WMMA **2.1250 ms** vs FMA **10.110 ms**（**4.76×**）；256³ WMMA **0.9175 ms** vs FMA **1.0798 ms**（WMMA 快 17.7%）；8×256³ CPU 11.906 ms vs WMMA 1.2202 ms（9.76×）；device-resident 链 WMMA **8.495 ms/8 步 = 1.062 ms/步** vs FMA **67.908 ms/8 步 = 8.489 ms/步**（7.99×），对比 API 路径 2.125 ms 可见每调用约 1.06 ms 为 host 固定开销。
* 构建期：fatbin 生成进 CI 门禁（mcc 缺失 → 空库）。

#### 5.7 质量维度对标（ISO/IEC 25010）

| 维度 | 设计对应 |
| :--- | :--- |
| 功能性正确性 | 数值 vs 独立参考路径互证；全内核族数值契约设备无关；错误面收窄（编译错误构建期暴露；无 blas/dnn 错误类） |
| 性能效率 | WMMA 内核吃 Tensor Core（本项目 1024³ 实测 **4.76×** vs FMA、256³ **1.18×**、device-resident 链 **7.99×**；官方博客 ~15× 是路径上界）+ submission 设备驻留链 |
| 可维护性 | 单文件 unsafe 岛、构建期类型化注册表、模块与 cuda/metal 同构 |
| 可移植性 | 条件编译空库、mcc 缺失退化空库、workspace 全平台绿 |
| **安全性** | **全仓最小 unsafe 面（musa=2，自写 FFI 2024 下限）**、驱动交接函数薄壳、SAFETY 注释 + checkunsafe.sh 机械强制 |
| 可测试性 | 构造器区分路径、kernelcount 观测、真机门禁（M6） |

#### 5.8 全局统一（跨后端，见 `docs/gpu-backend-unification.md` v1.0）

* **统一终态**：三个 GPU 后端共同收敛到「构建期预编译 + 运行时零编译 + 最小 unsafe 面」——cuda=1（规划目标）、musa=2、metal=3（2026-10-17 现值；musa=2 为 2024 edition 自写 FFI 理论下限）。
* **cuda 的 JIT 消除已验证可行**：cudarc 0.19.10 `load_module(PtxKind::Binary)` → `cuModuleLoadData`（源码确认）；nvcc 构建期预编译 CUBIN 内嵌 + WMMA matmul 内核（构建机有完整 CUDA 工具链，WMMA 头可编译）→ cuda=1。
* **metal**：JIT 消除 = xcrun metal 构建期 metallib（P3，`newLibraryWithData` 待尖刺）；unsafe=3（2026-10-17 合并为驱动交接块后的现值，§21.8 实测）。
* 阶段互斥：P1（musa）→ P2（cuda 收敛）→ P3（metal JIT 消除），各为独立可测提交。

**状态声明**：**P1b 已实现**（提交 `0b0f7cf`，2026-10-16）：`crates/musa` 完整后端落地——`build.rs` 在 Linux 且探测到 `/usr/local/musa-3.1.0/bin/mcc` 时把 `kernels/*.cu`（11 个，含新增减最大式 `softmax.cu`，`scan_parallel.cu` 的两处 `min(t0+chunk,n)`→三元已按 M0 冒烟结论适配）预编译为 ELF fatbin 经 `include_bytes!` 嵌入（`registry.rs`）；`ffi.rs` 的 `musa=2` 面落地（`unsafe extern "C"` 声明 11 个 `mu*` 符号 + 单一惰性驱动交接 `dispatch`，handle 为 `usize` 保持 cache `Send+Sync`）；`executor.rs` 实现全部 13 个 `api::compute` 算子 + host-side `musasubmission`（scan/scan_cached 在 submission 镜像 cuda 报 `unsupportedoperation`）；`scan_cached` 在 compute 路径**受支持**（n=1 串行内核续算）、`rope_at`/`attention_cached` 走**专用内核**（不复制 cuda 的 host 拼接）；`tests/it/musacontract.rs` 全量数值契约（13 算子 vs 标量参考、host 侧拒绝路径、submission 链、共享 device-free 向量，`E4`/`requiredevice` 探测协议）；`benches/musa.rs` 镜像 cuda 工作负载并**新增 scan**。本地已验证：`cargo check --workspace --all-targets`、`cargo clippy --workspace --all-targets -D warnings`、`cargo fmt --check` 全绿，`scripts/checkunsafe.sh` musa 2/2 exact，musa 交叉 check 绿。**真机已验证（2026-10-16，MTT S4000 / MP_22 / MUSA 3.1.0）**：`musacontract` **28/28 全绿**（13 算子对照标量参考 1e-5 容差，含 scan 串行/并行/缓存、attention 全量/增量、batch matmul、rope_at）；`benches/musa.rs` 出数并回填 `docs/conformance.md`（matmul 256 1.26 ms / softmax 512×4096 2.52 ms / attention seq128 1.46 ms 快于 CPU；launch 受限小算子慢于 CPU——方向性基线，无性能声称）。

**WMMA matmul 已实现并完成首轮优化（2026-10-17，MTT S4000 真机）。** `crates/musa/kernels/matmul.cu` 在保留 FMA `matmul` 符号的同时新增 TF32 `matmul_wmma` 符号；`musaexecutor::new` 默认走 WMMA，`newfma` 是独立参考路径（`matmulpath` 枚举，submission 同样按路径选择）。内核以 128 线程 = 一个 wave128 组，每块计算 64×32 tile（两个 32×32×8 累加器）；完整 tile 用 `float4` 向量装载，下一 K tile 在 MMA 前预取到寄存器、compute 后写回同一 stage；块索引按 8 行分组 CTA Swizzle；输出 stride 对齐时直接 `store_matrix_sync` 写全局，边缘 tile 走 guarded shared staging，结果 staging 复用操作数缓冲（共享内存 8.5 KiB）；每元素执行 `wmma::__float_to_tf32`。submission 的 `recordmatmul` 也改为设备驻留：输入只上传一次、中间结果留设备 slot、`apply` 一次同步下载，新增 `musa/matmulchain` 基准（8×1024³ 每步 1.062 ms，7.99× vs FMA）。契约 39/39、unsafe 2/2。优化迭代（1024³ WMMA）：32-lane 16×16×8 初版 5.7208 ms → wave128 32×32 2.9479 ms → `float4` 内点装载 2.1530 ms → 64×32 + 缓冲复用 2.0568 ms → 寄存器预取 + CTA Swizzle 约 2.0 ms；经典双 stage 双缓冲因共享内存翻倍、占用率减半而回退（1024 +9.5%、256 +38.6% 一次运行）。关键教训：`m32n32k8` 必须整 wave（128 线程）执行，32 线程或拆 4 组的用法在真机上只得到零/部分输出。

**M0 调研尖刺已于 2026-10-16 在 MTT S4000 真机完成并收敛**：host 层 = 独立符号（自写 FFI 岛，非 cudarc 复用）；**5.x/MTRTC 升级评估完成（结论：不升级）**；**性能路径对比完成（结论 A1+B1+C2，后经消除 unsafe 设计修订为 A1+WMMA+C2）**；**消除 unsafe 终极架构完成（musa=2：2024 edition 自写 FFI 理论下限——`unsafe extern` 声明 + 单一驱动交接块；不引入 muBLAS/muDNN，WMMA 内核吃 Tensor Core——`include/crt/mma.h` 真机验证 203 重载 mma_sync）**。M2–M6 中，M2（musart FFI）与 M3（内核族预编译）已由 P1b 覆盖；M4（muBLAS/muDNN）按终态设计**取消**，其 WMMA 替代路径已于 2026-10-17 **实现并真机验证**；M5（算子全量）与 M6（真机+基准+文档）已由 P1b 与 2026-10-17 WMMA 收尾覆盖。剩余为后续性能移植（B 预转置/异步双缓冲/CTA Swizzle）与 `rmsnormmatmul` 后端移植，均不在本次范围。


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

**结论（唯一最优，已被 §5.1 修订）**：**A1（mcc 预编译 + `mu*` driver）+ B1（muBLAS/muDNN）+ C2（submission 批处理）**。——B1 最终由 WMMA 自研 `matmul` 内核 + 自研 `softmax` 内核替代（`musa` 3→2，见 §5.1/§5.4）；本节其余分析保留为决策过程记录。
- 综合最优 = A1：零升级风险；稳态性能与 A2 等价；host 层比 A3 薄且为未来 MTRTC 同层接口（A1 ⊃ A3）。
- 性能最优 = B1×C2（Tensor Core 计算差 ~14× × 固定开销差 ~35×）；A1/A2 在性能上无差异，A2 无任何增益。
- 性能预估（外推，非实测；真实差距待 M6 基准）：最差路径（B2+C1）1× → 最优路径（B1+C2）预估 **15–30×**（固定开销 + 计算双杠杆）。
