# CUDA / Intel 核显内核配置候选档位对照（调研）

- 版本：v1.1
- 日期：2026-11-26
- 调研主机：Apple M5 Pro / macOS arm64 / rustc 1.100.0；win 实测 192.168.0.65（Windows 10.0.26300）
- 数据来源：联网厂商文档与公开调优指南（见末节「来源」）＋ 仓内设备档案 `crates/prof/src/nvidia/a3000.toml`、`crates/prof/src/intel/intel.toml` ＋ win 实测枚举读数
- 配套冻结口径：[`docs/adr/0056-cutile-prof-dsl-ir-spp.md`](adr/0056-cutile-prof-dsl-ir-spp.md) G1/G2/G3/G5

## 一、结论先行

1. 调研收敛出两平台的分块、线程/warp、流水线、向量化候选档位；仓内目前只登记了
   `[simt].group`（warp/subgroup 宽度）与族级 `block` 占位种子，**warp 数、stage 深度、
   向量宽度、work-group 多档位尚未进入候选空间**。
2. 真机验证现状：本机 M5 Pro 仅 Metal 后端；win 上 A3000（sm86）与 UHD750 双硬件在场、
   nvcc 13.3、MSVC 14.51 cl 与 Intel OpenCL/Level-Zero 运行时均可见，最小 CUDA 内核已实测
   编译并在 A3000 上回读正确；但 win 仓内快照落后本机、prof 产物错位，当前口径对拍用例未跑成，
   Intel 仓内真机对拍证据仍为空。
3. 本表是候选空间的**设计输入**，不代表已落地、可发射、可计时的配置。typed IR 设备执行
   （G2）、真机计时择优（G5）仍在冻结边界内；非 f32 档贯通属 G1。

## 二、CUDA（A3000 sm86 / Ampere）候选档位

A3000 实测硬件约束（`a3000.toml` 注释，2026-11-25 `cudaGetDeviceProperties`）：
32 SM、warpSize 32、1536 thread/SM、sharedMem/block 49152 B、regs/block 65536（32-bit）、
192-bit 显存总线、mem 时钟 7001 MHz、SM 最高 1665 MHz；工具链 nvcc 13.3。

| 算子族 | CTA / 分块 tile | 线程 / warp | 流水线 stages | 向量化 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 逐点 Elementwise | block 128 / 256 / 512（32 的倍数，256 默认） | warp 数 1 / 2 / 4 / 8 | 2–5 | 每线程 1 / 4 元素（按对齐） | occupancy 候选 1/2/4/8；逐点、归约起步取 4 |
| 归约 Reduce | BLOCK 取 2 的幂，64–4096；block 256 / 512 / 1024 | warp=32，warp 内归约 | 2–5 | 向量 1 / 4 | tile 大小受寄存器压力约束 |
| GEMM f32 | CTA `128×128×32`、`128×64×32`、`64×128×32`、`64×32×32`；小 M/N 走 split-K | num_warps 2 / 4 / 8（小 tile 4，大 tile 8） | 3 / 4 / 5 | 按 K 对齐取 4/8 | cluster{2,1}、rasterization 属更后级旋钮 |
| GEMM BF16/FP16（参考，未落地） | `128×256×64`、`64×256×32` 等 | 大 tile 配 8 warps | 3–4 | — | 依赖 tensor core；本仓 G1 未贯通 |

要点：block 大小没有通用最优解，按 32 倍数起步、256 默认，用 occupancy（活跃 warp 占比）
与寄存器压力折中；GEMM 元参数含 CTA/warp/指令三级 tile、stages、split-K、cluster，先用
启发式把数千组合剪到小集合再实测，而非暴力全空间搜索。

## 三、Intel Xe-LP（UHD750 gen12）候选档位

UHD750 实测约束（`intel.toml`）：32 EU、subgroup 16、`groups_max=32`、SLM 64 KiB/16 bank、
REG 32 KiB、OpenCL 2.0、tools clang 17.0；f32 峰值 665.6 GFLOPS、带宽 51.2 GB/s。

| 算子族 | work-group 尺寸 | subgroup（SIMD） | 向量 / tile | 备注 |
| --- | --- | --- | --- | --- |
| 逐点 / 向量 | 128 / 256 / 384 / 512（Xe-LP 上限 512；Arc/PVC 可到 1024） | 8 / 16 / 32（Xe 常用 16） | SIMD8 向量；向量化 8 时线程上限约 64 | 线程上下文优先于 SIMD 向量；sg 加宽类同展开，代价是延迟与寄存器压力 |
| 归约 | 256 / 512 | 16 | 线程数 × sg 宽度受 112 thread-contexts/EU 约束 | wg 内工作项必须落在同一 Xe-core |
| GEMM BF16/FP16（参考，未落地） | — | 显式 `sub_group_size<16>`，错配会静默出错 | 起步 `256×256×32`；寄存器溢出再降 M/N；memory-bound 加深 K-tile | f32 退回 16×16 / 32×32 小 tile |

要点：Xe-core 每核 112 个 thread context；向量化 8 时每 wg 线程上限 64（512/8）；subgroup
尺寸是硬件约束，必须用设备查询 `sub_group_sizes` 确认，不能写死；work-group 超过设备上限、
或全局尺寸不能被 wg 整除会直接启动失败。

## 四、映射到仓内现有轴

| 配置轴 | 仓内现状（位置） | 调研建议候选 | 缺口归属 |
| --- | --- | --- | --- |
| warp / subgroup 宽度 | `[simt].group`：nvidia=32、intel=16 | 固定宽度已登记，不参与遍历 | 无缺口 |
| block / work-group | `GeometrySeed.block=[256,1,1]` 族级占位；`TileCombo` 分块轴来自算子声明 | CUDA 128/256/512；Intel 256/384/512 | 候选空间扩充，待 P3/G5 |
| 流水线 stages | 仅 `load_async` 能力行固定 `1`；`pipe<2>/<3>` 为调度侧建模，不进能力签名 | 2/3/4/5 | stage 轴未进 IR/签名（G2/G5） |
| 向量化宽度 | P4 主机侧档位已收口，设备发射未贯通 | 向量 1/4/8 | G1 设备档 + G2 发射 |
| 真机择优 | `autotune` opt-in 骨架，默认关、无入口、不落盘 | 启发式剪枝后小集合实测取最优 | G5，依赖 G3 真机计时 |

## 五、双机验证环境盘点

### 本机（Apple M5 Pro, macOS arm64）

- GPU：Apple M5 Pro，Metal 4；无 NVIDIA、无 Intel 核显；`nvcc`、`/usr/local/cuda`、
  OpenCL ICD 均不存在。
- 可跑：host C 与 Metal 语法门 / 逐点族真机对拍；**CUDA、Intel OpenCL 档无法在本机验证**。

### SSH `win`（192.168.0.65，user fei，Windows 10.0.26300 / i7-11700）

2026-11-26 上线实测：

| 项 | 读数 |
| --- | --- |
| NVIDIA | RTX A3000 12GB Laptop GPU，驱动 621.14，**sm 8.6**，12288 MiB |
| Intel | UHD Graphics 750（gen12），驱动 32.0.101.7088 |
| Rust | rustc / cargo **1.100.0-nightly**（符合固定版本） |
| CUDA 工具链 | nvcc **13.3 V13.3.33**；MSVC **在位**：VS18 BuildTools `VC\Tools\MSVC\14.51.36231` cl 19.51.36256（`Hostx64\x64\cl.exe`，vswhere `-latest` 命中），另 `D:\VSBuildTools\MSVC\14.44.35207`；`D:\BuildTools` 为仅含 LLVM format/tidy 的残壳 |
| Intel 运行时 | oneAPI 2026.0；`sycl-ls` 枚举到 `level_zero:0` UHD750 12.1.0 [1.6.34728]、`opencl:0` UHD750 **OpenCL 3.0 NEO** [32.0.101.7088]，另有两条 CPU OpenCL |

CUDA 链实测（最小 `hello.cu`，sm_86）：默认 ssh 环境下 `nvcc -ccbin <14.51 cl>` 直编 **EXIT=0**（无需预
source vcvarsall）；产物在 A3000 运行回读 `1.0`、退出码 0，设备内核执行 + 回读正确。前序"MSVC 缺失"为
漏查 D 盘两个 VS 实例导致的误判。

历史证据：ADR 0056 G 段记录此前 CUDA prof 见证层 18/18 真机对拍闭合、`max_rel_err` 全 0、`scope=Device`；
本轮最小内核已复核编译-执行链通，仓内见证对拍待代码同步后复跑。

仓内快照状态（`C:\Users\fei\llm-sync`、`llm-a3000-91199ce0`、`llm-intel`）：均非 git 跟踪，
停在 P1/P2 阶段（含 `witness_tests.rs` 通道，但无本轮 P3 `geometry_seed`）；`D:\llm` 为更旧的
分片调度快照。`llm-a3000` 当前 prof 内嵌生成产物仍引用旧 `ProfileView`，与已并库的 prof 源码错位，
`cargo test` 编译断裂——快照落后本机，不能作为当前口径验证基线。

仓内可用性判定（本机代码口径）：

- CUDA：`nvcc_available()` 要求 nvcc **且** Windows 上 `find_cl()` 命中；`find_cl` 经 vswhere
  `-products * -find VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe` 命中 14.51 cl（文件校验通过），判定**可用**。
- OpenCL：`OpenClDevice::probe()` 枚举设备名含 `Intel` 即选中；运行时在场，代码同步后应命中
  UHD750（注意 CPU OpenCL 设备名也含 "Intel"，平台/设备顺序需在接通时核对）。

### 仓内通道限制

五个 Emit 前端的语法门均为 `-fsyntax-only`，CUDA 另带 `--cuda-host-only -nocudainc
-nocudalib`；设备真编译/launch 走独立 runner（`CudaBackend`，win 上历史已跑通见证层）。
typed IR 接设备执行仍属 G2；在该通道贯通前，仓内新配置只能过语法门与 host C 对拍，出不了
两平台设备性能数字。

## 六、剩余阻塞与验证步骤

1. **代码同步**：把本机当前工作区同步到 win（三份快照均落后本机若干代，prof 生成产物与源码已错位、
   对拍用例名/结构不一致）；同步后再跑当前口径用例。
2. **跑当前口径对拍**：`cargo test -p llm-shard --test backends` 下的
   `every_nvidia_witness_runs_on_the_cuda_device`（CUDA，带 skip 门控）与
   `every_intel_witness_runs_on_uhd750`（OpenCL，无 skip，探测不到即红）。MSVC/nvcc 编译链
   本轮已用最小内核复核通过，无需再补宿主编译器。
4. 边界：见证层对拍 ≠ typed IR 设备执行；无 roofline 读数前不签 L3 性能档；MUSA/Ascend/
   ROCm 维持无证据口径。

## 七、来源

联网调研：

- NVIDIA GTC25 S72575《How You Should Write a CUDA C++ Kernel》：block 取 32 倍数、256 默认、occupancy 与寄存器折中。<https://www.nvidia.com/en-us/on-demand/session/gtc25-s72575/>
- NVIDIA 博客《Improving GEMM Kernel Auto-Tuning Efficiency with Heuristics and CUTLASS 4.2》：CTA/warp/instr tile、stages、split-K、cluster；启发式剪枝。<https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/>
- CUTLASS 文档《Efficient GEMM in CUDA》：threadblock/warp tile、sliced-k。<https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html>
- Triton 教程 03 Matrix Multiplication：8 组 `BLOCK_M/N/K`、`num_stages`、`num_warps` 候选。<https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
- Triton TileIR Performance Tuning Tips：occupancy 1/2/4/8、逐点/归约起步 4、stages 3/4/6。<http://www.shturl.cc/414903c22390409cce2c8de14b57b5fa>
- Triton block 参数调优 skill：归约 BLOCK 64–4096 幂次、stages 2–5、warps 1/2/4/8、smem 上限。<http://www.shturl.cc/022163a210dd8993ca78c38be9b0ae7c>
- oneAPI GPU Optimization Guide（2025.0）：112 thread contexts/EU、向量 8 时 64 线程、sg 8/16/32、wg 512。<https://cdrdv2-public.intel.com/834474/oneapi_optimization-guide-gpu_2025.0-771772-834474.pdf>
- Intel《Thread Mapping and GPU Occupancy》：work-group/subgroup 选择口径。<https://www.intel.co.uk/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/thread-mapping.html>
- Intel《Sub-Groups and SIMD Vectorization》：subgroup 8/16/32、设备查询口径。<https://www.intel.sg/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/sub-group.html>
- oneAPI GPU Optimization Guide（2025.2）：wg 128/384/1024 占用率表。<https://cdrdv2-public.intel.com/859422/oneapi_optimization-guide-gpu_2025.2-771772-859422.pdf>
- sycl-tla Intel Performance Guide：Xe subgroup 16、GEMM tile 256×256×32、K-tile 加深。<http://www.shturl.cc/641baa115696e4ac182483a1c8680514>
- Intel oneAPI《Test and Troubleshoot Your Installation》：`clinfo -l` 枚举 OpenCL 设备。<https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-hpc-cluster/2023-2/test-and-troubleshoot-your-installation.html>

仓内档案与证据：

- `crates/prof/src/nvidia/a3000.toml`（sm86 实测参数）
- `crates/prof/src/intel/intel.toml`（UHD750 OpenCL 档案）
- `docs/evidence/cuda-backward-probes/README.md`（A3000 独立探针对拍记录）
- `docs/adr/0056-cutile-prof-dsl-ir-spp.md`（G1/G2/G3/G5 冻结口径）
