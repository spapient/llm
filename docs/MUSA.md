# MUSA 真机测量通道

S4000（MTT S4000 / MUSA 5.2 / muBLAS 5.2）上的测量记录与复现要点。**代码**在
[`crates/musa`](../crates/musa)：一个 `musa` 二进制加三条裸探针；本文件只放结论与工具链事实。

## 四段命令

| 段 | 命令 | 做什么 |
|---|---|---|
| 生成 | `musa gen <op> [m] [n] --out DIR` | 声明 → 协商 → 规划 → 设备源 + 标量参考（不需要 MUSA） |
| 编译 | `musa build <src.cu> --out BIN [--root DIR] [--lane plain\|mublas]` | `mcc` 一次编译，转发头随源落地 |
| 计时 | `musa bench <src.cu> --out SRC [--reps N]` | 夹住「首个发射 → 末次回读」，`cudaEvent` 取 100 次均值 |
| 对拍 | `musa verify <打印> <参考> [--tol T]` | 逐元素相对误差，一个超阈即失败（退出码 2） |

`musa run <op> [m] [n]` 把四段按数据流串起来：产物默认落临时目录、跑完删除，`--keep DIR` 才留。

## 工具链三条要点

`MCC=/usr/local/musa/bin/mcc`（clang-14）；完整头在 `/usr/local/musa-5.2.0/include`
（`/usr/local/musa/include` 的 `mublas.h` 是残缺包装）。

1. `precision::tf32` 只在 `-D__MUSA_QY2_MMA__=1` 下声明；
2. S4000（QY2/mp_22）的 tf32 MMA 是 **m16n16k8**，`fragment<32,32,8>` 是 **wave128 形态**
   （整个 CTA 128 线程共持，每 warp 只持象限）——16×16×8 片段在该工具链上数值错误；
3. 默认 target mp_21 无 MMA，WMMA 必须 `--cuda-gpu-arch=mp_22`。

`crates/musa/assets/shim` 的两个转发头（`cuda_runtime.h` → `musa_runtime`、`cuda_fp16.h` →
`musa_fp16.h`）就够跨工具链：框架产出的是 CUDA 方言，MUSA 编译器只差这两个名字映射。

## 精度口径

| 对象 | 短 K norm | 长 K（14336）norm | 结论 |
|---|---:|---:|---|
| muBLAS 5.2 | 1.56e-6 | 4.23e-6 | 纯 FP32 累加，长 K 只正常增长 2.7× |
| WMMA 纯 tf32 | 2.79e-4 | — | tf32 级，不满足 `_f32` 契约 |
| 3xTF32 | 1.23e-5 | 1.64e-4（13× 超线性） | 主场景出局 |

`gemm_f32` 的 f32 契约只有 FP32 SIMT 路径满足；WMMA 按 tf32 预算 `norm < 1e-3` 验收。

## 三路对照（2026-11-19，TFLOPS）

| 路线 | cube | prefill | n1536 | n768 |
|---|---:|---:|---:|---:|
| muBLAS 5.2 | 6.55 | 14.40 | 17.77 | 15.64 |
| WMMA v8 纯 tf32 | 10.60 | 13.24 | 10.18 | 8.61 |
| FP32 SIMT v4 | 6.32 | 6.74 | 9.81 | 9.68 |

已穷尽的方向：双缓冲、grid 交换、`launch_bounds`、BK8/12/16、16 累加器、3xTF32、`fast_math`、
`fmaf`、多 tile。硬件边界：S4000（mp_22）**无 `cp.async`**（SINCE_MP_43）；5.2 工具链的 SIMT
调度相对旧树 3.1 退化（prefill 10.74 无法复现）。

## 探针

`musa list` 打印这三条。它们测的是**硬件与工具链**，与框架数字只做数量级对照。

| 探针 | 测什么 |
|---|---|
| `fma` | FP32 FMA 峰值（`fmaf` 乘加链，单次发射） |
| `simt-gemm` | 手写 128×128 分块 FP32 GEMM（寄存器块 + 共享内存） |
| `mublas` | muBLAS FP32 GEMM（厂商库对照） |

## 框架产出的真机对拍（2026-11-20 记录）

以下算子经同一通道（生成 → `mcc` → 运行 → 逐元素对拍）在 S4000 上通过：
`silu` / `gelu` / `add` / `mul`（逐点）、`rmsnorm` / `softmax`（两段/三段分块，4 块折叠）、
`gemv`（一维循环注入，逐位相同）、`gemm`（二维循环注入 16×16，2.79e-10）、`scan`（分块扫描 4 块，
2.9e-10）、`gemm` WMMA（64×64×8，tf32）与 WMMA 非整除守卫（100×70×8）。

框架侧已无 `Unsupported` 骨架：逐点、分块归约、分块归一化、循环注入、分块扫描、WMMA 特化全部落地。
两种垫片头即可让框架的设备源在 MUSA 上起跑——这也是「第三后端」的接入点所在：不在 crate 结构，
而在**编译器与两个转发头**。

## 与生态做法的对照（2026-11-21 调研）

搬运时拿两条外部惯例核对过，结论是"跟主流一致，且省掉了最多代码"：

| 外部做法 | 出处 | 本通道的选择 |
|---|---|---|
| 每个样例自成一个可执行程序，自带 `main` 与运行说明，没有 shell 编排层 | `NVIDIA/cuda-samples` README 的样例目录约定 | 四段收进**一个** `musa` 二进制，`run` 按数据流串起来；不再有 `build.sh` 这类胶水 |
| CUDA 源码靠**名字翻译**（`cuda*` → `hip*`）迁到别的厂商栈，而不是改生成器 | `ROCm/HIP` README 与 `ROCm/HIPIFY`（"tools to translate CUDA source into portable HIP C++") | 生成器只产出 CUDA 方言，由两个转发头完成映射——**一处映射**胜过在每个后端里写一遍方言分支 |

两条都指向同一个判断：减少代码的地方不在"多写一层抽象"，而在**去掉重复的入口与重复的方言分支**。
