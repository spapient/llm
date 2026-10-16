# MUSA（摩尔线程）真机验证

验证 `crates/cuda/kernels/scan_parallel.cu` 的三段式并行扫描内核在摩尔线程
MTT S4000 上的数值正确性。内核源码逐字复用（仅两处语法适配：CUDA 内置
`min` 改为三元表达式、头文件改为 MUSA 对应项），host 用 MUSA runtime API
（`musaMalloc`/`musaMemcpy`/`musaLaunchKernel`）。

## 验证对象与结论

| 项目 | 结果 |
|---|---|
| 主机 | 远程 `sha1`（`connect.sha1.seetacloud.com`，Ubuntu 22.04 x86_64） |
| GPU | **MTT S4000**（摩尔线程第三代 MUSA，MP_22，48 GB，cc=2.2） |
| 工具链 | MUSA Toolkit 3.1.0（`mcc` = MTGPU clang-14），驱动 2.7.0，musart 运行时 |
| 用例 | n=1000/d=3/N=4（4 块非整除）、n=16384/d=256/N=64（64 块大序列）、n=128/d=8/N=8（2 块） |
| 容差 | abs 1e-3 / rel 1e-5（与 `cudacontract` 16K 用例同一契约，P·ε·‖B‖ 界） |
| 数值 | **SMOKE PASSED**：16K max|abs|=5.2e-6、非整除 max|abs|=1.9e-7、小用例 3.0e-7 |
| 性能 | 16K 三段并行含同步 59.2 ms（含首次 launch/JIT 开销；CPU 串行参考同规模约 90 ms 量级） |

## 编译与运行（在 MTT 主机上）

```bash
# 关键点：必须用 -mtgpu 进入 MUSA 路径，--offload-arch 与硬件匹配
# MTT S4000 = mp_22（MUSA 3.1.0 版本文档亦见 mp_21/mp_22 多架构编译）
/usr/local/musa-3.1.0/bin/mcc -mtgpu --offload-arch=mp_22 \
  -I/usr/local/musa-3.1.0/include -L/usr/local/musa-3.1.0/lib -lmusart \
  -o smoke_scan smoke_scan.cu
LD_LIBRARY_PATH=/usr/local/musa-3.1.0/lib ./smoke_scan
```

工具链就绪步骤（若新环境）：安装 MUSA Toolkit（含 mcc/musart/`mthreads-gmi`），
确认 `/dev/dri/cardN + renderD*` 存在、`mthreads-gmi` 能列出 MTT 设备。

## 与项目 CUDA 后端的关系（如实标注）

- **内核可移植性已验证**：三段式并行扫描（`scan_agg`/`scan_prefix`/`scan_apply`）
  的 CUDA C 源码经 `mcc -mtgpu` 直接编译并在 MTT S4000 数值正确。
- **cudarc 运行时未适配（OPEN）**：项目 `crates/cuda` 后端基于 `cudarc`
  （依赖 `libcuda.so` driver API + NVRTC 运行时编译）。MTT 主机上**无
  `libcuda.so`、无 NVRTC**，cudarc 无法直接运行；MUSA 提供的是
  `libmusa.so`/`libmusart.so`（MUSA 原生 API）与 `mcc`（编译期工具链）。
  在 MTT 上跑完整 `cudacontract` 需要 MUSA 后端适配（MUSA driver/runtime
  API 替代 cudarc 调用面 + `mcc` 预编译替代 NVRTC 运行时编译），列为独立阶段。

## 环境要点

- `mthreads-gmi`（摩尔线程系统管理工具）确认设备：MTT S4000，48 GB，MPC 可用。
- OpenCL 3.0（Moore Threads OpenCL 平台）亦可作为替代执行路径。
- MUSA 3.1.0 的 `device_launch_parameters.h` 定义 `threadIdx/blockIdx` 等，
  用 `-mtgpu` 模式时自动生效；`-nocudainc`/`--cuda-path` 的 clang CUDA 模式
  因缺 `cuda.h` 兼容头不可用（MUSA 未提供 CUDA 头文件兼容层）。
