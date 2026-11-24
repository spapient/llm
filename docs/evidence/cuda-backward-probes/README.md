# CUDA 设备反向梯度真机对拍探测（A3000）

本目录是设备反向通路在 CUDA 真机上的**独立见证程序**：每个 `.cu` 自带主机装配
（`cudaMalloc`/拷贝/发射/回读）与 f64 参考（中心差分或解析公式），不依赖仓内 Rust 执行器。
用途是在 win 全仓工作区同步完成前，先把各算子族设备反向核的数值正确性钉在真 GPU 上；
仓内 `device::backward` 规约通道契约、双路发射器与门控 Rust 测试仍为待接切片（见末节）。

## 环境

- 设备：`NVIDIA RTX A3000 12GB Laptop GPU`（UUID `GPU-fc44195b-6612-39b9-bc3f-f88ea2c737f3`）。
- 工具链：nvcc `CUDA 13.3 V13.3.33`；主机 MSVC `14.44.35207`；非交互 ssh 会话需先
  `call vcvarsall.bat x64` 再 `nvcc -ccbin cl.exe`（`INCLUDE`/`LIB` 默认空）。
- 容差：相对误差阈值 `5e-2`，分母地板 `1e-3`；参考在 f64 求值（逐点族中心差分步长 `1e-6`）。

## 结果（2026-11-24，A3000 真机回读）

| 探测 | 算子族 | 形状 | 梯度输出 | max rel |
| --- | --- | --- | --- | --- |
| `pointwise.cu` | sigmoid | N=96 | dx | 0.00000 |
| `pointwise.cu` | tanh | N=96 | dx | 0.00001 |
| `pointwise.cu` | silu | N=96 | dx | 0.00000 |
| `pointwise.cu` | gelu | N=96 | dx | 0.00005 |
| `rmsnorm.cu` | RMSNorm | N=96 | dx | 0.00000 |
| `rmsnorm.cu` | RMSNorm | N=96 | dγ | 0.00005 |
| `matmul.cu` | gemv | R=6, C=8 | dW | 0.00002 |
| `matmul.cu` | gemv | R=6, C=8 | dx | 0.00001 |
| `matmul.cu` | gemm | M=4, K=5, N=6 | dA | 0.00001 |
| `matmul.cu` | gemm | M=4, K=5, N=6 | dB | 0.00001 |
| `conv2d.cu` | conv2d | 4×4×2 → 3，K3/S1/P1，16 输出格 | dW | 0.00001 |
| `conv2d.cu` | conv2d | 同上 | dbias | 0.00014 |
| `conv2d.cu` | conv2d | 同上 | dcol | 0.00001 |
| `conv2d.cu` | conv2d | 同上 | dx | 0.00002 |
| `attention.cu` | 稠密 attention | Q=8, hist=8, d=8 | dQ | 0.00000 |
| `attention.cu` | 稠密 attention | 同上 | dK | 0.00000 |
| `attention.cu` | 稠密 attention | 同上 | dV | 0.00001 |

全部 `< 5e-2` 判定 PASS；实测量级在 `1e-5` 以内（conv dbias 最差 `1.4e-4`）。

## 公式口径（与 `train/src/backprop/formulas.rs` 同式）

- 逐点：sigmoid `y(1−y)`、tanh `1−y²`、silu `σ+xσ(1−σ)`、gelu `Φ(x)+x·φ(x)`；
  erf 走 A&S 7.1.26 多项式（MSL/CUDA 均无 `erf`/`erff` 内建依赖）。
- RMSNorm：`rms=√(mean(x²)+ε)`，`ε=1e-6`；
  `dx[j]=g[j]γ[j]/r − x[j]·weighted/(n·r³)`，`dγ[j]=g[j]x[j]/r`，`weighted=Σ gγx`。
- gemv `y=Wx`：`dW[n,k]=dout[n]x[k]`，`dx[k]=Σ_n dout[n]W[n,k]`。
- gemm `Y=AB`：`dA[m,k]+=g[m,n]B[k,n]`，`dB[k,n]+=A[m,k]g[m,n]`。
- conv2d：im2col 展开；`dW+=colᵀ·dy`，`dbias=Σ dy`，`dcol=dy·Wᵀ`，col2im 累加回 `dx`
  （重叠位用 `atomicAdd`）。权重按 patch 主序 `w[patch·cout+o]`。
- attention：`p=softmax(QKᵀ/√d)`，`dP=dO·V`，`dS=p(dP−Σ p·dP)`；
  `dV+=p·dO`，`dQ+=dS·K/√d`，`dK+=dS·Q/√d`。

## 待接仓内通道（非本探测覆盖）

- 仓内 `backward.rs` 现为四槽定长单输出逐点契约；规约族（归一化/矩阵乘/卷积/注意力）需要
  多输出、保存统计量、两趟规约与散射累加的新契约通道，Metal/CUDA 双路发射器尚未落地。
- 上述探测为独立 `.cu` + nvcc 回读回路；仓内门控 Rust 测试（复用 `compile_run_cuda`
  Driver/`device_available` 门控）要等 win 全仓工作区可编译后接入。
- layernorm、latent/decode 注意力变体未单列探测。
