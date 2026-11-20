# MUSA S4000 真机测量 harness

2026-11-19/20 在 autodl 容器(sha1,MTT S4000 / MUSA 5.2.0 / muBLAS 5.2)上建立的测量通道。
全部数据与结论见记忆 `musa-5.2-measurements`(会话外)与 [../docs/adr/0049-refactor-summary-2026-11.md](../docs/adr/0049-refactor-summary-2026-11.md)。

## 编译(5.2 工具链要点)

```bash
MCC=/usr/local/musa/bin/mcc        # → clang-14
INC=/usr/local/musa-5.2.0/include  # 完整头文件在这里(/usr/local/musa/include 的 mublas.h 是残缺包装)
# 普通 CUDA 语法 kernel:
$MCC -O3 -x musa <file>.cu -o <out> -L/usr/local/musa/lib -lmusart
# WMMA(tf32):
$MCC -O3 -x musa -D__MUSA_QY2_MMA__=1 --cuda-gpu-arch=mp_22 <file>.cu \
     -o <out> -I$INC -L/usr/local/musa/lib -lmusart
# muBLAS 5.2(handle 风格,与 cuBLAS v2 同构):
$MCC -O3 -x musa t_mublas.cu -o t_mublas -I$INC -L/usr/local/musa/lib -lmusart -lmublas
```

三个编译坑:① `precision::tf32` 只在 `__MUSA_QY2_MMA__` 宏下声明;② S4000(QY2/mp_22)的
tf32 MMA 是 **m16n16k8** 片段且 `fragment<32,32,8>` 是 **wave128 形态**(整个 CTA 128 线程
共持,每 warp 只持象限——16×16×8 片段的 mma 在该工具链上数值错误);③ 默认 target mp_21 无
MMA,必须 `--cuda-gpu-arch=mp_22`。tf32 转换内建名 `__musa_float_to_tf32`。

## 文件

| 文件 | 内容 |
|---|---|
| `t_wmma_v8.cu` | **最优正确形态**:64×64 四单元 wave128、BK8 单缓冲、10.60/13.24/10.18/8.61 TF |
| `t_debug7.cu` | 32×32 最小正确例(0/1024 错),wave128 用法参考 |
| `t_wmma_3xtf32.cu` | 3xTF32(精度达标 norm 1.23e-5,性能 5.8-6.8) |
| `t_simt_db.cu` / `t_simt_fma.cu` / `t_simt_opt.cu` | FP32 SIMT 变体(BK8 双缓冲 / fmaf / swap+LB;最优 v4 在 t_musa.cu 族) |
| `t_musa.cu` | FP32 SIMT v4(128×128/8acc/BK16 单缓冲,6.32/6.74/9.81/9.68) |
| `t_mublas.cu` | muBLAS 5.2 对照(6.55/14.40/17.77/15.64 @ FP32 级精度) |
| `t_accuracy2.cu` / `t_accuracy3.cu` | 精度对拍 harness(v8 纯 tf32 / 3xTF32 vs muBLAS vs f64 参考) |
| `t_peak.cu` / `t_diag.cu` / `t_barrier.cu` | FMA 峰值 21 TF / 载入-mma 分段诊断 / 屏障成本 |

## 终局数据矩阵(2026-11-19)

| 路线 | cube | prefill | n1536 | n768 | 精度 norm |
|---|---:|---:|---:|---:|---:|
| muBLAS 5.2 | 6.55 | 14.40 | 17.77 | 15.64 | 1.56e-6 |
| WMMA v8 纯 tf32 | 10.60 | 13.24 | 10.18 | 8.61 | 2.79e-4 |
| FP32 SIMT v4 | 6.32 | 6.74 | 9.81 | 9.68 | FP32 级 |

已穷尽的方向:双缓冲、grid 交换、launch_bounds、BK8/12/16、16 累加器、3xTF32、fast_math、
fmaf、多 tile。硬件边界:S4000(mp_22)**无 cp.async**(SINCE_MP_43);5.2 工具链 SIMT 调度
退化(旧树 3.1 的 prefill 10.74 无法复现)。

## 长 K 精度(2026-11-20 补测,K=14336)

| 对象 | 短 K norm | 长 K norm | 长 K max_abs | 结论 |
|---|---:|---:|---:|---|
| muBLAS 5.2 | 1.56e-6 | 4.23e-6 | 7.71e-4 ✓ | 2.7× 正常 FP32 累加增长——**纯 FP32 路径** |
| 3xTF32 | 1.23e-5 | 1.64e-4 | 2.98e-2 ✗ | 13× 超线性——**主场景出局** |

主场景(14336 长度)精度合格的唯一路线 = FP32 SIMT v4(全精度)。


## 框架产出 → MUSA(2026-11-20,里程碑)

本仓框架(cutile 流水线 + engine CUDA codegen)产出的 GPU 程序,经 mcc 编译后在 S4000 真机
执行,与框架标量参考**逐位对拍通过**——首次证明「框架 → MUSA」通道存在且成本极低。

### 通道

1. **生成**(Windows,有 nvcc 使能 nvidia 设备协商):`gen-cuda/` 调公开 API
   `cutile::execution::cuda_grid_program`(设备源)+ `engine::codegen::cuda_grid_driver`(host 驱动)
   → 单文件 + `(model.reference)` 标量参考。
   - 注意:生成器 Cargo.toml 末尾的 `[workspace]` 是必需的——win 家目录存在旧版
     `C:\Users\fei\Cargo.toml` workspace 根,不加会被误当成员。
   - 用法:`cargo run -- <op> [m] [n]`(默认 64×1)。
2. **编译**(S4000):`mcc -O3 -x musa -Ishim fw.cu -o fw -L/usr/local/musa/lib -lmusart`。
   `cuda-shim/` 提供两个转发头:`cuda_runtime.h`(cuda→musa 的 API/类型/错误码宏)、
   `cuda_fp16.h`(→ `musa_fp16.h`)。**仅需这两个垫片**。
3. **对拍**:程序打印 f32 位模式,`check_bits.py` 与参考逐项比较。

### 覆盖与结果

| 算子 | 形态 | 结果 | max_rel |
|---|---|---|---|
| `silu_f32` | 逐点 | OK | 1.15e-07 |
| `gelu_f32` | 逐点 | OK | 3.86e-07 |
| `add_f32` | 逐点 | OK | 0 |
| `mul_f32` | 逐点 | OK | 0 |
| `rmsnorm_f32` | 两段分块(4 块折叠) | OK(1024 元素) | 1.17e-07 |
| `softmax_f32` | 三段分块(4 块折叠) | OK(1024 元素) | 1.19e-07 |
| `gemv_f32` | 一维循环注入(`[grid] dims=["n:N"]`) | OK(64×64) | **0(逐位)** |
| `gemm_f32` | 二维循环注入(`[grid] dims=["i:M","j:N"]`) | OK(16×16) | 2.79e-10 |
| `scan_f32` | 分块扫描(局部 + 折叠 + 校正,4 块) | OK(64 元素) | 2.9e-10 |
| `gemm_f32`(WMMA) | 张量核特化(64×64 块、BK8、4 片段) | OK(64×64×8) | tf32: max_abs 3.18e-4 / norm 3.02e-4 |
| `gemm_f32`(WMMA,非整除) | 守卫路径(100×70×8) | OK | tf32: norm 4.88e-4 |

归一化由 `engine::norm::cuda_rmsnorm_driver` / `cuda_softmax_driver` 把多段设备源与 host
串联驱动拼成单文件(部分和/最大值回读到 CPU 折叠后再派发);`gen-cuda` 已内置这条分支。
**框架侧已无 `Unsupported` 骨架**:逐点、分块归约、分块归一化(两段/三段)、循环注入(gemv/gemm)、
分块扫描、WMMA gemm 特化全部落地。

- 循环注入:设备源 `cuda_gemm_program`(含各维网格)+ `engine::codegen::cuda_loop_driver`。
- 分块扫描:设备源 `cuda_scan_program`(两段 + decay)+ `engine::scan::cuda_scan_driver`。
- WMMA 特化:`cuda_wmma_gemm_program` + `engine::wmma::gemm_driver`;形状在生成期写死,`M/N` 整除
  64 时无守卫,否则载入按行列守卫、写出经共享内存 staging 拷回。编译需
  `-D__MUSA_QY2_MMA__`(设备源内已定义)与 `--cuda-gpu-arch=mp_22 -I/usr/local/musa-5.2.0/include`。
- **精度口径**:WMMA 是 tf32 级(与裸 v8 的 2.79e-4 一致),不满足 `gemm_f32` 的 f32 契约;
  按 tf32 预算 norm<1e-3 验收。要 f32 契约需 3xTF32(长 K 亦不达标,见研究线)。

### 意义

- 框架生成的设备代码与 MUSA 的 CUDA 方言二进制兼容;两种垫片头即可跨工具链。
- 这是「第三后端」落地的可行性证据:接入点不在 crate 结构,而在**编译器与两个转发头**。
- 与裸 harness 线(上方数据矩阵)互补:那条线探 kernel 性能上限,这条线证框架链路可用。


## 算子性能(框架产出,S4000)

| 优化 | 前后 | 口径 |
|---|---|---|
| 归一化段并行化(块内串行 → 逐元素网格) | rmsnorm 0.2847 → **0.2215 ms/iter(-22.2%)** | 1024 元素、2000 次迭代、cudaEvent 包住 dispatch 循环 |

计时方法:在生成程序的 `int G = ` 前插入事件与迭代循环,回读前收尾打印;垫片补 `cudaEvent*` 映射。
逐点算子已是每线程一元素;gemm/gemv/attention 的 CUDA lowering 尚未落地(研究线 WMMA v8 为裸 kernel)。


## 全算子性能基线(S4000,框架产出,2026-11-20)

计时口径:`bench_patch.py` 在生成程序里包住「首个发射 → 最后一次回读」整条链路(含主机折叠),
`cudaEvent` 计 100–200 次迭代均值。小形状下 launch/回读固定开销占主导。

| 算子 | 形状 | 基线 ms/iter | 调优后 |
|---|---|---|---|
| silu_f32(逐点) | 16384 | 0.0975 | —(每线程 1 元素,已满并行) |
| gemv_f32(一维注入) | 256×256 | 0.0920 | — |
| reduce_sum_f32(分块) | 16384 | 0.1201 | **0.0923(chunk 256→32,-23%)** |
| scan_f32(两段+折叠) | 16384 | 0.4071 | 保持 256(chunk 小反而更差) |
| rmsnorm_f32(两段) | 16384 | 0.2355 | 保持 256(同上) |

调优结论:分块粒度必须按形态分别验证——归约的 host 折叠是纯加法,O(G) 便宜、GPU 并行收益为主;
scan/rmsnorm 的折叠含 `a^len` 逐次乘法与整段回读,O(G) 成本超过收益。
