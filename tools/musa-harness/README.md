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
