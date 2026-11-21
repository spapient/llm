# 数值算子三件套（定义 / 复杂度 / 误差界）

> 每一个数值算子在这里交齐三件套：**定义**（输入域、输出域、代数性质）、**复杂度**（紧界 + 达到界的构造）、**误差界**（前向误差或实测容差，指向钉住它的测试）。位级一致性单独列一栏，指向用 `to_bits()` 逐位对拍的用例。
>
> 符号：`u` 为单元舍入（f32 `u = 2^-24 ≈ 5.96e-8`）；`γ_n = n·u / (1 − n·u)`。表中「实测界」是测试里真实 assert 的容差，不是外推值。

| 算子 | 定义 | 复杂度（紧界 + 构造） | 误差界 | 位级用例 |
| --- | --- | --- | --- | --- |
| **GEMM f32**（`crates/gemm`） | `C[M,N] = A[M,K]·B[K,N]`；输入域 `M,N,K ≥ 1`，输出域 `M×N`；代数性质：双线性、对分块可分、按行/列尺度不变（`α·(AB) = (αA)B`）。 | 时间 `Θ(MNK)` 乘加（直接/tiled 路径；代数下界 `Ω(MNK)`，Strassen/Winograd 未采用）；tiled 版 I/O 达 Hong–Kung 下界 `Θ(MNK/√Z)`（`Z` = 缓存字）。空间 `Θ(MK+KN+MN)` + 打包缓冲 `Θ(k_c·n_c)`。 | 每个输出是长 `K` 的内积：前向误差 `≤ γ_K·Σ|a_i b_i|`。`K=512` 时 `γ_512 ≈ 3.05e-5`，与实测容差一致 —— `packed_test_support.rs` assert `1e-5`（GEMM）、`1e-6`（反量化点积）。tiled 求和重排不改变逐内积的误差界（每个内积内部顺序不变）。 | 见 `crates/dtype/tests/*.rs` 的 `to_bits()` 对拍（标量/位级地板）。 |
| **GEMM i8**（`crates/gemm`） | `C = A_i8·B_i8` 整数累加后按 scale 反量化；输入域 i8 量化张量，输出域 f32 或 i32 累加器。 | 同上 `Θ(MNK)` 整数 MAC，无浮点舍入直到反量化一步。空间同 f32 且打包缓冲减半。 | 整数累加**精确**（无舍入），误差全部来自末尾一次反量化：`gemv_i8.rs` / `packed_test_support.rs` assert `1e-6`。 | `gemm/tests/dequant_i8_tests.rs` 对拍 i8 路径与标量参考。 |
| **GEMV f32 decode**（`crates/gemm`） | `y[M] = W[M,K]·x[K]`（`K=4096` 权重侧）；每输出一个长内积。 | 时间 `Θ(MK)`；**带宽界**：算术强度 `2/(4·1)=0.5 FLOP/byte`，落在 roofline 的 DRAM 区 —— `gemv_f32/decode_1x4096x4096` 实测 `72.65 GiB/s` = 内存读上限 `83.68 GiB/s` 的 `86.8%`。空间 `Θ(MK+K+M)`。 | 同 GEMM，`γ_K·Σ|w_i x_i|`；`gemv.rs` 与标量参考 assert `1e-5` 量级。 | —（GEMV 单链无 SIMD 特化位级断言，见开口清单）。 |
| **GEMV i8 量化**（`crates/gemm`） | 反量化折叠进点积：`Σ w_i·x_i · (s_w·s_x)`，scale 乘到累加器末尾一次。 | 时间 `Θ(MK)` 整数 MAC + 一次反量化；带宽界（i8 强度翻倍为 `1 FLOP/byte`）。 | 折叠是**严格代数等价**（scale 为常数可提公因子）；`gemv_i8.rs` assert `1e-6`。 | — |
| **Softmax**（`crates/gemm`） | `softmax(x)_i = exp(x_i − m) / Σ_j exp(x_j − m)`；输入域任意 f32 行，输出域和为 1 的分布；性质：平移不变、单调、正值。 | 时间 `Θ(n)`（max 一遍、exp 求和一遍、归一化一遍）；空间 `Θ(n)`，在线重标定版 `Θ(1)` 额外。 | **max 相减必要性**：`exp(x)` 在 `x>88.7` 溢出 f32，减去 `m=max(x)` 保证 `x_i−m ≤ 0`，`exp ≤ 1`，和永不为 0。误差：`softmax.rs` assert 每行和 `=1` 容差 `1e-5`（exp 近似 + 求和重排的总效应）。 | — |
| **RMSNorm**（`crates/gemm`） | `y = x / rms(x) · γ`，`rms = sqrt(mean(x²))`；性质：尺度不变（`x→cx` 输出不变）。 | 时间 `Θ(n)`（平方和一遍、归一化一遍）；空间 `Θ(1)` 额外。 | rsqrt 近似 + 平方求和：`rmsnorm.rs` assert 全零行输出 `→0` 容差 `1e-6`；`norm_tests.rs` 组均值 `1e-5`、组方差 `1e-4`。 | — |
| **Attention（flash/tiled）**（`crates/conv`） | `Attn = softmax(QKᵀ/√d)·V`；GQA 共享 KV 头。 | naive 空间 `Θ(n²)`（分数矩阵物化）；flash/tiled 空间 `Θ(n·d)`（分块 + 在线 softmax 重标定，不物化分数）。时间都是 `Θ(n²·d)`；tiled 访存量是 naive 的 `Θ(d/n)` 比。 | 在线重标定：每块用局部 max 重标定、块间乘 `exp(m_old − m_new)`，误差上界与 softmax 同阶。`attention_tests.rs` `approx_eq(flash, plain, 1e-5)`；实测 tiled 加速 `1.76×`、flash decode `2.30×`（误差界不松）。 | — |
| **Conv**（`crates/conv`） | direct `Θ(C_in·C_out·K²·H_out·W_out)`；im2col 物化 `Θ(C_in·K²·H_out·W_out)` 后转 GEMM；Winograd（未采用）乘数 ↓2.25× 但加数与误差 ↑。 | direct 时间 `Θ(C_in·C_out·K²·H_out·W_out)`；空间 `Θ(C_in·K²·H_out·W_out)`（im2col）。 | `conv_tests.rs` `assert_close 1e-4/1e-3`；补零剔除后左上单元 `1e-6`。direct/im2col 逐位一致（同顺序求和）。 | — |
| **Einsum**（`crates/spec`） | 广义收缩 `out = Σ 收缩下标 Π 操作数`；收缩顺序是组合优化（NP 难）。 | 代价模型：`Formula::cost()` 给出 `flops`（有规约轴时每规约元素 `2 flops`，否则只算输出元素）+ 两个方向流量；**不声称最优顺序**（A3），只交可核对的 `flops/流量` 两数。 | `sch_tests.rs` 标量 `1e-12`；`dropout_tests.rs` `1e-5`。 | `spec/tests/einsum.rs` 对拍。 |
| **Adam**（`crates/train/src/optimizer/adam.rs`） | `m=β1·m+(1−β1)g`，`v=β2·v+(1−β2)g²`，偏差修正 `m̂=m/(1−β1ᵗ)`，`x −= lr·m̂/(√v̂+ε)`；`β=(0.9,0.999)`，`ε=1e-8`。 | 时间 `Θ(n)`/步，空间 `Θ(2n)`。 | `ε` 防除零、无精度语义（仅稳定性）；混合精度下 `v` 用 f32 累加不溢出（`g²≤(1e3)²=1e6 ≪ f32 max`）。`optimizer_tests.rs` 一步后参数 `1e-7`。 | — |
| **Muon**（`crates/train/src/optimizer/muon.rs`） | 动量 `μ=0.95` + Nesterov；方向做 Newton–Schulz 正交化后 `x −= lr·orth(d)/‖d‖`。 | NS 迭代 `NS_STEPS=5`，三次多项式 `(3.4445,−4.7750,2.0315)` 逼近 `1/√x`；正交化 `Θ(n·d²)`（SVD 自由）。 | 正交性 `‖XᵀX−I‖` 由测试间接钉住：对角谱 `3:2:1` 被压到 `0.5..=1.8` 窄带，非对角项 `<1e-6`；尺度不变（乘 1000 一步不动）——`optimizer_tests.rs`。 | — |
| **rANS 熵编码**（`crates/weights/src/rans.rs`） | 16 位量化范围编码器（`LOG2_M=16`，`STATE_MIN=1<<23`），8 路 SIMD。 | 时间 `Θ(n)`（`divrem` 用倒数乘替代除）；空间 `Θ(256·2^16)=64KiB` 符号 LUT。 | 状态机不变量：区间分割 `[0, TOTAL_FREQ)` 不重叠；`normalize` 必须吸收全部舍入误差（`debug_assert_eq!(diff, 0)`）。压缩率=熵 + 归一化舍入上界。 | — |
| **PTQ 量化**（`crates/spec`、`crates/weights`、`crates/dtype`） | 块量化：`QType`（GGML 块）与 `MxType`（微缩放）闭集；scale 选取见 `weights/tests/quant.rs`。 | 反量化时间 `Θ(n)`；块粒度决定 scale 密度。 | 重建误差按 bit 宽分层：`spec/tests/quant_tests.rs`、`weights/tests/quant.rs`、`weights/tests/round_trip.rs` 钉住往返误差；`dtype/tests/*.rs` `to_bits()` 钉住块编码位级。 | `dtype/tests/qtype_tests.rs`、`mxtype_tests.rs`。 |

## 位级一致性（K12）覆盖

逐位对拍（`assert_eq!(x.to_bits(), y.to_bits())`）已覆盖：`crates/algorithm/tests/spike_tests.rs`（INT8 脉冲 LIF 整数累加 + 代理梯度），`crates/dtype/tests/{dtype,qtype,mxtype}_tests.rs`（块量化编码），`crates/tensor/tests/tensor_tests.rs`。这些是「整数语义 / 编码」路径，天然可逐位；浮点 SIMD 求和路径（GEMM/GEMV/softmax）因重排顺序不承诺逐位相等，按 A1 以 `γ_n` 界 + 实测容差（上表）为契约，不造假逐位。

## 开口（仍未闭合，及闭合前提）

1. **GEMV 单链无 SIMD 位级对拍**：decode GEMV 是纯标量累加链，无 SIMD 特化，故无「SIMD vs 标量」逐位用例可写；闭合前提 = 引入多累加链/SIMD 路径后再补对拍。
2. **浮点求和路径逐位**：GEMM/GEMV/softmax 的 SIMD 路径与顺序标量参考**不逐位相等**（求和重排）；已按 A1 用 `γ_n` 界 + 测试容差钉住，不做位级承诺。闭合前提 = 把累加器固定为顺序标量（牺牲峰值，与本指令 K7/K8 冲突）。
3. **Winograd / Strassen 未采用**：Conv 只有 direct/im2col，无 Winograd 误差界可交；闭合前提 = 引入 Winograd 路径后补。
4. **投机解码**：本仓推理路径未实现 draft/accept 循环，无期望加速比可交；闭合前提 = 先有投机解码实现。
5. **einsum 收缩顺序**：`Formula::cost()` 是「单次全缩」代价，不做 NP 难的最优路径搜索（A3）；闭合前提 = 需要时给出贪心/DP 的保守值 + 下界。
