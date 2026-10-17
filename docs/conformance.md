# 文档一致性记录

状态: 拟定
范围: `docs/` 下四份规范文档（`architecture.md`、`engineering.md`、`design.md`、本记录）及其定义的工作区规则的一致性状态
读者: 工程师、评审者、维护者、AI 编码代理

---

本记录是一致性判定的唯一场所。每份规范文档在自身中定义其判定词汇；本记录记录当前判定及其背后的证据。

文档的存在不构成一致性的证据。判定为「已验证」MUST 引用已执行的检查或已核对的源码。

本记录随 2026-10-10 的文档合并更新：原先记录在 14 份顶层文档与 19 份 ADR 上的判定，已按其主题归属映射到合并后的 `architecture.md`、`engineering.md`、`design.md` 的对应章节；判定结论与证据沿用合并前经执行的检查。2026-10-10 的对抗性修正重测了静态门，并清除了黑盒测试迁移后失效的计数（见「本轮重测证据」）；通过/失败汇总未在本轮重跑，凡引用具体通过/失败数字者均以其所属章节的日期快照为准。性能相关的判定保持「拟定」，因为没有执行任何优化或前后对比。

2026-10-11 新增 `crates/metal` 与 `crates/cuda` 两个 Layer 1 GPU 后端（工作区成员 crate 由 19 增至 21），并首次引入受限的 `unsafe` 豁免模型与机械门禁 `scripts/checkunsafe.sh`。本节判定按 2026-10-11 的实测更新：Metal 的运行时数值验证已在本机执行并通过；CUDA 的运行时数值验证因本机与默认 CI 均无 NVIDIA GPU 而判定为**受阻**，只有编译门禁与设备无关的校验用例可作为静态证据。同日稍后 CUDA 侧补齐了 `compute` 契约的全部算子（cuBLAS `matmul`、cuDNN `softmax`、NVRTC CUDA C++ 的其余七个），`unsafe` 登记数由 2 增至 3，`cudacontract` 由 9 例增至 33 例。**2026-10-12 更新**：CUDA 的运行时数值验证已在一台 RTX 2080 Ti 主机上真机执行并通过（`cudacontract` 36/36、`cargo test --workspace` 758 passed / 0 failed、0 条 `E4 未执行`），故下表 `层 1：cuda` 的判定由**受阻**改为**已验证**；基准与固定成本分解见 `docs/design.md` §21.10.3。

**2026-10-17 更新**：metal 的 `ffi.rs` 四个按类 unsafe 站点合并为三个驱动交接块（提交 79f3595，同步路径 `dispatch` 一块 + 累积路径编码块/提交块），登记数 metal 4→3；`scripts/checkunsafe.sh` 现登记 `metal=3/cuda=3/musa=2`，`docs/design.md` §21.8 登记表、`AGENTS.md` §6、`docs/architecture.md` 与各 crate 文档同步对齐。同日 GPU 三后端各落一项削减与验证：metal `rope_at` 专核 + submission 输入上传去重（`gpu/decode` −76.4%/−73.7%，M5 Pro）；musa batch matmul 一次 dispatch（8×256³ −76.0%，MTT S4000，musacontract 29/29）；cuda cuBLAS 句柄缓存 + strided-batched 批处理 + 自研 NVRTC softmax 路径（交叉 clippy 干净，unsafe 3/3，真机复测待 WSL2）。CPU 侧全部数值 SIMD 路径迁移至 `archmage` 安全 intrinsics（fearless_simd 移除），rmsnorm 在服务器 x86（AVX2）实测 1.6644 → 1.2178 ms（1.37×），softmax deg-7 泰勒 `exp2` 快路径 2.5391 → 1.0522 ms（2.41×）。

**2026-10-15 更新**：实施 `docs/design.md` §21.13 的累加式执行上下文（`api::submission`/`stepref`/`operand`，`compute` 冻结）并完成三后端的运行时数值验证——`cpu`（`submissioncontract` 8/8）、`metal`（`metalcontract` 45/45，Apple M5 Pro）、`cuda`（`cudacontract` 43/43，NVIDIA T600 / WSL2 Ubuntu 24.04 真机）；三后端「同一链经 `submission` 与经 `compute` 逐位相同」均成立，`unsafe` 站点不变（metal 4/4、cuda 3/3）。另将平台无关的 `geometry` 从 `metal`/`cuda` 各一份复制抽到 `api::geometry`，消除 607 行重复。**性能验收（`gpu/modelforward` 10× 改善）已达成**：`metal` 52.60 ms → `metalsubmission` 2.81 ms（改善 ≈18.7×），§21.13 性能判定翻「已验证」。详情见「本轮重测证据（2026-10-15）」。

## 判定词汇

- **拟定（Proposed）** —— 要求已规定但尚未与实现对验。
- **已验证（Verified）** —— 源码审阅与相关检查支持该要求。
- **不一致（Non-conforming）** —— 存在已知违规。
- **受阻（Blocked）** —— 验证需要不可用的代码、工具或环境。

## 本轮重测证据（2026-10-10）

环境：macOS 27.0.1（build 26A434，arm64）、`rustc 1.99.0 (b940084d7 2026-09-28)`、git HEAD `978c57ee`。

- `bash scripts/check.sh` —— 报告 `All static quality checks passed.`（`cargo fmt --check`、`cargo check --workspace`、`clippy --workspace --all-targets -- -D warnings`、`checknaming.sh`、`checksrctests.sh`、`checkthresholds.sh`、`cargo doc` 全部通过），exit 0。
- `bash scripts/checksrctests.sh` —— `inline-test ratchet: 0 file(s) still pending migration.`，exit 0；`scripts/srctestbaseline.txt` 仅含一个换行，故为**空基线**，棘轮禁令绝对。
- `cargo test --workspace --locked -- --list` —— 枚举 **736** 个测试用例（含 doctest）；该命令只编译并列测试，不运行测试体。本记录不据此重报通过/失败总数。

## 本轮重测证据（2026-10-11）

环境：macOS 27.0.0（arm64，Apple M5 Pro）、`rustc 1.99.0 (b940084d7 2026-09-28)`。本轮唯一新增的交叉编译目标是 `x86_64-unknown-linux-gnu`（由 `rustup target add` 安装的 `rust-std`）。

- `cargo test --workspace` —— 44 个测试目标全部 `test result: ok`，**0 failed**（exit 0），在具备 GPU 访问的环境下连续 3 次运行结果一致。其中 `metalcontract` 的 30 个用例在本机 Apple GPU 上真实运行并通过，覆盖九个算子的数值对照与形状/边界/错误路径（E1）。同一二进制在拒绝设备访问的沙箱内运行时，需要派发的用例报 `executionfailed` 而宿主侧定案的拒绝用例仍通过；`tests/it/metalcontract.rs` 的环境假设已记录该差异，故该套件必须在允许 GPU 访问的环境下运行。
- **Metal 全算子补齐（E1）。** `crates/metal` 的 `compute` 契约现全部实现（`matmul`、`add`、`mul`、`silu`、`rmsnorm`、`softmax`、`rope`、`embedding`、`attention`），MSL 全部内嵌于 `source.rs` 并在运行期以 `newLibraryWithSource` 编译，`metalcontract` 由 6 例增至 30 例。首轮运行发现 `rope` 内核把局部变量命名为 `half`（与 MSL 内建标量类型同名）导致编译失败、调用报 `executionfailed`；改名 `pairs` 后通过，失败与修复均在本机复现。`unsafe` 站点计数未随算子数增长：所有新内核都复用「上传 / 下载 / 绑定缓冲 / 绑定标量参数」这同一组四站点（`checkunsafe.sh` 仍报 `metal 4/4`）。
- `bash scripts/check.sh` —— 报告 `All static quality checks passed.`，exit 0（`fmt --check`、`cargo check --workspace --all-targets --locked`、`clippy --workspace --all-targets --locked -- -D warnings`、`checknaming.sh`、`checksrctests.sh`、`checkunsafe.sh`、`checkthresholds.sh`、`cargo doc -D warnings`）。
- `bash scripts/checkunsafe.sh` —— exit 0，输出 `crates/metal/src/ffi.rs: 4/4 unsafe site(s) (exact)` 与 `crates/cuda/src/ffi.rs: 3/3 unsafe site(s) (exact)`，即两处豁免的站点数与 `docs/design.md` §21.8 登记表的条目数**恰好相等**，且每站均有前置 `// SAFETY:` 注释（E1）。
- `bash scripts/checkunsafe.sh` 的**证伪性验证**（本日新增）：门在本次修订中由「站点数 ≤ 上界」收紧为「站点数 == 登记数」，因 §21.8 原文已声明门禁断言「站点计数与本表条目数一致」，而实现此前弱于文档。验证方式为在 gitignored 的 `work/` 内复制该脚本并分别把 `metal` 的登记数改为 5 与 3：两个方向均以 `exit 1` 失败并报出 `4 unsafe site(s) but docs/design.md §21.8 registers {5|3}`；未修改的副本 `exit 0`。三个探针脚本随后删除（E1）。
- `bash scripts/checknaming.sh` —— exit 0。本轮修正了该门的一处**误报**：其原先把内嵌在 `r#"..."#` 原文串里的 MSL / CUDA C 源码当作 Rust 标识符扫描，因而对 `device const float* a` 报出 `const 'float' must be uppercase compact`。新增的原料串状态机使其只扫描 Rust 代码。
- 命名门**证伪性验证** —— 临时加入一个含 `const badlower: usize = 1;` 与一个含 `const float*` 的原料串的文件：门正确报出 `badlower` 一处违规（exit 1）且未报原料串内的 `const float`；探针文件随后删除（E1）。
- `cargo check -p cuda --target x86_64-unknown-linux-gnu --all-targets` —— exit 0；`cargo clippy -p cuda --target x86_64-unknown-linux-gnu --all-targets -- -D warnings` —— exit 0（E1）。这证明 Linux 侧的 CUDA 代码在没有 CUDA toolkit、没有 NVIDIA 驱动的 macOS 主机上也能完成类型检查与 lint。
- `cargo check -p llm --target x86_64-unknown-linux-gnu --test it` 与 `--bench cuda` —— exit 0。这证明 `tests/it/cudacontract.rs` 的全部 33 个用例与 `benches/cuda.rs` 在 Linux 上均可编译（**仅为编译证据；该命令不运行测试体，也未产生任何通过/失败结果**）。
- `cargo clippy -p cuda --target x86_64-unknown-linux-gnu --lib --locked -- -D clippy::cognitive_complexity -D clippy::too_many_lines` —— exit 0，即 CUDA 侧在 Linux 上也满足 DS3 预算（宿主 macOS 的 `checkthresholds.sh` 看不到这些被 `cfg(target_os = "linux")` 门控的代码，故单列）。
- **已执行（E1，2026-10-12 真机）** —— CUDA 的运行时数值验证。在一台 NVIDIA GeForce RTX 2080 Ti 主机（Linux x86_64、CUDA runtime 13.0、驱动 580.105.08、`rustc 1.99.0`）上，以 `LLM_CUDA_REQUIRE_DEVICE=1` 执行 `cargo test -p llm --test it cudacontract` → **36 passed / 0 failed**；`cargo test --workspace --locked` → **758 passed / 0 failed**，输出中 **0 条 `E4 未执行`**，即 15 个需要设备的用例确实运行并与 `testkit::reference` 的标量参考对照；`bash scripts/check.sh` → `All static quality checks passed.`（exit 0）。首次真机执行立即暴露一个潜伏缺陷：非方阵 `matmul` 对照用例以 shape `[7, 4]` 描述右操作数却只给 20 个元素（`lengthmismatch { expected: 28, actual: 20 }`）；该套件整体 `#[cfg(target_os = "linux")]` 且在无设备时提前返回，故自 67ff611 起从未被执行，Linux 目标上的编译绿灯掩盖了它。已修正（提交 `test(cuda): 修正非方阵 matmul 对照用例的右操作数长度`）。基准、每调用固定成本分解与基线见 `docs/design.md` §21.10.3。`.github/workflows/ci.yml` 的 `cuda-test` job 在自托管 GPU runner 上设置 `LLM_CUDA_REQUIRE_DEVICE=1`，届时缺设备会成为硬失败而非静默通过；该 job 目前仍仅在手动触发时运行（自托管 runner 尚未注册，见 `docs/design.md` §21 的未决项），故本次执行是手工完成的。

## 本轮重测证据（2026-10-12）

环境：macOS 27.0.1（build 26A434，arm64，Apple M5 Pro）、`rustc 1.99.0`、基线 commit `55d8efb`。本轮是对 `docs/gpu-backend-optimization-plan.md` 的 G0/G1/G2 执行：先建立基线，再对齐 shape/预算校验，并补齐边界与数值测试。

- `bash scripts/check.sh` —— `All static quality checks passed.`，exit 0。
- `cargo test --workspace` —— 762 passed / **0 failed**（exit 0）；其中 `tests/it/metalcontract.rs` 的 43 例在本机 Apple GPU 上真实运行（上一轮为 30 例）。
- **可观测分类三后端一致（E1）。** `tests/it/gpucontract.rs` 提供 30 个定案于宿主侧的契约向量；`metal` 与 `cuda` 两个套件各自断言「与列表一致」与「与 `cpu` 逐项相同」。整改前 `metal`/`cuda` 把「秩超过声明的最大秩」分类为 `executionfailed`，而 `cpu` 分类为 `resourceexhausted`——同一操作数在不同设备上得到不同公共错误。现由 `api::checkbudget`/`checkgatherbudget` 单点决定，三个后端一致。
- **`matmul` 结果预算（E1）。** 三个后端此前只校验操作数；`m*k ≤ max` 且 `k*n ≤ max` 时仍可要求 `m*n` 级别的输出分配。现结果形状在任何分配前过同一预算谓词，`matmul/outelements`、`matmul/outbytes` 与其 inclusive 边界用例（`k == 0`，无设备即可定案）覆盖该行为。
- **证伪性验证（E1）。** ① 把秩分类改回 `executionfailed` → `metal` 的两个向量用例变红并报出具体用例名与两侧值；② 删除 `metal` executor 的结果预算校验 → `matmul/outelements` 变红。两处探针随后还原。
- **端到端代价与架构缺口（E1，2026-10-12 第三轮）。** `benches/metal.rs` 新增 `gpu/modelforward`：同一微型 Llama（2 层、`dim=64`、`seq=16`）前向在同一进程内跑两个 provider，median **CPU 129–130 µs / Metal 50.5–53.5 ms（约 390–414×）**。该前向是一条约 35 次 `compute` 调用的链，≈1.5 ms/调用，与固定往返地板一致。随后用四组对照排除其他解释：负载大小（6 元素与 2^20 元素同为约 1.9–2.1 ms）、一次命令缓冲内的 dispatch 数（1 与 16 次同为约 2.2 ms）、等待方式（忙等 `status()` 1754–2001 µs，与 `waitUntilCompleted` 相同）、存储模式（`StorageModePrivate` 1920 µs 对 `StorageModeShared` 2037 µs）；宿主 `kern.hv_vmm_present = 0`，即裸机而非虚拟化。结论：**在内核层面继续优化不会改变 GPU 在小/中工作负载上慢于 CPU 的事实**；唯一的一阶杠杆是把多次 dispatch 合入同一命令缓冲。据此在 `docs/design.md` §21.13 提出累加式执行上下文的**提议**设计（所有权、请求隔离、预算、错误归属、被否决方案、三阶段实施顺序与验收标准），**尚未实施**。
- **Metal 运行时编译缓存与 dispatch 几何（E1，2026-10-12 第二轮）。** 逐阶段插桩（随后移除）证明 `silu` 的稳态成本由 `waitUntilCompleted` 支配（2088 µs），且与负载大小（6 元素与 2^20 元素同为约 2.1 ms）和一次命令缓冲内的 dispatch 次数（1 与 16 次同为约 2.2 ms）**均无关**；`commit` 仅 20 µs，缓冲完成后的第二次 `wait` 仅 4–16 µs。据此实施两项：① 设备与 MSL pipeline 缓存由 executor 拥有（不是全局单例），九个算子的 median 提升 15–34%；② `matmul` 的全部 batch block 合入一次 dispatch（batch 作为网格第三维），8×256³ 由 13.969 ms 降至 2.878 ms（−79.4%，×4.85），是唯一一处 Metal 快于 CPU（7.519 ms）的测量点。`metalcontract` 由 40 增至 43 例，新增「九个算子恰好九个 pipeline 且重复调用不增长」「clone 共享缓存」「`Send + Sync` 且四线程并发结果一致」三例；`unsafe` 站点仍为 4/4。CUDA 当时未随动，因为其无实测基线；该基线已于 2026-10-12 在真机建立（`docs/design.md` §21.10.3：每调用约 12.3 ms，其中约 10.8 ms 是 NVRTC 编译），故 CUDA 侧的同类缓存列入下一步。详见 `docs/design.md` §21.10.2 与 §21.6。
- **Metal 全算子基准与运行间方差（E1）。** `benches/metal.rs` 由只测 `matmul` 扩为九个算子（工作负载与组名与 `benches/cuda.rs` 一致），`cargo bench -p llm --bench metal -- --warm-up-time 0.5 --measurement-time 1.0 --sample-size 10` 测得 Metal 在每个工作负载上都慢于 CPU；同一二进制的 `gpu/matmul/metal256` 四次 median 为 2.337/2.617/2.672/2.850 ms（散布约 ±10%），而 `cpu256` 四次为 954.9/956.4/958.6/974.3 µs（约 ±1%）。未声明任何加速（见 `docs/design.md` §21.10.1）。
- `cargo check -p llm --target x86_64-unknown-linux-gnu --test it --locked`、`cargo clippy -p cuda --target x86_64-unknown-linux-gnu --all-targets --locked -- -D warnings`、同目标 `--lib` 的 DS3 阈值检查 —— 均 exit 0（E1，编译证据）。
- **已执行（E1，2026-10-12 真机）** —— CUDA 运行时数值验证：在 RTX 2080 Ti 主机上以 `LLM_CUDA_REQUIRE_DEVICE=1` 得到 `cudacontract` 36 passed / 0 failed 与 `cargo test --workspace` 758 passed / 0 failed（0 条 `E4 未执行`），`check.sh` exit 0；首次执行即暴露并修正了一个用例缺陷（非方阵 matmul 对照用例的元素数不足）。见 `docs/design.md` §21.10.3。
- **CUDA 与 `cpu`/`metal` 的一处不一致（E1，已修）。** 首次真机执行发现 `softmax` 在整行 `-inf` 的输入上与契约不符：cuDNN 的 `CUDNN_SOFTMAX_ACCURATE` 返回全零，而契约（`crates/cpu/src/softmax.rs` 与 `metal` 的实现）传播 NaN。现由「输入含任何非有限值时不委托 cuDNN，改用与 `cpu`/`metal` 相同的标量式」修正；`tests/it/cudacontract.rs` 以两例钉住：全 `-inf` 行传播 NaN，五个非有限行（全 `-inf`、含 `+inf`、含 NaN、`-inf` 与有限混合）与 `cpu` 逐元素一致。同机 A/B 测得该守卫在 2^21 元素上约 +1.9 ms（4.9417 ms → 6.8184 ms），并记录了它的替代方案与下一步（`docs/design.md` §21.10.3 第 6 条）。
- **CUDA 的运行时编译缓存（E1，2026-10-12 实施）。** device 0 的已 retain 主上下文与已解析内核由 executor 拥有的 `ffi::cache` 持有（clone 经 `Arc` 共享，非全局单例），执行流故意不缓存。同机前后对比：由 NVRTC 服务的七个工作负载 median 下降 74–92%（`rope` 12.402 → 0.95834 ms、`attention` 14.309 → 1.2974 ms、`embedding` 15.941 → 4.1326 ms），两个 cuBLAS 工作负载不变；CUDA 快于 CPU 的工作负载由 3 个增至 5 个。缓存后固定成本 NVRTC 0.98–1.08 ms/调用，cuBLAS 1.47–1.59 ms、cuDNN 2.85–2.98 ms（由句柄创建主导，记为下一项）。`cudacontract` 以三例钉住缓存语义（编译一次、clone 共享、四线程并发只编译一次），`unsafe` 站点仍为 3/3。见 `docs/design.md` §21.6 与 §21.10.3 第 7 条。

## 本轮重测证据（2026-10-15）

环境：macOS 27.0.1（arm64，Apple M5 Pro）、`rustc 1.99.0 (b940084d7 2026-09-28)`。本轮背景：Claude Code 桌面 App 的会话沙箱封锁了 `cargo`（sccache 的 Unix socket 被拒、rsproxy.cn 出网被白名单锁死、`mktemp` 的 `/var/folders` 被拒）；三条均已在用户侧绕过或就地修复，随后在本机 M5 Pro 完成 Metal 验证、在用户的 Windows + WSL2（Ubuntu 24.04）上完成 CUDA 真机验证。

- **沙箱三连修复（E1，非代码改动）。** ① `~/.cargo/config.toml` 的 `rustc-wrapper = "sccache"` 已注释（sccache 禁 Unix socket，报 `Operation not permitted`），`cargo` 改由 rustc 直接编译；② rsproxy.cn 被会话级白名单锁死、`allowed_domains` 等本地配置均无法放开，改为 `cargo vendor --respect-source-config`（139 MB，含 proptest）并把项目级 `.cargo/config.toml` 指向 vendored-sources 离线源，原命令不加任何参数完全离线跑通；③ `scripts/checknaming.sh` 的 `mktemp` 从默认 `/var/folders/...` 改为项目内 `"$ROOT_DIR/.checknaming.XXXXXX"`，跑通且无残留。
- **Metal 45/45（E1，2026-10-15）。** `cargo test -p llm --test it metalcontract` 在本机 Apple GPU 上 45 例全部通过。首轮运行 44 过、1 崩（SIGTRAP），崩在 `submission_rejects_an_unrecorded_result_reference`（故意拒绝路径）：`crates/metal/src/ffi.rs` 的 `session` 无 `impl Drop`，任何中途错误路径 drop session 时 encoder 未 `endEncoding` 就被释放，Metal 在 dealloc 时断言炸进程。修复：`session` 增加 `ended` 标志，`apply` 先 `endEncoding` 再置位，新增 `impl Drop` 兜底补调；`apply` 因 Drop 类型不能 move 字段改用 `core::mem::take`。修复后 45/45 全过，`submission_reproduces_a_compute_chain_bit_for_bit` 无数值不匹配；`cargo bench -p llm --bench metal -- gpu/submissionchain` 无回归（`submission32` ≈2.42–2.49 ms、1.65–1.69 Melem/s）。
- **CUDA 43/43（E1，2026-10-15，第二台真机）。** 在用户的 Windows 11 + WSL2 的 Ubuntu 24.04（glibc）新建发行版并装齐工具链：rustup 1.29.1 → rustc 1.99.0（与项目 `rust-toolchain.toml` 一致）、`nvidia-cuda-toolkit` 12.0.140（提供 nvcc/libcublas/libnvrtc）、cudnn 9.3.0.75 官方 redist tarball 整目录装入（含 `libcudnn.so.9` 与全部 `libcudnn_*` 兄弟库）；`LLM_CUDA_REQUIRE_DEVICE=1 cargo test -p llm --test it cudacontract` → **43 passed / 0 failed**（`nvidia-smi -L` = NVIDIA T600）。两个执行侧要点：① `hasdevice` 探针用 `Command::new("nvidia-smi")` 按 PATH 查找，而 nvidia-smi 位于 `/usr/lib/wsl/lib`——运行环境必须把该目录留在 PATH 上，否则 22 个设备用例以「no CUDA device was detected」误报失败（此时 cuInit 实际成功、`cuDeviceGetCount=1`）；② 编译期不需要 cudnn，但运行期 cudarc 的 `cudnn = ["cudnn-09021"]` 需要 `libcudnn.so.9`，且 cudnn 9 tarball 需整目录拷贝（漏拷 `libcudnn_graph.so.9` 会以 `Cannot load symbol cudnnCreate` 崩溃）。

## 本轮重测证据（2026-10-16）

环境：macOS 27.0.1（arm64，Apple M5 Pro）、`rustc 1.99.0 (b940084d7 2026-09-28)`；远程真机为 Windows 11 + WSL2（Ubuntu 24.04、CUDA 12.0.140、cudnn 9.3.0.75）的 NVIDIA T600。本轮为 KV cache 接线：attention 契约扩展 + `forward_cached` 增量前向 + `infer` 的 `kvstate` 接线 + GPU 用例与 decode 基准。

- **attention 契约扩展（E1）。** `attend` 标量参考此前假设 `qseq == kvseq`，现支持 `offset`（query 起点），三后端一致；`tests/it/metalcontract.rs` 与 `tests/it/cudacontract.rs` 各增 `attention_accepts_a_shorter_query_than_the_keys`（qseq=1 < kvseq=3）。Metal **46/46**（M5 Pro）、CUDA **44/44**（T600 真机）全绿。
- **KV cache 接线（E1）。** `llamamodel::forward_cached`/`llamablock::forward_cached` 读并追加 `kvcache`（仅处理新 token 位置），`infer` 的 `kvstate` 接线（`decode.rs`/`engine.rs`/`generation.rs`/`prefill.rs`）。CPU 侧 `incrementalattention` 2、`kvcacheforward` 1、`generation` 10 全绿；`forward_cached` 与全量 `forward` 逐位相同。
- **decode 基准（submission 版，2026-10-16 复测）。** compute 路径首测不对称（`metalcached` 把 prefill 计入计时：1.671 s vs `metalrecompute` 0.825 s），`iter_batched` 对称化后 868.85 ms vs 830.37 ms（同步往返主导、无收益）；`forwardsubmission_cached` 接入后复测：`metalcached` **45.03 ms** vs `metalrecompute` **47.26 ms**（相对 compute 版整组 −94.8%），**增量 vs 全量快 ≈4.7%**。结论：单次提交收益兑现（每前向 35 次 submit+wait → 1 次提交，≈52 ms → ≈2.8 ms）；seq=16 规模下增量 vs 全量的计算差被每前向固定开销压制，**加速比由长序列（seq=128/256）阶段兑现**。

## 本轮重测证据（2026-10-17）

环境：远程真机为 AutoDL x86_64 容器（MTT S4000、MUSA 3.1.0、`rustc 1.99.0 (b940084d7 2026-09-28)`、cargo 1.99.0），git HEAD `496b43f`；服务器树由本地 HEAD 同步（原服务器实验态备份于 `/root/llm-backup-20261017001044`）。本轮为 MUSA 后端算子契约与性能基线的真机复验。

- **算子契约（E1）。** `LLM_MUSA_REQUIRE_DEVICE=1 cargo test -p llm --test it musacontract -- --nocapture` → **29 passed / 0 failed**（较 β scan 行记录的 28/28 多 1 例：`scan_matches_the_scalar_reference_across_blocks` 跨块并行用例；13 算子含 scan 串行/并行/缓存、attention 全量/增量、batch matmul、rope_at 均对照标量参考 1e-5 容差）。输出无 `E4 未执行`。
- **unsafe 门（E1）。** `bash scripts/checkunsafe.sh` → exit 0，`musa/src/ffi.rs` **2/2 (exact)**、`metal` 3/3、`cuda` 3/3，无未登记站点。
- **性能基线复验（criterion 0.7.0 中位数，musa 与 cpu 同组同 run 对照）。** matmul 256³：**1.284 ms vs 2.329 ms**（记录 1.26 vs 2.51）；matmulbatch 8×256³：**8.958 ms vs 18.863 ms**（新记录）；silu：**1.328 ms vs 3.481 ms**（记录 1.54 vs 3.46）；softmax 512×4096：**2.128 ms vs 8.916 ms**（记录 2.52 vs 8.89）；attention seq128：**1.573 ms vs 2.489 ms**（记录 1.46 vs 3.33）；仍慢于 CPU 的方向不变：add 1.836 ms vs 497.5 µs、mul 1.805 ms vs 485.5 µs、rmsnorm 2.213 ms vs 1.664 ms、rope 752.5 µs vs 100.7 µs、embedding 3.125 ms vs 19.2 µs、scan 17.210 ms vs 5.210 ms——「每调用 launch + 缓冲分配开销主导」的既有结论复现，方向与量级一致；无性能声称（方向性基线）。

## 本轮重测证据（2026-10-17 CUDA RTX 3090）

环境：AutoDL x86_64 容器（nmb2，NVIDIA GeForce RTX 3090、驱动 5.15.0-136-generic、`rustc 1.99.0`），git HEAD 依次含三阶段优化（cuBLAS 句柄缓存、strided-batched 批处理、自研 NVRTC softmax 路径）与 Linux 无 MUSA 工具链主机的链接门控修复。

- **契约（E1）。** `LLM_CUDA_REQUIRE_DEVICE=1 cargo test -p llm --test it cudacontract -- --nocapture` → **49 passed / 0 failed**（含新增 `nvrtc_softmax_matches_the_scalar_reference_per_row` 与既有 batch/strided 用例），0 条 `E4 未执行`。为达成此次链接，修复两处跨平台缺陷：① `crates/cpu/src/softmax.rs` 的 AVX2 快路径从未在非 aarch64 目标编译（token 绑定为下划线却引用、`_mm256_cmplt_epi32` 无 archmage 安全包装改用 `cmpgt` 交换操作数）；② `musa` 在 Linux 无工具链主机链接失败（空库契约未覆盖 Linux）——新增 `musa_toolkit` cfg（musa build.rs 发射 + 根包 build.rs 对 tests/benches 发射，musacontract/benches/musa 同门控）。
- **基准（E1，criterion 中位数，cuda 与 cpu 同组同 run）。** matmul 256³：**151.41 µs vs 1.5646 ms**（10.3×）；matmulbatch 8×256³：**2.5913 ms vs 11.992 ms**（4.6×，对照三阶段前按记录固定成本推估约 13 ms——每块 151 µs + 每调用约 1.5 ms 句柄创建 × 8 块）；silu：**1.9335 ms vs 3.7737 ms**；softmax 512×4096：**1.0502 ms vs 5.5266 ms**（cuDNN 路径；同装置 softmax 路径 A/B：cuDNN 5.5588 ms vs NVRTC 3.4353 ms，**−38.2%**——每调用 cuDNN 句柄创建成本被 NVRTC 路径消除）；rmsnorm 2.1376 ms vs 2.1183 ms（近持平）；add/mul/rope/embedding 仍慢于 CPU（launch 主导，方向与既有记录一致）。无性能声称（方向性基线）。

## 当前判定

### architecture.md

| 章节 / 主题 | 判定 | 证据 |
| --- | --- | --- |
| 架构边界 | 已验证 | 全部 21 个 `crates/*` 成员 crate 与根 `llm` package 已实现（工作区共 22 个 Cargo package，其中 21 个为成员 crate）。解析后的工作区依赖图无环（`cargo check --workspace --all-targets` exit 0；`cargo metadata` exit 0——若有环将解析失败），且 `scripts/check.sh`（fmt、`clippy --workspace --all-targets -- -D warnings`、`rustdoc -D warnings`）干净。`metal` 与 `cuda` 都是「后端 → api」方向，`api` 未新增指向任何具体后端的边。 |
| crate 归属与依赖图 | 已验证 | 清单与 `cargo metadata` 显示生产边无环且方向正确；根 `llm` package 无生产依赖，其 dev-dependencies 覆盖全部成员 crate（仅测试用），`grep crates/*/Cargo.toml` 未发现指向 `cli`/`serve` 的产品边，`cli`/`serve` 不是任何生产计算 crate 的依赖。逐 crate 的实际生产边见 `architecture.md` §13.2。`objc2-*` 仅在 macOS 条件下、`cudarc` 仅在 Linux 条件下被引入，故默认 feature 不引入任何 GPU 依赖。 |
| 工作区布局 | 已验证 | 根 `Cargo.toml` 是 package + workspace 根（无 lib/bin）；跨 crate 集成测试在根 `tests/`（43 个套件），基准在根 `benches/`（9 个目标）；`README.md`、`LICENSE`（`LICENSE-APACHE`、`LICENSE-MIT`）与 `clippy.toml` 位于仓库根；`criterion` 在根声明一次。`cargo bench --workspace --no-run` 构建所有基准目标（`benches/metal.rs` 在非 macOS 上、`benches/cuda.rs` 在非 Linux 上均编译为 `fn main() {}`）。 |

### engineering.md

| 章节 / 主题 | 判定 | 证据 |
| --- | --- | --- |
| 工作流与执行协议 | 已验证 | 其余 crate（P1–P12）均按此工作流实现：阅读契约、计划、实现、执行范围化验证、记录 ADR；合并后的验证运行了 `scripts/check.sh` 与 `scripts/test.sh`。 |
| 错误处理标准 | 已验证 | 每个 crate 拥有 `thiserror` 错误枚举（如 `clierror`、`trainerror`、`serveerror`、`opterror`、`infererror`）；工作区拒绝 `unwrap_used`/`expect_used`/`panic`/`todo`/`unimplemented`，且 `cargo clippy --workspace --all-targets -- -D warnings` 通过，因此可恢复失败被返回而非 panic。 |
| 命名标准与机械门 | 已验证 | `bash scripts/checknaming.sh`（禁用同义词、常量/宏大小写、豁免条款检查，覆盖 `crates/*/src` 去掉 `#[cfg(test)]`）exit 0；`bash scripts/checkthresholds.sh` 对 `DS3` 与 `A4` 均 exit 0。8 个构造动词已收敛（`architecture::build`→`newmodel` 等），`nn::init` 工厂规范化为 `fill*`/`init*`，两处 `DS3` 生产违规重构到 80 行以下。`bash scripts/check.sh` 报告 "All static quality checks passed"（本轮重测，见「本轮重测证据」）。2026-10-11 修正了该门对内嵌原料串的误报，并做了证伪性验证（见「本轮重测证据（2026-10-11）」）。 |
| 公开 API 设计标准 | 已验证 | 每个 crate 都有公开 API；`RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps` 干净，每个文档化的公开项均可解析。`metal`/`cuda` 的 crate 文档不链接被 `cfg` 门控掉或私有的条目（`ffi`、`cudaexecutor::newnvrtc` 用代码跨度而非文档链接），因此交叉平台构建文档仍然干净。 |
| 测试策略与质量门 | 已验证 | 黑盒迁移完成：所有 `crates/*/src` 零内联 `#[cfg(test)]` 模块（`rg -l '#\[cfg\(test\)\]' crates/*/src` 为空；21/21 成员 crate 已迁移）。43 个根 `tests/` 套件仅调用公开 `pub` API，`benches/` 保留 9 个基准目标。`scripts/srctestbaseline.txt` 仅含一个换行（空基线），`scripts/checksrctests.sh` 的棘轮禁令绝对（本轮重测：`0 file(s) still pending migration.`，exit 0）。`bash scripts/check.sh` 干净。GPU 套件同为黑盒：`metalcontract` 与 `cudacontract` 只经 `pub` 接口触发，且各自用 `#![cfg]` 限定平台。 |
| 黑盒契约优先测试 | 已验证 | 决策、门与诊断缝均就位。21 个成员 crate 已迁移（0 `#[cfg(test)]`，基线为空），由 `scripts/migrateinline.py`、`scripts/migratecrates.py`、`scripts/finalizecrate.py` 驱动。缝在 `infer`、`train`、`checkpoint`、`serve`、`cli` 中接线，`EnvFilter` 订阅者由 `cli::runcli` 与 `httpserver::serve`/`serveone` 安装；`tests/diagnostics.rs` 证明无需重编的运行期切换（同一 `loadcheckpoint` 调用在 `DEBUG` 订阅者下发出 "checkpoint loaded"，在 `WARN` 下抑制）。 |
| 集成测试与基准整合 | 已验证 | 每个 crate 局部集成套件移到仓库根 `tests/`（14 个移动文件，合并 `common/`），`optim` 基准移到根 `benches/`；根 `llm` package 声明仅测试的 dev-dependencies。`scripts/check.sh` 与 `scripts/test.sh` 通过（本轮重测 `check.sh` exit 0；通过/失败汇总未在本轮重跑），`cargo bench --workspace --no-run` 构建 8 个目标。其「crate 局部覆盖现仅内联单元测试」的后果已被黑盒测试决策取代。 |
| unsafe 豁免与机械门禁 | 已验证 | 其余 19 个 crate 的 `#![forbid(unsafe_code)]` 一行未改；`metal`、`cuda` 与 `musa` 用 `#![deny(unsafe_code)]`，各自恰好一个文件 `src/ffi.rs` 声明 `#![allow(unsafe_code)]`。`bash scripts/checkunsafe.sh` exit 0，报 `metal 3/3 (exact)`、`cuda 3/3 (exact)`、`musa 2/2 (exact)`，与 `docs/design.md` §21.8 登记表的条目数恰好相等（非上界语义：多一处或少一处均失败，已由双向证伪性探针验证）；工作区 lint `unsafe_code = "deny"` 未变动。**2026-10-17 更新**：metal 由 4 站点合并为 3 个驱动交接块（提交 79f3595），登记表与门禁同步更新。豁免条款、白名单、站点登记表与其依据（GOV7）与回滚方案见 `AGENTS.md` §6、`docs/engineering.md` 规则 14 与 `docs/design.md` §21.8。 |
| 性能工程标准 | 拟定 | 每个被基准的 crate 在仓库根有可见基线：`benches/{cpuops,kernels,metadata,autograd,nn,models,optim,metal,cuda}.rs`，共享工作负载在 `crates/bench`。`benches/metal.rs` 已在同一轮内测量 CPU 与 Metal 的全部九个算子（256³ matmul：CPU ≈ 951 µs、Metal ≈ 1.779 ms；8×256³ batch：CPU ≈ 7.61 ms、Metal ≈ 3.19 ms），并如实记录原因（每次调用一次命令缓冲往返，与负载和 dispatch 数无关；见 `docs/design.md` §21.10.2）。batch 工作负载上 Metal 快于 CPU 是测量结果，不是优化目标。`benches/cuda.rs` 已建立并在宿主与 Linux 目标上编译通过，但**没有任何 CUDA 测量**（本机无 NVIDIA GPU），故同样不声称任何改进。§21.13 的 `forwardsubmission` 已达成性能验收（`metal` 52.60 ms → `metalsubmission` 2.81 ms，改善 ≈18.7×，见「层 1：执行上下文与 dispatch 批处理（§21.13）」行）。 |

### design.md

| 章节 / 主题 | 判定 | 证据 |
| --- | --- | --- |
| 层 0：lib / tensor / api | 已验证 | `lib`、`tensor`、`api` 已按文档化契约实现；根 `tensorcontract`、`apicontract`、`crosscrate` 套件通过。 |
| 层 0：张量布局契约 | 拟定 | 该契约首次成文于设计文档（不再是指向其它文档的占位），依据是对 `crates/tensor/src` 的源码核对；尚未作为独立规范单独验证。 |
| 层 0：api 计算契约扩展 | 已验证 | `api` 暴露冻结的 `kernel<T>` 以及增量的 `binarykernel`/`weightedkernel`/`gatherkernel`/`compute` 契约；`crates/cpu` 实现它们，根 `cpucontract` 套件从 crate 外部触发它们。 |
| 层 0：api 接口宽度（A4）与 β scan 立项评审 | 拟定 | `docs/beta-scan-plan.md` §4.3：`api` crate root 导出项已 18/20（余量 2），`compute` 加 `scan`/`scan_cached`（11→13）不新增导出项，但下一算子（γ 脉冲等）落地前须立项评审「上调 A4 上界或为算子集拆分」。该评审尚未执行。 |
| β scan 算子（B0–B6，`docs/beta-scan-plan.md`） | 已验证（cpu+metal+cuda） | `compute` 加 `scan`/`scan_cached`、`backenderror` 加 `invalidparameter`；`cpu` 串行 `cpuscan`（vs `testkit::reference::scan` 逐位，`scanunits` 8/8）+ 分段 `apply_chunked`（§5.10 容差，`scanunits` 2/2）；**GPU 三段式并行扫描**（Metal `scan_parallel.metal` / CUDA `scan_parallel.cu`：块内累积 + 块间前缀 + 块内应用，串行内核保留为宽状态回退）——Metal 数值真机绿（`metalcontract::scan` 16/16 含 16K/64 块并行用例）、**CUDA 数值真机绿（NVIDIA T600，`cudacontract` 48/48 含 16K/64 块并行用例，容差按 P·ε·‖B‖ 界）**；**MUSA 数值契约真机全绿（MTT S4000 / MP_22 / MUSA 3.1.0，`musacontract` 28/28：13 算子含 scan 串行/并行/缓存、attention 全量/增量、batch matmul、rope_at，均对照标量参考 1e-5 容差）；基准 `benches/musa.rs` 实测（同装置对照 CPU release）：matmul 256 1.26 ms vs 2.51 ms、softmax 512×4096 2.52 ms vs 8.89 ms、attention seq128 1.46 ms vs 3.33 ms、silu 1.54 ms vs 3.46 ms 快于 CPU；add/mul/rope/embedding/scan 慢于 CPU（embedding 3.52 ms vs 19.5 µs、scan 28.3 ms vs 5.06 ms）——每调用 launch + 缓冲分配开销主导，无性能声称（记录为方向性基线）。**2026-10-17 真机复验：`musacontract` 29/29（新增跨块 scan 用例）、unsafe 2/2、基准方向与量级一致**（见「本轮重测证据（2026-10-17）」）**；`nn::ssm` + `registry::architecture::mamba`（`nnlayers`/`modelsunits` 3/3）；基准 serial 线性 $O(n)$ 实测 + **M1 Should 同装置闭合：16K scan 26.6 ms vs 16K 稠密 attention 2747 ms ≈103× ≥ 5×**（M5 Pro，`benches/metal.rs`）。 |
| 层 1：cpu | 已验证 | `crates/cpu` 实现 `backend`、`executor` 与算子模块（`matmul`、`attention`、`embedding`、`normalization`、`rotary`、`softmax`、`activation`、`elementwise`）；单元测试现由仓库根 `tests/cpuunits.rs` 经 `pub` 接口触发（`crates/cpu/src` 零内联 `#[cfg(test)]`），契约测试在根 `tests/cpucontract.rs`，`scripts/check.sh` 干净。基准在仓库根：`benches/cpuops.rs`（执行器级）、`benches/kernels.rs`（原始算子）、`benches/metadata.rs`（张量元数据）。 |
| 层 1：存储 dtype 与计算 precision | 已验证 | `dtype::f64` 与 `impl element for f64` 存在且 `sizeinbytes() == 8`；根 `tests/tensorunits.rs` 与 `tests/tensorcontract.rs` 覆盖 element/dtype 宽度不变量与 `f64` 视图往返；`clippy -D warnings` 与 `cargo doc -D warnings` 干净。 |
| 层 1：metal（GPU 后端） | 已验证 | `crates/metal` 实现 `api::backend` 与 `api::compute` 的**全部**算子：`matmul` 与 `add`/`mul`/`silu`/`rmsnorm`/`softmax`/`rope`/`embedding`/`attention` 均由内嵌于 `source.rs` 的 MSL 经 `newLibraryWithSource` 在运行期编译执行，无需 Xcode；`attention` 是单个融合内核（不物化 score 矩阵）。根 `tests/it/metalcontract.rs` 的黑盒用例由 30 增至 43、2026-10-15 再增至 **45**，全部在本机 Apple M5 Pro 上真实运行并通过（2026-10-15 修复了 `session` 的 encoder 生命周期缺陷后，见「本轮重测证据（2026-10-15）」），覆盖九个算子的数值对照（对照 `testkit::reference`）与形状/边界/错误路径：非方阵且互异维度的 matmul、宽度 1000 的 `rmsnorm`/`softmax`、`heads == kvheads` 的 attention、位置 0 的 `rope`、全 `-inf` 行的 softmax、9 个共享无设备契约向量与 `cpu` 的逐项比对，以及「九个算子恰好九个 pipeline 且重复调用不增长」「clone 共享缓存」「`Send + Sync` 且四线程并发结果一致」三例缓存/并发断言；`unsafe` 站点仍为 4/4。基准 `benches/metal.rs` 测量九个算子与 batch 8 的 matmul（见「本轮重测证据（2026-10-12）」）。 |
| 层 1：cuda（GPU 后端） | 已验证 | `crates/cuda` 实现 `api::backend` 与 `api::compute` 的**全部**算子：`matmul` 默认走 cuBLAS `gemm`（并经 `cudaexecutor::newnvrtc` 提供 NVRTC 内核作独立参考路径），`softmax` 走 cuDNN，`add`/`mul`/`silu`/`rmsnorm`/`rope`/`embedding`/`attention` 走内嵌 CUDA C++ 经 NVRTC 运行期编译。**编译侧已验证**：`cargo check`/`clippy -p cuda --target x86_64-unknown-linux-gnu --all-targets -- -D warnings` 与 `-p llm --test it --bench cuda` 均 exit 0，且不需要 CUDA toolkit 或 cuDNN 头文件（`cudarc` 使用 `dynamic-loading`）。**运行侧已分别在两台真机验证（E1）**：2026-10-12 在 RTX 2080 Ti（`sm_75`）、CUDA runtime 13.0、驱动 580.105.08、`rustc 1.99.0` 上，以 `LLM_CUDA_REQUIRE_DEVICE=1` 运行 `tests/it/cudacontract.rs` 的 36 例全部通过（18 例宿主侧定案 + 15 例走设备并与 `testkit::reference` 标量参考对照 + 3 例与共享向量列表一致），`cargo test --workspace --locked` 758 passed / 0 failed、输出 0 条 `E4 未执行`、`check.sh` exit 0；2026-10-15 在 Windows 11 + WSL2（Ubuntu 24.04、`nvidia-cuda-toolkit` 12.0.140、cudnn 9.3.0.75）的 NVIDIA T600 上复验，同一命令 **43 passed / 0 failed**（套件同期增至 43 例，见「本轮重测证据（2026-10-15）」）。首次执行即暴露并修正了一个用例缺陷。基准与每调用固定成本分解见 `docs/design.md` §21.10.3。`cuda-test` job 仍待自托管 runner 注册。 |
| 层 1：执行上下文与 dispatch 批处理（§21.13） | 已验证 | `api::submission`/`stepref`/`operand` 在 `api` 定义，`compute` 冻结一行未改；`cpu` 以「按序重放 `compute`」实现（`submissioncontract` 8/8），`metal` 以「一个 command encoder + 一次 commit/wait、中间结果驻留设备」实现（`metalcontract` 45/45，Apple M5 Pro），`cuda` 以「一个 stream 串行 + 一次 synchronize」实现（`cudacontract` 43/43，NVIDIA T600 / WSL2 Ubuntu 24.04 真机）。三后端「同一链经 `submission` 与经 `compute` 逐位相同」均成立；`unsafe` 站点不变（metal 4/4、cuda 3/3）。**性能验收已达成**：`forwardsubmission` 接入后 `gpu/modelforward` 的 `metal` 52.60 ms → `metalsubmission` 2.81 ms（改善 ≈18.7×，远超 10×），`submissionforward` 2 例在 CPU 上钉住 `forwardsubmission` 与 `forward` 逐位相同。 |
| 层 0：geometry 公共化 | 已验证 | 平台无关的 `geometry`（`matmulshape`/`axisshape`/`ropeshape`/`embeddingshape`/`attentionshape` + `binaryshape` + 7 个 op 常量）从 `metal`/`cuda` 各一份复制抽到 `api::geometry`，`attentionshape::new` 增 `maxheaddim` 参数承载唯一设备差异（Metal 4096 vs CUDA 8192）；消除 607 行重复。`api` 导出宽度 15→16（用 `pub mod geometry` 而非 `pub use` 以守住 A4 上界）。重构后 `metalcontract` 45/45、`cudacontract` 43/43 仍全过。 |
| 层 1：autograd | 已验证 | `crates/autograd` 实现图、变量与所支持算子的 VJP；根 `autograd` 套件对每个算子用有限差分参考检查，`cargo bench --no-run` 构建 `benches/autograd.rs`。 |
| 层 2：nn | 已验证 | `crates/nn` 实现参数、模块与层集；根 `nnlayers` 套件用 `testkit` 的独立标量参考检查每一层。 |
| 层 2：models（模型支持契约） | 已验证 | `crates/models` 实现因果 LM 契约与 Llama config/weights/block/model；根 `modelforward` 套件钉住确定性前向 logits。模型支持契约首次成文于设计文档，依据是对 `crates/models/src` 的源码核对。 |
| 层 2/3：KV cache 接线（`forward_cached`） | 已验证（正确性 + 性能方向） | `llamamodel::forward_cached`/`llamablock::forward_cached` 增量前向读/追加 `kvcache`，`infer` 的 `kvstate` 接线；attention 契约扩展支持 `qseq ≤ kvseq`，三后端一致。`metalcontract` 46/46（M5 Pro）、`cudacontract` 44/44（T600 真机）、CPU `incrementalattention` 2 + `kvcacheforward` 1 + `generation` 10 全绿。**性能**：接入 `forwardsubmission_cached` 后 `gpu/decode` 整组 −94.8%（每前向 52→2.8 ms），seq=16 增量快 ≈4.7%（方向正确；加速比由长序列阶段兑现，见「本轮重测证据（2026-10-16）」）。 |
| 层 2：tokenizer | 已验证 | `crates/tokenizer` 实现词表、合并、特殊词元、对话模板、编码/解码与引擎；根 `tokens` 套件通过。 |
| 层 2：data | 已验证 | `crates/data` 实现记录、数据集、打包、批次、采样器与加载器；根 `pipeline` 套件端到端组合它们。 |
| 层 2：format 与 checkpoint（检查点格式契约） | 已验证 | `crates/format` 与 `crates/checkpoint` 实现解析/校验与原子 save/load/兼容性；根 `artifact` 与 `roundtrip` 套件通过。检查点格式契约首次成文于设计文档，依据是对两个 crate 源码的核对。 |
| 层 2：format 与 checkpoint（.u 增量检查点契约 §12.12） | 已验证 | `.u` 格式的合并/恢复逻辑与哈希/校验体系已在 `checkpoint` crate 实现并有测试覆盖：内容寻址分块（CDC）、增量创建/合并/按需读取、gate rANS 逐块熵编码、链压缩与溯源（`compact`/`verifyprovenance`）、`checkpointerror` 损坏分类（`iscorruption()`），以及 Ed25519 签名（`encode_signed`/`verifysignature`，`FLAGSIGNED` 标志位）、AES-256-GCM 分块加密（`encode_encrypted`/`decode_encrypted`，`FLAGENCRYPTED` 标志位）、多排布 layout 持久化（`setlayout`/`tryfrombyteswithlayout`）、稀疏存储（`addsparse`/`desparsify`）、对称 int8 量化内核（`quantize`/`dequantize`）、多文件分片（`shards::split`/`join`）与 mmap 零拷贝（`mmapsource`，safe API）。根 `tests/it/checkpointunits.rs` 的 blocks/delta/merge/reader/compact/compression/quant/signature/encryption 用例通过；`cargo clippy --workspace --all-targets -D warnings`、`checknaming.sh`、`checkunsafe.sh`（零新增 unsafe 站点）通过。固定 64 字节字段布局仍为草案。 |
| 层 3：optim | 已验证 | `crates/optim` 在拥有的张量上实现 AdamW、动量 SGD、梯度裁剪与 LR 调度；根 `optim` 套件通过，整合的 `benches/optim.rs` 可构建。 |
| 层 3：context | 已验证 | `crates/context` 实现取消、预算、指标、执行上下文与 worker；根 `context` 套件通过。 |
| 层 3：train（训练契约） | 已验证 | `crates/train` 实现配置、损失、梯度累积、步进、评估与恢复；根 `lossdecrease`、`trainingresume`、`cancellation` 套件通过。经 `nn`/`models` 的反向是声明的限制。训练契约首次成文于设计文档，依据是对 `crates/train/src` 的源码核对。 |
| 层 3：infer（推理契约） | 已验证 | `crates/infer` 实现引擎、请求/生成状态、prefill/decode、KV cache、采样、停止与流式；根 `generation`/`isolation` 套件通过。推理契约首次成文于设计文档，依据是对 `crates/infer/src` 的源码核对。 |
| 层 4：cli | 已验证 | `crates/cli` 实现解析、分发与中央退出码表；根 `cli` 套件覆盖解析、错误映射与退出码，`cli train` 对声明的限制返回文档化退出码 4。 |
| 层 4：serve | 已验证 | `crates/serve` 实现 std-only HTTP/1.1 适配器、路由、准入控制与 deadline→cancellation 映射；根 `service` 与 `server` 套件（含真实 loopback socket）通过。 |
| 基础设施：testkit、bench | 已验证 | 两者均为依赖叶子且非生产依赖；`testkit` 仅提供夹具/断言/参考实现/微型模型，`bench` 提供共享确定性工作负载；根基准目标消费它们并成功构建。 |

### conformance.md（本记录）

| 主题 | 判定 | 证据 |
| --- | --- | --- |
| 一致性记录本身 | 拟定 | 判定随 2026-10-10 文档合并由 33 份文档映射到 4 份文档；判定与证据沿用合并前已执行的检查，并由 2026-10-10 对抗性修正补记静态门重测（见「本轮重测证据」）。每当某文档的主题被实现、测量或发现违反其规则时，本记录 MUST 更新。 |

本记录 MUST 在文档主题被实现、测量或发现违反其自身规则时更新。
