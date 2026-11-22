//! 职责：真机正确性判据——独立参考、可证误差界、边界形状、写满保证、布局前置。
//!
//! 与 `gemm.rs` / `tensor.rs` / `attention.rs` 的分工：那三份用例主要出吞吐，对拍用的是夹具的
//! [`matmul_values`] / [`attention_inputs`]，取值是 `0.25·k` 这类**可精确表示**的格点。取这样的值
//! 时每个乘积在 f32 里都是精确的，最佳累加顺序下误差恒为 `0.00e0`——**任何累加顺序错误都会被
//! 掩盖**，`误差 0.00e0` 并不是「算对了」的证据。本文件的输入一律用满尾数伪随机值，判据换成
//! 本身可证的误差界，并补上三处此前完全未验证的面：
//!
//! - 大形状（`M·N·K > 1024³` 时夹具不算参考，只打印 `误差 NaN`）；
//! - 张量核读的 `Bt`（B 的 half 转置）——布局前置本身从未被对拍过，它错则整批 GEMM 全错；
//! - 输出是否**被写满**、同一派发是否**可逐位复现**（漏写与竞争只在初值上露头）。

use super::*;

/// SplitMix64：确定性伪随机位模式。免依赖、跨平台一致，用例失败时可照序号复算。
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// 满尾数 f32，落在 `[-0.5, 0.5)`：23 位尾数全随机，不落在任何 2 的幂格点上，
    /// 因而乘积必须真正舍入，累加顺序的错误会显现。
    fn f32(&mut self) -> f32 {
        let bits = (self.bits() >> 40) as u32;
        f32::from_bits(0x3f80_0000 | (bits & 0x007f_ffff)) - 1.5
    }

    /// 落在 `[0.25, 0.5)` 的满尾数 f32（随机符号）：half 的正规区间，避开次正规与溢出的舍入差异。
    fn half_normal(&mut self) -> f32 {
        let bits = (self.bits() >> 40) as u32;
        let magnitude = 0.25 + 0.25 * ((bits & 0x007f_ffff) as f32 / 8_388_608.0);
        if bits & 0x0080_0000 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    fn values(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.f32()).collect()
    }
}

/// `γ_n = n·u / (1 − n·u)`，`u = 2^-24`（f32 就近取偶的单位舍入误差）。
///
/// 内积 `Σ a·b` 有 `k` 个乘积与 `k` 次加法，且每个乘积自身也要舍入一次，故取 `n = 2k`。
fn gamma(n: usize) -> f64 {
    let nu = n as f64 * 2f64.powi(-24);
    nu / (1.0 - nu)
}

/// 单个输出元素的 f64 参考与其前向误差界：返回 `(Σa·b, γ_{2k}·Σ|a·b|)`。
///
/// 参考在 f64 里独立重算，与设备的任何一条路径不共享一行算术（对照见算子侧 F6 自我对拍）。
fn reference_and_bound(a: &[f32], b: &[f32], n: usize, k: usize, i: usize, j: usize) -> (f64, f64) {
    let mut want = 0.0f64;
    let mut magnitude = 0.0f64;
    for p in 0..k {
        let x = f64::from(a[i * k + p]);
        let y = f64::from(b[p * n + j]);
        want += x * y;
        magnitude += x.abs() * y.abs();
    }
    (want, gamma(2 * k) * magnitude)
}

/// 逐元素按可证误差界判定：`|fl(Σa·b) − Σa·b| ≤ γ_{2k}·Σ|a·b|`。
///
/// 界随数据缩放——`Σ|a·b|` 大就放宽、接近零就收紧。这正是不需要手调常数的原因：
/// 定值阈值（如 `1e-3`）在真值量级跨十几个数量级时必然要么太松、要么误报。
fn assert_within_bound(
    label: &str,
    got: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(got.len(), m * n, "{label} 的输出元素数应与 M·N 一致");
    let mut worst = (0.0f64, 0usize, 0usize);
    for i in 0..m {
        for j in 0..n {
            let (want, bound) = reference_and_bound(a, b, n, k, i, j);
            let error = (f64::from(got[i * n + j]) - want).abs();
            assert!(
                error <= bound,
                "{label} 的元素 ({i},{j})：|差|={error:.3e} 超出误差界 {bound:.3e}\
                 （f64 参考 {want:.6e}、设备 {:.6e}，Σ|a·b| 决定界宽）",
                got[i * n + j]
            );
            let used = if bound > 0.0 { error / bound } else { 0.0 };
            if used > worst.0 {
                worst = (used, i, j);
            }
        }
    }
    println!(
        "{label}：最差元素 ({},{}) 用掉误差界的 {:.0}%",
        worst.1,
        worst.2,
        worst.0 * 100.0
    );
}

/// 快照设备缓冲（内核已同步完成，共享内存可直接读）。
fn snapshot(buf: &mut MetalBuf) -> Vec<u8> {
    buf.as_mut_bytes().to_vec()
}

/// 把输出缓冲填成 NaN：内核若漏写任何一块，NaN 会原样留下，不会被「旧值是 0」蒙混过去。
fn fill_nan(buf: &mut MetalBuf) {
    for slot in buf.as_mut_bytes().chunks_exact_mut(4) {
        slot.copy_from_slice(&f32::NAN.to_ne_bytes());
    }
}

/// half 位模式还原成 f32（本文件只喂正常区间的值，不涉及次正规与溢出）。
fn half_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let magnitude = match exponent {
        0 => mantissa as f32 * 2f32.powi(-24),
        31 if mantissa == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + mantissa as f32 / 1024.0) * 2f32.powi(exponent - 15),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

/// 注意力输出的相对 Frobenius 误差 `‖got − want‖ / ‖want‖`。
///
/// 分母取整批范数：逐元素相对误差在真值接近 0 时会被灾难性抵消放大成 O(1)，
/// 那不是「算错了」而是「分母选错了」。
fn frobenius(got: &[f32], want: &[f64]) -> f64 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (got, want) in got.iter().zip(want) {
        let difference = f64::from(*got) - want;
        numerator += difference * difference;
        denominator += want * want;
    }
    (numerator / denominator).sqrt()
}

/// 注意力 f64 参考：与内核同一数学（`softmax(QKᵀ/√D)·V`），最大项在 f64 里扣除。
fn attention_reference(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f64> {
    let scale = (d as f64).sqrt();
    let mut out = vec![0.0f64; s * d];
    let mut scores = vec![0.0f64; s];
    for i in 0..s {
        let mut row_max = f64::NEG_INFINITY;
        for j in 0..s {
            let mut dot = 0.0f64;
            for p in 0..d {
                dot += f64::from(q[i * d + p]) * f64::from(k[j * d + p]);
            }
            scores[j] = dot / scale;
            row_max = row_max.max(scores[j]);
        }
        let mut sum = 0.0f64;
        for score in scores.iter_mut() {
            *score = (*score - row_max).exp();
            sum += *score;
        }
        for h in 0..d {
            let mut acc = 0.0f64;
            for j in 0..s {
                acc += scores[j] * f64::from(v[j * d + h]);
            }
            out[i * d + h] = acc / sum;
        }
    }
    out
}

/// 上传一路 Q/K/V 并开一路输出缓冲：注意力四路缓冲的固定装配。
fn attention_bufs(
    device: &MetalDevice,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    len: usize,
) -> (MetalBuf, MetalBuf, MetalBuf, MetalBuf) {
    (
        upload(device, q, "Q"),
        upload(device, k, "K"),
        upload(device, v, "V"),
        output_buf(device, len, "Y"),
    )
}

/// f32 GEMM 的逐元素前向误差界：96×80×96，输入满尾数、非退化，标量网格与 threadgroup 分块
/// 两条路径都必须落在 `γ_{2k}·Σ|a·b|` 内。
///
/// 为什么不用夹具的 [`matmul_inputs`]：那套取值是 `0.25`/`0.5` 的整数倍，乘积在 f32 里**精确**，
/// 最佳累加顺序下误差恒为 `0.00e0`，累加顺序错了也测不出来。这里换成满尾数输入，判据也从
/// 定值阈值换成可推导的误差界。
#[test]
fn gemm_stays_within_forward_error_bound_on_full_mantissa_inputs() {
    const M: usize = 96;
    const N: usize = 80;
    const K: usize = 96;

    let Some((device, queue)) = bench_device("前向误差界 GEMM") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-bound");

    let mut rng = Rng::new(0x5eed_1234_abcd_0001);
    let a = rng.values(M * K);
    let b = rng.values(K * N);
    let a_buf = upload(&device, &a, "A");
    let b_buf = upload(&device, &b, "B");
    let mut y = output_buf(&device, M * N, "Y");
    let grid = [M as u32, N as u32, 1];

    // 路径一：标量网格（每线程算一个输出元素）。
    let grid_pipe = grid_pipeline(&device, &catalogue, &dirs, "matmul_f32", M, N, "标量网格");
    assert!(
        queue.execute(&grid_pipe, &[&a_buf, &b_buf, &y], grid, [1, 1, 1]),
        "标量网格派发应成功"
    );
    assert_within_bound(
        "标量网格 96×80×96",
        &read_back(y.as_mut_bytes()),
        &a,
        &b,
        M,
        N,
        K,
    );

    // 路径二：threadgroup 分块（16×16，M/N/K 都是 16 的整数倍，前置条件成立）。
    let tiled = program(
        &catalogue,
        &dirs,
        "matmul_f32",
        Metal,
        Mode::tiled(16, 1),
        M,
        N,
    )
    .and_then(|stages| stages.one("分块"))
    .expect("分块 MSL 应能生成");
    assert!(
        tiled.contains("threadgroup float As["),
        "96/80/96 都是 16 的倍数，应走分块 lowering"
    );
    let tiled_pipe = device.compile(&tiled).expect("分块 MSL 应能编译");
    assert!(
        queue.execute(&tiled_pipe, &[&a_buf, &b_buf, &y], grid, [16, 16, 1]),
        "分块派发应成功"
    );
    assert_within_bound(
        "分块 16×16 96×80×96",
        &read_back(y.as_mut_bytes()),
        &a,
        &b,
        M,
        N,
        K,
    );
}

/// 边界形状：`1×1×1`、`3×5×7`、`17×33×65`、`129×7×3` 都要与 f64 参考落在误差界内；
/// 同时验证 `Mode::tiled` 在不整除时**必须回落**，不能算出一半分块就交付。
///
/// `65 = 4·16 + 1`、`33 = 2·16 + 1` 专打尾块：分块内核最容易错的地方是最后一块不满时的守卫，
/// 而整除形状（`64`、`80`）永远走不到那条分支。
#[test]
fn gemm_handles_ragged_tail_shapes_and_refuses_to_pretend_it_tiled() {
    let Some((device, queue)) = bench_device("边界形状 GEMM") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-ragged");

    for (m, n, k) in [(1, 1, 1), (3, 5, 7), (17, 33, 65), (129, 7, 3)] {
        let mut rng = Rng::new(0x5eed_0000_0000_0002 ^ ((m * 1000 + n * 10 + k) as u64));
        let a = rng.values(m * k);
        let b = rng.values(k * n);
        let a_buf = upload(&device, &a, "A");
        let b_buf = upload(&device, &b, "B");
        let mut y = output_buf(&device, m * n, "Y");
        let label = format!("{m}×{n}×{k}");

        let grid_pipe = grid_pipeline(&device, &catalogue, &dirs, "matmul_f32", m, n, "标量网格");
        assert!(
            queue.execute(
                &grid_pipe,
                &[&a_buf, &b_buf, &y],
                [m as u32, n as u32, 1],
                [1, 1, 1]
            ),
            "{label} 标量网格派发应成功"
        );
        assert_within_bound(&label, &read_back(y.as_mut_bytes()), &a, &b, m, n, k);

        // 不整除时 `Mode::tiled` 的前置条件不成立：必须回落到标量路径。
        let fallback = program(
            &catalogue,
            &dirs,
            "matmul_f32",
            Metal,
            Mode::tiled(16, 1),
            m,
            n,
        )
        .and_then(|stages| stages.one("分块"))
        .expect("不整除时应能回落到标量路径");
        assert!(
            !fallback.contains("threadgroup float As["),
            "{label} 不被 16 整除，不得走分块 lowering"
        );
    }
}

/// 写满 + 可复现：输出缓冲先填 NaN，派发后**每个元素都必须变成有限值**，且同一输入重复派发要
/// 逐位相同（含张量核双缓冲路径）。
///
/// 这条用例抓的是两类只在特定初值下露头的缺陷：漏写（尾块/越界守卫写少了）、竞争（共享内存
/// 少一次 `threadgroup_barrier`，或 ping-pong 两片被复用得过早）。二者在「输出缓冲本就是 0」
/// 的夹具里都不会显形。
#[test]
fn gemm_writes_every_output_and_repeats_bit_exactly() {
    const M: usize = 96;
    const N: usize = 80;
    const K: usize = 96;

    let Some((device, queue)) = bench_device("写满与可复现") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-repeat");

    let mut rng = Rng::new(0x5eed_0000_0000_0003);
    let a = rng.values(M * K);
    let b = rng.values(K * N);
    let a_buf = upload(&device, &a, "A");
    let b_buf = upload(&device, &b, "B");
    let mut y = output_buf(&device, M * N, "Y");
    let grid = [M as u32, N as u32, 1];
    let block = [16, 16, 1];

    let tiled = program(
        &catalogue,
        &dirs,
        "matmul_f32",
        Metal,
        Mode::tiled(16, 1),
        M,
        N,
    )
    .and_then(|stages| stages.one("分块"))
    .expect("分块 MSL 应能生成");
    let tiled_pipe = device.compile(&tiled).expect("分块 MSL 应能编译");

    fill_nan(&mut y);
    assert!(
        queue.execute(&tiled_pipe, &[&a_buf, &b_buf, &y], grid, block),
        "分块派发应成功"
    );
    let first = snapshot(&mut y);
    assert!(
        read_back(&first).iter().all(|value| value.is_finite()),
        "分块内核必须写满每个输出元素，NaN 残留即漏写"
    );
    assert_within_bound("分块写满 96×80×96", &read_back(&first), &a, &b, M, N, K);

    // 再填一遍 NaN 再派发：初值不影响结果，两次必须逐位相同。
    fill_nan(&mut y);
    assert!(
        queue.execute(&tiled_pipe, &[&a_buf, &b_buf, &y], grid, block),
        "分块重复派发应成功"
    );
    assert_eq!(first, snapshot(&mut y), "同一输入的两次派发应逐位相同");

    // 张量核双缓冲：ping-pong 两片缓冲与「一次 barrier」的软件流水最容易出竞争。
    use op::sig::Signature;

    let geometry = TensorGeometry::new(128, 64, 96, 8);
    let mut stage = tensor_stage(&device, &queue, (256, 256, 1024), true);
    let signature = Signature::parse(
        "bench",
        "void cutile_bench_kernel(const float *a, const float *b, float *y, int M, int N, int K) {}",
    )
    .expect("自建签名应能解析");
    let msl = crate::tiled::metal_tensor_gemm(
        "bench",
        &signature,
        &tensor_sample(256, 256, 1024),
        geometry,
    )
    .expect("张量核 MSL 应能生成");
    assert!(
        msl.contains("threadgroup half sb[2]"),
        "两片 B 片装得下 32 KiB 时应走双缓冲"
    );
    let pipe = tensor_pipe(&device, &queue, &stage, (256, 256), geometry, &msl);

    fill_nan(&mut stage.y);
    assert!(
        queue.execute(
            &pipe.pipe,
            &[&stage.a, &stage.bt, &stage.y],
            pipe.grid,
            pipe.block
        ),
        "双缓冲派发应成功"
    );
    let first = snapshot(&mut stage.y);
    assert!(
        read_back(&first).iter().all(|value| value.is_finite()),
        "双缓冲内核必须写满每个输出元素，NaN 残留即漏写"
    );
    assert_eq!(
        first,
        snapshot(&mut stage.y),
        "双缓冲的两次派发应逐位相同（ping-pong 复用过早会读出半成品）"
    );
    let error = stage.error("双缓冲写满 256×256×1024");
    assert!(
        error < TENSOR_TOLERANCE,
        "双缓冲的误差 {error:.2e} 超过 {TENSOR_TOLERANCE:.0e}"
    );
}

/// int8 GEMM 的判据是**逐位相等**，不是误差界：`i8×i8` 的乘积与 `i32` 累加都不会舍入，
/// 因而设备输出必须与 CPU 的 `i32` 参考完全一致（`|acc| ≤ 2.1e6 < 2^24`，转 f32 亦精确）。
///
/// 两档形状各自的契约同时验证：`24×40×128` 被 `tile=8` 整除走 int8 分块；
/// `24×40×120` 的 `M` 不被 `tile=16` 整除，必须回落到标量路径
/// ——回落后的结果仍然要逐位正确，否则「回落」就成了「算错」的别名。
#[test]
fn int8_gemm_is_bit_exact_against_i32_accumulation() {
    let Some((device, queue)) = bench_device("int8 精确对拍") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-int8-exact");

    for (m, n, k, tile) in [(24usize, 40usize, 128usize, 8usize), (24, 40, 120, 16)] {
        let label = format!("int8 {m}×{n}×{k} tile={tile}");
        let mut rng = Rng::new(0x5eed_0000_0000_0004 ^ ((m * n * k + tile) as u64));
        // 满量程 i8：字节在 0..=255 全取到，转 i8 后覆盖 -128..=127。
        let a: Vec<u8> = (0..m * k).map(|_| rng.bits() as u8).collect();
        let b: Vec<u8> = (0..k * n).map(|_| rng.bits() as u8).collect();
        let (signed_a, signed_b): (Vec<i8>, Vec<i8>) = (
            a.iter().map(|&value| value as i8).collect(),
            b.iter().map(|&value| value as i8).collect(),
        );
        let mut acc = vec![0i32; m * n];
        op::matmul_i8::forward_tiled(&signed_a, &signed_b, m, n, k, &mut acc)
            .expect("CPU i32 参考");

        let a_buf = upload_bytes(&device, &a, "A-int8");
        let b_buf = upload_bytes(&device, &b, "B-int8");
        let mut y = output_buf(&device, m * n, "Y-int8");

        let msl = program(
            &catalogue,
            &dirs,
            "matmul_i8",
            Metal,
            Mode::int8(tile),
            m,
            n,
        )
        .and_then(|stages| stages.one("int8"))
        .unwrap_or_else(|error| panic!("{label} 的 MSL 应能生成：{error}"));
        let tiled = msl.contains("threadgroup int As[");
        match (m % tile, n % tile, k % tile) {
            (0, 0, 0) => assert!(tiled, "{label} 三个轴都整除，应走 int8 分块 lowering"),
            _ => assert!(!tiled, "{label} 有不整除的轴，应回落到标量路径"),
        }

        let pipe = device.compile(&msl).expect("int8 MSL 应能编译");
        assert!(
            queue.execute(
                &pipe,
                &[&a_buf, &b_buf, &y],
                [m as u32, n as u32, 1],
                [tile as u32, tile as u32, 1]
            ),
            "{label} 派发应成功"
        );
        let got = read_back(y.as_mut_bytes());
        for (idx, (got, want)) in got.iter().zip(&acc).enumerate() {
            assert_eq!(
                got.to_bits(),
                (*want as f32).to_bits(),
                "{label} 的元素 ({},{})：设备 {got:.1e}、i32 参考 {want}（int8 路径不允许任何舍入）",
                idx / n,
                idx % n
            );
        }
        println!(
            "{label}：{} 个元素逐位相等（{}）",
            m * n,
            if tiled { "分块" } else { "回落标量" }
        );
    }
}

/// 注意力在**分数溢出区间**仍然正确：`exp` 的底数必须扣掉行最大值，否则 `exp(5080)` 直接变
/// `inf`，整行输出成 NaN。
///
/// 输入构造成可解析的形式而不是随机大数：`Q = 320·e₀`、`K[j] = j·e₀`，于是
/// `score_j = 320·j/√64 = 40·j`，最大分数 `40·(S−1) = 5080`。正确答案因此是闭式的——
/// 只有 `j = S−1` 的权重非零（次大项是 `exp(−40) ≈ 4e−18`），输出必须等于 `V[S−1]`。
/// 这样既逼出了溢出，又有一个与实现无关的精确期望值，不受浮点敏感度影响。
#[test]
fn attention_survives_score_overflow_and_matches_closed_form() {
    const S: usize = 128;
    const D: usize = 64;
    const THREADS: usize = 64;

    let Some((device, queue)) = bench_device("注意力溢出") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-attn-overflow");

    let mut rng = Rng::new(0x5eed_0000_0000_0005);
    let mut q = vec![0.0f32; S * D];
    let mut k = vec![0.0f32; S * D];
    for i in 0..S {
        q[i * D] = 320.0;
    }
    for j in 0..S {
        k[j * D] = j as f32;
    }
    let v = rng.values(S * D);
    let (q_buf, k_buf, v_buf, mut y_buf) = attention_bufs(&device, &q, &k, &v, S * D);

    // 闭式期望：第 S−1 个键独占全部权重。
    let expected: Vec<f64> = (0..D).map(|h| f64::from(v[(S - 1) * D + h])).collect();
    let check = |label: &str, y: &mut MetalBuf| {
        let got = read_back(y.as_mut_bytes());
        assert!(
            got.iter().all(|value| value.is_finite()),
            "{label} 输出出现 NaN/inf：行最大值没有被扣除，exp 溢出了"
        );
        let error = frobenius(&got, &expected);
        assert!(
            error < 1e-5,
            "{label} 的相对 Frobenius 误差 {error:.2e} 超过 1e-5（分数已饱和，输出应等于 V[S−1]）"
        );
        println!("{label}：最大分数 5080 未溢出，相对误差 {error:.2e}");
    };

    // 路径一：每查询行一线程的标量网格。
    let grid_pipe = grid_pipeline(
        &device,
        &catalogue,
        &dirs,
        "attention_f32",
        S,
        D,
        "网格注意力",
    );
    assert!(
        queue.execute(
            &grid_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            [S as u32, 1, 1],
            [1, 1, 1]
        ),
        "网格注意力派发应成功"
    );
    check("网格注意力 S=128 D=64", &mut y_buf);

    // 路径二：flash（在线 softmax 的分块 rescale 在极端分数下最容易失稳）。
    let flash = program(
        &catalogue,
        &dirs,
        "attention_f32",
        Metal,
        Mode::flash(THREADS, 64),
        S,
        D,
    )
    .and_then(|stages| stages.one("flash"))
    .expect("flash MSL 应能生成");
    let flash_pipe = device.compile(&flash).expect("flash MSL 应能编译");
    assert!(
        queue.execute(
            &flash_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            [(S * THREADS) as u32, 1, 1],
            [THREADS as u32, 1, 1]
        ),
        "flash 派发应成功"
    );
    check("flash 注意力 S=128 D=64", &mut y_buf);
}

/// flash 的键块边界：`S` 不被 `block` 整除时必须走带守卫的尾块，且结果与 f64 参考一致；
/// `S` 整除时则不得生成尾块守卫（守卫存在本身会拖慢热路径）。
///
/// `200` 对 `64`/`128`/`192` 都余数非零（`8`/`72`/`8`），三个块大小的尾块路径全被覆盖；
/// `192` 对 `64` 整除，用来验证生成器确实会去掉守卫——否则「尾块守卫」这条判据就成了空话。
#[test]
fn flash_attention_respects_key_block_tails() {
    const D: usize = 64;
    const THREADS: usize = 64;

    let Some((device, queue)) = bench_device("flash 块边界") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-flash-tail");

    for s in [200usize, 192] {
        let mut rng = Rng::new(0x5eed_0000_0000_0006 ^ s as u64);
        // 幅度 4：分数标准差约 13，最大约 33——有区分度又不至于溢出。
        let q: Vec<f32> = rng.values(s * D).iter().map(|x| x * 4.0).collect();
        let k: Vec<f32> = rng.values(s * D).iter().map(|x| x * 4.0).collect();
        let v = rng.values(s * D);
        let reference = attention_reference(&q, &k, &v, s, D);
        let (q_buf, k_buf, v_buf, mut y_buf) = attention_bufs(&device, &q, &k, &v, s * D);
        let launch = [(s * THREADS) as u32, 1, 1];
        let block = [THREADS as u32, 1, 1];

        for key_block in [64usize, 128, 192] {
            let msl = program(
                &catalogue,
                &dirs,
                "attention_f32",
                Metal,
                Mode::flash(THREADS, key_block),
                s,
                D,
            )
            .and_then(|stages| stages.one("flash"))
            .unwrap_or_else(|error| panic!("S={s} block={key_block} 的 MSL 应能生成：{error}"));
            let tail = !s.is_multiple_of(key_block);
            assert_eq!(
                msl.contains("if (kb + j < S)"),
                tail,
                "S={s} block={key_block}：尾块守卫{}，生成结果不符",
                if tail { "应存在" } else { "不应存在" }
            );
            let pipe = device.compile(&msl).expect("flash MSL 应能编译");
            assert!(
                queue.execute(&pipe, &[&q_buf, &k_buf, &v_buf, &y_buf], launch, block),
                "S={s} block={key_block} 派发应成功"
            );
            let got = read_back(y_buf.as_mut_bytes());
            assert!(
                got.iter().all(|value| value.is_finite()),
                "S={s} block={key_block} 输出出现 NaN/inf"
            );
            let error = frobenius(&got, &reference);
            assert!(
                error < 2e-4,
                "S={s} block={key_block} 的相对 Frobenius 误差 {error:.2e} 超过 2e-4"
            );
            println!(
                "flash S={s} block={key_block}（{}）：相对误差 {error:.2e}",
                if tail { "尾块" } else { "整除" }
            );
        }
    }
}

/// 张量核的布局前置 `Bt[j][i] = half(B[i][j])` 逐元素**位相等**。
///
/// 这条前置此前完全未被对拍：`tensor_stage` 跑了转置，但只把 GEMM 的输出与参考比，而大形状的
/// 参考又会被跳过（`误差 NaN`）。若转置错位一列，整批 GEMM 全错却可能仍然「误差很小」地自洽。
///
/// `100×70` 两个方向都不被 32 整除，`32×32` 分块的边界守卫与共享内存补零都被覆盖。
#[test]
fn transpose_half_layout_is_bit_exact_including_tail_tiles() {
    const K: usize = 100;
    const N: usize = 70;

    let Some((device, queue)) = bench_device("half 转置布局") else {
        return;
    };
    let mut rng = Rng::new(0x5eed_0000_0000_0007);
    // 落在 half 正规区间：避开次正规与溢出的舍入差异，使「位相等」是唯一正确的期望。
    let b: Vec<f32> = (0..K * N).map(|_| rng.half_normal()).collect();

    let source = upload(&device, &b, "B");
    let mut transposed = device.new_buf(N * K * 2).expect("Bt 缓冲");
    let msl = crate::tiled::metal_transpose_half(K, N).expect("转置 MSL 应能生成");
    let pipe = device.compile(&msl).expect("转置 MSL 应能编译");
    // 派发几何与 `tensor_stage` 完全一致：线程组 [⌈N/32⌉, ⌈K/32⌉]、组内 32×8。
    assert!(
        queue.execute(
            &pipe,
            &[&source, &transposed],
            [(N as u32).div_ceil(32) * 32, (K as u32).div_ceil(32) * 8, 1],
            [32, 8, 1]
        ),
        "转置派发应成功"
    );

    let bytes = transposed.as_mut_bytes();
    assert_eq!(bytes.len(), N * K * 2, "Bt 应按 half 分配 N·K 个元素");
    let mut checked = 0usize;
    for j in 0..N {
        for i in 0..K {
            let at = (j * K + i) * 2;
            let got = half_to_f32(u16::from_ne_bytes([bytes[at], bytes[at + 1]]));
            let want = half_precision(b[i * N + j]);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "Bt[{j}][{i}] 与 half(B[{i}][{j}]) 不一致：{got:.8e} ≠ {want:.8e}"
            );
            checked += 1;
        }
    }
    println!("half 转置 {K}×{N}：{checked} 个元素逐位相等（含尾块）");
}

/// 大形状的抽样校验：`2048×2048×4096`（`M·N·K = 1.7e10 > 1024³`）在夹具里只打印 `误差 NaN`，
/// 即**吞吐跑得再多也没有任何正确性证据**。这里取 512 个散点、每点 `O(K)` 的 f64 参考，
/// 用与张量核小形状同一判据（[`TENSOR_TOLERANCE`]）判定，把盲区换成实数。
///
/// 抽样而非全量：全量 f64 参考是 `1.7e10` 次乘法（分钟级），而「布局错位、尾块漏算、半个 tile
/// 没写」这类缺陷必然在 512 个散点上暴露——它不是精度认证，是**覆盖认证**。
#[test]
fn tensor_gemm_large_shape_is_verified_on_sampled_elements() {
    const M: usize = 2048;
    const N: usize = 2048;
    const K: usize = 4096;
    const SAMPLES: usize = 512;

    let Some((device, queue)) = bench_device("大形状抽样校验") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-large-sample");

    // 夹具上传的输入就是 `matmul_values` 的确定性输出，这里重新算一遍拿到同一批数。
    let mut stage = tensor_stage(&device, &queue, (M, N, K), false);
    let (a, b) = matmul_values(M, N, K);

    let geometry = crate::tiled::tensor_geometry(M, N, K);
    let msl = program(
        &catalogue,
        &dirs,
        "matmul_f32",
        Metal,
        Mode::tensor(geometry),
        M,
        N,
    )
    .and_then(|stages| stages.one("张量核"))
    .expect("张量核 MSL 应能生成");
    assert!(
        msl.contains("mpp::tensor_ops::matmul2d"),
        "Gemm 骨架应走张量核 lowering"
    );
    // `tensor_pipe` 内部即完成派发并断言成功，这里只需要它的副作用（输出已落在 `stage.y`）。
    let _pipe = tensor_pipe(&device, &queue, &stage, (M, N), geometry, &msl);

    let got = read_back(stage.y.as_mut_bytes());
    assert_eq!(got.len(), M * N, "输出元素数应与 M·N 一致");
    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    for sample in 0..SAMPLES {
        // 互质步长散点，覆盖全部 tile 行/列（含 2048 与 4096 的边界附近）。
        let i = (sample * 7919) % M;
        let j = (sample * 104_729) % N;
        let mut want = 0.0f64;
        for p in 0..K {
            // 设备侧 A 与 B 都按 half 精度进入张量核，参考按同一精度舍入后再在 f64 里累加。
            want +=
                f64::from(half_precision(a[i * K + p])) * f64::from(half_precision(b[p * N + j]));
        }
        let value = got[i * N + j];
        assert!(
            value.is_finite(),
            "抽样元素 ({i},{j}) 不是有限值：张量核输出有 NaN/inf"
        );
        let error = (f64::from(value) - want).abs() / want.abs().max(1.0);
        if error > worst {
            worst = error;
            worst_at = (i, j);
        }
    }
    assert!(
        worst < TENSOR_TOLERANCE,
        "大形状抽样误差 {worst:.2e}（元素 {worst_at:?}）超过 {TENSOR_TOLERANCE:.0e}"
    );
    println!(
        "张量核 {M}×{N}×{K}（M·N·K={:.1e}）：{SAMPLES} 个抽样点最差相对误差 {worst:.2e} @ {worst_at:?}",
        (M * N * K) as f64
    );
}
