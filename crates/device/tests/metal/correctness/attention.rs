//! 职责：注意力分数溢出与键块尾块守卫的正确性判据。

use super::*;

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
