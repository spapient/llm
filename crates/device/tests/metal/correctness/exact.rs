//! 职责：int8 GEMM 与 i32 累加的位级一致性判据。

use super::*;
use tens::View;

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
        nn::matmul_i8::forward_tiled(
            View::matrix(m, k, &signed_a).expect("a 二维"),
            View::matrix(k, n, &signed_b).expect("b 二维"),
            &mut acc,
        )
        .expect("CPU i32 参考");

        let a_buf = upload_bytes(&device, &a, "A-int8");
        let b_buf = upload_bytes(&device, &b, "B-int8");
        let mut y = output_buf(&device, m * n, "Y-int8");

        let msl = program_sample(
            &catalogue,
            &dirs,
            "matmul_i8",
            Metal,
            Mode::int8(tile),
            &nn::matmul_i8::sample_k(m, n, k),
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
