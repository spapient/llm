//! `ledger` 模块的单元测试。
use super::*;

use diagnostic::Code;
use storage::leb::Cursor;

/// diag(3, 2, 1, 0.5)：谱隙 1、1、0.5 都远离机器精度，判定结果可逐点预算。
fn diag4() -> Vec<f32> {
    let mut w = vec![0.0f32; 16];

    for (i, d) in [3.0f32, 2.0, 1.0, 0.5].iter().enumerate() {
        w[i * 4 + i] = *d;
    }

    w
}

/// 6×6 精确秩 2：两行线性无关，其余行全零，因此 σ3 = … = σ6 = 0。
///
/// 行取整数是为了让 f64 → f32 的重构精确：夹具自己不能引入误差，否则测的就不是证书。
fn rank_two() -> Vec<f32> {
    let mut w = vec![0.0f32; 36];

    w[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    w[6..12].copy_from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

    w
}

testkit::test_cases! {
fn stability_threshold_is_strict() {
    assert_eq!(stable(1.0, 0.1), Stable::Pass);

    assert_eq!(stable(0.2, 0.1), Stable::Fail);

    assert_eq!(stable(0.2, 0.099), Stable::Pass);

    assert_eq!(stable(0.2, 0.1), Stable::Fail, "等号不算严格大于");
}

fn certified_rank_stops_at_the_spectral_gap() {
    let sigma = [10.0, 5.0, 1.0, 0.0];

    assert_eq!(certified_rank(&sigma, 0.1), Some(3), "三处谱隙都过 2ε");

    assert_eq!(
        certified_rank(&sigma, 0.6),
        Some(2),
        "1 − 0 = 1 不大于 2·0.6，只好退回 r = 2"
    );

    assert_eq!(
        certified_rank(&[1.0, 1.0], 0.1),
        None,
        "没有谱隙就没有可判秩，不许瞎选一个"
    );
}

fn the_payload_audit_rejects_a_certificate_that_names_another_measurement() {
    let cert = certify(&diag4(), 4, 4).expect("可认证").expect("有谱隙");

    audit(&cert.note, 4, 4, &cert.low, &cert.rest).expect("真实载荷必须自洽");

    // 三条读端复核各自独立：改一条就得被这条载荷自己拒掉。

    for tampered in [
        Note {
            method: EpsMethod::Candidate,

            ..cert.note.clone()
        },
        Note {
            bound: Bound::Candidate,

            stable: Stable::Open,

            equal: false,

            lossless: false,

            ..cert.note.clone()
        },
        Note {
            comp: 1e-9,

            ..cert.note.clone()
        },
    ] {
        assert_eq!(
            audit(&tampered, 4, 4, &cert.low, &cert.rest)
                .unwrap_err()
                .code(),
            Code::Invalid,
            "{tampered:?}"
        );
    }
}

fn certify_reports_the_gated_rank_and_a_lossless_note() {
    let cert = certify(&diag4(), 4, 4).expect("可认证").expect("有谱隙");

    assert_eq!(cert.rank, 3, "diag(3,2,1,0.5) 的谱隙在这里断开");

    assert_eq!(cert.note.rank, 3);

    assert_eq!(cert.note.stable, Stable::Pass);

    assert_eq!(cert.note.mode, Mode::Residual);

    assert!(
        cert.note.equal && cert.note.lossless,
        "low + rest 必须逐位还原"
    );

    assert!((cert.note.gap - 0.5).abs() < 1e-9);

    // 显式残差补偿不计 σ_{r+1}：E_decomp 只是诊断，无损由逐位往返判定。

    assert!((cert.note.decomp - 0.5).abs() < 1e-9);

    assert!(
        cert.note.eps < 1e-12,
        "后验残差应在机器精度量级：{}",
        cert.note.eps
    );

    assert_eq!(cert.low.len(), 16);

    assert_eq!(cert.rest.len(), 16);
}

/// 精确秩 2 的矩阵：认证出的秩就是 2，残差到机器精度、往返逐位。
fn a_low_rank_matrix_certifies_at_its_exact_rank() {
    let cert = certify(&rank_two(), 6, 6).expect("可认证").expect("有谱隙");

    assert_eq!(
        cert.rank, 2,
        "σ3 = σ4 = 0 没有谱隙，秩必须停在 2，而不是溜进噪声底"
    );

    assert!(cert.note.lossless && cert.note.equal);

    assert!(
        cert.note.decomp < 1e-10,
        "秩 2 处残差应到机器精度：{}",
        cert.note.decomp
    );

    assert!(
        cert.rest.iter().all(|x| x.abs() < 1e-5),
        "残差全是数值噪声，不该有可见量"
    );
}

fn a_matrix_without_a_gap_is_rejected() {
    // 全零：所有奇异值为 0，谱隙 0 不大于 2ε = 0 → 拒压，而不是假装无损。

    assert!(certify(&[0.0f32; 16], 4, 4).expect("形状合法").is_none());

    // min(m,n) = 1：`1 <= r < 1` 无解，秩选择不适用。

    assert!(certify(&[1.0, 2.0, 3.0], 1, 3).expect("形状合法").is_none());
}

fn a_bad_shape_is_an_error() {
    assert_eq!(
        certify(&[1.0, 2.0, 3.0], 2, 2).unwrap_err().code(),
        Code::Invalid
    );

    assert_eq!(certify(&[], 0, 4).unwrap_err().code(), Code::Invalid);

    assert_eq!(
        certify(&[1.0], usize::MAX, 2).unwrap_err().code(),
        Code::Invalid
    );
}
}

/// 一张自洽的证书：L0 诊断界、残差方法、无补偿误差。
fn valid() -> Note {
    Note {
        rank: 2,

        used: true,

        mode: Mode::Residual,

        stable: Stable::Pass,

        method: EpsMethod::Residual,

        bound: Bound::Diagnostic,

        gap: 1.0,

        eps: 0.1,

        decomp: 0.0,

        comp: 0.0,

        equal: true,

        lossless: true,
    }
}

testkit::test_cases! {
fn note_roundtrips_through_bytes() {
    let note = valid();

    assert!(check_note(&note).is_ok());

    let bytes = note.to_bytes();

    let mut cur = Cursor::new(&bytes);

    let back = Note::decode(&mut cur).expect("解码");

    assert_eq!(back, note);

    assert!(cur.remaining().is_empty(), "编码不许留尾巴");
}

fn an_unknown_method_or_bound_code_is_rejected() {
    // 秩 2 只占 1 字节，于是布局是 rank|used|mode|stable|method|bound|…：方法码在 4、界码在 5。

    let bytes = valid().to_bytes();

    assert_eq!((bytes[4], bytes[5]), (1, 0), "残差方法 + L0 诊断界");

    for (offset, code) in [(4, 3u8), (5, 3u8)] {
        let mut broken = bytes.clone();

        broken[offset] = code;

        let mut cur = Cursor::new(&broken);

        assert!(Note::decode(&mut cur).is_err(), "offset {offset}");
    }
}

fn the_synthesis_error_adds_the_two_disjoint_terms() {
    // §4.8：`E_decomp` 与 `E_comp` 来自不同产物，相加就是 `ε_synth`，不重复计入。

    let note = Note {
        mode: Mode::Residual,

        decomp: 1e-3,

        comp: 2e-3,

        // 补偿项非零就不可能逐位往返。
        equal: false,

        lossless: false,

        ..valid()
    };

    assert!(check_note(&note).is_ok());

    assert!((note.synth() - 3e-3).abs() < 1e-18);
}

fn a_candidate_only_measurement_cannot_sign_a_pass() {
    // §4.2.1：幂迭代这类只够排序的 `ε` 不能当证书。

    let note = Note {
        method: EpsMethod::Candidate,

        ..valid()
    };

    assert!(check_note(&note).is_err());

    // 退到 Open 就不矛盾：它本来就没说“稳定”。

    let open = Note {
        stable: Stable::Open,

        ..note
    };

    assert!(check_note(&open).is_ok());
}

fn an_l0_prime_candidate_bound_cannot_sign_lossless() {
    let note = Note {
        bound: Bound::Candidate,

        ..valid()
    };

    assert!(check_note(&note).is_err());

    let open = Note {
        stable: Stable::Open,

        equal: false,

        lossless: false,

        ..note
    };

    assert!(check_note(&open).is_ok());
}

fn a_rank_choice_must_name_its_bound() {
    let note = Note {
        bound: Bound::None,

        ..valid()
    };

    assert!(check_note(&note).is_err());

    // 没用秩选择时，“未声明界”只是如实记录。

    let unused = Note {
        used: false,

        decomp: 0.0,

        ..note
    };

    assert!(check_note(&unused).is_ok());
}

fn a_compensation_error_only_belongs_to_the_residual_mode() {
    // 纯截断只丢低秩项，没有补偿产物，就没有补偿误差。

    let trunc = Note {
        mode: Mode::Trunc,

        decomp: 0.0,

        comp: 1e-9,

        ..valid()
    };

    assert!(check_note(&trunc).is_err());
}

fn a_bitwise_roundtrip_forbids_counting_the_compensation_twice() {
    let note = Note {
        comp: 1e-9,

        ..valid()
    };

    assert!(check_note(&note).is_err());
}

fn every_name_is_stable_and_distinct() {
    let methods: Vec<&str> = [
        EpsMethod::Backward,
        EpsMethod::Residual,
        EpsMethod::Candidate,
    ]
    .into_iter()
    .map(EpsMethod::as_str)
    .collect();

    assert_eq!(
        methods,
        [
            "verified_backward_error",
            "verified_residual",
            "candidate_only"
        ]
    );

    let bounds: Vec<&str> = [Bound::Diagnostic, Bound::Candidate, Bound::None]
        .into_iter()
        .map(Bound::as_str)
        .collect();

    assert_eq!(bounds, ["l0_diagnostic", "l0prime_candidate_open", "none"]);
}

fn invariant_violations_are_rejected() {
    let base = Note {
        gap: 0.1, // Pass 但 gap 不 > 2eps

        ..valid()
    };

    assert!(check_note(&base).is_err());

    let contradictory = Note {
        lossless: true,

        decomp: 1.0,

        equal: false,

        ..base
    };

    assert!(check_note(&contradictory).is_err());
}
}

// ---- Jacobi SVD ----

fn frob_resid(a: &[f64], back: &[f64]) -> f64 {
    a.iter()
        .zip(back)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

testkit::test_cases! {
fn diagonal_singular_values_are_exact() {
    // diag(3,2,1,0.5) 行主序。
    let mut a = vec![0.0f64; 16];
    let diag = [3.0, 2.0, 1.0, 0.5];
    for (i, d) in diag.iter().enumerate() {
        a[i * 4 + i] = *d;
    }
    let svd = compute_svd(&a, 4, 4).expect("svd");
    for (got, want) in svd.sigma.iter().zip(diag.iter()) {
        assert!((got - want).abs() < 1e-10, "got {got} want {want}");
    }
}

fn full_reconstruction_residual_is_machine_order() {
    let mut rng = 0x1234_5678u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 40) as f64 / (1u64 << 24) as f64 - 0.5
    };
    for (m, n) in [(8, 8), (12, 6), (6, 12), (20, 10)] {
        let a: Vec<f64> = (0..m * n).map(|_| next()).collect();
        let svd = compute_svd(&a, m, n).expect("svd");
        let k = m.min(n);
        assert_eq!(svd.sigma.len(), k);
        let back = svd.reconstruct(k);
        let scale = svd.sigma[0];
        assert!(
            frob_resid(&a, &back) < 1e-10 * scale * (m as f64).sqrt(),
            "残差应在机器精度量级 m={m} n={n}"
        );
    }
}

fn truncated_reconstruction_is_monotonic() {
    let mut a = vec![0.0f64; 36];
    for i in 0..6 {
        a[i * 6 + i] = 6.0 - i as f64;
    }
    let svd = compute_svd(&a, 6, 6).expect("svd");
    let mut prev = f64::INFINITY;
    for r in 1..=6 {
        let err = frob_resid(&a, &svd.reconstruct(r));
        assert!(err <= prev, "增加秩不应让误差变大");
        prev = err;
    }
}

fn singular_values_are_descending_and_nonnegative() {
    let mut rng = 99u64;
    let a: Vec<f64> = (0..64)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 40) as f64 / (1u64 << 24) as f64
        })
        .collect();
    let svd = compute_svd(&a, 8, 8).expect("svd");
    for pair in svd.sigma.windows(2) {
        assert!(pair[0] >= pair[1]);
        assert!(pair[0] >= 0.0);
    }
}

fn bad_shape_is_an_error() {
    assert!(compute_svd(&[1.0], 2, 2).is_err());
    assert!(compute_svd(&[], 0, 4).is_err());
}

/// 信号秩选择：结构化的秩边界有相对谱隙 → 秩 10；满秩噪声平坦谱 → 拒压；单一奇异值无从比较 → 拒压。
fn signal_rank_separates_structure_from_noise() {
    let mut structured = vec![10.0f64; 10];
    structured.extend(vec![0.1f64; 6]);
    assert_eq!(signal_rank(&structured), Some(10));

    let noise: Vec<f64> = (0..16).map(|k| 10.0 - k as f64 * 0.1).collect();
    assert_eq!(signal_rank(&noise), None);

    assert_eq!(signal_rank(&[1.0]), None);
}

/// SVD 分解/重建吞吐（`#[ignore]`）：
/// `cargo test -p llm-weights --lib -- svd_throughput --release --ignored --nocapture`。
///
/// 确定性 f64 方阵的 `compute`（Jacobi）与 `reconstruct(r=10)` 计时，报秒/矩阵。5 轮取中位数，
/// 只出数字、不做断言——吞吐随机器浮动。`compute` 是 `compensate` 路径的主成本，此表用于
/// 定位该路径的优化重点。
#[ignore = "吞吐基准：cargo test -p llm-weights --lib -- svd_throughput --release --ignored --nocapture"]
fn svd_throughput() {
    for n in [256usize, 512, 1024] {
        let a = svd_matrix(n, 0xabcd_1234);
        let mut c_samples = [0.0f64; 5];
        for slot in &mut c_samples {
            let start = std::time::Instant::now();
            let svd = compute_svd(&a, n, n).expect("svd");
            std::hint::black_box(&svd);
            *slot = start.elapsed().as_secs_f64();
        }
        c_samples.sort_by(f64::total_cmp);
        let svd = compute_svd(&a, n, n).expect("svd");
        let mut r_samples = [0.0f64; 5];
        for slot in &mut r_samples {
            let start = std::time::Instant::now();
            let back = svd.reconstruct(10);
            std::hint::black_box(&back);
            *slot = start.elapsed().as_secs_f64();
        }
        r_samples.sort_by(f64::total_cmp);
        println!(
            "svd n={n} compute={:.4}s reconstruct(r=10)={:.5}s",
            c_samples[2], r_samples[2]
        );
    }
}
}

/// 确定性 f64 方阵（幅度约 [−0.5,0.5)）。
fn svd_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n * n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 40) as f64 / (1u64 << 24) as f64 - 0.5
        })
        .collect()
}
