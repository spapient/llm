use super::*;
use diagnostic::Code;
use std::cell::Cell;

/// 永远接受的入口。
struct Accept;

impl Target for Accept {
    fn name(&self) -> &str {
        "accept"
    }

    fn run(&self, _input: &[u8]) -> Result<(), Error> {
        Ok(())
    }
}

/// 永远拒绝、但给出说明的入口。
struct Reject;

impl Target for Reject {
    fn name(&self) -> &str {
        "reject"
    }

    fn run(&self, _input: &[u8]) -> Result<(), Error> {
        Err(Error::new(Code::Invalid, "这段字节不是一个合法的键值对"))
    }
}

/// 同一次核对里对同一输入给出两种结论——正是第 2 条契约要抓的东西。
struct Unstable(Cell<bool>);

impl Target for Unstable {
    fn name(&self) -> &str {
        "unstable"
    }

    fn run(&self, _input: &[u8]) -> Result<(), Error> {
        let flip = self.0.get();
        self.0.set(!flip);
        if flip {
            Err(Error::new(Code::Invalid, "第二次才拒绝"))
        } else {
            Ok(())
        }
    }
}

/// 拒绝但消息为空（或全空白）的入口：契约三要抓的对象；消息由构造参数给出。
struct Silent(&'static str);

impl Target for Silent {
    fn name(&self) -> &str {
        "silent"
    }

    fn run(&self, _input: &[u8]) -> Result<(), Error> {
        Err(Error::new(Code::Invalid, self.0))
    }
}

/// 一跑就 panic 的入口：契约一的对象，消息带输入长度。
struct Explodes;

impl Target for Explodes {
    fn name(&self) -> &str {
        "explodes"
    }

    fn run(&self, input: &[u8]) -> Result<(), Error> {
        panic!("在 {} 个字节上崩了", input.len());
    }
}

#[test]
fn a_seed_derives_the_same_case_every_time() {
    let seed = Seed::new(b"gemv_i4");
    assert_eq!(seed.derive(7), seed.derive(7), "确定性是复现的前提");
    assert_ne!(seed.derive(7), seed.derive(8), "不同用例不该撞在一起");
}

#[test]
fn adjacent_cases_differ_when_mutated() {
    let seed = Seed::new(b"0123456789abcdef");
    for case in 0..32 {
        let left = seed.derive(case);
        let right = seed.derive(case + 1);
        // 约一成用例原样保留范例：两条相邻未变异用例相同是允许的；只要至少一侧被扰动，
        // 两侧就不该撞成同一串字节。
        if left != seed.bytes() || right != seed.bytes() {
            assert_ne!(left, right, "相邻的两条变异用例不该落到同一串字节");
        }
    }
}

#[test]
fn mutations_shrink_and_grow_the_exemplar() {
    let seed = Seed::new(b"0123456789abcdef");
    let lengths: Vec<usize> = (0..128).map(|case| seed.derive(case).len()).collect();
    assert!(
        lengths.iter().any(|&len| len < seed.len()),
        "删除/截断族必须收出更短的用例"
    );
    assert!(
        lengths.iter().any(|&len| len > seed.len()),
        "插入/追加/复制族必须长出更长的用例"
    );
}

#[test]
fn a_one_byte_change_in_the_seed_spreads_to_the_stream() {
    let left = Seed::new(b"gemv_i4");
    let right = Seed::new(b"gemv_i5");
    assert_ne!(left.derive(0), right.derive(0), "一位翻转应当改变派生流");
}

#[test]
fn an_empty_seed_is_a_legal_case() {
    let seed = Seed::new(b"");
    assert!(seed.is_empty());
    assert_eq!(seed.len(), 0);
    assert_eq!(seed.derive(3), seed.derive(3), "空种子同样要可复现");
    assert!(seed.derive(3).len() <= 2, "空范例只造最短边界");
}

/// 派生长度有可预期的上界：最多三步复制/追加，不会无限膨胀，遍历成本可预期。
#[test]
fn derived_length_stays_within_a_bounded_window() {
    let seed = Seed::new(b"0123456789abcdef");
    for case in 0..256 {
        let out = seed.derive(case);
        assert!(
            out.len() <= seed.len() * 3 + 8,
            "派生长度失控：{}",
            out.len()
        );
    }
}

/// 任意种子字节（空、非 UTF-8、重复字节）派生都不 panic。
#[test]
fn derive_handles_any_seed_bytes_without_panic() {
    let cases: [&[u8]; 4] = [&[], &[0xFF, 0x00], &[0xE2, 0x82, 0xAC], &[7u8; 64]];
    for bytes in cases {
        let seed = Seed::new(bytes.to_vec());
        for case in 0..16 {
            let _ = seed.derive(case);
        }
    }
}

#[test]
fn acceptance_and_rejection_are_both_legal_outcomes() {
    let seed = Seed::new(b"abc");
    assert!(sweep(&Accept, &seed, 3).is_ok());
    assert!(sweep(&Reject, &seed, 3).is_ok(), "拒绝不是失败");
}

#[test]
fn zero_cases_touches_nothing() {
    assert!(sweep(&Unstable(Cell::new(false)), &Seed::new(b"x"), 0).is_ok());
}

#[test]
fn an_empty_rejection_message_is_a_finding() {
    let finding = sweep(&Silent("   "), &Seed::new(b"abc"), 1).expect_err("空消息");
    assert_eq!(finding.contract(), "拒绝时必须给出非空消息");
    assert_eq!(
        finding.behaviour(),
        &Behaviour::Rejected {
            code: Code::Invalid,
            message: "   ".to_string()
        }
    );
}

/// 第 1 条契约也必须由本 crate 抓：驱动方不必自己 `catch_unwind`，也不会丢掉用例号。
#[test]
fn a_panicking_target_is_a_finding_with_its_case_and_input() {
    let seed = Seed::new(b"abc");
    let finding = sweep(&Explodes, &seed, 3).expect_err("panic 是违约");
    assert_eq!(
        finding.contract(),
        "不 panic：任何输入都必须被解析成结论，而不是打死宿主"
    );
    assert_eq!(finding.case(), 0, "第一条证据就是第 0 个用例");
    assert_eq!(finding.input(), seed.derive(0).as_slice());
    assert_eq!(finding.target(), "explodes");
    let Behaviour::Panicked { message } = finding.behaviour() else {
        panic!("必须是 Panicked，实际 {:?}", finding.behaviour());
    };
    assert!(
        message.contains("个字节上崩了"),
        "payload 消息要留住：{message}"
    );
}

/// `panic_any` 的数值载荷同样是证据，按 Debug 留住——不能因此退回成"通过"。
#[test]
fn a_panic_any_payload_is_still_a_finding() {
    struct SilentPanic;

    impl Target for SilentPanic {
        fn name(&self) -> &str {
            "silent-panic"
        }

        fn run(&self, _input: &[u8]) -> Result<(), Error> {
            std::panic::panic_any(7u8);
        }
    }

    let finding = sweep(&SilentPanic, &Seed::new(b"x"), 1).expect_err("panic 是违约");
    let Behaviour::Panicked { message } = finding.behaviour() else {
        panic!("必须是 Panicked，实际 {:?}", finding.behaviour());
    };
    assert_eq!(message, "7", "数值 payload 按 Debug 留住：{message}");
}

#[test]
fn an_unstable_target_is_caught_with_its_case_and_input() {
    let seed = Seed::new(b"abc");
    let finding = sweep(&Unstable(Cell::new(false)), &seed, 2).expect_err("结论不稳定");
    assert_eq!(finding.contract(), "同一输入必须给出同一个结论");
    assert_eq!(finding.case(), 0, "第一条证据就是第 0 个用例");
    assert_eq!(finding.input(), seed.derive(0).as_slice());
}

/// `exercise` 单条输入跑三条契约，违约以证据返回；harness 再经 `exercise_or_panic` 转 panic。
#[test]
fn exercise_runs_the_same_contracts_as_sweep() {
    let finding = Unstable(Cell::new(false))
        .exercise(b"abc")
        .expect_err("harness 入口也要抓不稳定结论");
    assert_eq!(finding.contract(), "同一输入必须给出同一个结论");
    assert_eq!(finding.case(), 0, "模糊输入没有派生用例号");
}

/// `exercise_or_panic` 把违约转成 panic，载荷里带完整报告（目标名、用例号、输入）。
#[test]
fn exercise_or_panic_panics_with_the_finding_report() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Unstable(Cell::new(false)).exercise_or_panic(b"abc");
    }));
    let payload = result.expect_err("不稳定结论必须 panic");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(
        message.contains("unstable"),
        "panic 载荷要带目标名：{message}"
    );
    assert!(message.contains("case #0"), "报告要带用例号：{message}");
}

/// `sweep_all` 收齐全部违约证据，不提前退出。
#[test]
fn sweep_all_collects_every_violation() {
    let findings = sweep_all(&Silent(""), &Seed::new(b"abc"), 3);
    assert_eq!(findings.len(), 3, "三条用例各自一份证据");
    assert!(
        findings
            .iter()
            .all(|f| f.contract() == "拒绝时必须给出非空消息")
    );
}

/// 证据的文本呈现能直接贴进 issue：目标、用例号、结论与可复现输入都在。
#[test]
fn a_finding_renders_a_reproducible_report() {
    let finding = sweep(&Explodes, &Seed::new(b"abc"), 1).expect_err("panic 是违约");
    let report = finding.to_string();
    assert!(report.contains("explodes"), "报告要带目标名：{report}");
    assert!(report.contains("case #0"), "报告要带用例号：{report}");
    assert!(report.contains("panicked"), "报告要带结论：{report}");
}

#[test]
fn repro_renders_utf8_as_literal_and_binary_as_hex() {
    assert_eq!(repro(b"hi").to_string(), "\"hi\"");
    assert_eq!(repro(&[0xFF, 0x00]).to_string(), "0xff00");
}
