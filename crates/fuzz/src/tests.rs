use super::*;
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

/// 同一次 `sweep` 里对同一输入给出两种结论——正是第 2 条契约要抓的东西。
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

#[test]
fn a_seed_derives_the_same_case_every_time() {
    let seed = Seed::new(b"gemv_i4");
    assert_eq!(seed.derive(7), seed.derive(7), "确定性是复现的前提");
    assert_ne!(seed.derive(7), seed.derive(8), "不同用例不该撞在一起");
}

#[test]
fn an_empty_seed_is_a_legal_case() {
    let seed = Seed::new(b"");
    assert!(seed.is_empty());
    assert_eq!(seed.derive(3), Vec::<u8>::new());
    assert_eq!(seed.len(), 0);
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
    struct Silent;

    impl Target for Silent {
        fn name(&self) -> &str {
            "silent"
        }

        fn run(&self, _input: &[u8]) -> Result<(), Error> {
            Err(Error::new(Code::Invalid, "   "))
        }
    }

    let finding = sweep(&Silent, &Seed::new(b"abc"), 1).expect_err("空消息");
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
    struct Explodes;

    impl Target for Explodes {
        fn name(&self) -> &str {
            "explodes"
        }

        fn run(&self, input: &[u8]) -> Result<(), Error> {
            panic!("在 {} 个字节上崩了", input.len());
        }
    }

    let seed = Seed::new(b"abc");
    let finding = sweep(&Explodes, &seed, 3).expect_err("panic 是违约");
    assert_eq!(
        finding.contract(),
        "不 panic：任何输入都必须被解析成结论，而不是打死宿主"
    );
    assert_eq!(finding.case(), 0, "第一条证据就是第 0 个用例");
    assert_eq!(finding.input(), seed.derive(0).as_slice());
    let Behaviour::Panicked { message } = finding.behaviour() else {
        panic!("必须是 Panicked，实际 {:?}", finding.behaviour());
    };
    assert!(
        message.contains("个字节上崩了"),
        "payload 消息要留住：{message}"
    );
}

/// 不带字符串的 panic 同样是证据，只是消息是占位——不能因此退回成"通过"。
#[test]
fn a_panic_without_a_message_is_still_a_finding() {
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
    assert_eq!(
        finding.behaviour(),
        &Behaviour::Panicked {
            message: "panic 没有携带字符串说明".to_string()
        }
    );
}

#[test]
fn an_unstable_target_is_caught_with_its_case_and_input() {
    let seed = Seed::new(b"abc");
    let finding = sweep(&Unstable(Cell::new(false)), &seed, 2).expect_err("结论不稳定");
    assert_eq!(finding.contract(), "同一输入必须给出同一个结论");
    assert_eq!(finding.case(), 0, "第一条证据就是第 0 个用例");
    assert_eq!(finding.input(), seed.derive(0).as_slice());
}
