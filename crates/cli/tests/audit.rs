//! 结构门与分配门：把 S++ 审计里"靠人看"的几条变成会红的断言。
//!
//! 结构门盯的是**形状**，不是风格——文件行数上限、`src/` 平铺、生产目录里没有测试代码。
//! 分配门盯的是**启动路径**：`main` 每次跑都要走一遍解析，这条路上多一次 `String` 复制
//! 就是每个进程都白付一次；次数用全局分配计数器数出来，因此判据是整数相等而非计时。

// 分配计数只能通过全局分配器钩子实现，那是 unsafe trait；本文件只在测试里跑，不进产物。
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};

/// 生产文件上限：400 行以内不需要内聚证明（`docs/PROMPT-S++-AUDIT.md` N3）。
const PRODUCTION_LIMIT: usize = 400;

/// 单个测试文件上限（N13）。
const TEST_LIMIT: usize = 800;

struct Counting;

// 计数按线程分开：测试并行跑，别的线程的分配不该记到这条断言上。
thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn counted() {
    let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        counted();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        counted();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// 本次调用在当前线程上分配了多少次。
fn allocations<T>(run: impl FnOnce() -> T) -> (usize, T) {
    let read = || ALLOCATIONS.with(Cell::get);
    let before = read();
    let value = run();
    (read() - before, value)
}

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// 递归收集 `*.rs`，跳过 `target/`。
fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

fn lines(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{} 应当可读：{error}", path.display()))
        .lines()
        .count()
}

fn relative(path: &Path) -> String {
    path.strip_prefix(crate_dir())
        .unwrap_or(path)
        .display()
        .to_string()
}

#[test]
fn every_production_file_fits_the_density_budget() {
    let offenders: Vec<String> = rust_files(&crate_dir().join("src"))
        .into_iter()
        .filter(|path| lines(path) > PRODUCTION_LIMIT)
        .map(|path| format!("{}（{} 行）", relative(&path), lines(&path)))
        .collect();
    assert!(
        offenders.is_empty(),
        "生产文件超过 {PRODUCTION_LIMIT} 行：{offenders:#?}"
    );
}

#[test]
fn every_test_file_fits_the_density_budget() {
    let offenders: Vec<String> = rust_files(&crate_dir().join("tests"))
        .into_iter()
        .filter(|path| lines(path) > TEST_LIMIT)
        .map(|path| format!("{}（{} 行）", relative(&path), lines(&path)))
        .collect();
    assert!(
        offenders.is_empty(),
        "测试文件超过 {TEST_LIMIT} 行：{offenders:#?}"
    );
}

/// `src/` 只有一层：再往下就是分组，不是平铺。
#[test]
fn the_production_tree_stays_flat() {
    let source = crate_dir().join("src");
    let mut nested = Vec::new();
    let mut stack = vec![source.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let is_entry_dir = path == source.join("bin");
            if dir != source || !is_entry_dir {
                nested.push(relative(&path));
            }
            stack.push(path);
        }
    }
    assert!(nested.is_empty(), "`src/` 下出现了分组目录：{nested:#?}");
}

/// 文件名即模块名、单词、无下划线（N4）。
#[test]
fn module_file_names_are_single_words() {
    let offenders: Vec<String> = rust_files(&crate_dir().join("src"))
        .into_iter()
        .filter(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem.contains('_'))
        })
        .map(|path| relative(&path))
        .collect();
    assert!(offenders.is_empty(), "文件名带下划线：{offenders:#?}");
}

/// 测试代码不住在生产目录：`src/` 里只允许 `#[path = "..."]` 钩子。
#[test]
fn production_files_only_hook_tests_in() {
    let offenders: Vec<String> = rust_files(&crate_dir().join("src"))
        .into_iter()
        .filter(|path| {
            let text = std::fs::read_to_string(path).unwrap_or_default();
            text.contains("mod tests {") || text.contains("#[test]")
        })
        .map(|path| relative(&path))
        .collect();
    assert!(
        offenders.is_empty(),
        "`src/` 里出现了内联测试，应移到 `tests/` 并用 `#[path]` 挂回：{offenders:#?}"
    );
}

fn args(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| word.to_string()).collect()
}

/// 解析一次的分配次数是**契约**：改文法或取值类型时，先想清楚多的那次分配值不值。
#[test]
fn parsing_a_bare_command_allocates_nothing() {
    let argv = args(&["status"]);
    let (count, invocation) =
        allocations(|| llm_cli::Invocation::parse(&argv).expect("`status` 是合法命令"));
    assert_eq!(count, 0, "无取值命令不该分配：{invocation:?}");
}

/// 取值一律借用 `argv`，因此取值个数不影响分配次数：多一个选项就多一次分配的写法在这里是缺陷。
#[test]
fn parsing_borrows_every_value_instead_of_copying_it() {
    for words in [
        &["run", "--weights", "w.safetensors", "--input", "1,2"][..],
        &["train", "--lr", "0.01", "--warmup", "10", "--steps", "20"][..],
        &["import", "--in", "a.safetensors", "--out", "a.u"][..],
    ] {
        let argv = args(words);
        let (count, invocation) =
            allocations(|| llm_cli::Invocation::parse(&argv).expect("文法接受这些选项"));
        assert_eq!(count, 0, "{words:?} 不该分配：{invocation:?}");
    }
}

/// 吞吐探针：默认忽略，避免计时抖动进断言；要数字时用
/// `cargo test -p llm-cli --release --test audit -- --ignored --nocapture` 跑一次。
#[test]
#[ignore = "计时探针，只在需要数字时手动跑"]
fn parse_throughput() {
    let argv = args(&[
        "run",
        "--weights",
        "w.safetensors",
        "--input",
        "1,2",
        "--device",
        "cpu",
    ]);
    let rounds = 2_000_000;
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        std::hint::black_box(llm_cli::Invocation::parse(std::hint::black_box(&argv)).ok());
    }
    let elapsed = start.elapsed();
    println!(
        "Invocation::parse：{} ns/次（{rounds} 次取总时长）",
        elapsed.as_nanos() / rounds
    );
}
