//! 词汇表契约的可执行版本（`docs/ARCHITECTURE.md` 第三节「闭集与位集」）。
//!
//! 两条规则，各一个用例：
//!
//! 1. **闭集表只有一份**：`prof` 下游的成员把名字表写成 `impl Closed for ..` 里的切片，
//!    `ALL` / `as_str` / `parse` 由 `prof::inherent_names!` 转发；手写 `pub const ALL: [Self; N]`
//!    只允许出现在够不到 `Closed` 的层（`llm-prof` 是 L2 基座，反向依赖会成环）。
//! 2. **位集只有一份**：`struct FooSet(u32)` 一律不许新增，位集只有
//!    `prof::atom::Set<E, WORDS>`（`WORDS` 是 const 参数，容量在编译期自证）。
//!
//! 白名单在 `lints/vocabulary.toml`，**不在源码里**：`#[allow]` 只能豁免一次检查，而配置是
//! 一份可审计的清单——每条豁免必须写理由，且清单里的 crate 一旦不再违规就必须删掉条目
//! （下面有用例守这一点，否则白名单会变成"永久免检区"）。
//!
//! 扫描器自己也被守：文件数低于下限即失败。一条什么都没扫到却通过的门禁，比没有门禁更误导。

use std::fs;
use std::path::PathBuf;

#[path = "support/paths.rs"]
mod paths;

use paths::{rust_files, workspace_root as root};

/// 扫描下限：近 30 个 crate、80+ 个源文件；低于此数说明遍历逻辑坏了（而不是"没有违规"）。
const MIN_SOURCES: usize = 70;

/// 一份扫描结果：`(crate, 文件, 行号, 该行原文)`。
type Hit = (String, PathBuf, usize, String);

/// `crates/<crate>/src/**/*.rs` 下的全部源文件。
fn sources() -> Vec<(String, PathBuf)> {
    let crates = root().join("crates");
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(&crates) else {
        return out;
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in names {
        let mut files = rust_files(&crates.join(&name).join("src"));
        files.sort();
        for file in files {
            out.push((name.clone(), file));
        }
    }
    out
}

/// 命中 `predicate` 的行。
fn hits(predicate: impl Fn(&str) -> bool) -> Vec<Hit> {
    let mut out = Vec::new();
    for (name, file) in sources() {
        let Ok(text) = fs::read_to_string(&file) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            if predicate(line) {
                out.push((
                    name.clone(),
                    file.strip_prefix(root()).unwrap_or(&file).to_path_buf(),
                    index + 1,
                    line.trim().to_string(),
                ));
            }
        }
    }
    out
}

/// 白名单里的 crate 与理由；配置来自 `tool` 构建期生成的词汇表（同一份文件只解析一次）。
fn exempt(section: &str) -> Vec<(String, String)> {
    let table = tool::vocabulary()
        .table(section)
        .unwrap_or_else(|| panic!("`lints/vocabulary.toml` 缺 `[{section}]`"));
    table
        .table_array("exempt")
        .unwrap_or_else(|| panic!("`lints/vocabulary.toml` 缺 `{section}.exempt`"))
        .iter()
        .map(|entry| {
            let name = entry
                .string("crate")
                .unwrap_or_else(|| panic!("`{section}.exempt` 的条目缺 `crate`"));
            let reason = entry
                .string("reason")
                .unwrap_or_else(|| panic!("`{section}.exempt` 的 `{name}` 缺 `reason`"));
            assert!(
                reason.chars().count() >= 8,
                "`{section}.exempt` 的 `{name}` 理由太短（{reason:?}）：豁免必须说清为什么"
            );
            (name.to_string(), reason.to_string())
        })
        .collect()
}

/// 注解文法里的闭集表：`const <名字>: [&str; N] = [ "..", .. ];`。
///
/// 只能读源码：proc-macro crate 不导出非宏项，`syntax::DTYPES` 在 crate 外
/// 不可见。扫描只认 `= [ .. ]` 之间的逗号分隔字面量，排版怎么折行都不影响；声明里的 `N`
/// 一并核对，防止“表删了一项、长度还写着 9”。
fn annotation_list(name: &str) -> Vec<String> {
    let path = root().join("crates/annotation/src/syntax.rs");
    let text = fs::read_to_string(&path).expect("`crates/annotation/src/syntax.rs` 必须存在");
    let head = format!("const {name}: [&str;");
    let start = text.find(&head).unwrap_or_else(|| {
        panic!("注解文法里没有 `{name}`：它是注解与词汇表之间契约的一半，不能改名或删除")
    });

    let after = &text[start..];
    let open = start + after.find("= [").expect("闭集表必须写成 `= [ .. ]`") + 3;
    let close = open + text[open..].find(']').expect("闭集表必须闭合");
    let items: Vec<String> = text[open..close]
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            item.strip_prefix('"')
                .and_then(|rest| rest.strip_suffix('"'))
                .unwrap_or_else(|| panic!("`{name}` 的条目必须是字符串字面量，得到 {item:?}"))
                .to_string()
        })
        .collect();
    assert!(!items.is_empty(), "`{name}` 是空表：闭集表不能为空");

    let length_start = start + after.find("; ").expect("长度必须写在 `[&str; N]` 里") + 2;
    let length_end = length_start + text[length_start..].find(']').expect("长度必须闭合");
    let declared: usize = text[length_start..length_end]
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("`{name}` 的长度不是整数"));
    assert_eq!(
        declared,
        items.len(),
        "`{name}` 声明了 {declared} 个元素，实际写了 {} 个",
        items.len()
    );
    items
}

fn describe(hits: &[Hit]) -> String {
    hits.iter()
        .map(|(name, file, line, text)| format!("\n  {name} {}:{line}: {text}", file.display()))
        .collect()
}

#[test]
fn closed_vocabularies_are_declared_once() {
    // 手写的闭集表：数组形式 `pub const ALL: [Self; N]`。切片形式 `&'static [Self]` 是
    // `Closed` 的实现（`inherent_names!` 转发的那一份），不在此列。
    // 行内任意位置都算：`impl X { pub const ALL: [Self; N] = ..; }` 这种一行写法不该逃过。
    // 切片形式 `const ALL: &'static [Self]` 不以 `[` 紧接，所以不会被误判。
    let hits = hits(|line| line.contains("const ALL: ["));
    assert!(
        sources().len() >= MIN_SOURCES,
        "扫描器没看到源文件：{} < {MIN_SOURCES}",
        sources().len()
    );

    let allowed = exempt("closed");
    let names: Vec<&str> = allowed.iter().map(|(name, _)| name.as_str()).collect();
    let offenders: Vec<&Hit> = hits
        .iter()
        .filter(|(name, ..)| !names.contains(&name.as_str()))
        .collect();
    assert!(
        offenders.is_empty(),
        "这些闭集表该用 `impl prof::atom::Closed for ..` + `prof::inherent_names!`：{}",
        offenders
            .iter()
            .map(|(name, file, line, text)| format!("\n  {name} {}:{line}: {text}", file.display()))
            .collect::<String>()
    );

    // 白名单不能过期：每条豁免都必须在今天仍然真的手写着闭集表。
    for (name, _) in &allowed {
        assert!(
            root().join("crates").join(name).is_dir(),
            "`lints/vocabulary.toml` 里的 `{name}` 不是成员"
        );
        assert!(
            hits.iter().any(|(crate_name, ..)| crate_name == name),
            "`lints/vocabulary.toml` 里的 `{name}` 已经没有手写闭集表，删除这条豁免"
        );
    }

    // 契约的另一半：`Closed` 的实现必须写成切片，这样才能与 `ALL` 的顺序、`index` 的下标一致。
    let prof = root().join("crates/prof/src/atom.rs");
    let text = fs::read_to_string(&prof).expect("`crates/prof/src/atom.rs` 必须存在");
    assert!(
        text.contains("const ALL: &'static [Self]"),
        "`prof::atom` 的闭集表必须写成切片"
    );
    assert!(
        text.contains("macro_rules! inherent_names"),
        "转发宏 `inherent_names!` 是这份契约的入口"
    );
}

/// 执行前端必须登记进 `cutile::device` 的描述符表，且四个槽位齐全：语言身份 `emit`、探测
/// `available`、程序生成 `program`、执行回读 `run`。
///
/// 表驱动之后，执行前端不再有 `Device` trait 或单独的前端类型：差异全是表内的函数指针行，
/// "写了实现却忘了进表"这一事故在结构上消失——要执行就必须出现在表的一行里。文本契约守另一类
/// 事故：表行漏了某个槽位（缺探测 / 缺程序 / 缺回读），或 `emit` 没绑到 `Emit` 闭集。
/// 加设备只在表内加一行，不改任何 `match`。
#[test]
fn every_device_front_end_is_entered() {
    let path = root().join("crates/cutile/src/device.rs");
    let text = fs::read_to_string(&path).expect("`crates/cutile/src/device.rs` 必须存在");
    assert!(
        text.contains("const SPECS"),
        "执行前端描述符表 `const SPECS` 必须存在"
    );

    let mut checked = 0;
    let mut failures: Vec<String> = Vec::new();
    let mut slots: Vec<&str> = Vec::new();
    let mut in_spec = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Spec {") {
            in_spec = true;
            slots.clear();
            continue;
        }
        if !in_spec {
            continue;
        }
        if trimmed == "}," || trimmed == "}" {
            checked += 1;
            if !slots
                .iter()
                .any(|line| line.starts_with("emit:") && line.contains("Emit::"))
            {
                failures.push("执行前端表行缺少 `emit: Emit::<..>` 语言身份槽位".to_string());
            }
            for slot in ["available", "program", "run"] {
                if !slots
                    .iter()
                    .any(|line| line.starts_with(&format!("{slot}:")))
                {
                    failures.push(format!("执行前端表行缺少 `{slot}` 函数指针槽位"));
                }
            }
            in_spec = false;
            slots.clear();
        } else {
            slots.push(trimmed);
        }
    }
    assert!(
        checked >= 2,
        "扫描器只看到 {checked} 个执行前端：扫描逻辑或文件布局坏了，而不是\"没有违规\""
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn bitsets_have_exactly_one_implementation() {
    // 手写位集：`struct FooSet(u32)` 一类。
    let hits = hits(|line| line.contains("struct ") && line.contains("Set(u32)"));
    let allowed = exempt("bitset");
    let names: Vec<&str> = allowed.iter().map(|(name, _)| name.as_str()).collect();
    let offenders: Vec<&Hit> = hits
        .iter()
        .filter(|(name, ..)| !names.contains(&name.as_str()))
        .collect();
    assert!(
        offenders.is_empty(),
        "位集只允许 `prof::atom::Set<E, WORDS>` 一份实现：{}",
        describe(&offenders.into_iter().cloned().collect::<Vec<Hit>>())
    );

    let atom = fs::read_to_string(root().join("crates/prof/src/atom.rs"))
        .expect("`crates/prof/src/atom.rs` 必须存在");
    assert!(
        atom.contains("pub struct Set<E: Closed, const WORDS: usize>"),
        "唯一实现 `Set<E, WORDS>` 不见了：这条规则的替代品是空的"
    );
}

/// 注解的闭集必须由词汇表与对拍口径**推出来**，而不是第三份手抄表。
///
/// `DTYPES` 与「[`cutile::reference::tolerance`] 给出非零阈值」的 `DType` 集合逐项相等——
/// 注解能写出的输出类型必须正好是对拍能给结论的那些：多一个，`verify` 会拿 0 当阈值，
/// 少一个，那门输出就永远无法被声明成算子。
///
/// 变体名再核一遍：`codegen` 按 `DType::variant` 把名字翻成变体，
/// 名字与变体一旦分家，生成代码会指向不存在的变体——错误会出现在被标注的 crate 里，
/// 而不是写错字的那张表上。
#[test]
fn annotation_names_come_from_the_vocabulary() {
    // 输出类型：与「对拍有阈值」的那一组相等。
    let dtypes = annotation_list("DTYPES");
    let mut comparable: Vec<&str> = cutile::atom::DType::ALL
        .iter()
        .filter(|dt| cutile::reference::tolerance(**dt) > 0.0)
        .map(|dt| dt.name())
        .collect();
    comparable.sort_unstable();
    let mut declared: Vec<&str> = dtypes.iter().map(String::as_str).collect();
    declared.sort_unstable();
    assert_eq!(
        declared, comparable,
        "`DTYPES` 必须与 `cutile::reference::tolerance` 给出非零阈值的类型逐项相等"
    );
    for dtype in &dtypes {
        let dt = cutile::atom::DType::parse_name(dtype)
            .unwrap_or_else(|| panic!("`DTYPES` 里的 `{dtype}` 不在 `DType` 闭集里"));
        assert_eq!(dt.name(), *dtype, "`{dtype}` 必须能往返解析");
        // 变体名的唯一事实源是 `DType::variant`（缩写词全大写规则住在 dtype 里）。
        // `Debug` 输出必须与它一致，否则 annotation 生成的 `DType::{variant}` 编不过。
        assert_eq!(
            format!("{dt:?}"),
            dt.variant(),
            "`{dtype}` 的 Debug 名与 variant 不一致"
        );
    }
}

/// 语义表的**唯一生产者**：注册表里的 `MODEL` 必须由注解生成，注解标注的格式必须在注册表里。
///
/// 类型系统守住了"注解说的是哪份格式"（`<别名 as gemv::Format>`，写错就编译不过），守不住两件
/// 纯文本事故：
///
/// 1. 标注了 `#[annotation::model(..)]` 却忘了在 `REGISTRY` 里补一行——`MODEL` 常量照样生成，
///    但注册表里没有这条算子，`verify --op <名字>` 报"未知算子"，而写算子的人以为已经接上了；
/// 2. 注册表里的 `model:` 手写成 `Model { .. }` 字面量——语义表就有了第二份来源，生成的那一份
///    与它漂移时没有任何判据（这正是 P0-4 之前 `call_args` 的老毛病）。
///
/// 两条规则互为反面，都没有合法例外（标注了却不注册永远错；注册的语义表不是常量也永远错），
/// 所以不进白名单：出现例外说明契约本身该改，而不是这个 crate 该豁免。
#[test]
fn the_registry_and_the_model_annotations_are_mirrors() {
    // 规则的另一种写法：新增一副骨架时，把它的注解名加进这里，规则跟着扩。
    const PRODUCERS: [&str; 1] = ["#[annotation::model("];

    let lib = fs::read_to_string(root().join("crates/op/src/lib.rs"))
        .expect("`crates/op/src/lib.rs` 必须存在");
    // 注册表是一张表：`registry! { "名字" => 语义模块, .. }`，一行一对。字段由宏展开成
    // `Registration { name, template, model: <模块>::MODEL }`，所以表里剩下的就是"哪条算子用
    // 哪个模块的 `MODEL`"——这正是本测试要核对的那件事，逐字抄 22 份结构不携带任何额外信息。
    let start = lib
        .find("registry! {")
        .expect("`REGISTRY` 不见了：它是语义表的唯一入口");
    let open = start + lib[start..].find('{').expect("注册表必须是一张表");
    let close = open + lib[open..].find("\n}").expect("注册表必须闭合");
    let mut names: Vec<String> = Vec::new();
    let mut models: Vec<String> = Vec::new();
    for line in lib[open..close].lines() {
        let Some((name, module)) = line.trim().split_once("=>") else {
            continue;
        };
        let name = name
            .trim()
            .trim_end_matches(',')
            .trim_matches('"')
            .to_string();
        let module = module.trim().trim_end_matches(',');
        models.push(format!("{module}::MODEL"));
        names.push(name);
    }
    assert_eq!(
        names.len(),
        models.len(),
        "注册表的条目必須同时写 `name` 与 `model`：{names:?} vs {models:?}"
    );
    assert!(
        names.len() >= 2,
        "只看到 {} 条注册：扫描逻辑坏了",
        names.len()
    );

    // 注解侧：`name` 是第一个键（参数顺序由 `annotation::syntax` 定死，这里只读不猜）。
    let mut annotated: Vec<(String, String)> = Vec::new();
    for (crate_name, file) in sources() {
        if crate_name != "op" {
            continue;
        }
        let Ok(text) = fs::read_to_string(&file) else {
            continue;
        };
        for line in text.lines() {
            let Some(position) = line.find(PRODUCERS[0]) else {
                continue;
            };
            let rest = &line[position..];
            let Some(offset) = rest.find("name = \"") else {
                panic!("注解必须写成 `name = \"..\"`：{line}");
            };
            let value = &rest[offset + "name = \"".len()..];
            let end = value
                .find('"')
                .unwrap_or_else(|| panic!("注解里的名字字面量没有闭合：{line}"));
            annotated.push((
                value[..end].to_string(),
                file.display().to_string() + ":" + &line_number(&text, line),
            ));
        }
    }
    assert!(
        annotated.len() >= 2,
        "只看到 {} 条 `#[annotation::model]`：扫描逻辑或文件布局坏了",
        annotated.len()
    );

    // 方向一：注解 → 注册表。
    let mut failures: Vec<String> = Vec::new();
    for (name, at) in &annotated {
        if !names.contains(name) {
            failures.push(format!(
                "{at}: 标了 `#[annotation::model]` 生成 `MODEL`，但 `REGISTRY` 里没有 `{name}`：\
                 发布出去的二进制里没有这条算子"
            ));
        }
    }

    // 方向二：注册表 → 注解生成的常量。
    for (name, model) in names.iter().zip(&models) {
        let Some(module) = model.strip_suffix("::MODEL") else {
            failures.push(format!(
                "`{name}` 的语义表是 `{model}`，不是 `::MODEL` 常量：\
                 语义表必须有唯一生产者（注解生成），不能是手写的 `Model {{ .. }}`"
            ));
            continue;
        };
        let file = root().join("crates/op/src").join(format!(
            "{}.rs",
            module.rsplit("::").next().unwrap_or(module)
        ));
        let text = fs::read_to_string(&file).unwrap_or_else(|_| {
            panic!(
                "`{name}` 的语义表指向 `{module}`，但 {} 读不到",
                file.display()
            )
        });
        if !PRODUCERS.iter().any(|producer| text.contains(producer)) {
            failures.push(format!(
                "`{name}` 的 `MODEL` 在 {} 里不是注解生成的：\
                 `#[annotation::model]` 是它与注册表之间唯一的绑定点",
                file.display()
            ));
        }
        if !annotated.iter().any(|(annotated, _)| annotated == name) {
            failures.push(format!(
                "`{name}` 在 `REGISTRY` 里，但它指向的 `MODEL` 没有同名注解：\
                 注解是语义表的唯一生产者"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 模板签名 / 注册表语义表 / 注解声明三者的参数类型必须逐项对齐。
///
/// 每条注册都走同一条链：从 `.k` 解析签名 → 用注册表语义表的 `sample` 构造确定性样本 →
/// [`op::sig::Signature::check`] 对齐缓冲同名同类型、标量有取值、样本没有多出签名之外的名字；
/// 再核对可写输出的 C 类型与 `Model::dtype` 指同一个类型（前者说内核写什么，后者说回读按什么
/// 解释）。错配在测试期拒绝，而不是等到 `cc` 报 C 的指针类型不兼容。生成的 `__model_check`
/// 逐条再守一遍；本测试是注册表级的整表扫，防止某条注解展开被剥离后只剩单条防线。
#[test]
fn every_registered_kernel_signature_matches_its_model_sample() {
    let mut checked = 0;
    let mut failures: Vec<String> = Vec::new();
    for entry in op::REGISTRY {
        let signature = match op::sig::Signature::parse(entry.name, entry.template) {
            Ok(signature) => signature,
            Err(error) => {
                failures.push(format!("`{}`：签名解析失败：{error}", entry.name));
                continue;
            }
        };
        let sample = (entry.model.sample)(entry.model.shape.0, entry.model.shape.1);
        if let Err(error) = signature.check(&sample) {
            failures.push(format!("`{}`：样本与 `.k` 签名不一致：{error}", entry.name));
        }
        let output_dtype = signature.output_type().and_then(|ctype| ctype.dtype());
        if output_dtype != Some(entry.model.dtype) {
            failures.push(format!(
                "`{}`：`.k` 可写输出是 `{}`，语义表 dtype 是 `{:?}`，两者必须是同一个类型",
                entry.name,
                signature.output_type().map_or("-", op::sig::CType::c_name),
                entry.model.dtype
            ));
        }
        checked += 1;
    }
    assert!(
        checked >= 2,
        "扫描器只看到 {checked} 条注册：扫描逻辑或表布局坏了"
    );
    assert_eq!(checked, op::REGISTRY.len(), "注册表每条都必须过签名对齐");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 一条命中的行号（从 1 开始），用于把失败指回源码。
fn line_number(text: &str, line: &str) -> String {
    text.lines()
        .position(|candidate| candidate == line)
        .map_or_else(|| "-".to_string(), |index| (index + 1).to_string())
}
