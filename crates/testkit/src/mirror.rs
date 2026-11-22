//! 注册表与注解的镜像关系：语义表只有一个生产者，签名、样本、`dtype` 逐项对齐。

use std::fs;

use crate::architecture::{dependents, source_text, workspace_root as root};
use crate::contracts::{in_op_family, locate_module, sources};

/// 语义表的**唯一生产者**：注册表里的 `MODEL` 必须由注解生成，注解标注的格式必须在注册表里。
#[test]
fn the_registry_and_the_model_annotations_are_mirrors() {
    // 规则的另一种写法：新增一副骨架时，把它的注解名加进这里，规则跟着扩。
    const PRODUCERS: [&str; 1] = ["#[annotation::model("];

    let lib = fs::read_to_string(root().join("crates/op/src/lib.rs"))
        .expect("`crates/op/src/lib.rs` 必须存在");
    // 注册表现在由 `for_each_operator!` 生成（`REGISTRY` 是它的投影）：名字、可微性与语义表都由
    // `Operator` 派生（`Operator::NAME` / `DIFFERENTIABLE` / `model_of`），文本里不再有手写的注册名与
    // `Model` 字面量。锚点因此从 `registry! {` 换到那张宏表：每行是
    // `名字: $crate::模块::类型 => "骨架"`。本测试核对「注解仍是语义表的唯一生产者」这条事实
    // 没有被表结构改变。
    let start = lib
        .find("macro_rules! for_each_operator")
        .expect("`for_each_operator!` 不见了：它是 `REGISTRY` 的唯一来源");
    let open = start + lib[start..].find("$m! {").expect("宏表必须是一张表") + "$m! {".len();
    let close = open + lib[open..].find("\n        }").expect("宏表必须闭合");
    let mut types: Vec<String> = Vec::new();
    for line in lib[open..close].lines() {
        let Some((_, rest)) = line.trim().split_once(": $crate::") else {
            continue;
        };
        let Some((ty, _skeleton)) = rest.split_once(" =>") else {
            continue;
        };
        types.push(ty.trim().to_string());
    }
    assert!(
        types.len() >= 2,
        "只看到 {} 条注册：扫描逻辑坏了",
        types.len()
    );

    // 注解侧：`name` 是第一个键（参数顺序由 `annotation::syntax` 定死，这里只读不猜）。
    let mut annotated: Vec<(String, String)> = Vec::new();
    // 注解散在算子四层里，且这四层的目录名会改：判据必须是"谁依赖 `annotation`"，
    // 不是目录名前缀——前缀判断在改名当刻静默归零。
    let producers = dependents("annotation");
    for (crate_name, file) in sources() {
        if !producers.contains(&crate_name) {
            continue;
        }
        let Some(text) = source_text(&file) else {
            continue;
        };
        for line in text.lines() {
            let Some(position) = line.find(PRODUCERS[0]) else {
                continue;
            };
            let rest = &line[position..];
            // 宏体里的 `name = $op` 不是字面量注解（名字由宏参数展开），扫描器只认字面量；
            // 宏算子由 Direction 2 的 `invokes_model_macro` 补认。
            let Some(offset) = rest.find("name = \"") else {
                continue;
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

    let names: Vec<&str> = nn::registry::REGISTRY.iter().map(|entry| entry.name).collect();
    assert_eq!(
        names.len(),
        types.len(),
        "注册表每行必须是一份 `类型 => 模板`：{types:?}"
    );

    // 方向一：注解 → 注册表（名字取自 `Operator::NAME`，不是文本）。
    let mut failures: Vec<String> = Vec::new();
    for (name, at) in &annotated {
        if !names.contains(&name.as_str()) {
            failures.push(format!(
                "{at}: 标了 `#[annotation::model]` 生成 `MODEL`，但 `REGISTRY` 里没有 `{name}`：\
             发布出去的二进制里没有这条算子"
            ));
        }
    }

    // 方向二：注册表 → 注解生成的语义表。类型路径的模块文件里必须有字面量注解，或调用一个
    // 会展开出 `#[annotation::model]` 的宏；`model_of` 的路径正确性由编译期 bound 保证。
    let model_macros = model_macros();
    for (name, ty) in names.iter().zip(&types) {
        let Some((module_path, _ty)) = ty.rsplit_once("::") else {
            failures.push(format!(
                "`{name}` 的类型 `{ty}` 不是 `模块::类型` 形式：注册表必须指向 Operator 实现"
            ));
            continue;
        };
        let module = module_path.split("::").next().unwrap_or(module_path);
        let file = locate_module(module)
            .unwrap_or_else(|| panic!("`{name}` 指向 `{ty}`，但 `pub mod {module}` 读不到"));
        let text = fs::read_to_string(&file).expect("读模块文件");
        let via_macro = invokes_model_macro(&text, &model_macros);
        if !PRODUCERS.iter().any(|producer| text.contains(producer)) && !via_macro {
            failures.push(format!(
                "`{name}` 的类型在 {} 里没有注解：`#[annotation::model]` 是语义表的唯一生产者",
                file.display()
            ));
        }
        if !annotated.iter().any(|(annotated, _)| annotated == name) && !via_macro {
            failures.push(format!(
                "`{name}` 在 `REGISTRY` 里，但它的模块没有同名注解：注解是语义表的唯一生产者"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `op/src` 里会展开出 `#[annotation::model]` 的宏名：宏体里的 `name = $op` 不是字面量，
/// 扫描器不认，但调用这些宏的算子模块同样满足"注解是唯一生产者"。判据是宏体里出现了
/// `#[annotation::model(name = $..)`，再沿**宏调用链**取传递闭包——装配骨架一层层加间接
/// （`window_op!` → `classic_op!` → 注解），只认一跳的话，每加一层间接算子就被判成"没有注解"。
fn model_macros() -> std::collections::BTreeSet<String> {
    let bodies = macro_bodies();
    let mut macros: std::collections::BTreeSet<String> = bodies
        .iter()
        .filter(|(_, body)| {
            body.lines()
                .any(|line| line.contains("#[annotation::model(") && line.contains("name = $"))
        })
        .map(|(name, _)| name.clone())
        .collect();
    loop {
        let reached: Vec<String> = bodies
            .iter()
            .filter(|(name, body)| {
                !macros.contains(*name)
                    && macros
                        .iter()
                        .any(|known| body.contains(&format!("{known}!")))
            })
            .map(|(name, _)| name.clone())
            .collect();
        if reached.is_empty() {
            return macros;
        }
        macros.extend(reached);
    }
}

/// 每个 `macro_rules!` 的宏体：定义行到收尾的 `}`，不展开。
fn macro_bodies() -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    for (crate_name, file) in sources() {
        if !in_op_family(&crate_name) {
            continue;
        }
        let Some(text) = source_text(&file) else {
            continue;
        };
        let mut current: Option<(String, String)> = None;
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix("macro_rules! ") {
                if let Some((name, body)) = current.take() {
                    out.insert(name, body);
                }
                let name = rest
                    .split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
                    .next()
                    .unwrap_or("")
                    .to_string();
                current = Some((name, String::new()));
                continue;
            }
            if let Some((name, body)) = current.as_mut() {
                // 收尾的 `}` 必须顶格：宏体内部函数的右括号在缩进里，认错会截断半个宏体。
                if line == "}" && !body.is_empty() {
                    out.insert(name.clone(), std::mem::take(body));
                    current = None;
                    continue;
                }
                body.push_str(line);
                body.push('\n');
            }
        }
        if let Some((name, body)) = current.take() {
            out.insert(name, body);
        }
    }
    out
}

/// 模块文件是否调用了某个会展开出 `#[annotation::model]` 的宏（`crate::binary_f32!` 这类）。
fn invokes_model_macro(text: &str, macros: &std::collections::BTreeSet<String>) -> bool {
    macros.iter().any(|name| text.contains(&format!("{name}!")))
}

/// 模板签名 / 注册表语义表 / 注解声明三者的参数类型必须逐项对齐。
#[test]
fn every_registered_kernel_signature_matches_its_model_sample() {
    let mut checked = 0;
    let mut failures: Vec<String> = Vec::new();
    for entry in nn::registry::REGISTRY {
        // 模板正文归目录（`prof::catalogue`），注册表只留名字与语义表。
        let template = prof::catalogue::template(entry.name).expect("注册过的名字都有模板");
        let signature = match nn::sig::Signature::parse(entry.name, template) {
            Ok(signature) => signature,
            Err(error) => {
                failures.push(format!("`{}`：签名解析失败：{error}", entry.name));
                continue;
            }
        };
        let sample = (entry.model.sample)(entry.model.shape.0, entry.model.shape.1);
        if let Err(error) = signature.check_sample(&sample) {
            failures.push(format!("`{}`：样本与 `.k` 签名不一致：{error}", entry.name));
        }
        let output_dtype = signature.output_type().and_then(|ctype| ctype.dtype());
        if output_dtype != Some(entry.model.dtype) {
            failures.push(format!(
                "`{}`：`.k` 可写输出是 `{}`，语义表 dtype 是 `{:?}`，两者必须是同一个类型",
                entry.name,
                signature.output_type().map_or("-", nn::sig::CType::c_name),
                entry.model.dtype
            ));
        }
        checked += 1;
    }
    assert!(
        checked >= 2,
        "扫描器只看到 {checked} 条注册：扫描逻辑或表布局坏了"
    );
    assert_eq!(checked, nn::registry::REGISTRY.len(), "注册表每条都必须过签名对齐");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 一条命中的行号（从 1 开始），用于把失败指回源码。
fn line_number(text: &str, line: &str) -> String {
    text.lines()
        .position(|candidate| candidate == line)
        .map_or_else(|| "-".to_string(), |idx| (idx + 1).to_string())
}
