//! 注册表与注解的镜像关系：语义表只有一个生产者，签名、样本、`dtype` 逐项对齐。
//!
//! 类型系统守得住「注解说的是哪份格式」，守不住两件纯文本事故：标了注解却没进注册表，
//! 或注册表里手写了第二份 `Model` 字面量。两条规则都没有合法例外——出现例外说明契约本身
//! 该改，而不是这里该豁免。

#[cfg(test)]
pub(crate) mod tests {
    use std::fs;

    use crate::architecture::tests::workspace_root as root;
    use crate::contracts::tests::{locate_module, sources};

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

        let names: Vec<&str> = op::REGISTRY.iter().map(|entry| entry.name).collect();
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
    /// `#[annotation::model(name = $..)`——逐文件追踪最近的 `macro_rules!` 名，不展开宏。
    fn model_macros() -> std::collections::BTreeSet<String> {
        let mut macros = std::collections::BTreeSet::new();
        for (crate_name, file) in sources() {
            if crate_name != "op" {
                continue;
            }
            let Ok(text) = fs::read_to_string(&file) else {
                continue;
            };
            let mut current: Option<String> = None;
            for line in text.lines() {
                if let Some(rest) = line.trim().strip_prefix("macro_rules! ") {
                    current = Some(
                        rest.split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
                            .next()
                            .unwrap_or("")
                            .to_string(),
                    );
                }
                if line.contains("#[annotation::model(")
                    && line.contains("name = $")
                    && let Some(name) = &current
                {
                    macros.insert(name.clone());
                }
            }
        }
        macros
    }

    /// 模块文件是否调用了某个会展开出 `#[annotation::model]` 的宏（`crate::binary_f32!` 这类）。
    fn invokes_model_macro(text: &str, macros: &std::collections::BTreeSet<String>) -> bool {
        macros.iter().any(|name| text.contains(&format!("{name}!")))
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
}
