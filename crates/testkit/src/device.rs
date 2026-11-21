//! 设备片段层的契约：一台设备的 P0 片段、见证入口与档案供给必须互相闭合。
//!
//! 判据只认三样东西——目录里的文件、生成的 `FRAGMENTS` / `PRIMS` 表、档案里的 `[[provides]]`。
//! 六台设备共用这一套，所以"这台设备少了什么"不靠人读注释，靠 `cargo test` 说。
//!
//! 每个判据**只产出缺陷**（`Result<(), String>`），不自己失败：失败是测试的事。这样
//! `crates/tools` 的零 panic 门不必为夹具开口子，测试也不必靠 `catch_unwind` 观察断言。
//!
//! 这里刻意不出现 `prof` 的类型：设备 crate 在测试里把档案摊成 [`Provision`]，于是 `testkit`
//! 仍然只有 `std` 一个依赖，"夹具不进生产二进制"的边界不被打破。

/// 一条供给：档案里"这台设备上这条原语怎么写"的最小事实。
#[derive(Debug, Clone, Copy)]
pub struct Provision<'a> {
    /// 原语名，如 `load`。
    pub prim: &'a str,
    /// 实现名，如 `scalar_copy`。
    pub impl_name: &'a str,
    /// 见证入口与期望值；`None` 表示这条供给还没有可执行的见证。
    pub witness: Option<Witness<'a>>,
}

/// 一条见证：入口名 + 期望输出。
#[derive(Debug, Clone, Copy)]
pub struct Witness<'a> {
    /// 入口函数名。
    pub func: &'a str,
    /// 期望输出，长度即函数的返回值。
    pub expect: &'a [f32],
}

/// 一台设备在片段层的自述：由设备 crate 的生成表与档案拼出来。
#[derive(Debug, Clone, Copy)]
pub struct Contract<'a> {
    /// 档案 id，失败信息里用它指认是哪台设备。
    pub id: &'a str,
    /// 片段与见证的扩展名。
    pub ext: &'a str,
    /// 片段目录（设备 crate 的 `ROOT`）。
    pub dir: &'a str,
    /// P0 闭集：模板真的会用到的原语名。
    pub p0: &'a [&'a str],
    /// 片段：`(原语, 正文)`，来自生成的 `FRAGMENTS`。
    pub fragments: &'a [(&'a str, &'a str)],
    /// 入口：`(原语, 见证函数名)`，来自生成的 `PRIMS`。
    pub prims: &'a [(&'a str, &'a str)],
    /// 档案里这台设备的全部供给。
    pub provides: &'a [Provision<'a>],
    /// 见证源码（`DEVICE.source`）。
    pub source: &'a str,
}

impl<'a> Contract<'a> {
    /// 片段必须**恰好**覆盖 P0 闭集：一个不多（多余的原语没有模板用），一个不少（缺一条模板拼不出来）。
    ///
    /// # Errors
    ///
    /// 有两条片段抢同一个原语，或片段集合与 P0 闭集不相等时返回原因。
    pub fn check_p0_set(&self) -> Result<(), String> {
        let mut names: Vec<&str> = self.fragments.iter().map(|(prim, _)| *prim).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        if before != names.len() {
            return Err(format!("`{}` 有两条片段抢同一个原语", self.id));
        }
        let mut want = self.p0.to_vec();
        want.sort_unstable();
        if names == want {
            return Ok(());
        }
        Err(format!(
            "`{}` 的片段层与 P0 闭集不一致：多出来的片段没有模板会用到，缺的原语会让模板拼不出内核",
            self.id
        ))
    }

    /// 每条签名的 `(原语, impl)` 必须是档案里的一条供给，且那条供给有见证——签名是承诺，没有见证
    /// 的签名是"看起来实现了"。入口名还必须与档案的见证是同一条。
    ///
    /// # Errors
    ///
    /// 第一条不闭合的签名返回原因；`Ok` 表示九条签名逐条对上了。
    pub fn check_signatures(&self) -> Result<(), String> {
        for (prim, body) in self.fragments {
            self.check_signature(prim, body)?;
        }
        Ok(())
    }

    /// 见证文件里的每个 `int witness_*(float *out)` 都必须被某条供给认领，每个入口都必须在文件里
    /// 定义一次：多出来的是死代码，少一条是镜像里点名点空。
    ///
    /// # Errors
    ///
    /// 入口缺席、重复，或有见证函数无人认领时返回原因。
    pub fn check_witness_entries(&self) -> Result<(), String> {
        let defined = definitions(self.source);
        for (prim, func) in self.prims {
            let count = defined.iter().filter(|name| **name == *func).count();
            if count != 1 {
                return Err(format!(
                    "`{}` 的 `{prim}` 入口 `{func}` 在见证文件里出现了 {count} 次：一条签名一个函数",
                    self.id
                ));
            }
        }
        for name in &defined {
            if self.claimed(name) {
                continue;
            }
            return Err(format!(
                "`{}` 的见证函数 `{name}` 没有任何供给认领：它执行的算法归属不到一条签名",
                self.id
            ));
        }
        Ok(())
    }

    /// 没有片段正文的供给必须带见证。扫描器一台设备一种语言、一个原语至多一份正文，所以
    /// 硬件具备但 P0 模板用不上的能力只能停在档案里；这类供给若连见证都没有，就只剩一行
    /// 声明——正文说"怎么写"、见证说"写成什么"，两样都缺的能力不该出现在档案里。
    ///
    /// # Errors
    ///
    /// 第一条"正文与见证都缺"的供给返回原因。
    pub fn check_evidence(&self) -> Result<(), String> {
        for provide in self.provides {
            let fragmented = self
                .fragments
                .iter()
                .any(|(_, body)| sign(body).is_ok_and(|sign| sign == (provide.prim, provide.impl_name)));
            if fragmented || provide.witness.is_some() {
                continue;
            }
            return Err(format!(
                "`{}` 的 `{}/{}` 既没有片段正文也没有见证：这条能力无凭无据",
                self.id, provide.prim, provide.impl_name
            ));
        }
        Ok(())
    }

    /// 目录是闭集：只有 P0 的九个 `<prim>.<ext>` 与一个 `witness.<ext>`。`prof::codegen` 的扫描器
    /// 只认这两种名字，多出来的设备源文件会在构建期被拒——这里提前说清是哪个文件。
    ///
    /// `.rs` 不算内容：片段目录就是 crate 的 `src/`，`lib.rs` 与它住在一起。
    ///
    /// # Errors
    ///
    /// 目录不可读，或目录里出现了闭集之外的名字时返回原因。
    pub fn check_closed_directory(&self) -> Result<(), String> {
        let entries = std::fs::read_dir(self.dir)
            .map_err(|error| format!("`{}` 的片段目录不可读：{error}", self.id))?;
        let mut found = Vec::new();
        for entry in entries {
            let entry = entry
                .map_err(|error| format!("`{}` 的片段目录项不可读：{error}", self.id))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".rs") {
                found.push(name);
            }
        }
        found.sort();
        let mut want = self.expected_names();
        want.sort();
        if found == want {
            return Ok(());
        }
        Err(format!(
            "`{}` 的片段目录不是闭集：只允许 P0 的 `<prim>.{}` 与 `witness.{}`，实得 {found:?}",
            self.id, self.ext, self.ext
        ))
    }

    /// 一条签名与档案的对账：文件名、署名、供给、见证入口四者必须是同一个词。
    fn check_signature(&self, prim: &str, body: &str) -> Result<(), String> {
        let (signed_prim, signed_impl) = sign(body)?;
        if signed_prim != prim {
            return Err(format!(
                "`{}` 的 `{prim}.{}` 签名写着 `{signed_prim}`：文件名即原语，两者必须是同一个词",
                self.id, self.ext
            ));
        }
        let provide = self.provide(signed_prim, signed_impl)?;
        let witness = provide.witness.ok_or_else(|| {
            format!(
                "`{}` 的 `{signed_prim}/{signed_impl}` 有片段正文却没有见证：这条签名不可执行，等级只能是 L0",
                self.id
            )
        })?;
        let entry = self.entry(signed_prim)?;
        if entry == witness.func {
            return Ok(());
        }
        Err(format!(
            "`{}` 的 `{signed_prim}` 入口 `{entry}` 与档案的见证 `{}` 不是同一条",
            self.id, witness.func
        ))
    }

    /// 档案里认领 `prim/impl_name` 的那条供给。
    fn provide(&self, prim: &str, impl_name: &str) -> Result<&'a Provision<'a>, String> {
        let provides: &'a [Provision<'a>] = self.provides;
        let found = provides
            .iter()
            .find(|provide| provide.prim == prim && provide.impl_name == impl_name);
        found.ok_or_else(|| {
            format!(
                "`{}` 的片段署名 `{prim}/{impl_name}` 在档案里没有对应供给：正文写的实现无人认领",
                self.id
            )
        })
    }

    /// 生成表里 `prim` 的见证入口名。
    fn entry(&self, prim: &str) -> Result<&'a str, String> {
        let prims: &'a [(&'a str, &'a str)] = self.prims;
        let found = prims.iter().find(|(name, _)| *name == prim);
        found.map(|(_, func)| *func).ok_or_else(|| {
            format!("`{}` 的 `{prim}` 有片段却没有入口：镜像里查不到它的见证", self.id)
        })
    }

    /// 某条见证函数是否被档案里的一条供给认领。
    fn claimed(&self, name: &str) -> bool {
        self.provides
            .iter()
            .any(|provide| provide.witness.is_some_and(|witness| witness.func == name))
    }

    /// 闭集里的全部名字：P0 的 `<prim>.<ext>` 加一个 `witness.<ext>`。
    fn expected_names(&self) -> Vec<String> {
        let mut want: Vec<String> = self
            .p0
            .iter()
            .map(|prim| format!("{prim}.{}", self.ext))
            .collect();
        want.push(format!("witness.{}", self.ext));
        want
    }
}

/// 取片段首行的签名声明：`/* cutile-sign: <prim>/<impl> */`。
///
/// # Errors
///
/// 片段没有签名声明、声明没有闭合，或声明不是两段时返回原因。
fn sign(body: &str) -> Result<(&str, &str), String> {
    let (_, rest) = body
        .split_once("cutile-sign:")
        .ok_or_else(|| String::from("片段缺签名声明 `/* cutile-sign: <prim>/<impl> */`"))?;
    let (line, _) = rest
        .split_once("*/")
        .ok_or_else(|| String::from("签名声明没有闭合的 `*/`"))?;
    let line = line.trim();
    let (prim, implementation) = line
        .split_once('/')
        .ok_or_else(|| format!("签名 `{line}` 不是 `<prim>/<impl>` 两段"))?;
    Ok((prim.trim(), implementation.trim()))
}

/// 见证文件里的 `int witness_*(float *out)` 定义名。注释里那种 `int witness_<name>(...)` 因为
/// 名字里带着 `<`，在这里被过滤掉。
fn definitions(source: &str) -> Vec<&str> {
    let mut names = Vec::new();
    for (index, _) in source.match_indices("int ") {
        let name = defined_name(&source[index + 4..]);
        if let Some(name) = name {
            names.push(name);
        }
    }
    names
}

/// 从 `int ` 之后的那段正文里读出函数名：只认 `witness_*` 且参数表里有 `float *out` 的定义。
fn defined_name(rest: &str) -> Option<&str> {
    let (head, tail) = rest.split_once('(')?;
    let name = head.trim();
    let (args, _) = tail.split_once(')')?;
    (is_witness_name(name) && args.contains("float *out")).then_some(name)
}

/// 见证入口的命名形状：`witness_` 前缀，其余全是标识符字符（于是 `witness_<name>` 不算）。
fn is_witness_name(name: &str) -> bool {
    name.starts_with("witness_")
        && name
            .chars()
            .all(|letter| letter.is_ascii_alphanumeric() || letter == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGNED: &str = "/* cutile-sign: load/scalar_copy */\n$0 = $1;";

    /// 一台最小的探针：一条 `load/scalar_copy` 供给，见证源码里一个 `witness_load`。
    const PROVIDES: [Provision<'_>; 1] = [Provision {
        prim: "load",
        impl_name: "scalar_copy",
        witness: Some(Witness {
            func: "witness_load",
            expect: &[1.0],
        }),
    }];

    fn probe<'a>(fragments: &'a [(&'a str, &'a str)], source: &'a str) -> Contract<'a> {
        Contract {
            id: "probe",
            ext: "c",
            dir: ".",
            p0: &["load"],
            fragments,
            prims: PRIMS_OF_PROBE,
            provides: &PROVIDES,
            source,
        }
    }

    const PRIMS_OF_PROBE: &[(&str, &str)] = &[("load", "witness_load")];

    #[test]
    fn a_signature_is_read_from_the_first_comment() {
        assert_eq!(sign(SIGNED).ok(), Some(("load", "scalar_copy")));
        assert!(sign("$0 = $1;").is_err(), "缺签名声明必须被发现");
        assert!(sign("/* cutile-sign: load */").is_err(), "少一段必须被发现");
    }

    #[test]
    fn only_real_definitions_are_witnesses() {
        let source = "// 统一 ABI：`int witness_<name>(float *out)` 返回个数\n\
                      int witness_dot(float *out) { out[0] = 1.0f; return 1; }\n\
                      int helper(int x) { return x; }\n";
        assert_eq!(definitions(source), ["witness_dot"]);
    }

    #[test]
    fn a_well_formed_probe_passes_every_judgement() {
        let fragments = [("load", SIGNED)];
        let source = "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n";
        let contract = probe(&fragments, source);
        assert_eq!(contract.check_p0_set(), Ok(()));
        assert_eq!(contract.check_signatures(), Ok(()));
        assert_eq!(contract.check_witness_entries(), Ok(()));
        assert_eq!(contract.check_evidence(), Ok(()));
    }

    #[test]
    fn a_signature_the_archive_does_not_claim_is_a_defect() {
        let fragments = [("load", "/* cutile-sign: load/vectorized */\n$0 = $1;")];
        let defect = probe(&fragments, "").check_signatures();
        assert!(
            defect.as_ref().is_err_and(|reason| reason.contains("没有对应供给")),
            "无人认领的实现必须被发现：{defect:?}"
        );
    }

    #[test]
    fn an_unclaimed_witness_is_a_defect() {
        let fragments = [("load", SIGNED)];
        let source = "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n\
                      int witness_orphan(float *out) { out[0] = 1.0f; return 1; }\n";
        let defect = probe(&fragments, source).check_witness_entries();
        assert!(
            defect.as_ref().is_err_and(|reason| reason.contains("没有任何供给认领")),
            "没人认领的见证必须被发现：{defect:?}"
        );
    }

    #[test]
    fn a_provision_without_any_evidence_is_a_defect() {
        let fragments = [("load", SIGNED)];
        let bare = [
            PROVIDES[0],
            Provision {
                prim: "mma",
                impl_name: "cube",
                witness: None,
            },
        ];
        let contract = Contract {
            provides: &bare,
            ..probe(&fragments, "")
        };
        let defect = contract.check_evidence();
        assert!(
            defect.as_ref().is_err_and(|reason| reason.contains("无凭无据")),
            "只有一行声明、正文与见证都缺的供给必须被发现：{defect:?}"
        );
    }

    #[test]
    fn a_fragment_set_that_overruns_p0_is_a_defect() {
        let fragments = [("load", SIGNED), ("load", SIGNED)];
        let defect = probe(&fragments, "").check_p0_set();
        assert!(
            defect.as_ref().is_err_and(|reason| reason.contains("抢同一个原语")),
            "同一原语两份正文必须被发现：{defect:?}"
        );
    }
}
