//! 设备片段层的契约：一台设备的 P0 片段、见证入口与档案供给必须互相闭合。
//!
//! 判据只认三样东西——目录里的文件、生成的 `FRAGMENTS` / `PRIMS` 表、档案里的 `[[provides]]`。
//! 六台设备共用这一套，所以"这台设备少了什么"不靠人读注释，靠 `cargo test` 说。
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
    pub fn assert_p0_set(&self) {
        let mut names: Vec<&str> = self.fragments.iter().map(|(prim, _)| *prim).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "`{}` 有两条片段抢同一个原语", self.id);
        let mut want = self.p0.to_vec();
        want.sort_unstable();
        assert_eq!(
            names, want,
            "`{}` 的片段层与 P0 闭集不一致：多出来的片段没有模板会用到，缺的原语会让模板拼不出内核",
            self.id
        );
    }

    /// 每条签名的 `(原语, impl)` 必须是档案里的一条供给，且那条供给有见证——签名是承诺，没有见证
    /// 的签名是"看起来实现了"。
    pub fn assert_signatures(&self) {
        for (prim, body) in self.fragments {
            let (signed_prim, signed_impl) = sign(body);
            assert_eq!(
                signed_prim, *prim,
                "`{}` 的 `{prim}.{}` 签名写着 `{signed_prim}`：文件名即原语，两者必须是同一个词",
                self.id, self.ext
            );
            let provide = self
                .provides
                .iter()
                .find(|provide| provide.prim == signed_prim && provide.impl_name == signed_impl)
                .unwrap_or_else(|| {
                    panic!(
                        "`{}` 的片段署名 `{signed_prim}/{signed_impl}` 在档案里没有对应供给：正文写的实现无人认领",
                        self.id
                    )
                });
            let witness = provide.witness.unwrap_or_else(|| {
                panic!(
                    "`{}` 的 `{signed_prim}/{signed_impl}` 有片段正文却没有见证：这条签名不可执行，等级只能是 L0",
                    self.id
                )
            });
            let entry = self
                .prims
                .iter()
                .find(|(prim, _)| *prim == signed_prim)
                .map(|(_, func)| *func)
                .unwrap_or_else(|| {
                    panic!(
                        "`{}` 的 `{signed_prim}` 有片段却没有入口：镜像里查不到它的见证",
                        self.id
                    )
                });
            assert_eq!(
                entry, witness.func,
                "`{}` 的 `{signed_prim}` 入口 `{entry}` 与档案的见证 `{}` 不是同一条",
                self.id, witness.func
            );
        }
    }

    /// 见证文件里的每个 `int witness_*(float *out)` 都必须被某条供给认领，每个入口都必须在文件里
    /// 定义一次：多出来的是死代码，少一条是镜像里点名点空。
    pub fn assert_witness_entries(&self) {
        let defined = definitions(self.source);
        for (prim, func) in self.prims {
            let count = defined.iter().filter(|name| **name == *func).count();
            assert_eq!(
                count, 1,
                "`{}` 的 `{prim}` 入口 `{func}` 在见证文件里出现了 {count} 次：一条签名一个函数",
                self.id
            );
        }
        for name in &defined {
            assert!(
                self.provides
                    .iter()
                    .any(|provide| provide.witness.is_some_and(|witness| witness.func == *name)),
                "`{}` 的见证函数 `{name}` 没有任何供给认领：它执行的算法归属不到一条签名",
                self.id
            );
        }
    }

    /// 没有片段正文的供给必须带见证。扫描器一台设备一种语言、一个原语至多一份正文，所以
    /// 硬件具备但 P0 模板用不上的能力只能停在档案里；这类供给若连见证都没有，就只剩一行
    /// 声明——正文说"怎么写"、见证说"写成什么"，两样都缺的能力不该出现在档案里。
    pub fn assert_evidence(&self) {
        for provide in self.provides {
            let fragmented = self
                .fragments
                .iter()
                .any(|(_, body)| sign(body) == (provide.prim, provide.impl_name));
            assert!(
                fragmented || provide.witness.is_some(),
                "`{}` 的 `{}/{}` 既没有片段正文也没有见证：这条能力无凭无据",
                self.id,
                provide.prim,
                provide.impl_name
            );
        }
    }

    /// 目录是闭集：只有 P0 的九个 `<prim>.<ext>` 与一个 `witness.<ext>`。`prof::codegen` 的扫描器
    /// 只认这两种名字，多出来的设备源文件会在构建期被拒——这里提前说清是哪个文件。
    ///
    /// `.rs` 不算内容：片段目录就是 crate 的 `src/`，`lib.rs` 与它住在一起。
    pub fn assert_closed_directory(&self) {
        let mut found: Vec<String> = std::fs::read_dir(self.dir)
            .unwrap_or_else(|error| panic!("`{}` 的片段目录不可读：{error}", self.id))
            .map(|entry| {
                entry
                    .expect("目录项可读")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .filter(|name| !name.ends_with(".rs"))
            .collect();
        found.sort();
        let mut want: Vec<String> = self
            .p0
            .iter()
            .map(|prim| format!("{prim}.{}", self.ext))
            .collect();
        want.push(format!("witness.{}", self.ext));
        want.sort();
        assert_eq!(
            found, want,
            "`{}` 的片段目录不是闭集：只允许 P0 的 `<prim>.{}` 与 `witness.{}`",
            self.id, self.ext, self.ext
        );
    }
}

/// 取片段首行的签名声明：`/* cutile-sign: <prim>/<impl> */`。
fn sign(body: &str) -> (&str, &str) {
    let rest = body
        .split_once("cutile-sign:")
        .map(|(_, rest)| rest)
        .unwrap_or_else(|| panic!("片段缺签名声明 `/* cutile-sign: <prim>/<impl> */`"));
    let line = rest
        .split_once("*/")
        .map(|(line, _)| line)
        .unwrap_or_else(|| panic!("签名声明没有闭合的 `*/`"));
    let line = line.trim();
    line.split_once('/')
        .map(|(prim, implementation)| (prim.trim(), implementation.trim()))
        .unwrap_or_else(|| panic!("签名 `{line}` 不是 `<prim>/<impl>` 两段"))
}

/// 见证文件里的 `int witness_*(float *out)` 定义名。注释里那种 `int witness_<name>(...)` 因为
/// 名字里带着 `<`，在这里被过滤掉。
fn definitions(source: &str) -> Vec<&str> {
    let mut names = Vec::new();
    for (index, _) in source.match_indices("int ") {
        let rest = &source[index + 4..];
        let Some(open) = rest.find('(') else {
            continue;
        };
        let name = rest[..open].trim();
        if name.is_empty() || !name.starts_with("witness_") {
            continue;
        }
        if !name
            .chars()
            .all(|letter| letter.is_ascii_alphanumeric() || letter == '_')
        {
            continue;
        }
        let tail = &rest[open..];
        let Some(close) = tail.find(')') else {
            continue;
        };
        if !tail[..close].contains("float *out") {
            continue;
        }
        names.push(name);
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_signature_is_read_from_the_first_comment() {
        assert_eq!(
            sign("/* cutile-sign: load/scalar_copy */\n$0 = $1;"),
            ("load", "scalar_copy")
        );
    }

    #[test]
    fn only_real_definitions_are_witnesses() {
        let source = "// 统一 ABI：`int witness_<name>(float *out)` 返回个数\n\
                      int witness_dot(float *out) { out[0] = 1.0f; return 1; }\n\
                      int helper(int x) { return x; }\n";
        assert_eq!(definitions(source), ["witness_dot"]);
    }

    #[test]
    fn an_unclaimed_witness_is_a_defect() {
        let fragments = [("load", "/* cutile-sign: load/scalar_copy */\n$0 = $1;")];
        let prims = [("load", "witness_load")];
        let provides = [Provision {
            prim: "load",
            impl_name: "scalar_copy",
            witness: Some(Witness {
                func: "witness_load",
                expect: &[1.0],
            }),
        }];
        let contract = Contract {
            id: "probe",
            ext: "c",
            dir: ".",
            p0: &["load"],
            fragments: &fragments,
            prims: &prims,
            provides: &provides,
            source: "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n\
                     int witness_orphan(float *out) { out[0] = 1.0f; return 1; }\n",
        };
        contract.assert_signatures();
        let failed = std::panic::catch_unwind(|| contract.assert_witness_entries());
        assert!(failed.is_err(), "没人认领的见证必须被发现");
    }

    #[test]
    fn a_quoted_include_in_a_witness_is_a_defect() {
        let judge = |source: &str| {
            Contract {
                id: "probe",
                ext: "c",
                dir: ".",
                p0: &[],
                fragments: &[],
                prims: &[],
                provides: &[],
                source,
            }
            .assert_self_contained();
        };
        judge("#include <math.h>\nint witness_a(float *out) { out[0] = 1.0f; return 1; }\n");
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            judge(
                "#include \"preamble.h\"\nint witness_a(float *out) { out[0] = 1.0f; return 1; }\n",
            )
        }))
        .expect_err("引号包含在副本旁边解析不到，必须被发现");
    }

    #[test]
    fn a_provision_without_any_evidence_is_a_defect() {
        let fragments = [("load", "/* cutile-sign: load/scalar_copy */\n$0 = $1;")];
        let prims = [("load", "witness_load")];
        let judge = |provides: &[Provision<'_>]| {
            Contract {
                id: "probe",
                ext: "c",
                dir: ".",
                p0: &["load"],
                fragments: &fragments,
                prims: &prims,
                provides,
                source: "",
            }
            .assert_evidence();
        };
        let copied = Provision {
            prim: "load",
            impl_name: "scalar_copy",
            witness: Some(Witness {
                func: "witness_load",
                expect: &[1.0],
            }),
        };
        let bare = [
            copied,
            Provision {
                prim: "mma",
                impl_name: "cube",
                witness: None,
            },
        ];
        let backed = [
            copied,
            Provision {
                prim: "mma",
                impl_name: "cube",
                witness: Some(Witness {
                    func: "witness_mma",
                    expect: &[13.0],
                }),
            },
        ];
        judge(&backed);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| judge(&bare)))
            .expect_err("只有一行声明、正文与见证都缺的供给必须被发现");
    }
}
