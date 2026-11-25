//! 设备片段契约与可执行见证：P0 片段、档案供给、入口和本机读数必须闭合。

use std::collections::{BTreeMap, BTreeSet};

use crate::witness::{Call, MAX_OUT, Reading, run};

/// 见证对拍的容差，与 `cutile::witness` 同一个口径。
pub const TOLERANCE: f64 = 1e-5;
/// P0 闭集：`crates/prof/src/<vendor>/kernels/*.k` 会 `@` 到的八个原语。
pub const P0: [&str; 8] = [
    "cvt", "dot", "load", "load_async", "mma", "quant", "reduce", "store",
];

/// Rust、TOML 与共享参考由生成器单独校验，其余设备源文件进入目录闭集核对。
fn is_device_fragment(name: &str) -> bool {
    !(name.ends_with(".rs") || name.ends_with(".toml") || name.ends_with(".k"))
}

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

/// 一台设备在片段层的自述：由设备模块的生成表与档案拼出来。
#[derive(Debug, Clone, Copy)]
pub struct Contract<'a> {
    /// 档案 id，失败信息里用它指认是哪台设备。
    pub id: &'a str,
    /// 片段与见证的扩展名。
    pub ext: &'a str,
    /// 片段目录（设备模块的 `ROOT`）。
    pub dir: &'a str,
    /// P0 闭集：模板真的会用到的原语名。
    pub p0: &'a [&'a str],
    /// 片段：`(原语, 正文)`，来自生成的 `FRAGMENTS`。
    pub fragments: &'a [(&'a str, &'a str)],
    /// 入口：`(原语, 见证函数名)`，来自生成的 `PRIMS`。
    pub prims: &'a [(&'a str, &'a str)],
    /// 档案里这台设备的全部供给。
    pub provides: &'a [Provision<'a>],
    /// 写回尾块的文件名主干（`prof::sign` 的那一份）。
    pub epilogue_stem: &'a str,
    /// 见证源码（`DEVICE.source`）。
    pub source: &'a str,
}

impl<'a> Contract<'a> {
    /// 片段必须**恰好**覆盖 P0 闭集，且同一原语不得有两份正文。
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

    /// 每条签名都必须对应一条带见证的档案供给，且入口一致。
    pub fn check_signatures(&self) -> Result<(), String> {
        self.fragments.iter().try_for_each(|(prim, body)| {
            let signed = sign(body)
                .map_err(|reason| format!("`{}` 的 `{prim}.{}`：{reason}", self.id, self.ext))?;
            self.check_signed(prim, signed)
        })
    }

    /// 文件名与署名必须指向同一个原语，然后交给档案对账。
    fn check_signed(&self, prim: &str, signed: (&str, &str)) -> Result<(), String> {
        let (signed_prim, signed_impl) = signed;
        if signed_prim != prim {
            return Err(format!(
                "`{}` 的 `{prim}.{}` 签名写着 `{signed_prim}`：文件名即原语，两者必须是同一个词",
                self.id, self.ext
            ));
        }
        self.check_provide(signed_prim, signed_impl)
    }

    /// 署名在档案里必须有供给、见证和同一个入口。
    fn check_provide(&self, prim: &str, impl_name: &str) -> Result<(), String> {
        let Some(provide) = self.find(prim, impl_name) else {
            return Err(format!(
                "`{}` 的片段署名 `{prim}/{impl_name}` 在档案里没有对应供给：正文写的实现无人认领",
                self.id
            ));
        };
        let Some(witness) = provide.witness else {
            return Err(format!(
                "`{}` 的 `{prim}/{impl_name}` 有片段正文却没有见证：这条签名不可执行，等级只能是 L0",
                self.id
            ));
        };
        let entry = self.entry(prim)?;
        if entry == witness.func {
            return Ok(());
        }
        Err(format!(
            "`{}` 的 `{prim}` 入口 `{entry}` 与档案的见证 `{}` 不是同一条",
            self.id, witness.func
        ))
    }

    /// 每个生成表入口恰好定义一次，每个见证函数也恰好被一条供给认领。
    pub fn check_witness_entries(&self) -> Result<(), String> {
        let defined = definitions(self.source);
        let mut counts = BTreeMap::new();
        for name in defined {
            *counts.entry(name).or_insert(0usize) += 1;
        }
        self.check_one_each(&counts)?;
        self.check_unclaimed(&counts)
    }

    /// 用一次建好的定义索引核对所有入口，避免每个原语重扫整个见证文件。
    fn check_one_each(&self, defined: &BTreeMap<&str, usize>) -> Result<(), String> {
        self.prims.iter().try_for_each(|(prim, func)| {
            let count = defined.get(func).copied().unwrap_or_default();
            if count == 1 {
                return Ok(());
            }
            Err(format!(
                "`{}` 的原语 `{prim}` 入口 `{func}` 在见证文件里出现了 {count} 次：一条签名一个函数",
                self.id
            ))
        })
    }

    /// 见证文件里每个函数都得有档案认领。
    fn check_unclaimed(&self, defined: &BTreeMap<&str, usize>) -> Result<(), String> {
        defined.keys().try_for_each(|name| {
            if self.claimed(name) {
                return Ok(());
            }
            Err(format!(
                "`{}` 的见证函数 `{name}` 没有任何供给认领：它执行的算法归属不到一条签名",
                self.id
            ))
        })
    }

    /// 没有片段正文的供给必须带见证；两项都缺的能力没有证据。
    pub fn check_evidence(&self) -> Result<(), String> {
        let signed: BTreeSet<(&str, &str)> = self
            .fragments
            .iter()
            .filter_map(|(_, body)| sign(body).ok())
            .collect();
        self.provides.iter().try_for_each(|provide| {
            let signature = (provide.prim, provide.impl_name);
            if signed.contains(&signature) || provide.witness.is_some() {
                return Ok(());
            }
            Err(format!(
                "`{}` 的 `{}/{}` 既没有片段正文也没有见证：这条能力无凭无据",
                self.id, provide.prim, provide.impl_name
            ))
        })
    }

    /// 目录是闭集：P0、见证/尾块之外，只允许前置声明与已声明算子的特化。
    pub fn check_closed_directory(&self) -> Result<(), String> {
        let found = self.found_names()?;
        let mut want = self.expected_names();
        want.extend(
            found
                .iter()
                .filter(|name| self.is_known_auxiliary(name))
                .cloned(),
        );
        want.sort();
        if found == want {
            return Ok(());
        }
        Err(format!(
            "`{}` 的片段目录不是闭集：只允许 P0、见证/尾块、前置声明与已声明算子的特化正文，实得 {found:?}",
            self.id
        ))
    }

    /// 前置声明与共享 `ops/<stem>.k` 的特化正文都是已登记辅助文件。
    fn is_known_auxiliary(&self, name: &str) -> bool {
        let Some(stem) = name.strip_suffix(&format!(".{}", self.ext)) else {
            return false;
        };
        if stem == "preamble" {
            return true;
        }
        std::path::Path::new(self.dir)
            .parent()
            .is_some_and(|src| src.join("ops").join(format!("{stem}.k")).is_file())
    }

    /// 片段目录里的全部文件名（子目录与生成器另行校验的文件除外）。
    fn found_names(&self) -> Result<Vec<String>, String> {
        let entries = std::fs::read_dir(self.dir)
            .map_err(|error| format!("`{}` 的片段目录不可读：{error}", self.id))?;
        let mut found = Vec::new();
        for entry in entries {
            self.record_name(entry, &mut found)?;
        }
        found.sort();
        Ok(found)
    }

    /// 收录单个目录项。
    fn record_name(
        &self,
        entry: std::io::Result<std::fs::DirEntry>,
        found: &mut Vec<String>,
    ) -> Result<(), String> {
        let entry = entry.map_err(|error| format!("`{}` 的片段目录项不可读：{error}", self.id))?;
        if entry.file_type().map_err(|error| error.to_string())?.is_dir() {
            return Ok(());
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_device_fragment(&name) {
            found.push(name);
        }
        Ok(())
    }

    /// 档案里认领 `prim/impl_name` 的那条供给。
    fn find(&self, prim: &str, impl_name: &str) -> Option<&'a Provision<'a>> {
        self.provides
            .iter()
            .find(|provide| provide.prim == prim && provide.impl_name == impl_name)
    }

    /// 生成表里 `prim` 的见证入口名。
    fn entry(&self, prim: &str) -> Result<&'a str, String> {
        self.prims
            .iter()
            .find(|(name, _)| *name == prim)
            .map(|(_, func)| *func)
            .ok_or_else(|| format!("`{}` 的 `{prim}` 有片段却没有入口：镜像里查不到它的见证", self.id))
    }

    /// 某条见证函数是否被档案里的一条供给认领。
    fn claimed(&self, name: &str) -> bool {
        self.provides
            .iter()
            .any(|provide| provide.witness.is_some_and(|witness| witness.func == name))
    }

    /// 闭集里的全部名字：P0 的 `<prim>.<ext>`、一个见证和一份尾块。
    fn expected_names(&self) -> Vec<String> {
        let mut want = Vec::with_capacity(self.p0.len() + 2);
        want.extend(self.p0.iter().map(|prim| format!("{prim}.{}", self.ext)));
        want.push(format!("witness.{}", self.ext));
        want.push(format!("{}.{}", self.epilogue_stem, self.ext));
        want
    }

    /// 见证源码必须自足：副本编译时无法解析设备目录里的引号包含。
    pub fn check_self_contained(&self) -> Result<(), String> {
        for (idx, line) in self.source.lines().enumerate() {
            let Some((_, rest)) = line.split_once("#include") else {
                continue;
            };
            let target = rest.trim();
            if target.starts_with('"') {
                return Err(format!(
                    "`{}` 的见证第 {} 行用引号包含了 `{target}`：副本编译时相对路径不存在",
                    self.id,
                    idx + 1
                ));
            }
        }
        Ok(())
    }

    /// 每条期望值必须写得进 `1..=MAX_OUT` 的回读缓冲。
    pub fn check_expectations(&self) -> Result<(), String> {
        self.provides.iter().try_for_each(|provide| {
            let Some(witness) = provide.witness else {
                return Ok(());
            };
            self.check_length(provide.prim, provide.impl_name, witness.expect)
        })
    }

    /// 一条期望值的长度校验。
    fn check_length(&self, kind: &str, name: &str, expect: &[f32]) -> Result<(), String> {
        let len = expect.len();
        if (1..=MAX_OUT).contains(&len) {
            return Ok(());
        }
        Err(format!(
            "`{}` 的 `{kind}/{name}` 期望值有 {len} 个：见证必须写 1..={MAX_OUT} 个元素",
            self.id
        ))
    }

    /// 每条供给的见证都要在本机算出档案写的值。
    pub fn check_witnesses(&self) -> Result<(), String> {
        let calls = self.calls();
        let Some(readings) = execute(self, &calls)? else {
            return Ok(());
        };
        judge(self.id, &calls, &readings)
    }

    /// 档案里带见证的供给，摊成执行器要的 `Call`。
    fn calls(&self) -> Vec<Call<'a>> {
        self.provides
            .iter()
            .filter_map(|provide| provide.witness)
            .map(|witness| Call {
                func: witness.func,
                expect: witness.expect,
            })
            .collect()
    }
}

/// 取片段首行的签名声明：`/* cutile-sign: <prim>/<impl> */`。
fn sign(body: &str) -> Result<(&str, &str), String> {
    let Some((_, rest)) = body.split_once("cutile-sign:") else {
        return Err(String::from("片段缺签名声明 `/* cutile-sign: <prim>/<impl> */`"));
    };
    let Some((line, _)) = rest.split_once("*/") else {
        return Err(String::from("签名声明没有闭合的 `*/`"));
    };
    let line = line.trim();
    let Some((prim, implementation)) = line.split_once('/') else {
        return Err(format!("签名 `{line}` 不是 `<prim>/<impl>` 两段"));
    };
    Ok((prim.trim(), implementation.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::{Call, run};

    const SIGNED: &str = "/* cutile-sign: load/scalar_copy */\n$0 = $1;";
    const FRAGMENTS: [(&str, &str); 1] = [("load", SIGNED)];
    const PRIMS: [(&str, &str); 1] = [("load", "witness_load")];
    const PROVIDES: [Provision<'_>; 1] = [Provision {
        prim: "load",
        impl_name: "scalar_copy",
        witness: Some(Witness {
            func: "witness_load",
            expect: &[1.0],
        }),
    }];

    fn probe<'a>(provides: &'a [Provision<'a>], source: &'a str) -> Contract<'a> {
        with_fragments(&FRAGMENTS, provides, source)
    }

    fn with_fragments<'a>(
        fragments: &'a [(&'a str, &'a str)],
        provides: &'a [Provision<'a>],
        source: &'a str,
    ) -> Contract<'a> {
        Contract {
            id: "probe",
            ext: "c",
            dir: ".",
            p0: &["load"],
            fragments,
            prims: &PRIMS,
            provides,
            epilogue_stem: "epilogue",
            source,
        }
    }

    #[test]
    fn a_signature_is_read_from_the_first_comment() {
        assert_eq!(sign(SIGNED).ok(), Some(("load", "scalar_copy")));
        assert!(sign("$0 = $1;").is_err(), "缺签名声明必须被发现");
        assert!(sign("/* cutile-sign: load */").is_err(), "少一段必须被发现");
    }

    #[test]
    fn only_real_definitions_are_witnesses() {
        let source = "// `int witness_<name>(float *out)` 只是注释\n\
                      int witness_dot(float *out) { out[0] = 1.0f; return 1; }\n\
                      int helper(int x) { return x; }\n";
        assert_eq!(definitions(source), ["witness_dot"]);
    }

    #[test]
    fn a_well_formed_probe_passes_every_judgement() {
        let source = "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n";
        let contract = probe(&PROVIDES, source);
        assert_eq!(contract.check_p0_set(), Ok(()));
        assert_eq!(contract.check_signatures(), Ok(()));
        assert_eq!(contract.check_witness_entries(), Ok(()));
        assert_eq!(contract.check_evidence(), Ok(()));
    }

    #[test]
    fn a_signature_the_archive_does_not_claim_is_a_defect() {
        let fragments = [("load", "/* cutile-sign: load/vectorized */\n$0 = $1;")];
        let defect = with_fragments(&fragments, &PROVIDES, "").check_signatures();
        assert!(defect.as_ref().is_err_and(|reason| reason.contains("没有对应供给")), "{defect:?}");
    }

    #[test]
    fn an_unclaimed_witness_is_a_defect() {
        let source = "int witness_load(float *out) { out[0] = 1.0f; return 1; }\n\
                      int witness_orphan(float *out) { out[0] = 1.0f; return 1; }\n";
        let defect = probe(&PROVIDES, source).check_witness_entries();
        assert!(defect.as_ref().is_err_and(|reason| reason.contains("没有任何供给认领")), "{defect:?}");
    }

    #[test]
    fn a_provision_without_any_evidence_is_a_defect() {
        let bare = [
            PROVIDES[0],
            Provision {
                prim: "mma",
                impl_name: "cube",
                witness: None,
            },
        ];
        let defect = probe(&bare, "").check_evidence();
        assert!(defect.as_ref().is_err_and(|reason| reason.contains("无凭无据")), "{defect:?}");
    }

    #[test]
    fn a_fragment_set_that_overruns_p0_is_a_defect() {
        let fragments = [("load", SIGNED), ("load", SIGNED)];
        let defect = with_fragments(&fragments, &PROVIDES, "").check_p0_set();
        assert!(defect.as_ref().is_err_and(|reason| reason.contains("抢同一个原语")), "{defect:?}");
    }

    #[test]
    fn a_host_witness_is_compiled_and_compared() {
        let source = "int witness_ok(float *out) { out[0] = 2.0f; return 1; }\n\
                      int witness_off(float *out) { out[0] = 2.5f; return 1; }\n";
        let calls = [
            Call { func: "witness_ok", expect: &[2.0] },
            Call { func: "witness_off", expect: &[2.0] },
        ];
        let Ok(readings) = run(source, "c", &calls) else {
            assert!(crate::gating::skip("本机没有可用的 C 编译器"));
            return;
        };
        assert_eq!(readings[0].count, 1);
        assert_eq!(readings[0].values, [2.0]);
        assert!(readings[0].max_rel_err <= 1e-6, "{:?}", readings[0]);
        assert!(readings[1].max_rel_err > 1e-3, "{:?}", readings[1]);
    }

    #[test]
    fn a_count_mismatch_is_not_a_near_miss() {
        let source = "int witness_short(float *out) { out[0] = 1.0f; return 1; }\n";
        let calls = [Call { func: "witness_short", expect: &[1.0, 1.0] }];
        let Ok(readings) = run(source, "c", &calls) else {
            assert!(crate::gating::skip("本机没有可用的 C 编译器"));
            return;
        };
        assert_eq!(readings[0].count, 1);
        assert!(readings[0].max_rel_err.is_infinite());
    }

    #[test]
    fn a_witness_the_host_rejects_is_an_error() {
        let source = "int witness_broken(float *out) { out[0] = ; return 1; }\n";
        let calls = [Call { func: "witness_broken", expect: &[0.0] }];
        let Err(error) = run(source, "c", &calls) else {
            panic!("语法错误的见证必须编译失败");
        };
        assert!(error.contains("拒绝了见证源码"), "{error}");
    }
}
