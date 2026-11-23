//! 设备片段层的契约：一台设备的 P0 片段、见证入口与档案供给必须互相闭合。

use std::path::Path;

/// P0 闭集：`crates/prof/ops/*.k` 真的会 `@` 到的八个原语，模板按名字取片段。
///
/// 这是**算子模板**的事实，不是某台设备的事实，因此六台设备共用一份。设备模块自带
/// 一份的写法会让 [`Contract::check_p0_set`] 的可信度落到被检查方手里：各设备缩一缩自己
/// 的闭集，门禁就绿了。闭集留在这里，设备只能选择覆盖它或暴露缺口。
pub const P0: [&str; 8] = [
    "cvt",
    "dot",
    "load",
    "load_async",
    "mma",
    "quant",
    "reduce",
    "store",
];

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
    /// 写回尾块的文件名主干（`prof::sign` 的那一份），闭集要拿它算文件名。
    pub epilogue_stem: &'a str,
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
            let signed = sign(body)
                .map_err(|reason| format!("`{}` 的 `{prim}.{}`：{reason}", self.id, self.ext))?;
            self.check_signed(prim, signed)?;
        }
        Ok(())
    }

    /// 一条签名：文件名与署名同一个词，然后交给档案对账。
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

    /// 署名在档案里必须有供给，那条供给必须有见证，且见证入口与生成表里的是同一个。
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

    /// 见证文件里的每个 `int witness_*(float *out)` 都必须被某条供给认领，每个入口都必须在文件里
    /// 定义一次：多出来的是死代码，少一条是镜像里点名点空。
    ///
    /// # Errors
    ///
    /// 入口缺席、重复，或有见证函数无人认领时返回原因。
    pub fn check_witness_entries(&self) -> Result<(), String> {
        let defined = crate::evidence::definitions(self.source);
        let prims = self.prims.iter().map(|(prim, func)| (*prim, *func));
        self.check_one_each(&defined, prims, "原语", "一条签名一个函数")?;
        self.check_unclaimed(&defined)
    }

    /// 每条命名了算法的条目在见证文件里必须正好有一段函数：少一段是漏说了，多一段是两份真相。
    fn check_one_each<'n>(
        &self,
        defined: &[&'n str],
        entries: impl Iterator<Item = (&'n str, &'n str)>,
        kind: &str,
        rule: &str,
    ) -> Result<(), String> {
        for (name, func) in entries {
            let count = defined.iter().filter(|seen| **seen == func).count();
            if count != 1 {
                return Err(format!(
                    "`{}` 的{kind} `{name}` 入口 `{func}` 在见证文件里出现了 {count} 次：{rule}",
                    self.id
                ));
            }
        }
        Ok(())
    }

    /// 反向的漏：见证文件里每个函数都得有档案认领，否则它算的东西归属不到任何一条。
    fn check_unclaimed(&self, defined: &[&str]) -> Result<(), String> {
        for name in defined {
            if !self.claimed(name) {
                return Err(format!(
                    "`{}` 的见证函数 `{name}` 没有任何供给认领：它执行的算法归属不到一条签名",
                    self.id
                ));
            }
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
            let fragmented = self.fragments.iter().any(|(_, body)| {
                sign(body).is_ok_and(|sign| sign == (provide.prim, provide.impl_name))
            });
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

    /// 目录是闭集：只有 P0 的八个 `<prim>.<ext>` 与一个 `witness.<ext>`。`prof::codegen` 的扫描器
    /// 只认这两种名字，多出来的设备源文件会在构建期被拒——这里提前说清是哪个文件。
    ///
    /// `.rs` 不算内容：片段目录就是 crate 的 `src/`，`lib.rs` 与它住在一起。
    ///
    /// # Errors
    ///
    /// 目录不可读，或目录里出现了闭集之外的名字时返回原因。
    pub fn check_closed_directory(&self) -> Result<(), String> {
        let found = self.found_names()?;
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

    /// 片段目录里的全部文件名（`.rs` 除外），排序后返回。
    fn found_names(&self) -> Result<Vec<String>, String> {
        let entries = std::fs::read_dir(self.dir)
            .map_err(|error| format!("`{}` 的片段目录不可读：{error}", self.id))?;
        let mut found = Vec::new();
        for entry in entries {
            let entry =
                entry.map_err(|error| format!("`{}` 的片段目录项不可读：{error}", self.id))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".rs") {
                found.push(name);
            }
        }
        found.sort();
        Ok(found)
    }

    /// 档案里认领 `prim/impl_name` 的那条供给。
    fn find(&self, prim: &str, impl_name: &str) -> Option<&'a Provision<'a>> {
        let provides: &'a [Provision<'a>] = self.provides;
        provides
            .iter()
            .find(|provide| provide.prim == prim && provide.impl_name == impl_name)
    }

    /// 生成表里 `prim` 的见证入口名。
    fn entry(&self, prim: &str) -> Result<&'a str, String> {
        let prims: &'a [(&'a str, &'a str)] = self.prims;
        let found = prims.iter().find(|(name, _)| *name == prim);
        match found {
            Some((_, func)) => Ok(func),
            None => Err(format!(
                "`{}` 的 `{prim}` 有片段却没有入口：镜像里查不到它的见证",
                self.id
            )),
        }
    }

    /// 某条见证函数是否被档案里的一条供给认领。
    fn claimed(&self, name: &str) -> bool {
        self.provides
            .iter()
            .any(|provide| provide.witness.is_some_and(|witness| witness.func == name))
    }

    /// 闭集里的全部名字：P0 的 `<prim>.<ext>`、一个 `witness.<ext>`、一份 `epilogue.<ext>`。
    fn expected_names(&self) -> Vec<String> {
        let mut want: Vec<String> = self
            .p0
            .iter()
            .map(|prim| format!("{prim}.{}", self.ext))
            .collect();
        want.push(format!("witness.{}", self.ext));
        // 一份写回尾块文本承载六种形式，文件名由 `prof::sign` 定，不随原语扩张。
        want.push(format!("{}.{}", self.epilogue_stem, self.ext));
        // 档案与片段语料住在同一目录（`src/<vendor>/<vendor>.toml`），它是目录闭集的一格。
        let vendor = Path::new(self.dir)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(self.id);
        want.push(format!("{vendor}.toml"));
        want
    }
}

/// 取片段首行的签名声明：`/* cutile-sign: <prim>/<impl> */`。
///
/// # Errors
///
/// 片段没有签名声明、声明没有闭合，或声明不是两段时返回原因。
fn sign(body: &str) -> Result<(&str, &str), String> {
    let Some((_, rest)) = body.split_once("cutile-sign:") else {
        return Err(String::from(
            "片段缺签名声明 `/* cutile-sign: <prim>/<impl> */`",
        ));
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
#[path = "../tests/device_tests.rs"]
mod device_tests;
