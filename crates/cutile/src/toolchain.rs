//! 工具链探测与身份：这台机器上哪台编译器/驱动可用、自报什么版本。
//!
//! 档案里的 `[meta].tool_ver` 是人写的登记值；代码生成的差异由**真正调用的那个编译器**决定，
//! 产物身份与报告一律用实测值。探测结果按进程级快照缓存：进程启动后新装的编译器要重启才看得见。

use std::fmt;
use std::process::Command;
use std::sync::OnceLock;

#[cfg(test)]
use testkit;

/// 一条被真正解析出来的工具链。产物身份与报告用它，而不是档案里的登记版本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toolchain {
    /// 命令名（`cc` / `clang` / `gcc` / `nvcc` / `xcrun metal`）。
    pub command: String,
    /// `--version` 里解析出的第一段版本号；解析不出时为 `None`（不猜）。
    pub version: Option<String>,
    /// `--version` 的首个非空行（原文，诊断与台账里要能对得上）。
    pub banner: String,
}

impl Toolchain {
    /// 身份：`命令@实测版本`，进产物身份与报告；解析不出版本时用 `-`，不假装知道版本。
    #[must_use]
    pub fn id(&self) -> String {
        format!(
            "{}@{}",
            self.command,
            self.version.as_deref().unwrap_or("-")
        )
    }

    /// 档案登记的版本是否与实测相容：只比**主版本号**（`clang-17` 的族名不算差异），不改判定等级。
    ///
    /// `"*"` 是显式的"任意版本"：`cpu` 档案登记的是 `cc`，而 `cc` 在 macOS 上是 clang、在
    /// 多数 Linux 上是 gcc，同一个可移植档案不可能写死一个主版本号。写死的结果是每台机器都报
    /// "不一致"——一条永远为真的告警等于没有告警。设备档案（`nvcc` / `mcc` / `xcrun`）仍记实测值。
    #[must_use]
    pub fn agrees_with(&self, declared: &str) -> bool {
        if declared.trim() == "*" {
            return true;
        }
        let Some(actual) = self.version.as_deref() else {
            return false;
        };
        let actual_major = actual.split('.').next().unwrap_or(actual);
        // 登记值可能带工具族前缀（`clang-17`）：只看最后一段。
        let declared = declared.trim().rsplit(['-', ' ', '_']).next().unwrap_or("");
        let declared_major = declared.split('.').next().unwrap_or("");
        !declared_major.is_empty() && declared_major == actual_major
    }
}

impl fmt::Display for Toolchain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.version {
            Some(version) => write!(formatter, "{} {version}", self.command),
            None => write!(formatter, "{}（版本未知：{}）", self.command, self.banner),
        }
    }
}

/// 从 `--version` 的输出里取第一段版本号：先看含 `version` / `release` 的行，再退回任意行；每行
/// 取第一段带点的"数字与点"——`-O2`、`sm_100` 里的孤立数字不是版本号。
pub fn parse_version(text: &str) -> Option<String> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let keyed = lines.iter().copied().filter(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("version") || lower.contains("release")
    });
    keyed.chain(lines.iter().copied()).find_map(|line| {
        line.split(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
            .map(|token| token.trim_end_matches('.'))
            .find(|token| token.contains('.'))
            .map(str::to_string)
    })
}

/// 进程内缓存的探测结果：同一台机器的工具链不会变，反复 `--version` 起子进程是纯浪费。
static C_TOOLCHAIN: OnceLock<Option<Toolchain>> = OnceLock::new();

/// 探测可用的 C 编译器（`cc` → `clang` → `gcc`）并读出它自报的版本；结果当作进程级快照缓存。
pub fn detect() -> Option<Toolchain> {
    C_TOOLCHAIN
        .get_or_init(|| ["cc", "clang", "gcc"].into_iter().find_map(probe))
        .clone()
}

/// 按**档案声明的工具**探测：除便携 C 之外，产物只能由档案里写的那个编译器编（`nvcc` / `mcc` /
/// `ccec` / `xcrun metal`），宿主的 `cc` 不认识 `--cuda-gpu-arch`，拿它去编只会把"这台设备没有
/// 工具链"报成一条 C 语法错误。声明不可用时返回 `None`（如实"没有"），不退回合主编译器——那会把
/// 缺工具链伪装成编译失败。
#[must_use]
pub fn for_profile(profile: &crate::prof::Profile) -> Option<Toolchain> {
    match profile.meta.emit {
        // 便携 C 的登记值是 `cc` 且版本记作 `*`：探测顺序（cc → clang → gcc）本身就是它的语义。
        crate::prof::Emit::C => detect(),
        _ => probe_declared(&profile.meta.tool),
    }
}

/// 按 `[meta].tool` 的**声明**探测：声明可以带固定前缀参数（`xcrun metal`），这仍是档案写的那条
/// 命令，不是退回合主编译器。命令名与参数以空白分隔，身份里保留声明原文，报告才能与档案对账。
#[must_use]
pub fn probe_declared(declared: &str) -> Option<Toolchain> {
    let mut parts = declared.split_whitespace();
    let program = parts.next()?;
    let leading: Vec<&str> = parts.collect();
    let mut tool = probe_with(program, &leading)?;
    tool.command = declared.trim().to_string();
    Some(tool)
}

/// 本机的 clang：先问 `clang`，再退到 `cc`（macOS 与多数发行版的 `cc` 就是 clang）。判据是
/// "这个程序是不是 clang"：`-x cuda` / `-x cl` 是 clang 的前端，gcc 没有。
#[must_use]
pub fn clang() -> Option<Toolchain> {
    probe("clang").or_else(|| detect().filter(|toolchain| toolchain.banner.contains("clang")))
}

/// 探测指定命令：跑一次 `--version`，成功才返回（版本解析不出也算可用）。
pub fn probe(command: &str) -> Option<Toolchain> {
    probe_with(command, &[])
}

/// 探测的实作：`program` 之后先放档案声明的固定前缀参数，再问版本。
fn probe_with(program: &str, leading: &[&str]) -> Option<Toolchain> {
    let output = Command::new(program)
        .args(leading)
        .arg("--version")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let text = if stdout.trim().is_empty() {
        String::from_utf8_lossy(&output.stderr).to_string()
    } else {
        stdout
    };
    let banner = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?
        .to_string();
    Some(Toolchain {
        command: program.to_string(),
        version: parse_version(&text),
        banner,
    })
}

/// 测试门控：没有主机 C 编译器时，编译/执行类用例在 CI 上不该伪装成通过；返回 `true` 表示跳过，
/// 原因经 [`testkit::skip`] 登记计数（标准测试框架没有正式 skip 信号）。
#[cfg(test)]
pub(crate) fn skip_without_host_compiler() -> bool {
    if detect().is_none() {
        testkit::skip("cutile: 未探测到主机 C 编译器（cc/clang/gcc），编译/执行用例未运行")
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_portable_profile_may_declare_any_version() {
        let tool = Toolchain {
            command: "cc".to_string(),
            version: Some("11.4.0".to_string()),
            banner: "cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0".to_string(),
        };
        assert!(tool.agrees_with("*"), "便携档案的 `*` 不能被报成不一致");
        assert!(tool.agrees_with("cc-11"), "同主版本相容");
        assert!(!tool.agrees_with("clang-17"), "跨主版本仍要说出来");
    }

    #[test]
    fn versions_are_read_from_what_the_compilers_actually_print() {
        // 三家真实输出：找的是同一段东西，而不是各自的字段名。
        assert_eq!(
            parse_version(
                "Apple clang version 21.0.0 (clang-2100.4.1.1)\nTarget: arm64-apple-darwin"
            ),
            Some("21.0.0".to_string())
        );
        assert_eq!(
            parse_version("gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0\nCopyright (C) 2023"),
            Some("13.2.0".to_string())
        );
        assert_eq!(
            parse_version(
                "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.3, V13.3.44"
            ),
            Some("13.3".to_string())
        );
        // 没有版本串时不猜：`-` 会出现在身份里，报告同时打印原文。
        assert_eq!(parse_version("cc (some vendor toolchain)\n"), None);
        // 孤立数字不是版本号（`-O2`、`sm_100` 常见于参数与设备名）。
        assert_eq!(parse_version("flags: -O2 sm_100"), None);
    }

    #[test]
    fn the_identity_carries_the_measured_version() {
        let tool = Toolchain {
            command: "cc".to_string(),
            version: Some("21.0.0".to_string()),
            banner: "Apple clang version 21.0.0".to_string(),
        };
        assert_eq!(tool.id(), "cc@21.0.0");
        assert!(tool.agrees_with("clang-21"), "登记值带族名也算一致");
        assert!(tool.agrees_with("21.4"), "只比主版本号");
        assert!(!tool.agrees_with("clang-17"), "换主版本就是另一条工具链");
        assert!(!tool.agrees_with(""), "空登记不能算一致");

        let unknown = Toolchain {
            command: "cc".to_string(),
            version: None,
            banner: "cc (some vendor toolchain)".to_string(),
        };
        assert_eq!(unknown.id(), "cc@-");
        assert!(!unknown.agrees_with("clang-21"), "不知道版本就不能声称一致");
    }

    #[cfg(unix)]
    #[test]
    fn a_probe_reads_the_version_of_the_command_it_was_given() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("cutile-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("临时目录");
        let fake = root.join("fake-cc.sh");
        std::fs::write(&fake, "#!/bin/sh\necho 'Fake clang version 99.1.2'\n").expect("写脚本");
        let mut perms = std::fs::metadata(&fake).expect("脚本").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&fake, perms).expect("可执行");
        let tool = probe(&fake.to_string_lossy()).expect("脚本可探测");
        assert_eq!(tool.version.as_deref(), Some("99.1.2"));
        assert_eq!(tool.id(), format!("{}@99.1.2", fake.to_string_lossy()));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 档案里可以写带前缀参数的命令（`xcrun metal`）：探测要把参数一起交给程序，身份保留声明原文。
    #[cfg(unix)]
    #[test]
    fn a_declared_command_may_carry_prefix_arguments() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("cutile-declared-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("临时目录");
        let fake = root.join("fake-xcrun.sh");
        // 只有前缀参数真的被传进来才自报版本：探测丢掉参数时这里就只剩 `None`。
        std::fs::write(
            &fake,
            "#!/bin/sh\nif [ \"$1\" = metal ]; then echo 'Apple metal version 32023.922'; fi\n",
        )
        .expect("写脚本");
        let mut perms = std::fs::metadata(&fake).expect("脚本").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&fake, perms).expect("可执行");
        let declared = format!("{} metal", fake.display());
        let tool = probe_declared(&declared).expect("带前缀参数的命令可探测");
        assert_eq!(tool.version.as_deref(), Some("32023.922"));
        assert_eq!(
            tool.id(),
            format!("{declared}@32023.922"),
            "身份保留声明原文"
        );
        assert!(
            probe_declared("cutile-no-such-compiler-0x").is_none(),
            "装不上的声明如实说没有，不退回合主编译器"
        );
        assert!(probe_declared("   ").is_none(), "空声明不是命令");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 设备档案只能探测它自己声明的编译器：本机没装就不能报成宿主的 `cc`。
    #[test]
    fn a_device_profile_probes_the_tool_it_declares() {
        let ascend = crate::prof::Profile::parse(prof::embedded("ascend").expect("ascend 档案"))
            .expect("档案可解析");
        match for_profile(&ascend) {
            Some(tool) => assert_eq!(tool.command, "ccec", "只认档案声明的编译器"),
            None => assert!(
                probe_declared("ccec").is_none(),
                "`ccec` 探得到就不能报成没有"
            ),
        }
    }

    #[test]
    fn missing_compiler_is_not_an_error_by_itself() {
        // 只是探测：返回 None 也算合法结果，调用方负责报"缺设备即失败"。
        let _ = detect();
    }

    #[test]
    fn parse_version_only_returns_dotted_tokens_and_never_panics() {
        let mut rng = testkit::Rng::new(0x7E57);
        for _ in 0..64 {
            let line = format!(
                "tool {} release {}.{} build {}",
                rng.below(999),
                rng.below(999),
                rng.below(999),
                rng.below(99999)
            );
            if let Some(version) = parse_version(&line) {
                assert!(version.contains('.'), "版本串必须带点：{version}");
            }
        }
    }
}
