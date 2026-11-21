//! 由 criterion 估读出的本地分析件：几何 / 上界 / 模型记录 / 吞吐与 roofline。

use super::*;

/// 带宽侧 S++ 线：实测吞吐相对本机实测搬运上界的达成率（%）不低于这个数，
/// 才算"顶到带宽上界"（`docs/PROMPT-S++-AUDIT.md` §8）。
pub(crate) const BANDWIDTH_PERCENT: f64 = 85.0;

/// 算力侧 S++ 线：实测相对**每内核声明上界**（登记基线）的达成率（%）不低于这个数。
///
/// 分母用登记基线而不是理论峰值：§8 明令禁止拿理论峰值冒充 roofline，登记表里那一行
/// 就是这台机器上这个内核已知能跑到的上界。
pub(crate) const COMPUTE_PERCENT: f64 = 90.0;

/// criterion 的 mean 点估计（`ns/iter`）。
///
/// 只依赖 `estimates.json` 里两个键名（`mean` 与 `point_estimate`）：criterion 的统计口径演进过，
/// 但这两个名字是它对外报告的核心。读不出来就明说"读不出"，不猜一个数。
pub(crate) fn mean_ns(text: &str) -> Option<f64> {
    let mean = text.find("\"mean\"")?;
    let key = text[mean..].find("\"point_estimate\"")?;
    let tail = &text[mean + key + "\"point_estimate\"".len()..];
    let start = tail.find(|c: char| c == '-' || c.is_ascii_digit())?;
    let number: String = tail[start..]
        .chars()
        .take_while(|c| {
            *c == '-' || *c == '.' || c.is_ascii_digit() || *c == 'e' || *c == '+' || *c == 'E'
        })
        .collect();
    number.parse().ok()
}

/// criterion 落盘名的长度上限，与 `criterion::report::MAX_DIRECTORY_NAME_LEN` 同值。
const NAME_LIMIT: usize = 64;

/// 分组名 / 用例名在磁盘上的形态：criterion 的 `report::make_filename_safe` 把
/// `? " / \ * < > : | ^` 逐个换成 `_`，再在 64 字节的字符边界处截断。
///
/// 这一层必须镜像，不能省：分组名带 `/`（如 `weights/load`）落盘成 `weights_load`，
/// 判定若按原名拼路径，就永远读不到测量、永远登记跳过——门禁假装绿了，其实那条基准
/// 一次都没被判过。
pub(crate) fn on_disk(name: &str) -> String {
    let mut safe: String = name
        .chars()
        .map(|c| match c {
            '?' | '"' | '/' | '\\' | '*' | '<' | '>' | ':' | '|' | '^' => '_',
            other => other,
        })
        .collect();
    let mut cut = safe.len().min(NAME_LIMIT);
    while !safe.is_char_boundary(cut) {
        cut -= 1;
    }
    safe.truncate(cut);
    safe
}

/// criterion 的输出目录：`target/criterion/<group>/<benchmark>/new/estimates.json`。
///
/// 目标目录在**工作区根**（`crates/benchmarks/../..`），不在本成员的目录下：criterion 由
/// `cargo bench` 调用，写的是调用方的 `target/`。
pub(crate) fn estimates(manifest: &Path, group: &str, benchmark: &str) -> PathBuf {
    manifest
        .join("../..")
        .join("target/criterion")
        .join(group)
        .join(on_disk(benchmark))
        .join("new/estimates.json")
}

/// 实测值；还没跑过基准就登记跳过并返回 `None`。
pub(crate) fn observed_ns(path: &Path) -> Option<f64> {
    let Ok(text) = std::fs::read_to_string(path) else {
        testkit::skip(format!(
            "benchmarks: 还没有基准测量（{}），先跑 cargo bench -p llm-benchmarks",
            path.display()
        ));
        return None;
    };
    Some(mean_ns(&text).unwrap_or_else(|| {
        panic!(
            "读不出 {} 的 mean.point_estimate：criterion 的输出格式变了，判定口径要跟着改",
            path.display()
        )
    }))
}

/// 搬 `bytes` 字节在该档上界下至少需要多少 ns。
pub(crate) fn bytes_in_ns(bytes: u64, ceiling: &Ceiling) -> f64 {
    bytes as f64 * ceiling.ns / ceiling.bytes as f64
}

/// 某档实测上界的 MB/s。
pub(crate) fn mbps(ceiling: &Ceiling) -> f64 {
    ceiling.bytes as f64 / ceiling.ns * NS_PER_SECOND / BYTES_PER_MB
}

/// 字节吞吐：bytes/iter ÷ ns/iter，十进制 GB/s（1 GB = 1e9 B）。
pub(crate) fn gb_per_s(bytes: u64, ns: f64) -> f64 {
    if ns > 0.0 {
        bytes as f64 / ns * NS_PER_SECOND / BYTES_PER_GB
    } else {
        0.0
    }
}

/// token 吞吐：tokens/iter ÷ ns/iter，tokens/s。
pub(crate) fn tokens_per_s(tokens: u64, ns: f64) -> f64 {
    if ns > 0.0 {
        tokens as f64 / ns * NS_PER_SECOND
    } else {
        0.0
    }
}

/// 一条基准的 roofline 口径：下界流量、所落的存储档位、该档实测上界、达成率。
///
/// 这是 Report 侧的本地分析件，不进 [`BenchRecord`]：达成率、瓶颈侧、MB/s、flops、档位都由
/// 算子语义与本机上界在此派生。
pub(crate) struct Roofline {
    /// 下界读入字节（`nn::Bounds`，由算子语义推出，不是手写常数）。
    pub(crate) read: u64,
    /// 下界写出字节。
    pub(crate) write: u64,
    /// 下界浮点运算次数。
    pub(crate) flops: f64,
    /// 工作集落在哪一档：`l1` / `l2` / `dram`。
    pub(crate) tier: &'static str,
    /// 该档实测读上界（MB/s）。
    pub(crate) read_ceiling_mbps: f64,
    /// 该档实测写上界（MB/s）。
    pub(crate) write_ceiling_mbps: f64,
    /// roofline 上界时间（ns）：两个方向各自换算后取慢的那一侧。
    pub(crate) ns: f64,
    /// 慢的那一侧：`read` / `write`。
    pub(crate) side: &'static str,
    /// 带宽侧达成率（%）：roofline 上界时间 / 实测时间。超过 100 说明上界登记过期，
    /// 不是"超越硬件"。
    pub(crate) achieved_percent: f64,
    /// 声明上界（ns）：登记基线，即这个内核在本机上已知能跑到的上界。
    pub(crate) declared_ns: f64,
    /// 算力侧达成率（%）：声明上界 / 实测时间。
    pub(crate) ceiling_percent: f64,
    /// 实测算力（MFLOP/s）：带宽达成率低时，这个数才说得清"代价花在算上"。
    pub(crate) mflops_per_s: f64,
}

impl Roofline {
    /// 由算子语义与本机上界推出一条基准的 roofline 口径。
    ///
    /// 工作集决定档位：16 KiB 量级的行宽算子按 `l1` 判，几百 KiB 的 `gemm` / `attention` 按
    /// `l2` 判，`gemv` 那 64 MiB 权重按 `dram` 判。档位定错会把"L1 里的往返"拿去比主存带宽，
    /// 达成率就没有意义。
    pub(crate) fn of(
        host: &HostProfile,
        scenario: &Scenario,
        declared_ns: f64,
        observed_ns: f64,
    ) -> Self {
        let bounds = nn::Bounds::of_context(scenario.op, scenario.bounds.0, scenario.bounds.1)
            .unwrap_or_else(|error| {
                panic!(
                    "{}：算子 `{}` 推不出下界（{}），roofline 就没有分子",
                    scenario.scenario, scenario.op, error
                )
            });
        let tier = tier(bounds.bytes());
        let read = ceiling_for(host.host, tier, "read").unwrap_or_else(|| {
            panic!(
                "主机 `{}` 没有登记 {tier} 档的读上界：\
                 先跑 cargo bench -p llm-benchmarks --bench mem_stream",
                host.host
            )
        });
        let write = ceiling_for(host.host, tier, "write").unwrap_or_else(|| {
            panic!(
                "主机 `{}` 没有登记 {tier} 档的写上界：\
                 先跑 cargo bench -p llm-benchmarks --bench mem_stream",
                host.host
            )
        });

        let read_ns = bytes_in_ns(bounds.read(), read);
        let write_ns = bytes_in_ns(bounds.write(), write);
        let (ns, side) = if read_ns >= write_ns {
            (read_ns, "read")
        } else {
            (write_ns, "write")
        };

        Self {
            read: bounds.read(),
            write: bounds.write(),
            flops: bounds.flops(),
            tier,
            read_ceiling_mbps: mbps(read),
            write_ceiling_mbps: mbps(write),
            ns,
            side,
            achieved_percent: if observed_ns > 0.0 {
                ns / observed_ns * 100.0
            } else {
                0.0
            },
            declared_ns,
            ceiling_percent: if observed_ns > 0.0 {
                declared_ns / observed_ns * 100.0
            } else {
                0.0
            },
            mflops_per_s: if observed_ns > 0.0 {
                bounds.flops() / observed_ns * 1e3
            } else {
                0.0
            },
        }
    }

    /// 是否已经顶到带宽上界：门槛与 `docs/lld.md` E7 的带宽达成率同一口径。
    pub(crate) fn bandwidth_bound(&self) -> bool {
        self.achieved_percent >= BANDWIDTH_PERCENT
    }

    /// 这一条基准该比哪一侧的达成率、门槛是多少：顶到带宽上界就比带宽，否则比声明上界。
    pub(crate) fn s_plus_line(&self) -> f64 {
        if self.bandwidth_bound() {
            BANDWIDTH_PERCENT
        } else {
            COMPUTE_PERCENT
        }
    }

    /// S++ 达成率（%）：与 [`Roofline::s_plus_line`] 同一侧的分母。
    pub(crate) fn s_plus_percent(&self) -> f64 {
        if self.bandwidth_bound() {
            self.achieved_percent
        } else {
            self.ceiling_percent
        }
    }

    /// 是否达到 §8 的 S++ 线。
    pub(crate) fn meets_s_plus(&self) -> bool {
        self.s_plus_percent() >= self.s_plus_line()
    }

    /// 瓶颈分类：只有顶到带宽上界才写 `bandwidth`，否则明确写 `not-bandwidth`。
    ///
    /// 不猜"是算力还是延迟"：那需要 prof 采样，达成率只够否定"带宽主导"。
    pub(crate) fn bottleneck(&self) -> &'static str {
        if self.bandwidth_bound() {
            "bandwidth"
        } else {
            "not-bandwidth"
        }
    }
}
