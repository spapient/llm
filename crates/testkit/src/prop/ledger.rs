//! 性质测试台账：每个工作区成员要么在 [`COVERED`] 里，要么有一条**具体**理由说明它为什么没有。
//!
//! 台账存在的理由：纪律门只回答了"用不用共享装置"，没回答"该不该有性质测试"。
//! 缺了后半个问题，`crate` 从有性质退化成没有性质的唯一症状是**没人发现**——
//! 那正是手抄配置曾经造成过的静默变弱。
//!
//! 三条硬约束由 [`crate::prop_discipline`] 的双向覆盖门强制：
//! 台账与源码扫描必须逐成员一致、每条理由必须具体且互不相同、已登记缺口的数量有上限
//! （否则"以后补"会变成永远不补的垃圾桶）。

/// 一条"为什么这个成员不需要性质测试"的分类理由。
pub struct Absence {
    /// 适用该理由的成员。
    pub members: &'static [&'static str],
    /// 理由本身：必须指向真实存在、可核查的替代覆盖。
    pub reason: &'static str,
}

/// 有性质测试套件的成员——与源码里实际出现的性质测试块一一对应。
pub const COVERED: &[&str] = &[
    "algorithm",
    "conv",
    "gemm",
    "nn",
    "rng",
    "shard",
    "tensor",
    "testkit",
    "train",
    "wire",
];

/// 已登记、尚未补上的缺口；每条都带一句"补什么、为什么还没补"。
///
/// 这是**唯一**允许出现"还没写"的名单，长度由 [`PLANNED_MAX`] 封顶：
/// 一旦它开始增长，就说明有人把"该有而没写"记成了"不需要"，那道门随即便失效。
pub const PLANNED: &[(&str, &str)] = &[
    (
        "autograd",
        "该断言\"每个算子恰好被访问一次\"与\"加法梯度是求和\"；但同一批公式的生产路径在 \
         `train`（刚补上梯度对拍），先坐实那边再回填这里的遍历纪律。",
    ),
    (
        "optim",
        "该复制 `train::optimizer` 的单步闭式解；动手前须确认 `optim` 与 `train` 不是同一份\
         实现的两次抄写，否则性质会在两处各写一遍、各漂一半。",
    ),
];

/// 缺口上限：只许减，不许随手加。
pub const PLANNED_MAX: usize = 2;

/// 有明确替代覆盖、因而**不该**再写性质测试的成员，按理由归类。
pub const ABSENT: &[Absence] = &[
    Absence {
        members: &["frag"],
        reason: "片段层：六台设备的片段正文与见证是**要被设备编译器读的代码**，Rust 侧只剩生成器给出的\
                 四张表，没有算法可断言。真值是 `crates/frag/tests/fragments.rs` 的逐条对账与见证执行，\
                 以及 `cutile::gate` 的语言前端解析；写成性质只剩\"结构等于结构\"，比编译器已保证的弱。",
    },
    Absence {
        members: &["cutile", "device"],
        reason: "发射路径：结论要么要真设备（`testkit::gating` 的跳过语义，本机恒定 skip 等于空转），\
                 要么是 PTX 文本形状——而它的真值是实测吞吐，断言文本只会锁死实现。",
    },
    Absence {
        members: &[
            "annotation",
            "diagnostic",
            "dtype",
            "op",
            "prim",
            "spec",
            "storage",
        ],
        reason: "词汇表：枚举、错误码与注册表行。唯一的不变量是\"每个取值都被处理过一次\"，\
                 而它由穷举 `match` 在编译期强制（加一个变体不处理就编译不过）——\
                 性质测试在这里提供的不是覆盖，是重复。",
    },
    Absence {
        members: &["benchmarks", "cli", "tools"],
        reason: "端到端工具：真值是 stdout、退出码与版面，已有逐字节快照用例。\
                 随机化输入只会变成\"输出等于输出\"，这条恒真、抓不到东西。",
    },
    Absence {
        members: &["engine", "prof", "serve"],
        reason: "调度与时序：要可控时钟与线程排练才能断言，现有场景用例已用假时钟钉死时间线；\
                 随机化调度只会让失败不可复现——在这类代码里不可复现等于没有失败。",
    },
    Absence {
        members: &["formats", "fuzz", "snapshot", "tokenizer", "weights"],
        reason: "任意字节鲁棒性由 `crates/fuzz` 的确定性靶覆盖——同一句话（不 panic、只返回 \
                 `Invalid`），样本量级远超几百个用例；这五个成员还各有真实样本的往返用例钉住正向路径。",
    },
];
