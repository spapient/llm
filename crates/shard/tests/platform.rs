use super::*;

/// 一台可以随便摆布证词的宿主：测试只摆事实，不依赖本机装了什么。
struct Fake {
    arch: Arch,
    device: bool,
    toolchain: Option<&'static str>,
}

impl Fake {
    const fn bare(arch: Arch) -> Self {
        Self {
            arch,
            device: false,
            toolchain: None,
        }
    }
}

impl Host for Fake {
    fn arch(&self) -> Arch {
        self.arch
    }

    fn device(&self, _row: Row) -> bool {
        self.device
    }

    fn toolchain(&self, _row: Row) -> Option<String> {
        self.toolchain.map(str::to_string)
    }
}

testkit::test_cases! {
fn specs_line_up_with_the_rows() {
    assert_eq!(Row::ALL.len(), SPECS.len(), "六行不缺不多");
    for (idx, row) in Row::ALL.iter().enumerate() {
        assert_eq!(*row as usize, idx, "判别式必须等于表下标");
        assert_eq!(
            row.as_str(),
            SPECS[idx].name,
            "{} 的名字与表错位",
            row.as_str()
        );
    }
}

fn the_six_rows_come_in_the_contract_order() {
    let names: Vec<&str> = Row::ALL.iter().map(|row| row.as_str()).collect();
    assert_eq!(names, ["cuda", "hip", "metal", "gpu", "x86", "arm"]);
}

fn gpu_rows_speak_homogeneous_abi_and_cpu_rows_ordered_dot() {
    for &row in Row::ALL {
        let expected = if row.compiler() || row == Row::Gpu {
            Abi::Homogeneous
        } else {
            Abi::OrderedDot
        };
        assert_eq!(row.abi(), expected, "{} 的 ABI 形态不符", row.as_str());
    }
    assert!(!Row::X86.primitive().is_empty(), "原语不能是空话");
}

fn vulkan_and_sycl_map_into_the_gpu_row() {
    for language in ["vulkan", "SYCL", " spirv ", "gpu"] {
        assert_eq!(Row::from_language(language), Some(Row::Gpu), "{language}");
    }
    assert_eq!(Row::from_language("hipcc"), Some(Row::Hip));
    assert_eq!(
        Row::from_language("xcrun metal"),
        None,
        "不猜：整串不是名字"
    );
    assert_eq!(Row::from_language("rocm"), Some(Row::Hip));
}

fn every_row_is_audited_even_when_the_host_has_nothing() {
    let host = Fake::bare(Arch::Other);
    let ledger = audit(&host);
    assert_eq!(ledger.len(), Row::ALL.len(), "缺设备也不许少一行");
    for (row, standing) in Row::ALL.iter().zip(&ledger) {
        assert_eq!(standing.as_str(), "open", "{} 不该凭空覆盖", row.as_str());
        assert_eq!(standing.evidence(), None);
        assert_eq!(
            standing,
            &Standing::Open { why: Why::NoDevice },
            "{} 缺的是设备",
            row.as_str()
        );
    }
}

fn a_cpu_row_is_covered_by_the_architecture_alone() {
    let host = Fake::bare(Arch::Aarch64);
    let arm = standing(Row::Arm, &host);
    assert!(arm.is_covered(), "架构对得上就该覆盖，不看工具链");
    let evidence = arm.evidence().expect("覆盖必带证据");
    assert!(evidence.contains("aarch64"), "{evidence}");
    assert!(evidence.contains(Row::Arm.primitive()), "{evidence}");
    assert_eq!(
        standing(Row::X86, &host),
        Standing::Open { why: Why::NoDevice },
        "aarch64 宿主不覆盖 x86 行"
    );
}

fn a_device_row_needs_both_the_device_and_the_compiler() {
    let device_only = Fake {
        arch: Arch::Aarch64,
        device: true,
        toolchain: None,
    };
    assert_eq!(
        standing(Row::Cuda, &device_only),
        Standing::Open {
            why: Why::NoToolchain
        },
        "有设备没编译器：理由必须是编译器，不是设备"
    );

    let both = Fake {
        arch: Arch::Aarch64,
        device: true,
        toolchain: Some("nvcc@12.6"),
    };
    let covered = standing(Row::Cuda, &both);
    let evidence = covered.evidence().expect("两件都有才覆盖");
    assert!(evidence.contains("nvcc@12.6"), "{evidence}");
    assert!(evidence.contains("CUTLASS/WGMMA"), "{evidence}");
}

fn the_gpu_row_self_certifies_on_the_vendor_primitive() {
    let host = Fake {
        arch: Arch::X86_64,
        device: true,
        toolchain: None,
    };
    let gpu = standing(Row::Gpu, &host);
    assert!(
        gpu.is_covered(),
        "GPU 行的一等原语就是厂商 MMA，不必另要编译器"
    );
    assert_eq!(gpu.evidence(), Some(Row::Gpu.primitive()));
}

fn require_reports_e_plat_instead_of_passing() {
    let host = Fake::bare(Arch::Aarch64);
    let error = require(Row::Cuda, &host).expect_err("缺设备必须开门失败");
    assert_eq!(error.code(), Code::Unsupported, "错误码就是 E_PLAT");
    let text = error.to_string();
    assert!(text.contains("cuda"), "{text}");
    assert!(text.contains("OPEN"), "{text}");
    assert!(text.contains(Why::NoDevice.sentence()), "{text}");
    require(Row::Arm, &host).expect("架构覆盖的行必须放行");
}

fn stable_short_names() {
    assert_eq!(
        Standing::Covered {
            evidence: String::new()
        }
        .as_str(),
        "covered"
    );
    assert_eq!(Standing::Open { why: Why::NoDevice }.as_str(), "open");
    assert_eq!(Why::NoDevice.as_str(), "no-device");
    assert_eq!(Why::NoToolchain.as_str(), "no-toolchain");
    assert_eq!(Arch::Other.name(), "other");
}

fn the_machine_tells_this_host_without_guessing() {
    let host = Machine;
    assert_eq!(host.arch(), Arch::of_host(), "架构只从一处读");
    let ledger = audit(&host);
    assert_eq!(ledger.len(), 6);
    for (row, standing) in Row::ALL.iter().zip(&ledger) {
        assert!(
            matches!(standing.as_str(), "covered" | "open"),
            "{} 只能是两种状态",
            row.as_str()
        );
        if standing.is_covered() {
            assert!(!standing.evidence().unwrap_or_default().is_empty());
        }
    }
    match host.arch() {
        Arch::Aarch64 => {
            assert!(
                ledger[Row::Arm as usize].is_covered(),
                "aarch64 宿主覆盖 ARM 行"
            );
            assert!(!ledger[Row::X86 as usize].is_covered());
        }
        Arch::X86_64 => {
            assert!(
                ledger[Row::X86 as usize].is_covered(),
                "x86_64 宿主覆盖 x86 行"
            );
            assert!(!ledger[Row::Arm as usize].is_covered());
        }
        Arch::Other => {
            assert!(!ledger[Row::Arm as usize].is_covered());
            assert!(!ledger[Row::X86 as usize].is_covered());
        }
    }
}
}
