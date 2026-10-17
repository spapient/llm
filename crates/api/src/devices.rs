/// Identifies the execution device class a backend runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum devicekind {
    /// Host CPU execution.
    cpu,
    /// Apple Metal execution.
    metal,
    /// NVIDIA CUDA execution.
    cuda,
    /// Moore Threads MUSA execution.
    musa,
    /// AMD / Hygon ROCm execution.
    rocm,
    /// Intel oneAPI SYCL execution.
    sycl,
    /// Huawei CANN execution.
    cann,
    /// Cambricon MLU execution.
    mlu,
}

/// An execution device handle.
///
/// The inner kind is private so that the set of devices is owned by this crate;
/// callers construct and inspect devices through the provided constructors and
/// accessors only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct device {
    kind: devicekind,
}

impl device {
    /// Returns the host CPU device.
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            kind: devicekind::cpu,
        }
    }

    /// Returns the Apple Metal device.
    #[must_use]
    pub const fn metal() -> Self {
        Self {
            kind: devicekind::metal,
        }
    }

    /// Returns the NVIDIA CUDA device.
    #[must_use]
    pub const fn cuda() -> Self {
        Self {
            kind: devicekind::cuda,
        }
    }

    /// Returns the Moore Threads MUSA device.
    #[must_use]
    pub const fn musa() -> Self {
        Self {
            kind: devicekind::musa,
        }
    }

    /// Returns the AMD / Hygon ROCm device.
    #[must_use]
    pub const fn rocm() -> Self {
        Self {
            kind: devicekind::rocm,
        }
    }

    /// Returns the Intel oneAPI SYCL device.
    #[must_use]
    pub const fn sycl() -> Self {
        Self {
            kind: devicekind::sycl,
        }
    }

    /// Returns the Huawei CANN device.
    #[must_use]
    pub const fn cann() -> Self {
        Self {
            kind: devicekind::cann,
        }
    }

    /// Returns the Cambricon MLU device.
    #[must_use]
    pub const fn mlu() -> Self {
        Self {
            kind: devicekind::mlu,
        }
    }

    /// Returns the kind of this device.
    #[must_use]
    pub const fn getkind(self) -> devicekind {
        self.kind
    }

    /// Reports whether this device is the host CPU.
    #[must_use]
    pub const fn iscpu(self) -> bool {
        matches!(self.kind, devicekind::cpu)
    }

    /// Reports whether this device is Apple Metal.
    #[must_use]
    pub const fn ismetal(self) -> bool {
        matches!(self.kind, devicekind::metal)
    }

    /// Reports whether this device is NVIDIA CUDA.
    #[must_use]
    pub const fn iscuda(self) -> bool {
        matches!(self.kind, devicekind::cuda)
    }

    /// Reports whether this device is Moore Threads MUSA.
    #[must_use]
    pub const fn ismusa(self) -> bool {
        matches!(self.kind, devicekind::musa)
    }

    /// Reports whether this device is AMD / Hygon ROCm.
    #[must_use]
    pub const fn isrocm(self) -> bool {
        matches!(self.kind, devicekind::rocm)
    }

    /// Reports whether this device is Intel oneAPI SYCL.
    #[must_use]
    pub const fn issycl(self) -> bool {
        matches!(self.kind, devicekind::sycl)
    }

    /// Reports whether this device is Huawei CANN.
    #[must_use]
    pub const fn iscann(self) -> bool {
        matches!(self.kind, devicekind::cann)
    }

    /// Reports whether this device is Cambricon MLU.
    #[must_use]
    pub const fn ismlu(self) -> bool {
        matches!(self.kind, devicekind::mlu)
    }
}
