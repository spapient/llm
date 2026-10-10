/// Identifies the execution device class a backend runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum devicekind {
    /// Host CPU execution.
    cpu,
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
}
