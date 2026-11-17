//! 设备后端边界：[`Executor`] / [`Allocator`] / [`Device`] 的具体实现侧。
//!
//! 运行时 trait 定义归 [`engine`] 与 [`storage`]（ADR 0025 §1）：`Executor` / `Device` 归
//! `engine::exec`，`Allocator` / `Buffer` 归 `storage`。本 crate 只持实现侧，依赖方向只能从边界
//! crate 指向运行时 trait，禁止反向（ADR 0025 §1、ADR 0035 §3）：本 crate 不定义第二套运行时
//! trait，也不反向依赖 `serve` / `nn`。
//!
//! # 模块边界（ADR 0035 预留）
//!
//! 逐厂商边界模块见 `metal` 与 `cuda`：设备句柄、命令队列、事件与缓冲映射未来只收在各自
//! 模块内（ADR 0035 §1 白名单四类：裸指针解引用、设备缓冲句柄、FFI 调用、mmap 映射）。
//! `#[allow(unsafe_code)]` 只出现在边界模块或边界函数，模块头注释写明越界理由与安全不变量
//! （ADR 0035 §2/§4）。本刀只落骨架与能力候选表，不接任何真实 FFI / 设备内核，全仓继续
//! `unsafe_code = "deny"`。
//!
//! 对外不暴露设备指针 / 句柄类型：trait 方法签名只出现运行时一侧的不透明标识与
//! [`storage::Buffer`]，模型、调度、算子代码拿不到厂商句柄（ADR 0035 §1）。

mod cuda;
mod metal;

use diagnostic::{Code, Error, Result};
use engine::exec::{Device, Executor, Graph, Stream};
use storage::{Allocator, Buffer};

pub use cuda::CudaBackend;
pub use metal::MetalBackend;

/// 未挂载设备后端的占位适配：同时实现 [`Executor`] 与 [`Allocator`]。
///
/// 提交、同步、分配、显式拷贝入口统一返回 [`Code::Unsupported`]；构造只绑一台登记用的
/// [`Device`] 标签，不触碰任何设备句柄。真实 Metal / CUDA 实现分别落在 `metal` / `cuda`
/// 边界模块；本类型是骨架期的 `Unsupported` 占位，不写厂商特判分支。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedBackend {
    device: Device,
}

impl UnsupportedBackend {
    /// 由设备标签构造；标签只作登记，不持有设备私有句柄。
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

/// 边界适配的统一失败：后端尚未挂载内核，指明入口类别与来源适配。
fn unsupported(site: &str, what: &str) -> Error {
    Error::new(
        Code::Unsupported,
        format!("后端未挂载：{what} 在该适配上不可用"),
    )
    .with(site)
}

impl Executor for UnsupportedBackend {
    fn device(&self) -> &Device {
        &self.device
    }

    fn submit(&mut self, _graph: &Graph, _stream: Stream) -> Result<()> {
        Err(unsupported("backend::UnsupportedBackend", "图提交"))
    }

    fn sync(&mut self, _stream: Stream) -> Result<()> {
        Err(unsupported("backend::UnsupportedBackend", "流同步"))
    }
}

impl Allocator for UnsupportedBackend {
    fn alloc(&mut self, _bytes: usize) -> Result<Buffer> {
        Err(unsupported("backend::UnsupportedBackend", "缓冲分配"))
    }

    fn free(&mut self, _buffer: Buffer) {}

    fn copy_into(&self, _src: &[u8], _dst: &mut [u8]) -> Result<()> {
        Err(unsupported("backend::UnsupportedBackend", "显式拷贝"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 设备标签回读：占位适配只转发登记标签，不保留第二份设备假设。
    #[test]
    fn device_label_round_trips() {
        let backend = UnsupportedBackend::new(Device::new("metal", "aarch64"));
        assert_eq!(backend.device().vendor(), "metal");
        assert_eq!(backend.device().arch(), "aarch64");
    }

    /// 提交与同步在占位适配上都是 `Unsupported`，不排队、不碰句柄。
    #[test]
    fn submit_and_sync_are_unsupported() {
        let mut backend = UnsupportedBackend::new(Device::new("cuda", "sm_90"));
        let graph = Graph::new();
        let stream = Stream::new(0);

        let submit = backend.submit(&graph, stream).expect_err("未挂载后端");
        assert_eq!(submit.code(), Code::Unsupported);
        let sync = backend.sync(stream).expect_err("未挂载后端");
        assert_eq!(sync.code(), Code::Unsupported);
    }

    /// 分配与拷贝是 `Unsupported`；`free` 按 trait 签名留空实现，不归还任何真实句柄。
    #[test]
    fn alloc_and_copy_are_unsupported_free_is_noop() {
        let mut backend = UnsupportedBackend::new(Device::new("cpu", "generic"));

        let alloc = backend.alloc(16).expect_err("未挂载后端");
        assert_eq!(alloc.code(), Code::Unsupported);
        let copy = backend.copy_into(&[0u8], &mut []).expect_err("未挂载后端");
        assert_eq!(copy.code(), Code::Unsupported);

        backend.free(Buffer::from_bytes(Vec::new()));
    }

    /// 同一 `Executor` / `Allocator` 端口下核对任意占位适配：五个运行入口都回
    /// [`Code::Unsupported`]，`free` 按 trait 签名留空、不归还任何真实句柄。
    fn assert_all_entries_unsupported<E: Executor + Allocator>(backend: &mut E) {
        let graph = Graph::new();
        let stream = Stream::new(0);

        let submit = backend
            .submit(&graph, stream)
            .expect_err("占位后端应拒绝图提交");
        assert_eq!(submit.code(), Code::Unsupported);
        let sync = backend.sync(stream).expect_err("占位后端应拒绝流同步");
        assert_eq!(sync.code(), Code::Unsupported);
        let alloc = backend.alloc(16).expect_err("占位后端应拒绝缓冲分配");
        assert_eq!(alloc.code(), Code::Unsupported);
        let copy = backend
            .copy_into(&[0u8], &mut [])
            .expect_err("占位后端应拒绝显式拷贝");
        assert_eq!(copy.code(), Code::Unsupported);
        backend.free(Buffer::from_bytes(Vec::new()));
    }

    /// 厂商标签只作登记：Metal / CUDA 占位构造后 vendor 与架构标签原样回读，不保留第二份设备假设。
    #[test]
    fn vendor_labels_round_trip_per_backend() {
        let metal = MetalBackend::new("aarch64");
        assert_eq!(metal.device().vendor(), "metal");
        assert_eq!(metal.device().arch(), "aarch64");

        let cuda = CudaBackend::new("sm_90");
        assert_eq!(cuda.device().vendor(), "cuda");
        assert_eq!(cuda.device().arch(), "sm_90");
    }

    /// 候选挂载算子必须都在全局算子注册表内，且不重复；名字只是能力登记，不代表已挂载内核。
    #[test]
    fn mounted_candidates_belong_to_the_registry() {
        let backends = [
            MetalBackend::new("aarch64").mounted(),
            CudaBackend::new("sm_90").mounted(),
        ];
        for mounted in backends {
            let mut seen = std::collections::BTreeSet::new();
            for name in mounted {
                assert!(op::contains(name), "能力表 `{name}` 不在 op 注册表内");
                assert!(seen.insert(name), "能力表 `{name}` 重复登记");
            }
        }
    }

    /// 三个占位后端在同一端口契约下入口行为一致：提交 / 同步 / 分配 / 拷贝全部 `Unsupported`。
    #[test]
    fn placeholders_share_the_same_unsupported_contract() {
        assert_all_entries_unsupported(&mut UnsupportedBackend::new(Device::new("cpu", "generic")));
        assert_all_entries_unsupported(&mut MetalBackend::new("aarch64"));
        assert_all_entries_unsupported(&mut CudaBackend::new("sm_90"));
    }
}
