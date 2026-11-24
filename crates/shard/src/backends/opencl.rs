//! Intel OpenCL 后端：见证装配、驱动内 JIT、网格发射与类型化回读。
//! 设备正文来自 prof 档案；OpenCL FFI 与句柄全部留在 `prof::runtime::opencl` 边界岛。

use crate::backend::{Backend, Codegen, Driver, Readback};
use crate::driver;
use crate::toolchain::Toolchain;
use crate::witness::{MAX_OUT, Scope, Selected};
use cutile::dirs::Dirs;
use diagnostic::Result;
use prof::prof::{Emit, Profile};
use prof::runtime::opencl::OpenClDevice;

pub(crate) struct OpenClBackend;

impl Codegen for OpenClBackend {
    fn emit(&self) -> Emit {
        Emit::Opencl
    }

    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let _ = scope;
        let mut program = profile.preamble.clone();
        driver::normalize_newline(&mut program);
        driver::push_sources(
            &mut program,
            &driver::witness_files(selected, dirs, driver::C_PARAMS)?,
        );
        program.push_str(&format!(
            "__kernel void cutile_witness(__global float *out, __global int *counts) {{\n\
             \x20   int tid = get_global_id(0);\n\
             \x20   if (tid >= get_global_size(0)) return;\n\
             \x20   float local[{MAX_OUT}];\n"
        ));
        driver::push_dispatch(&mut program, MAX_OUT, selected, "}\n");
        Ok(program)
    }
}

impl Driver for OpenClBackend {
    fn available(&self, _: &Profile) -> bool {
        OpenClDevice::probe().is_some()
    }

    fn run(
        &self,
        profile: &Profile,
        _: &Dirs,
        program: &str,
        total: usize,
        _: Scope,
    ) -> Result<Readback> {
        let device = OpenClDevice::new()?;
        let kernel = device.compile(program, &profile.flags.join(" "), "cutile_witness")?;
        let output = device.buffer(total * MAX_OUT * size_of::<f32>())?;
        let counts = device.buffer(total * size_of::<i32>())?;
        device.launch(&kernel, &[&output, &counts], [total, 1, 1], None)?;
        let counts = device.read_i32(&counts)?;
        let values = device.read_f32(&output)?;
        let mut readback = Vec::new();
        for (index, &count) in counts.iter().enumerate() {
            let count = usize::try_from(count).unwrap_or(0).min(MAX_OUT);
            let start = index * MAX_OUT;
            readback.extend_from_slice(&values[start..start + count]);
        }
        Ok((counts.into_iter().map(i64::from).collect(), readback))
    }

    fn toolchain(&self) -> Option<Toolchain> {
        OpenClDevice::probe().map(|info| Toolchain {
            command: "OpenCL".to_string(),
            version: Some(info.driver),
            banner: format!("{} · {}", info.name, info.version),
        })
    }
}

impl Backend for OpenClBackend {}
