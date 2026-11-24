from pathlib import Path

root = Path('/Users/fei/code/llm')
contract_path = root / 'crates/shard/src/backend.rs'
contract = contract_path.read_text()
backup = (root / '.tmp/backends-backup.rs').read_text()

body = backup.split('use crate::driver;', 1)[1]
body = 'use crate::driver;' + body
for line in [
    'use crate::toolchain::Toolchain;\n',
    'use crate::witness::{MAX_OUT, Scope, Selected};\n',
    'use cutile::dirs::Dirs;\n',
    'use diagnostic::{Code, Error, Result};\n',
    'use prof::prof::{Emit, Profile};\n',
    'use prof::runtime::opencl::OpenClDevice;\n',
]:
    body = body.replace(line, '')
body = body.replace(
    'static OPENCL_BACKEND: OpenClBackend = OpenClBackend;',
    '#[cfg(windows)]\nstatic OPENCL_BACKEND: opencl::OpenClBackend = opencl::OpenClBackend;',
)
body = body.replace(
    '    crate::backend::register(&OPENCL_BACKEND);',
    '    #[cfg(windows)]\n    crate::backend::register(&OPENCL_BACKEND);',
)
body = body.replace(
    'pub(crate) struct OpenClBackend;',
    '#[cfg(windows)]\nmod opencl {\n    use super::*;\n    use prof::runtime::opencl::OpenClDevice;\n\n    pub(super) struct OpenClBackend;',
)
body = body.replace(
    'impl Backend for OpenClBackend {}\n',
    'impl Backend for OpenClBackend {}\n}\n',
)
body = body.replace('crate::backend::register(', 'register(')
contract_path.write_text(
    contract
    + '\n\n// ---- 设备实现：配置表、单源后端与 Metal/OpenCL 执行 ----\n\n'
    + body
    + '\n#[cfg(test)]\n#[path = "../tests/backends.rs"]\nmod tests;\n'
)
print(f'wrote {contract_path} ({contract_path.stat().st_size} bytes)')
