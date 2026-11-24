from pathlib import Path
import hashlib

workspace = Path(__file__).resolve().parents[2]
source = workspace / "crates/prof/src/nvidia/kernels/matmul_f32.k"
raw = source.read_bytes()
text = raw.decode()
assert "{{op}}" in text
assert "/* cutile-unroll */" not in text
rendered = text.replace("{{op}}", "matmul_f32")
out = Path(__file__).with_name("matmul_f32_cublas.cu")
out.write_text(rendered)
print(f"nvidia_source_sha256={hashlib.sha256(raw).hexdigest()}")
print(f"nvidia_rendered_sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
