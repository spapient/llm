from pathlib import Path
import hashlib

workspace = Path(__file__).resolve().parents[2]
source = workspace / "crates/prof/src/musa/kernels/matmul_f32.k"
raw = source.read_bytes()
text = raw.decode()
assert "{{op}}" in text
assert text.count("/* cutile-unroll */") == 1
rendered = text.replace("{{op}}", "matmul_f32").replace(
    "/* cutile-unroll */", "#pragma unroll 4"
)
out = Path(__file__).with_name("matmul_f32.cu")
out.write_text(rendered)
print(f"source_sha256={hashlib.sha256(raw).hexdigest()}")
print(f"rendered_sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
