import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
paths = [
    "crates/prof/src/native.rs",
    "crates/prof/src/native/gemm8x8.rs",
    "crates/prof/src/native/i8.rs",
    "crates/prof/src/native/sme.rs",
    "crates/prof/src/runtime.rs",
    "crates/cutile/src/native.rs",
]
files = {}
public_lines = []
for name in paths:
    path = ROOT / name
    raw = path.read_bytes()
    text = raw.decode()
    files[name] = {
        "lines": len(text.splitlines()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if name.endswith(".rs"):
        public_lines.extend(
            f"{name}:{number}:{line.strip()}"
            for number, line in enumerate(text.splitlines(), 1)
            if re.match(
                r"^\s*pub(?:\([^)]*\))?\s+(?:fn|struct|enum|trait|const|type|mod|use)\b", line
            )
        )
result = {
    "schema": 1,
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "rustc": subprocess.check_output(["rustc", "--version"], cwd=ROOT, text=True).strip(),
    "files": files,
    "public_lines": public_lines,
    "native_lines": sum(files[name]["lines"] for name in paths[:4]),
    "native_bytes": sum(files[name]["bytes"] for name in paths[:4]),
}
(Path(__file__).parent / "baseline.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "head": result["head"],
            "rustc": result["rustc"],
            "native_lines": result["native_lines"],
            "native_bytes": result["native_bytes"],
            "public_lines": len(public_lines),
        },
        ensure_ascii=False,
    )
)
