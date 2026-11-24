from pathlib import Path
import subprocess

root = Path('/Users/fei/code/llm')
fsck = subprocess.run(
    ['git', 'fsck', '--no-reflogs', '--unreachable', '--no-progress'],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
).stdout
needles = (b'NativeSpec', b'OpenClBackend', b'compile_run_native')
found = []
for line in fsck.splitlines():
    parts = line.split()
    if len(parts) < 3 or parts[1] != 'blob':
        continue
    sha = parts[2]
    data = subprocess.run(
        ['git', 'cat-file', 'blob', sha], cwd=root, check=True, capture_output=True
    ).stdout
    if all(needle in data for needle in needles):
        found.append((sha, data))

if not found:
    raise SystemExit('no matching backend blob found')
sha, data = max(found, key=lambda item: len(item[1]))
out = root / '.tmp' / 'backends-recovered.rs'
out.write_bytes(data)
print(f'{sha} {len(data)} {out}')
