from collections import Counter, defaultdict
from pathlib import Path
import re

root = Path('/Users/fei/code/llm/crates/prof/src')
vendors = ('ascend', 'cpu', 'intel', 'metal', 'musa', 'nvidia')
files = {v: {p.stem: p for p in (root / v / 'kernels').glob('*.k')} for v in vendors}
stems = sorted(set().union(*(set(v) for v in files.values())))
exact = Counter()
semantic = Counter()
rows = []
for stem in stems:
    texts = {v: files[v][stem].read_text() for v in vendors if stem in files[v]}
    exact[len(set(texts.values()))] += 1
    normalized = {}
    for vendor, text in texts.items():
        lines = text.splitlines()
        lines = [line for line in lines if not line.lstrip().startswith('#pragma unroll')]
        if lines and lines[0].startswith('// cutile '):
            lines[0] = re.sub(r'^// cutile \S+', '// cutile <vendor>', lines[0])
        normalized[vendor] = '\n'.join(lines).strip()
    groups = len(set(normalized.values()))
    semantic[groups] += 1
    if groups > 1:
        rows.append((stem, groups, {v: len(t.splitlines()) for v, t in texts.items()}))
print('vendors:', len(vendors))
print('unique stems:', len(stems))
print('physical files:', sum(len(v) for v in files.values()))
print('physical lines:', sum(len(p.read_text().splitlines()) for v in files.values() for p in v.values()))
print('exact distinct-body histogram:', dict(sorted(exact.items())))
print('normalized distinct-body histogram:', dict(sorted(semantic.items())))
print('normalized differences:')
for row in rows:
    print(row)
print('duplicate physical lines beyond one canonical file per stem:', sum(
    (len(files) - 1) * len(next(iter(files.values())).read_text().splitlines())
    for files in ({v: files[v][stem] for v in vendors if stem in files[v]} for stem in stems)
))
