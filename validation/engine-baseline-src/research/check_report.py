import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report = root / "weight-format-architecture-research.md"
text = report.read_text(encoding="utf-8")
blocks = re.split(r"\n\s*\n", text)
short = []
for block in blocks:
    stripped = block.strip()
    if not stripped or stripped.startswith(("#", "|", "![", "```")):
        continue
    if len(stripped) < 100:
        short.append((len(stripped), stripped[:80]))
images = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
missing = [path for path in images if not (root / path).is_file()]
forbidden = re.findall(r"^##+\s+(?:References?|Sources|参考文献|参考资料)\s*$", text, re.M)
print(f"chars={len(text)} paragraphs={len(blocks)} images={images}")
print(f"short_paragraphs={short}")
print(f"missing_images={missing}")
print(f"forbidden_reference_sections={forbidden}")
assert not missing
assert not forbidden
assert not short
