import struct, sys, subprocess
got = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line: continue
    bits = int(line) & 0xFFFFFFFF
    got.append(struct.unpack('<f', struct.pack('<I', bits))[0])
want = [float(x) for x in open(sys.argv[2])]
if len(got) != len(want):
    print(f"FAIL len {len(got)} vs {len(want)}"); sys.exit(1)
worst = max(abs(g - w) / max(abs(w), 1e-30) for g, w in zip(got, want))
print(f"{'OK' if worst < 1e-5 else 'FAIL'} n={len(got)} max_rel={worst:.3e}")
sys.exit(0 if worst < 1e-5 else 1)
