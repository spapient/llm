# 给框架生成的程序注入 cudaEvent 计时:包住「第一条 <<< 发射」到「最后一次回读」之间的完整链路。
import sys
src, dst, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200
s = open(src).read()
launch = s.index("<<<")
line_start = s.rfind("\n", 0, launch) + 1
mark = "cudaMemcpy(h_y, d_y,"
end = s.rindex(mark)  # 最后一次回读前收尾
line_end = s.rfind("\n", 0, end) + 1
head = ("    cudaEvent_t t0, t1;\n    cudaEventCreate(&t0); cudaEventCreate(&t1);\n"
        "    cudaEventRecord(t0, 0);\n"
        f"    for (int rep = 0; rep < {reps}; ++rep) {{\n")
tail = ("    }\n    cudaEventRecord(t1, 0); cudaEventSynchronize(t1);\n"
        "    float bench_ms = 0; cudaEventElapsedTime(&bench_ms, t0, t1);\n"
        f"    fprintf(stderr, \"bench: %.4f ms/iter\\n\", bench_ms / {reps}.0f);\n")
out = s[:line_start] + head + s[line_start:line_end] + tail + s[line_end:]
open(dst, 'w').write(out)
print(f"bench -> {dst} (reps={reps})")
