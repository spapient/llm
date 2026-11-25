from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
BG = "#f8fafc"
plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})


def box(ax, center, width, height, text, color, size=17):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=0, facecolor=color, transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=size,
            fontweight=600, linespacing=1.35, transform=ax.transAxes)


def arrow(ax, start, end, text, rad=0.0):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                            linewidth=2.2, color="#94a3b8",
                            connectionstyle=f"arc3,rad={rad}", transform=ax.transAxes)
    ax.add_patch(patch)
    x = (start[0] + end[0]) / 2 + rad * 0.18
    y = (start[1] + end[1]) / 2 + rad * 0.14
    ax.text(x, y, text, ha="center", va="center", fontsize=12, color="#475569",
            backgroundcolor=BG, transform=ax.transAxes)


fig, ax = plt.subplots(figsize=(12, 7.2), dpi=100)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("共享格式与算子策略分层", fontsize=25, color="#0f172a", pad=28)
ax.text(0.5, 0.925, "单一 DType 事实源 · 文件产出 · 投影策略 · 无循环依赖",
        ha="center", fontsize=14, color="#64748b", transform=ax.transAxes)
box(ax, (0.16, 0.49), 0.24, 0.14, "dtype\n共享格式事实", "#2563eb", 19)
box(ax, (0.49, 0.70), 0.25, 0.14, "weights\n文件与量化配对", "#059669")
box(ax, (0.49, 0.28), 0.25, 0.14, "nn\n投影算子策略", "#d97706")
box(ax, (0.84, 0.49), 0.25, 0.14, "serve\n形状与执行装配", "#7c3aed")
arrow(ax, (0.25, 0.56), (0.37, 0.65), "复用 DType + group", 0.10)
arrow(ax, (0.25, 0.42), (0.37, 0.33), "同一 WeightFormat", -0.10)
arrow(ax, (0.49, 0.63), (0.49, 0.35), "weight_format()", 0.16)
arrow(ax, (0.62, 0.28), (0.72, 0.44), "operator_for()")
arrow(ax, (0.62, 0.70), (0.72, 0.55), "Entry / Scale", -0.14)
fig.savefig(ROOT / "weight-architecture.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)

labels = ["dtype()", "operator_for()"]
baseline = [9.175834, 137.791791]
split = [5.133459, 124.43725]
x = [0, 1]
width = 0.30
fig, ax = plt.subplots(figsize=(12, 7.2), dpi=100)
ax.set_title("WeightFormat release 微基准（越低越好）", fontsize=24, color="#0f172a", pad=28)
ax.text(0.5, 1.01, "同机 · 每轮 1800 万次调用 · 各 5 轮取中值 · 单位 ms",
        ha="center", fontsize=13, color="#64748b", transform=ax.transAxes)
bars1 = ax.bar([v - width / 2 for v in x], baseline, width, label="原始枚举实现", color="#94a3b8")
bars2 = ax.bar([v + width / 2 for v in x], split, width, label="DType 组合拆分", color="#2563eb")
ax.set_xticks(x, labels, fontsize=16)
ax.set_ylabel("耗时（ms）", fontsize=14)
ax.set_ylim(0, 155)
ax.grid(axis="y", color="#e2e8f0", linewidth=1)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#94a3b8")
ax.legend(loc="upper center", frameon=False, fontsize=13, ncol=2)
for bars in (bars1, bars2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                f"{bar.get_height():.3f}", ha="center", fontsize=14, color="#334155")
fig.tight_layout()
fig.savefig(ROOT / "weight-performance.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
