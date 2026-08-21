"""
生成最终整合图：k=1 vs k=2 多维度对比面板
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)  # 项目根目录（core/）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts 目录（main.py）
from main import clean_text, extract_concepts, build_transition_matrix
from diagnose_v2 import C1, C2  # 复用计数矩阵

# 重新计算关键数据
from main import stationary_distribution, effective_information, normalized_ei

RAW = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "daodejing_sample.txt")).read()
cleaned = clean_text(RAW)
seq = extract_concepts(cleaned)

P1, idx1, states1 = build_transition_matrix(seq, k=1)
P2, idx2, states2 = build_transition_matrix(seq, k=2)

π1 = stationary_distribution(P1)
π2 = stationary_distribution(P2)

ei1 = effective_information(P1, π1)
ei2 = effective_information(P2, π2)
nei1 = normalized_ei(P1, π1)
nei2 = normalized_ei(P2, π2)

# ---- 加载之前保存的图 ----
from matplotlib.image import imread

fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

# ---- 左上：k=1 热力图 ----
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(P1, cmap="YlOrRd", aspect="auto")
ax1.set_title("k=1 转移矩阵热力图\n(17×17, Laplace 平滑后)", fontsize=12, fontweight="bold")
ax1.set_xlabel("目标概念")
ax1.set_ylabel("源概念")
labels1 = list(idx1.keys())
ax1.set_xticks(range(len(labels1)))
ax1.set_yticks(range(len(labels1)))
ax1.set_xticklabels(labels1, rotation=45, ha="right", fontsize=8)
ax1.set_yticklabels(labels1, fontsize=8)
plt.colorbar(im1, ax=ax1, shrink=0.8)

# ---- 右上：k=2 计数矩阵热力图（活跃区）----
ax2 = fig.add_subplot(gs[0, 1])
active_r = np.any(C2 > 0, axis=1)
active_c = np.any(C2 > 0, axis=0)
C2_sub = C2[np.ix_(active_r, active_c)]
im2 = ax2.imshow(C2_sub, cmap="YlOrRd", aspect="auto")
ax2.set_title(f"k=2 原始计数矩阵\n(活跃区 {C2_sub.shape[0]}×{C2_sub.shape[1]}, 密度 2.96%)",
              fontsize=12, fontweight="bold")
ax2.set_xlabel("目标联合状态")
ax2.set_ylabel("源联合状态")
plt.colorbar(im2, ax=ax2, shrink=0.8)

# ---- 左中：EI 对比柱状图 ----
ax3 = fig.add_subplot(gs[1, 0])
x = ["k=1", "k=2"]
ei_vals = [ei1, ei2]
colors = ["#2196F3", "#FF5722"]
bars = ax3.bar(x, ei_vals, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, ei_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.001,
             f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax3.set_ylabel("有效信息 EI (bits)")
ax3.set_title("有效信息 EI 对比", fontsize=12, fontweight="bold")
ax3.set_ylim(0, max(ei_vals)*1.3)

# ---- 右中：归一化 EI 对比 + 充足线 ----
ax4 = fig.add_subplot(gs[1, 1])
nei_vals = [nei1, nei2]
bars2 = ax4.bar(x, nei_vals, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, v in zip(bars2, nei_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, v + 0.001,
             f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax4.set_ylabel("归一化 EI = EI / log(N)")
ax4.set_title("归一化有效信息对比（消除维度影响）", fontsize=12, fontweight="bold")
ax4.set_ylim(0, max(nei_vals)*1.3)

# ---- 左下：平稳分布对比 ----
ax5 = fig.add_subplot(gs[2, 0])
# k=1 平稳分布
order1 = np.argsort(π1)[::-1][:10]
y1 = np.arange(len(order1))
ax5.barh(y1, π1[order1], color="#2196F3", alpha=0.8, label="k=1")
ax5.set_yticks(y1)
ax5.set_yticklabels([list(idx1.keys())[i] for i in order1], fontsize=9)
ax5.set_xlabel("平稳概率 π")
ax5.set_title("k=1 平稳分布 Top-10", fontsize=12, fontweight="bold")
ax5.legend()

# ---- 右下：数据充足性说明面板 ----
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis("off")

summary_text = (
    "数据充足性诊断\n"
    "━" * 30 + "\n\n"
    f"当前样本: {len(seq)} 个概念观测\n\n"
    f"k=1: 17 状态, 每状态 ~{len(seq)//17} 次观测\n"
    f"      密度 13.5% → 勉强可用\n\n"
    f"k=2: 39 状态, 每状态 ~{len(seq)//39} 次观测\n"
    f"      密度 2.96% → 严重不足 ⚠️\n\n"
    "━" * 30 + "\n\n"
    "⚠️ k=2 的 EI 更低是数据不足的伪影\n"
    "  Laplace +1 平滑主导了概率估计\n\n"
    "✓ 公平对比需要完整 81 章文本\n"
    "  或滑动窗口扩充到 3000+ 观测"
)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", alpha=0.8))

plt.savefig(os.path.join(OUTPUT_DIR, "final_comparison_panel.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ 已保存 output/final_comparison_panel.png")
print("\n所有产出文件：")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fp) / 1024
    print(f"  {f:35s} ({size:.1f} KB)")
