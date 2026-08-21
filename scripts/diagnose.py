"""
稀疏性诊断 + 数据充足性分析
解释为什么示例文本下 k=2 的 EI 反而更低
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def diagnose(P, states, k, label):
    N = P.shape[0]
    total = N * N
    nnz = np.sum(P > 1e-12)
    density = nnz / total
    row_nnz = np.sum(P > 1e-12, axis=1)
    avg_out = np.mean(row_nnz)
    max_out = np.max(row_nnz)
    min_out = np.min(row_nnz)

    print(f"\n{'─'*55}")
    print(f"  {label}  稀疏性诊断")
    print(f"{'─'*55}")
    print(f"  矩阵大小:        {N} × {N}")
    print(f"  非零元素:        {nnz} / {total}")
    print(f"  密度:            {density:.4f} ({density*100:.2f}%)")
    print(f"  平均每行非零数:  {avg_out:.1f}")
    print(f"  最大/最小出度:  {max_out} / {min_out}")

    # 每个状态的"有效转移目标数"
    effective_targets = np.sum(P > 0.01, axis=1)
    print(f"  P>0.01 的平均目标数: {np.mean(effective_targets):.1f}")

    # 如果密度 < 5%，说明数据严重不足
    if density < 0.05:
        print(f"\n  ⚠️  密度仅 {density*100:.2f}%，远低于 5%")
        print(f"     → 转移矩阵极度稀疏，Laplace 平滑主导了概率估计")
        print(f"     → k=2 状态空间 {N} 远大于有效样本量 ~47")
        print(f"     → 结论：不是 k=2 没用，是数据不够支撑 k=2")
        print(f"     → 需要完整 81 章文本（~5000 字）才能公平对比")

    return density

# 加载之前保存的矩阵
P1 = np.load(os.path.join(OUTPUT_DIR, "P_k1.npy"))
P2 = np.load(os.path.join(OUTPUT_DIR, "P_k2.npy"))

# 重建 states 列表（简化版）
import json
from collections import Counter

RAW_TEXT = open(os.path.join(BASE_DIR, "daodejing_sample.txt")).read() if False else None

# 直接用 main.py 里的逻辑重建
import re
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录（core/）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts 目录（main.py）
from main import clean_text, extract_concepts, CONCEPT_DICT, REVERSE_MAP

# 用内置文本
RAW = """
道可道非常道。名可名非常名。无名天地之始，有名万物之母。
故常无欲以观其妙，常有欲以观其徼。此两者同出而异名，同谓之玄。
玄之又玄，众妙之门。

天下皆知美之为美，斯恶已。皆知善之为善，斯不善已。
故有无相生，难易相成，长短相形，高下相倾，音声相和，前后相随。
是以圣人处无为之事，行不言之教。
万物作焉而不辞，生而不有，为而不恃，功成而弗居。夫唯弗居，是以不去。

不尚贤，使民不争。不贵难得之货，使民不为盗。
不见可欲，使心不乱。是以圣人之治，虚其心，实其腹，弱其志，强其骨。
常使民无知无欲，使夫知者不敢为也。为无为，则无不治。
"""

cleaned = clean_text(RAW)
seq = extract_concepts(cleaned)
unique = sorted(set(seq))

states1 = unique
states2 = []
pair_to_id = {}
pid = 0
for t in range(len(seq)-1):
    p = (seq[t], seq[t+1])
    if p not in pair_to_id:
        pair_to_id[p] = pid
        states2.append(f"{p[0]}→{p[1]}")
        pid += 1

d1 = diagnose(P1, states1, 1, "k=1")
d2 = diagnose(P2, states2, 2, "k=2")

# ---- 可视化稀疏性对比 ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, P, label in [(axes[0], P1, "k=1 (17×17)"), (axes[1], P2, "k=2 (39×39)")]:
    N = P.shape[0]
    # 取上三角（不含对角线）画直方图
    iu = np.triu_indices(N, k=1)
    vals = P[iu]
    vals = vals[vals > 1e-12]  # 只画非零
    ax.hist(vals, bins=20, color="#2196F3" if "k=1" in label else "#FF5722", alpha=0.7)
    ax.set_title(f"转移概率分布\n{label}")
    ax.set_xlabel("P(i→j)")
    ax.set_ylabel("频次")
    ax.axvline(x=1/N, color="red", linestyle="--", label=f"均匀基线 1/N={1/N:.3f}")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sparsity_diagnosis.png"), dpi=150)
plt.close()
print("\n  ✓ 已保存 output/sparsity_diagnosis.png")

# ---- 数据充足性估算 ----
print("\n" + "="*55)
print("数据充足性估算")
print("="*55)
print(f"\n  当前样本: {len(seq)} 个概念, k=1 状态 {len(states1)}, k=2 状态 {len(states2)}")
print(f"\n  经验法则: 每个状态至少需要 20-50 次观测才能可靠估计转移概率")
print(f"\n  k=1 需要: ~{len(states1)*20}–{len(states1)*50} 个概念观测")
print(f"  k=2 需要: ~{len(states2)*20}–{len(states2)*50} 个概念观测")
print(f"\n  完整《道德经》(~5000字) 经概念抽取后约 800-1200 个概念观测")
print(f"  → 对 k=1 绰绰有余, 对 k=2 勉强可用")
print(f"  → 若用全文 + 滑动窗口扩充样本, k=2 对比将更可靠")
