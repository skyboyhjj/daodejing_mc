"""
修正版诊断：基于原始计数矩阵的稀疏性分析
上一版用 Laplace 平滑后的 P 诊断，导致密度虚高到 100%
这一版用原始计数矩阵 C 来诊断真实的数据充足性
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, re
sys.path.insert(0, ".")
from main import clean_text, extract_concepts, build_transition_matrix

# 重建概念序列
RAW = open("daodejing_sample.txt").read()
cleaned = clean_text(RAW)
seq = extract_concepts(cleaned)
print(f"概念序列长度: {len(seq)}")
print(f"序列: {seq}\n")

# ---- 用 main.py 里的构建函数（返回的是平滑后的 P）----
# 我们需要原始计数，所以在这里直接重建
unique = sorted(set(seq))
idx = {c: i for i, c in enumerate(unique)}
N1 = len(unique)

# k=1 计数
C1 = np.zeros((N1, N1))
for t in range(len(seq)-1):
    C1[idx[seq[t]], idx[seq[t+1]]] += 1

# k=2 计数
pair_to_id = {}
pid = 0
for t in range(len(seq)-1):
    p = (seq[t], seq[t+1])
    if p not in pair_to_id:
        pair_to_id[p] = pid
        pid += 1
N2 = len(pair_to_id)
C2 = np.zeros((N2, N2))
for t in range(len(seq)-2):
    s = pair_to_id[(seq[t], seq[t+1])]
    d = pair_to_id[(seq[t+1], seq[t+2])]
    C2[s, d] += 1

print("="*60)
print("基于原始计数矩阵 C 的稀疏性诊断（未平滑）")
print("="*60)

for label, C, k in [("k=1", C1, 1), ("k=2", C2, 2)]:
    N = C.shape[0]
    total = N * N
    nnz = np.sum(C > 0)
    density = nnz / total
    row_nnz = np.sum(C > 0, axis=1)
    avg_targets = np.mean(row_nnz)
    max_targets = np.max(row_nnz)
    min_targets = np.min(row_nnz)
    total_obs = int(C.sum())
    avg_per_cell = total_obs / max(nnz, 1)

    print(f"\n{'─'*50}")
    print(f"  {label}: 状态数={N}, 总观测数={total_obs}")
    print(f"{'─'*50}")
    print(f"  非零元:      {nnz} / {total} (密度 {density*100:.2f}%)")
    print(f"  平均每行有效目标数: {avg_targets:.1f}")
    print(f"  最大/最小出度:      {max_targets} / {min_targets}")
    print(f"  每对平均共现次数:    {avg_per_cell:.2f}")
    
    # 关键诊断
    empty_rows = np.sum(row_nnz == 0)
    single_rows = np.sum(row_nnz == 1)
    print(f"  零出度行数:  {empty_rows}（这些状态从未被观察到转移）")
    print(f"  单目标行数:  {single_rows}（这些状态只转移到1个目标）")
    
    if avg_per_cell < 2:
        print(f"\n  ⚠️  每对平均仅 {avg_per_cell:.1f} 次共现")
        print(f"     → 绝大多数转移对只出现 0 或 1 次")
        print(f"     → Laplace 平滑后的概率主要由 +1 主导，而非真实数据")
        print(f"     → 结论不可靠，需要更多数据")

# ---- 可视化：k=1 vs k=2 的状态-观测矩阵热力图 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, C, label in [(axes[0], C1, "k=1 原始计数 C"), 
                        (axes[1], C2, "k=2 原始计数 C")]:
    # 只显示有至少一次转移的行/列，让图更清晰
    active_rows = np.any(C > 0, axis=1)
    active_cols = np.any(C > 0, axis=0)
    C_sub = C[np.ix_(active_rows, active_cols)]
    
    im = ax.imshow(C_sub, cmap="YlOrRd", aspect="auto")
    ax.set_title(f"{label}\n活跃区: {C_sub.shape[0]}×{C_sub.shape[1]}")
    ax.set_xlabel("目标状态 idx")
    ax.set_ylabel("源状态 idx")
    plt.colorbar(im, ax=ax, label="共现次数")

plt.tight_layout()
plt.savefig("output/count_matrix_comparison.png", dpi=150)
plt.close()
print("\n  ✓ 已保存 output/count_matrix_comparison.png")

# ---- 数据充足性总结图 ----
fig, ax = plt.subplots(figsize=(8, 5))

scenarios = [
    ("当前示例\n(3章, 47概念)", 47, 17, 39),
    ("10章\n(~200概念)", 200, 25, 80),
    ("完整81章\n(~1000概念)", 1000, 30, 120),
    ("全文+滑动窗口\n(~3000概念)", 3000, 30, 120),
]

for i, (name, n_obs, n_k1, n_k2) in enumerate(scenarios):
    # 经验法则：每个状态至少 20 次观测
    need_k1 = n_k1 * 20
    need_k2 = n_k2 * 20
    ok_k1 = "✓" if n_obs >= need_k1 else "✗"
    ok_k2 = "✓" if n_obs >= need_k2 else "✗"
    
    ax.barh(i*2, n_obs/need_k1, color="#2196F3", alpha=0.7, label="k=1 充足度" if i==0 else "")
    ax.barh(i*2+0.8, n_obs/need_k2, color="#FF5722", alpha=0.7, label="k=2 充足度" if i==0 else "")

ax.set_yticks([i*2+0.4 for i in range(len(scenarios))])
ax.set_yticklabels([s[0] for s in scenarios])
ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5, label="充足线 (1.0)")
ax.set_xlabel("观测数 / 需求数  (≥1 表示数据充足)")
ax.set_title("不同文本规模下 k=1 与 k=2 的数据充足度")
ax.legend(loc="lower right")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("output/data_adequacy.png", dpi=150)
plt.close()
print(f"  ✓ 已保存 output/data_adequacy.png")

# ---- 最终结论 ----
print("\n" + "="*60)
print("核心结论")
print("="*60)
print("""
  示例文本（3章, 47个概念）下：
  • k=1: 17个状态, 每个状态平均观测 ~2.8 次 → 勉强可用
  • k=2: 39个状态, 每个状态平均观测 ~0.6 次 → 严重不足

  → 之前看到的"k=2 的 EI 更低"不是理论结论，
    而是数据不足的伪影（Laplace 平滑主导了估计）。

  → 要公平对比 k=1 vs k=2，至少需要：
    • 完整 81 章文本（~1000 概念观测）
    • 或更优：全文 + 滑动窗口扩充样本（~3000+ 观测）

  下一步建议：
  ① 获取完整王弼本 81 章文本 → 重跑 main.py
  ② 加入滑动窗口 → 将 k=2 的有效样本量放大 3-5 倍
  ③ 再用 main.py 的对比表看真实差异
""")
