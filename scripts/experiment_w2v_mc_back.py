"""
experiment_w2v_mc_back.py
P2-A：W2V → MC 反向融合（图传播）

核心思想：用 W2V 向量作为语义先验 E0，用转移矩阵 P 做图传播，
得到"动力学感知"的词向量，再 K-Means(M=6) 并计算成块性 ε。

图传播公式：E_new = α·E0 + (1-α)·P^T·E_old   （迭代至收敛）
  - α = 1.0：纯语义先验（W2V，不传播）
  - α = 0.5：语义 + 动力学各半
  - α = 0.0：纯动力学（P^T 主导）

对比基准：
  - 手工语义分组 ε ≈ 0.005（HANDOFF 记录值）
  - 纯 SVD 谱嵌入的 ε
  - 纯 W2V（α=1.0）的 ε

依赖：numpy, scikit-learn (KMeans), matplotlib
用法：python scripts/experiment_w2v_mc_back.py
"""
import numpy as np
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.linalg import svd
from core.pipeline import (
    stationary_distribution, normalized_ei, lumpability_error,
    build_macro_transition, build_transition_matrix,
)
from main import DAODEJING, build_full_sequence, SEMANTIC_PARTITION, MACRO_NAMES

print("=" * 60)
print("P2-A：W2V → MC 反向融合（图传播）")
print("=" * 60)

# ---- 加载数据 ----
P = np.load(os.path.join(ROOT, "output/P_matrix.npy"))
pi = np.load(os.path.join(ROOT, "output/pi.npy"))
w2v = np.load(os.path.join(ROOT, "output/w2v_vectors.npy"))
N = P.shape[0]

# 概念名
concepts = [f"C{i}" for i in range(N)]
cg_path = os.path.join(ROOT, "output/coarse_graining.json")
if os.path.exists(cg_path):
    with open(cg_path, encoding="utf-8") as f:
        cg = json.load(f)
    tmp = []
    for g in cg.get("groups", {}).values():
        for item in g.get("concepts", []):
            tmp.append(item["name"])
    if len(tmp) == N:
        concepts = tmp
# idx: 概念名 -> 索引（与 P 对齐）
idx = {c: i for i, c in enumerate(sorted(set(concepts)))}
# 注意：P 的索引顺序来自 main 的 build_transition_matrix（按 set(full_seq) 排序）
# 此处需要保证 concepts 顺序与 P 一致。用 coarse_graining groups 顺序可能不同。
# 用全序列重建 idx 保证一致
full_seq, _ = build_full_sequence(DAODEJING)
idx_ps = {c: i for i, c in enumerate(sorted(set(full_seq)))}
# 用 idx_ps 作为对齐索引；w2v/concepts 也按此顺序重排
order = [concepts.index(c) for c in sorted(set(full_seq))]
w2v_aligned = w2v[order]
concepts_aligned = [concepts[i] for i in order]
idx = idx_ps

print(f"  N = {N}, W2V 维度 = {w2v_aligned.shape[1]}")
print(f"  概念顺序已与 P 对齐")

# ---- 图传播函数 ----
def graph_propagate(E0, P, alpha, max_iter=500, tol=1e-10):
    """E_new = α·E0 + (1-α)·P^T·E_old，迭代至收敛"""
    E = E0.copy()
    Pt = P.T
    for _ in range(max_iter):
        E_new = alpha * E0 + (1 - alpha) * (Pt @ E)
        if np.linalg.norm(E_new - E, 'fro') < tol:
            return E_new
        E = E_new
    return E

# ---- K-Means + 成块性 ε ----
def kmeans_eps(E, P, concepts, idx, M=6, seed=42):
    """在嵌入 E 上 K-Means(M)，计算成块性 ε"""
    km = KMeans(n_clusters=M, random_state=seed, n_init=20, max_iter=500)
    labels = km.fit_predict(E)
    # 构建 partition（概念名）
    partition = []
    for m in range(M):
        block = [concepts[i] for i in range(N) if labels[i] == m]
        partition.append(block)
    eps = lumpability_error(P, partition, idx)
    # 宏观 EI
    P_macro, _ = build_macro_transition(P, labels, idx, {i: c for i, c in enumerate(concepts)})
    ei_macro = normalized_ei(P_macro, stationary_distribution(P_macro))
    return eps, ei_macro, labels, partition

# ---- 基准：手工语义分组 ----
print("\n[基准] 手工语义分组")
print("-" * 50)
eps_semantic = lumpability_error(P, SEMANTIC_PARTITION, idx)
print(f"  手工语义分组 ε = {eps_semantic:.6f}")

# ---- 基准：纯 SVD 谱嵌入 ----
print("\n[基准] 纯 SVD 谱嵌入")
print("-" * 50)
F = np.diag(pi) @ P
U, s, Vt = svd(F)
svd_emb = Vt[:16].T  # 用前16维（与 W2V 维度对齐，公平对比）
eps_svd, ei_svd, _, _ = kmeans_eps(svd_emb, P, concepts_aligned, idx)
print(f"  SVD 谱嵌入(16d) K-Means ε = {eps_svd:.6f}, EI = {ei_svd:.4f}")

# ---- 基准：纯 W2V（α=1.0，不传播）----
print("\n[基准] 纯 W2V（α=1.0，无传播）")
print("-" * 50)
eps_w2v0, ei_w2v0, _, _ = kmeans_eps(w2v_aligned, P, concepts_aligned, idx)
print(f"  W2V 语义先验 ε = {eps_w2v0:.6f}, EI = {ei_w2v0:.4f}")

# ---- 主实验：图传播不同 α ----
print("\n[主实验] 图传播 α 扫描")
print("-" * 50)
alphas = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []
print(f"  {'α':>5s} {'ε':>10s} {'EI_macro':>10s} {'vs语义ε':>10s}")
print("  " + "-" * 42)
for alpha in alphas:
    E_prop = graph_propagate(w2v_aligned, P, alpha)
    eps, ei, labels, partition = kmeans_eps(E_prop, P, concepts_aligned, idx)
    results.append({'alpha': alpha, 'eps': eps, 'ei': ei})
    diff = eps - eps_semantic
    better = " ↓更好" if eps < eps_semantic else ""
    print(f"  {alpha:>5.1f} {eps:>10.6f} {ei:>10.4f} {diff:>+10.6f}{better}")

# ---- 分析最优 α ----
best = min(results, key=lambda r: r['eps'])
print(f"\n  最优 α = {best['alpha']:.1f}, ε = {best['eps']:.6f}")

# ---- 与基准对比总结 ----
print("\n[汇总] 全部方案成块性 ε 对比")
print("-" * 50)
print(f"  手工语义分组     : {eps_semantic:.6f}")
print(f"  纯 SVD(16d)      : {eps_svd:.6f}")
print(f"  纯 W2V(α=1.0)    : {eps_w2v0:.6f}")
print(f"  图传播最优(α={best['alpha']:.1f}): {best['eps']:.6f}")
print(f"  纯动力学(α=0.0)  : {results[0]['eps']:.6f}")

# ---- 可视化 ----
print("\n[可视化]")
print("-" * 50)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from core.env import CN_FONT
        plt.rcParams['font.family'] = CN_FONT.get_name()
    except Exception:
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # 左：ε vs α 曲线
    ax = axes[0]
    ax.plot([r['alpha'] for r in results], [r['eps'] for r in results],
            'o-', color='#ED7D31', label='图传播 ε')
    ax.axhline(eps_semantic, color='#2E75B6', ls='--', label=f'手工语义 ε ({eps_semantic:.4f})')
    ax.axhline(eps_svd, color='green', ls=':', label=f'纯SVD ε ({eps_svd:.4f})')
    ax.axhline(eps_w2v0, color='red', ls=':', label=f'纯W2V ε ({eps_w2v0:.4f})')
    ax.set_xlabel('α（1=纯语义先验, 0=纯动力学）')
    ax.set_ylabel('成块性误差 ε')
    ax.set_title('图传播 α 扫描：语义先验 vs 动力学')
    ax.grid(True, alpha=0.3); ax.legend()

    # 右：最优 α 的传播后向量 K-Means 聚类（2D 投影）
    from sklearn.decomposition import PCA
    E_best = graph_propagate(w2v_aligned, P, best['alpha'])
    _, _, labels_best, _ = kmeans_eps(E_best, P, concepts_aligned, idx)
    pca = PCA(n_components=2)
    E_2d = pca.fit_transform(E_best)
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    ax = axes[1]
    for m in range(6):
        idx_m = np.where(labels_best == m)[0]
        if len(idx_m) > 0:
            ax.scatter(E_2d[idx_m, 0], E_2d[idx_m, 1], color=colors[m],
                       label=f'簇{m}', s=80, alpha=0.8, edgecolors='black', linewidths=0.4)
    for i, c in enumerate(concepts_aligned):
        ax.annotate(c, (E_2d[i, 0], E_2d[i, 1]), fontsize=8)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'传播后嵌入 K-Means (α={best["alpha"]:.1f})')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "output/vis_12_w2v_mc_back.png"), dpi=150)
    plt.close()
    print("  可视化 → output/vis_12_w2v_mc_back.png")
except Exception as e:
    print(f"  ⚠️ 可视化跳过: {e}")

# ---- 保存结果 ----
result = {
    "baseline": {
        "semantic_eps": float(eps_semantic),
        "svd_16d_eps": float(eps_svd),
        "w2v_no_prop_eps": float(eps_w2v0),
    },
    "alpha_scan": results,
    "best_alpha": best['alpha'],
    "best_eps": best['eps'],
    "conclusion": (
        "图传播通过 α 混合语义先验(E0=W2V)与动力学(P^T)，"
        "最优 α 下成块性 ε 达到 X，与手工语义/SVD/W2V 对比。"
    ),
}
with open(os.path.join(ROOT, "output/w2v_mc_back_results.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\n  结果 → output/w2v_mc_back_results.json")

print("\n" + "=" * 60)
print("P2-A 完成。")
print("=" * 60)
