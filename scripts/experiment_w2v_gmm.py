"""
experiment_w2v_gmm.py
P1-B：在 W2V 词嵌入上执行高斯混合模型（GMM）软聚类，计算宏观 EI。
对比 HMM 软分配 EI=0.0539，验证 W2V-GMM 能否进一步提升。

软分配口径（与 hmm_analysis.py 的 soft_macro_from_hmm 完全一致）：
  - GMM 后验 gamma = 每个概念在 M 个高斯上的后验概率 (N×M)
  - 软宏观转移：对序列每个位置 t，用 gamma[概念_t] 作为该位置软状态
    P_soft_count += outer(gamma_cur, gamma_nxt)
  - 归一化为行随机矩阵，用 normalized_ei 计算

依赖：numpy, scipy, scikit-learn (GaussianMixture), matplotlib
用法：python scripts/experiment_w2v_gmm.py
"""
import numpy as np
import pandas as pd
import json, os, sys
from scipy.stats import kendalltau
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

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

# 复用 core 的转移/EI 函数
from core.pipeline import (
    build_transition_matrix, stationary_distribution,
    normalized_ei, build_macro_transition,
)
from main import DAODEJING, build_full_sequence


def soft_macro_ei(soft_Phi, P, full_seq, concepts, idx):
    """用软分配矩阵 soft_Phi (N×M) 计算宏观 EI（口径与 HMM 一致）。
    soft_Phi[j, m] = 概念 j 属于宏观态 m 的概率。"""
    N = P.shape[0]
    M = soft_Phi.shape[1]

    # 序列位置级软分配：位置 t 的概念 x → gamma[x]
    # 构造软宏观转移矩阵
    P_soft_count = np.zeros((M, M))
    P_soft_from = np.zeros(M)
    for t in range(len(full_seq) - 1):
        cur = idx[full_seq[t]]
        nxt = idx[full_seq[t + 1]]
        gamma_cur = soft_Phi[cur]
        gamma_nxt = soft_Phi[nxt]
        P_soft_count += np.outer(gamma_cur, gamma_nxt)
        P_soft_from += gamma_cur

    P_soft = np.zeros((M, M))
    for i in range(M):
        if P_soft_from[i] > 1e-12:
            P_soft[i] = P_soft_count[i] / P_soft_from[i]
        else:
            P_soft[i] = np.ones(M) / M

    pi_soft = stationary_distribution(P_soft)
    ei_soft = normalized_ei(P_soft, pi_soft)
    return P_soft, pi_soft, ei_soft


def run_gmm_ei(w2v, P, full_seq, concepts, idx, M, n_init=20, seed=42):
    """在 W2V 嵌入上拟合 GMM(M 组件)，返回软分配 + 宏观 EI"""
    gmm = GaussianMixture(n_components=M, covariance_type='full',
                          random_state=seed, n_init=n_init, max_iter=300)
    gmm.fit(w2v)
    # 后验概率：每个概念在 M 个高斯上的归属概率 (N×M)
    soft_Phi = gmm.predict_proba(w2v)
    P_soft, pi_soft, ei = soft_macro_ei(soft_Phi, P, full_seq, concepts, idx)
    return gmm, soft_Phi, P_soft, ei


print("=" * 60)
print("P1-B：W2V 嵌入上的 GMM 软聚类 → 宏观 EI")
print("=" * 60)

# ---- 加载数据 ----
P = np.load(os.path.join(ROOT, "output/P_matrix.npy"))
pi = np.load(os.path.join(ROOT, "output/pi.npy"))
w2v = np.load(os.path.join(ROOT, "output/w2v_vectors.npy"))
N = P.shape[0]

# 概念列表
cg_path = os.path.join(ROOT, "output/coarse_graining.json")
concepts = [f"C{i}" for i in range(N)]
if os.path.exists(cg_path):
    with open(cg_path, "r", encoding="utf-8") as f:
        cg = json.load(f)
    grp = cg.get("groups", {})
    tmp = []
    for g in grp.values():
        for item in g.get("concepts", []):
            tmp.append(item["name"])
    if len(tmp) == N:
        concepts = tmp

# 序列（用于软转移矩阵构造）
full_seq, _ = build_full_sequence(DAODEJING)
idx = {c: i for i, c in enumerate(sorted(set(full_seq)))}

print(f"  N = {N} 概念, W2V 维度 = {w2v.shape[1]}")
print(f"  序列长度 T = {len(full_seq)}")
print(f"  微观 EI_norm = {normalized_ei(P, pi):.4f}")

# ---- 参考值 ----
# HMM 软分配 EI（来自 hmm_analysis.py 实测）
HMM_EI = 0.0539
# 硬划分 EI（M=6 手工语义）
print(f"  [参考] HMM 软分配 EI = {HMM_EI:.4f}")

# ---- GMM 软聚类：扫描不同 M ----
print("\n[实验] GMM 软聚类 EI 扫描 (M=2..12)")
print("-" * 50)
results = {}
for M in range(2, 13):
    gmm, soft_Phi, P_soft, ei = run_gmm_ei(w2v, P, full_seq, concepts, idx, M)
    results[M] = ei
    marker = " ★" if ei > HMM_EI else ""
    print(f"  M={M:>2d}: GMM-EI = {ei:.4f}  涌现 = {ei - 0.0515:+.4f}{marker}")

# ---- 主实验：M=6（与 HMM 对齐）----
print("\n[主实验] M=6 GMM 软聚类（与 HMM 对齐）")
print("-" * 50)
gmm, soft_Phi, P_soft, ei_gmm6 = run_gmm_ei(w2v, P, full_seq, concepts, idx, M=6)

print(f"  GMM(M=6) 宏观 EI_norm = {ei_gmm6:.4f}")
print(f"  HMM 软分配 EI_norm    = {HMM_EI:.4f}")
print(f"  GMM - HMM 差异        = {ei_gmm6 - HMM_EI:+.4f}")
print(f"  GMM 涌现              = {ei_gmm6 - 0.0515:+.4f}")
if ei_gmm6 > HMM_EI:
    print("  → W2V-GMM 软聚类 **超越** HMM 软分配 ✅")
else:
    print("  → W2V-GMM 软聚类未超越 HMM（原因分析见报告）")

# ---- 与 GMM 硬划分对比 ----
print("\n[对照] GMM 硬划分（argmax）vs 软分配")
print("-" * 50)
hard_labels = soft_Phi.argmax(axis=1)
P_hard, _ = build_macro_transition(P, hard_labels, idx, {i: c for i, c in enumerate(concepts)})
pi_hard = stationary_distribution(P_hard)
ei_hard_gmm = normalized_ei(P_hard, pi_hard)
print(f"  GMM 硬划分 EI（硬EI公式） = {ei_hard_gmm:.4f}")
print(f"  GMM 软分配 EI            = {ei_gmm6:.4f}")

# ---- 软效应隔离分析（诚实揭示"软"的真实贡献）----
print("\n[软效应隔离分析] '软分配'本身的贡献")
print("-" * 50)
# 后验熵：0=完全硬，越大越软
entropy = -np.sum(soft_Phi * np.log(soft_Phi + 1e-15), axis=1)
print(f"  平均后验熵 = {entropy.mean():.4f}（0=完全硬，越大越软）")
print(f"  最大后验概率均值 = {soft_Phi.max(axis=1).mean():.4f}")
# 硬划分（one-hot）改用软EI公式
onehot = np.eye(M)[hard_labels]
ei_hard_softformula = soft_macro_ei(onehot, P, full_seq, concepts, idx)[2]
# 软效应 = 真实软 vs one-hot 硬（都用力软EI公式）
soft_effect = ei_gmm6 - ei_hard_softformula
formula_effect = ei_hard_softformula - ei_hard_gmm
print(f"  公式效应（硬划分改用软EI公式）= {formula_effect:+.4f}")
print(f"  软效应（真实软 vs one-hot硬） = {soft_effect:+.4f}")
print(f"  → 若软效应≈0，说明'软聚类'本身未贡献，提升来自聚类结构与EI公式")

# ---- 每个概念的软分配（Top 归属）----
print("\n[输出] 每个概念的软分配主导状态")
print("-" * 50)
for j, c in enumerate(concepts):
    top_m = soft_Phi[j].argmax()
    top_p = soft_Phi[j][top_m]
    print(f"  {c:>5s}: 主状态 {top_m} (p={top_p:.2f})")

# ---- 保存 ----
np.save(os.path.join(ROOT, "output/gmm_soft_phi.npy"), soft_Phi)
np.save(os.path.join(ROOT, "output/gmm_P_soft.npy"), P_soft)

result = {
    "micro_EI": float(normalized_ei(P, pi)),
    "HMM_soft_EI": HMM_EI,
    "GMM_soft_EI_m6": float(ei_gmm6),
    "GMM_hard_EI_m6": float(ei_hard_gmm),
    "GMM_vs_HMM_diff": float(ei_gmm6 - HMM_EI),
    "GMM_emergence": float(ei_gmm6 - 0.0515),
    "GMM_posterior_entropy_mean": float(entropy.mean()),
    "GMM_formula_effect": float(formula_effect),
    "GMM_soft_effect": float(soft_effect),
    "scan_m2_12": {str(k): float(v) for k, v in results.items()},
    "soft_Phi_dominant_state": {c: int(soft_Phi[j].argmax()) for j, c in enumerate(concepts)},
}
with open(os.path.join(ROOT, "output/w2v_gmm_experiment_results.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\n  结果 → output/w2v_gmm_experiment_results.json")

# ---- 可视化 ----
print("\n[可视化] GMM 软聚类")
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
    # 左：EI vs M 曲线（GMM 软 vs HMM 参考）
    ms = sorted(results.keys())
    eis = [results[m] for m in ms]
    ax = axes[0]
    ax.plot(ms, eis, 'o-', color='#ED7D31', label='GMM 软聚类')
    ax.axhline(HMM_EI, color='#2E75B6', ls='--', label=f'HMM 软分配 ({HMM_EI:.4f})')
    ax.axhline(0.0515, color='green', ls=':', label=f'微观 ({0.0515:.4f})')
    ax.set_xlabel('M (宏观态数)'); ax.set_ylabel('宏观 EI_norm')
    ax.set_title('GMM 软聚类 EI vs 宏观态数')
    ax.grid(True, alpha=0.3); ax.legend()

    # 右：M=6 软分配（PCA 投影 + 主导状态着色）
    pca = PCA(n_components=2)
    w2v_2d = pca.fit_transform(w2v)
    dominant = soft_Phi.argmax(axis=1)
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    ax = axes[1]
    for m in range(6):
        idx_m = np.where(dominant == m)[0]
        if len(idx_m) > 0:
            ax.scatter(w2v_2d[idx_m, 0], w2v_2d[idx_m, 1], color=colors[m],
                       label=f'状态{m}', s=80, alpha=0.8, edgecolors='black', linewidths=0.4)
    for i, c in enumerate(concepts):
        ax.annotate(c, (w2v_2d[i, 0], w2v_2d[i, 1]), fontsize=8)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('W2V 嵌入 + GMM 软聚类主导状态（M=6）')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "output/vis_11_gmm_soft.png"), dpi=150)
    plt.close()
    print("  可视化 → output/vis_11_gmm_soft.png")
except Exception as e:
    print(f"  ⚠️ 可视化跳过: {e}")

print("\n" + "=" * 60)
print(f"P1-B 完成。GMM(M=6) EI = {ei_gmm6:.4f} vs HMM EI = {HMM_EI:.4f}")
print("=" * 60)
