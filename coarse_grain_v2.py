# -*- coding: utf-8 -*-
"""
改进版粗粒化：
1) 用层次聚类 (Ward) 在 P 的行空间 + 频率向量上做划分，强制 min_size>=2
2) 手工语义分组作为对照（可解释性优先）
3) 对每个 M 计算宏观 EI，绘制因果涌现曲线
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    DAODEJING, build_full_sequence, build_transition_matrix,
    stationary_distribution, effective_information, normalized_ei,
    lumpability_error, OUTPUT_DIR
)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json, os, pandas as pd

# 复用 main.py 的字体检测结果（兼容 Windows / Linux）
from main import CN_FONT, SEMANTIC_PARTITION
plt.rcParams['font.family'] = CN_FONT.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ===== 加载数据 =====
full_seq, chapter_seqs = build_full_sequence(DAODEJING)
P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
pi = stationary_distribution(P)
N = P.shape[0]
concepts = list(inv_idx.values())
print(f"N={N}, 总观测 T={len(full_seq)}")
print(f"微观 EI={normalized_ei(P,pi):.4f}")

# ===== 手工语义分组（先验，可解释性优先）=====
# 定义见 main.py（全链路统一使用，保证导出数据与最终结论一致）
sem_flat = [c for block in SEMANTIC_PARTITION for c in block]
assert set(sem_flat) == set(concepts), f"未覆盖: {set(concepts)-set(sem_flat)}"
assert len(sem_flat) == len(concepts), "重复元素"
print(f"\n✓ 手工分组覆盖全部 {N} 个概念")

# ===== 强制最小簇大小的层次聚类 =====
def ward_partition(min_size=2, K=6):
    """Ward 层次聚类，递归拆分直到每簇 >= min_size"""
    # 特征：P 的行 + pi 向量（拼接）
    feat = np.hstack([P, pi.reshape(-1,1)])
    # 距离矩阵
    D = pdist(feat, metric='cosine')
    Z = linkage(D, method='ward')
    
    # 从下往上剪枝：找满足 min_size 的最细划分
    # 策略：尝试不同距离阈值
    import scipy.cluster.hierarchy as sch
    # 用 fcluster 在不同阈值下搜索
    best_labels = None
    for t in np.linspace(0, Z[-1,2]*0.8, 50):
        labels = fcluster(Z, t=t, criterion='distance')
        sizes = np.bincount(labels)
        if sizes.min() >= min_size and len(sizes) <= K+2:
            best_labels = labels
        else:
            break
    if best_labels is None:
        # 退回：用 K 直接
        from sklearn.cluster import AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=K, linkage='ward')
        best_labels = ac.fit_predict(feat)
    return best_labels

labels_ward = ward_partition(min_size=2, K=6)
K_ward = len(set(labels_ward))
print(f"\nWard 聚类 K={K_ward}, 簇大小: {np.bincount(labels_ward)}")

# ===== 构建宏观转移矩阵（通用函数）=====
def build_macro(P, labels):
    M = len(set(labels))
    Phi = np.zeros((P.shape[0], M))
    for i, l in enumerate(labels):
        Phi[i, l] = 1.0
    Pm_un = Phi.T @ P @ Phi
    rs = Pm_un.sum(axis=1, keepdims=True)
    rs[rs==0] = 1.0
    return Pm_un / rs, Phi

def eval_partition(P, pi, labels, partition, name):
    Pm, Phi = build_macro(P, labels)
    pi_m = stationary_distribution(Pm)
    ei_raw = effective_information(Pm, pi_m)
    ei_norm = ei_raw / max(np.log(Pm.shape[0]), 1e-10)
    lump_err = lumpability_error(P, partition, idx)
    # 解释方差（用 Phi^T @ diag(pi) @ P 的 Frobenius）
    F = np.diag(pi) @ P
    recon = Phi @ Phi.T @ F
    expl = np.linalg.norm(recon, 'fro') / (np.linalg.norm(F, 'fro') + 1e-15)
    print(f"  {name:>20s}: M={len(set(labels)):>2d}  EI_norm={ei_norm:.4f}  lump_err={lump_err:.5f}  recon={expl:.4f}")
    return {'name':name,'M':len(set(labels)),'EI_norm':ei_norm,
            'lump_err':lump_err,'recon':expl,'Pm':Pm,'Phi':Phi,
            'labels':labels,'pi_m':pi_m}

# ===== 评估多种方案 =====
print("\n===== 粗粒化方案对比 =====")
micro_ei = normalized_ei(P, pi)
print(f"  微观基线:           EI_norm={micro_ei:.4f}")

results = {}

# 手工语义分组
sem_labels = np.zeros(N, dtype=int)
for m, block in enumerate(SEMANTIC_PARTITION):
    for c in block:
        sem_labels[idx[c]] = m
results['semantic'] = eval_partition(P, pi, sem_labels, SEMANTIC_PARTITION, "手工语义")

# Ward
ward_partition_list = []
ward_labels_map = {}
for i, l in enumerate(labels_ward):
    ward_labels_map.setdefault(l, []).append(inv_idx[i])
ward_blocks = list(ward_labels_map.values())
results['ward'] = eval_partition(P, pi, labels_ward, ward_blocks, "Ward 层次")

# K-Means (原方案，多 seed)
from sklearn.cluster import KMeans
for K in [4, 5, 6, 7, 8]:
    best_sil = -1
    best_lab = None
    for seed in range(20):
        km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=seed, max_iter=1000)
        lab = km.fit_predict(np.hstack([P, pi.reshape(-1,1)]))
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(np.hstack([P, pi.reshape(-1,1)]), lab)
        if sil > best_sil:
            best_sil = sil
            best_lab = lab
    blocks = [[inv_idx[i] for i in range(N) if best_lab[i]==m] for m in range(K)]
    results[f'kmeans_K{K}'] = eval_partition(P, pi, best_lab, blocks, f"KMeans K={K}")

# ===== 因果涌现曲线 =====
print("\n===== 因果涌现曲线 (手工语义分组，扫 M) =====")
M_range = range(2, min(16, N))
ei_curve = []
for M in M_range:
    # 用 Ward 做 M 簇
    from sklearn.cluster import AgglomerativeClustering
    ac = AgglomerativeClustering(n_clusters=M, linkage='ward')
    lab = ac.fit_predict(np.hstack([P, pi.reshape(-1,1)]))
    Pm = build_macro(P, lab)[0]
    pm = stationary_distribution(Pm)
    ei_curve.append(normalized_ei(Pm, pm))

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 图1: 涌现曲线
ax = axes[0]
ax.axhline(micro_ei, color='red', ls='--', lw=1.5, label=f'微观基线 {micro_ei:.4f}')
ax.plot(list(M_range), ei_curve, 'o-', color='#2E75B6', lw=2, ms=6)
ax.fill_between(list(M_range), ei_curve, alpha=0.1, color='#2E75B6')
best_M = list(M_range)[np.argmax(ei_curve)]
best_v = max(ei_curve)
ax.scatter([best_M], [best_v], color='red', s=200, marker='*', zorder=5)
ax.annotate(f'M*={best_M}\nEI={best_v:.4f}',
            xy=(best_M, best_v), xytext=(best_M+1, best_v*0.9),
            fontsize=11, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel('宏观状态数 M', fontsize=12)
ax.set_ylabel('归一化 EI', fontsize=12)
ax.set_title('因果涌现曲线（Ward 粗粒化）', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# 图2: 各方案 EI 对比
ax = axes[1]
names = list(results.keys())
vals = [results[n]['EI_norm'] for n in names]
colors = ['#ED7D31' if 'semantic' in n else '#5B9BD5' for n in names]
bars = ax.barh(names, vals, color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(micro_ei, color='red', ls='--', lw=1.5, label=f'微观 {micro_ei:.4f}')
for bar, v in zip(bars, vals):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2, f'{v:.4f}',
            va='center', fontsize=9)
ax.set_xlabel('归一化 EI', fontsize=12)
ax.set_title('各粗粒化方案 EI 对比', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='x')

# 图3: 手工语义分组的宏观转移热力图
ax = axes[2]
Pm_sem = results['semantic']['Pm']
pi_m_sem = results['semantic']['pi_m']
sem_names = [f"[{m}]{'+'.join(SEMANTIC_PARTITION[m])}" for m in range(len(SEMANTIC_PARTITION))]
short_names = ['道体论','无为法','辩证法','治术','民知欲','宇宙']
im = ax.imshow(Pm_sem, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=10)
ax.set_yticklabels(short_names, fontsize=10)
for i in range(6):
    for j in range(6):
        if Pm_sem[i,j] > 0.05:
            ax.text(j, i, f'{Pm_sem[i,j]:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if Pm_sem[i,j]>0.5 else 'black')
ax.set_title(f'手工语义分组 P\'(M=6)\nEI={results["semantic"]["EI_norm"]:.4f}  lump_err={results["semantic"]["lump_err"]:.5f}',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'vis_06_emergence_v2.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ vis_06_emergence_v2.png")

# ===== 保存最优方案（手工语义）=====
Pm = results['semantic']['Pm']
Phi = results['semantic']['Phi']
pi_m = results['semantic']['pi_m']
np.save(os.path.join(OUTPUT_DIR, 'P_macro.npy'), Pm)
np.save(os.path.join(OUTPUT_DIR, 'Phi.npy'), Phi)

# 导出综合数据
dashboard = {
    'basic_info': {
        'N_micro': int(N),
        'total_observations': len(full_seq),
        'chapters': 81,
        'concepts': concepts,
    },
    'metrics': {
        'micro_EI_raw': float(effective_information(P, pi)),
        'micro_EI_norm': float(micro_ei),
        'macro_EI_raw': float(effective_information(Pm, pi_m)),
        'macro_EI_norm': float(results['semantic']['EI_norm']),
        'causal_emergence': float(results['semantic']['EI_norm'] - micro_ei),
        'lumpability_error': float(results['semantic']['lump_err']),
    },
    'macro_states': [
        {
            'id': m,
            'name': f"道体论" if m==0 else
                   f"无为法" if m==1 else
                   f"辩证法" if m==2 else
                   f"治术" if m==3 else
                   f"民知欲" if m==4 else f"宇宙",
            'concepts': SEMANTIC_PARTITION[m],
            'pi': float(pi_m[m]),
        }
        for m in range(6)
    ],
    'P_macro': Pm.tolist(),
    'pi_macro': pi_m.tolist(),
    'concepts': concepts,
    'pi_micro': pi.tolist(),
    'P_micro': P.tolist(),
    'macro_names': ['道体论','无为法','辩证法','治术','民知欲','宇宙'],
}
with open(os.path.join(OUTPUT_DIR, 'dashboard_data.json'), 'w', encoding='utf-8') as f:
    json.dump(dashboard, f, ensure_ascii=False, indent=2)

# 打印最终分组
print(f"\n{'='*60}")
print(f"  最终采用：手工语义分组 (M=6)")
print(f"  微观 EI = {micro_ei:.4f}")
print(f"  宏观 EI = {results['semantic']['EI_norm']:.4f}")
print(f"  因果涌现 = {results['semantic']['EI_norm']-micro_ei:+.4f}")
print(f"  成块性误差 = {results['semantic']['lump_err']:.5f}")
print(f"{'='*60}")
for m in range(6):
    names = SEMANTIC_PARTITION[m]
    print(f"  [{m}] {'+'.join(names):<20s}  π'={pi_m[m]:.4f}")

# 打印宏观转移矩阵
print(f"\n宏观转移矩阵 P':")
df_pm = pd.DataFrame(Pm, index=['道体','无为','辩证','治术','民欲','宇宙'],
                       columns=['道体','无为','辩证','治术','民欲','宇宙'])
print(df_pm.round(3).to_string())
