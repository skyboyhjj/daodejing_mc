# -*- coding: utf-8 -*-
"""诊断聚类质量 + 测试不同 K 的 silhouette 分数"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录（core/）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts 目录（main.py）
from main import (
    DAODEJING, build_full_sequence, build_transition_matrix,
    stationary_distribution, OUTPUT_DIR
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

P, C, idx, inv_idx = build_transition_matrix(
    build_full_sequence(DAODEJING)[0], k=1)
pi = stationary_distribution(P)
F = np.diag(pi) @ P
_, s, Vt = np.linalg.svd(F)
N = P.shape[0]
concepts = list(inv_idx.values())
print(f"N={N}, concepts: {concepts}\n")

# 测试不同 K 和不同初始化
print("="*70)
print(f"{'K':>3} | {'init':<10} | {'seed':>5} | {'silhouette':>10} | {'min_cluster':>10} | {'max_cluster':>10}")
print("="*70)

results = []
for K in [4, 5, 6, 7, 8]:
    best = None
    for init in ['k-means++', 'random']:
        for seed in range(5):
            km = KMeans(n_clusters=K, init=init, n_init=20 if init=='k-means++' else 1,
                        random_state=seed, max_iter=1000, tol=1e-6)
            emb = Vt[:K, :].T
            labels = km.fit_predict(emb)
            sil = silhouette_score(emb, labels) if len(set(labels))>1 else -1
            sizes = np.bincount(labels, minlength=K)
            row = (K, init, seed, sil, sizes.min(), sizes.max(), labels)
            results.append(row)
            marker = " ← best so far" if (best is None or sil > best[3]) else ""
            if best is None or sil > best[3]:
                best = row
            print(f"{K:>3} | {init:<10} | {seed:>5} | {sil:>10.4f} | {sizes.min():>10d} | {sizes.max():>10d}{marker}")

# 选最优
results.sort(key=lambda r: -r[3])
top = results[0]
K_opt, init_opt, seed_opt, sil_opt, _, _, labels_opt = top
print(f"\n✓ 最优: K={K_opt}, init={init_opt}, seed={seed_opt}, silhouette={sil_opt:.4f}")

# 打印分组
print(f"\n--- 最优分组 (K={K_opt}) ---")
emb = Vt[:K_opt, :].T
km = KMeans(n_clusters=K_opt, init=init_opt, n_init=20 if init_opt=='k-means++' else 1,
            random_state=seed_opt, max_iter=1000, tol=1e-6)
labels = km.fit_predict(emb)
for m in range(K_opt):
    members = [(concepts[i], float(pi[i])) for i in range(N) if labels[i]==m]
    members.sort(key=lambda x:-x[1])
    names = ", ".join([f"{c}({p:.4f})" for c,p in members])
    print(f"  [{m}] ({len(members)}个) {names}")

# 保存最优标签
np.save(os.path.join(OUTPUT_DIR, 'best_labels.npy'), labels)
with open(os.path.join(OUTPUT_DIR, 'best_config.json'), 'w') as f:
    import json
    json.dump({'K':int(K_opt),'init':init_opt,'seed':int(seed_opt),
                'silhouette':float(sil_opt)}, f)
print(f"\n✓ 已保存 best_labels.npy (K={K_opt})")
