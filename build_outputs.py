# -*- coding: utf-8 -*-
"""
构建脚本：生成概念网络图和桑基图所需的全部数据
运行 main.py 之后运行此脚本
"""

import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    DAODEJING, build_full_sequence, build_transition_matrix,
    stationary_distribution, build_macro_transition,
    normalized_ei, effective_information, lumpability_error,
    semantic_macro_labels, SEMANTIC_PARTITION, MACRO_NAMES,
    OUTPUT_DIR
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def build_all():
    print("[构建] 生成可视化数据...")
    
    # 数据
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)
    N = P.shape[0]
    
    # ===== M=6 粗粒化：采用手工语义分组（项目最终方案）=====
    # 保证 network/sankey/dashboard 数据与 coarse_grain_v2.py 结论一致
    M = 6
    labels = semantic_macro_labels(idx, inv_idx)
    P_macro, Phi = build_macro_transition(P, labels, idx, inv_idx)
    
    # SVD 谱空间嵌入（仅用于散点图坐标，不参与分组）
    F = np.diag(pi) @ P
    _, s_vals, Vt = np.linalg.svd(F)
    explained = (s_vals[:M] ** 2).sum() / (s_vals ** 2).sum()
    embedding = Vt[:M, :].T  # N×M
    
    # macro_names（使用规范名称）
    macro_names = list(MACRO_NAMES)
    
    # macro_groups（按语义分组）
    macro_groups = {}
    for m, block in enumerate(SEMANTIC_PARTITION):
        macro_groups[m] = [(c, float(pi[idx[c]])) for c in block]
    
    # ===== 1. 网络图数据 (D3.js / Gephi 格式) =====
    nodes = []
    for i in range(N):
        nodes.append({
            'id': inv_idx[i],
            'pi': float(pi[i]),
            'macro': int(labels[i]),
            'degree_in': float(P[:, i].sum()),
            'degree_out': float(P[i, :].sum()),
        })
    
    edges = []
    for i in range(N):
        for j in range(N):
            if P[i, j] >= 0.03 and i != j:
                edges.append({
                    'source': inv_idx[i],
                    'target': inv_idx[j],
                    'weight': float(P[i, j]),
                })
    
    network_data = {
        'nodes': nodes,
        'edges': edges,
        'macro_names': macro_names,
        'macro_groups': {
            str(m): [c for c, _ in sorted(macro_groups.get(m, []), key=lambda x: -x[1])]
            for m in range(M)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'network_data.json'), 'w', encoding='utf-8') as f:
        json.dump(network_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ network_data.json ({len(nodes)} nodes, {len(edges)} edges)")
    
    # ===== 2. 桑基图数据 =====
    sankey_data = {
        'micro_nodes': list(inv_idx.values()),
        'macro_nodes': macro_names,
        'Phi': Phi.tolist(),  # N × M
        'P_macro': P_macro.tolist(),  # M × M
        'pi_micro': pi.tolist(),
        'pi_macro': stationary_distribution(P_macro).tolist(),
    }
    
    with open(os.path.join(OUTPUT_DIR, 'sankey_data.json'), 'w', encoding='utf-8') as f:
        json.dump(sankey_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ sankey_data.json")
    
    # ===== 3. 谱空间数据 =====
    spectral_data = {
        'concepts': list(inv_idx.values()),
        'svd_1': embedding[:, 0].tolist(),
        'svd_2': embedding[:, 1].tolist(),
        'svd_3': embedding[:, 2].tolist() if embedding.shape[1] > 2 else [0.0]*N,
        'labels': labels.tolist(),
        'pi': pi.tolist(),
        'singular_values': [float(s) for s in s_vals[:15]],
    }
    
    with open(os.path.join(OUTPUT_DIR, 'spectral_data.json'), 'w', encoding='utf-8') as f:
        json.dump(spectral_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ spectral_data.json")
    
    # ===== 4. 汇总仪表盘数据 =====
    pi_macro = stationary_distribution(P_macro)
    ei_micro = float(normalized_ei(P, pi))
    ei_macro = float(normalized_ei(P_macro, pi_macro))
    
    dashboard = {
        'basic_info': {
            'N_micro': N,
            'M_macro': M,
            'total_observations': len(full_seq),
            'total_concepts_unique': len(set(full_seq)),
            'chapters': 81,
        },
        'metrics': {
            'micro_EI_raw': float(effective_information(P, pi)),
            'micro_EI_norm': ei_micro,
            'macro_EI_raw': float(effective_information(P_macro, pi_macro)),
            'macro_EI_norm': ei_macro,
            'causal_emergence': ei_macro - ei_micro,
            'explained_variance': float(explained),
            'lumpability_error': float(lumpability_error(P, 
                [[inv_idx[i] for i in range(N) if labels[i]==m] for m in range(M)], idx)),
        },
        'macro_states': [
            {
                'id': m,
                'name': macro_names[m],
                'pi': float(pi_macro[m]),
                'concepts': [
                    {'name': c, 'pi': float(p)}
                    for c, p in sorted(macro_groups.get(m, []), key=lambda x: -x[1])
                ]
            }
            for m in range(M)
        ],
        'transition_micro': P.tolist(),
        'transition_macro': P_macro.tolist(),
        'concept_labels': list(inv_idx.values()),
        'macro_labels': macro_names,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'dashboard_data.json'), 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2)
    print(f"  ✓ dashboard_data.json")
    
    # ===== 打印摘要 =====
    print("\n" + "="*60)
    print("  构建完成！摘要：")
    print("="*60)
    print(f"  N = {N}, M = {M}")
    print(f"  微观 EI = {ei_micro:.4f}")
    print(f"  宏观 EI = {ei_macro:.4f}")
    print(f"  因果涌现 = {ei_macro - ei_micro:+.4f}")
    print(f"  解释方差 = {explained*100:.1f}%")
    print(f"\n  宏观态:")
    for m in range(M):
        print(f"    [{m}] {macro_names[m]} (π={pi_macro[m]:.4f})")
    
    return dashboard

# effective_information / normalized_ei / lumpability_error
# 已统一从 main.py 导入，避免重复定义（见文件顶部 import）

if __name__ == "__main__":
    build_all()
