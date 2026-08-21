# -*- coding: utf-8 -*-
"""
辅助脚本：导出可视化所需的全部数据
运行 main.py 之后运行此脚本
"""

import numpy as np
import pandas as pd
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录（core/）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts 目录（main.py）
from main import (
    DAODEJING, build_full_sequence, build_transition_matrix,
    stationary_distribution, build_macro_transition,
    normalized_ei, effective_information, lumpability_error,
    semantic_macro_labels, SEMANTIC_PARTITION, MACRO_NAMES,
    OUTPUT_DIR
)

def export_all():
    """导出所有数据供可视化使用"""
    print("[导出] 加载数据...")
    
    # 构建序列
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    
    # 构建转移矩阵
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)
    
    # ===== 粗粒化：采用手工语义分组（M=6，项目最终方案）=====
    # 注意：不用 SVD+K-Means，保证导出数据与 coarse_grain_v2.py 的最终结论一致
    labels = semantic_macro_labels(idx, inv_idx)
    P_macro, Phi = build_macro_transition(P, labels, idx, inv_idx)
    pi_macro = stationary_distribution(P_macro)
    
    N = P.shape[0]
    M = 6
    macro_names = list(MACRO_NAMES)
    
    # SVD 谱空间嵌入仅用于散点图坐标（不参与分组）
    F = np.diag(pi) @ P
    _, s_vals, Vt = np.linalg.svd(F)
    explained = (s_vals[:M] ** 2).sum() / (s_vals ** 2).sum()
    embedding = Vt[:M, :].T  # N×M
    
    # 构建 macro_groups（按语义分组）
    macro_groups = {}
    for m, block in enumerate(SEMANTIC_PARTITION):
        macro_groups[m] = [(c, float(pi[idx[c]])) for c in block]
    
    # ===== 导出 1: 概念序列 JSON =====
    concept_data = {
        'full_sequence': full_seq,
        'chapter_sequences': {str(k): v for k, v in chapter_seqs.items()},
        'unique_concepts': sorted(list(set(full_seq))),
        'total_concepts': len(full_seq),
    }
    with open(os.path.join(OUTPUT_DIR, 'concept_data.json'), 'w', encoding='utf-8') as f:
        json.dump(concept_data, f, ensure_ascii=False, indent=2)
    print("  ✓ concept_data.json")
    
    # ===== 导出 2: 转移矩阵 CSV =====
    df_P = pd.DataFrame(P, index=list(idx.keys()), columns=list(idx.keys()))
    df_P.to_csv(os.path.join(OUTPUT_DIR, 'transition_matrix.csv'), encoding='utf-8-sig')
    print("  ✓ transition_matrix.csv")
    
    # ===== 导出 3: 平稳分布 CSV =====
    df_pi = pd.DataFrame({
        'concept': list(idx.keys()),
        'pi': pi,
        'index': range(N)
    })
    df_pi.to_csv(os.path.join(OUTPUT_DIR, 'stationary_distribution.csv'),
                  encoding='utf-8-sig', index=False)
    print("  ✓ stationary_distribution.csv")
    
    # ===== 导出 4: SVD 嵌入 CSV =====
    df_svd = pd.DataFrame({
        'concept': list(idx.keys()),
        'svd_1': embedding[:, 0],
        'svd_2': embedding[:, 1],
        'svd_3': embedding[:, 2] if embedding.shape[1] > 2 else [0]*N,
        'macro_label': labels,
        'pi': pi,
    })
    df_svd.to_csv(os.path.join(OUTPUT_DIR, 'svd_embedding.csv'),
                  encoding='utf-8-sig', index=False)
    print("  ✓ svd_embedding.csv")
    
    # ===== 导出 5: 宏观转移矩阵 CSV =====
    df_Pmacro = pd.DataFrame(P_macro, index=macro_names, columns=macro_names)
    df_Pmacro.to_csv(os.path.join(OUTPUT_DIR, 'macro_transition.csv'),
                      encoding='utf-8-sig')
    print("  ✓ macro_transition.csv")
    
    # ===== 导出 6: 宏观分组 JSON =====
    partition_data = {
        'M': M,
        'macro_names': macro_names,
        'groups': {}
    }
    for m in range(M):
        items = sorted(macro_groups.get(m, []), key=lambda x: -x[1])
        partition_data['groups'][str(m)] = {
            'name': macro_names[m],
            'concepts': [{'name': c, 'pi': float(p)} for c, p in items],
            'macro_pi': float(pi_macro[m])
        }
    
    # 添加评估指标
    partition_data['metrics'] = {
        'micro_EI': float(effective_information(P, pi)),
        'micro_norm_EI': float(normalized_ei(P, pi)),
        'macro_EI': float(effective_information(P_macro, pi_macro)),
        'macro_norm_EI': float(normalized_ei(P_macro, pi_macro)),
        'explained_variance': float(explained),
        'singular_values': [float(s) for s in s_vals[:10]],
    }
    
    with open(os.path.join(OUTPUT_DIR, 'coarse_graining.json'), 'w', encoding='utf-8') as f:
        json.dump(partition_data, f, ensure_ascii=False, indent=2)
    print("  ✓ coarse_graining.json")
    
    # ===== 导出 7: 主题河流数据 CSV =====
    # 概念 → 宏观态 映射（一次性构建）
    concept_to_macro = {inv_idx[i]: labels[i] for i in range(N)}
    river_data = []
    for ch in sorted(chapter_seqs.keys()):
        seq = chapter_seqs[ch]
        counts = np.zeros(M)
        for concept in seq:
            if concept in concept_to_macro:
                counts[concept_to_macro[concept]] += 1
        total = counts.sum()
        if total > 0:
            counts = counts / total
        for m in range(M):
            river_data.append({
                'chapter': ch,
                'macro_state': m,
                'macro_name': macro_names[m],
                'density': counts[m]
            })
    
    df_river = pd.DataFrame(river_data)
    df_river.to_csv(os.path.join(OUTPUT_DIR, 'theme_river.csv'),
                      encoding='utf-8-sig', index=False)
    print("  ✓ theme_river.csv")
    
    # ===== 汇总报告 =====
    print("\n" + "="*60)
    print("  数据导出完成！")
    print("="*60)
    print(f"\n  N (微观状态数) = {N}")
    print(f"  M (宏观状态数) = {M}")
    print(f"  总概念观测数 = {len(full_seq)}")
    print(f"  微观 EI = {partition_data['metrics']['micro_norm_EI']:.4f}")
    print(f"  宏观 EI = {partition_data['metrics']['macro_norm_EI']:.4f}")
    print(f"  因果涌现 = {partition_data['metrics']['macro_norm_EI'] - partition_data['metrics']['micro_norm_EI']:+.4f}")
    print(f"  解释方差 = {explained*100:.1f}%")
    print(f"\n  宏观分组:")
    for m in range(M):
        print(f"    [{m}] {macro_names[m]} (π={pi_macro[m]:.4f})")
    
    return partition_data

if __name__ == "__main__":
    export_all()
