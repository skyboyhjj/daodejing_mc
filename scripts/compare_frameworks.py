# -*- coding: utf-8 -*-
"""
对比分析：默认义理分组 vs 网页主题框架分组

背景：用户询问能否借鉴 hui-skill.org《道德经》8 大板块主题
（道体论→辩证法→修身论→无为论→治国论 的"由体达用"认知层级）
来优化宏观态分组。

对比 2 种方案：
  A. 当前默认方案（义理类别导向，M=6）
  B. 网页上经认知层级方案（M=6，由体达用）—— SEMANTIC_PARTITION_WEB

评估指标：
  1. 成块性误差 ε（越小越自洽）
  2. 宏观 EI / 因果涌现
  3. 分组可解释性

运行：python scripts/compare_frameworks.py
"""

import os
import sys

# 【脚本位于 scripts/】项目根目录（core/）= 上一级；本脚本目录（main.py）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)          # core/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # main.py
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np

from core.env import setup_env
from core.pipeline import (
    build_transition_matrix, stationary_distribution,
    build_macro_transition, normalized_ei, lumpability_error,
    semantic_macro_labels,
)
from main import (
    DAODEJING, build_full_sequence,
    SEMANTIC_PARTITION, MACRO_NAMES,
    SEMANTIC_PARTITION_WEB, MACRO_NAMES_WEB,
)

setup_env()


def load_data():
    """加载全书序列与转移矩阵"""
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)
    return P, pi, idx, inv_idx


def evaluate(name, partition, macro_names, P, pi, idx, inv_idx):
    """评估一个分组的成块性与因果涌现"""
    labels = semantic_macro_labels(idx, inv_idx, partition)
    M = len(macro_names)

    # 构建 partition（用于成块性检验）
    blocks = []
    for m in range(M):
        blocks.append([inv_idx[i] for i in range(len(inv_idx)) if labels[i] == m])

    eps = lumpability_error(P, blocks, idx)
    P_macro, _ = build_macro_transition(P, labels, idx, inv_idx)
    pi_macro = stationary_distribution(P_macro)
    ei_micro = normalized_ei(P, pi)
    ei_macro = normalized_ei(P_macro, pi_macro)

    print(f"\n{'='*60}")
    print(f"【{name}】M={M}")
    print(f"{'='*60}")
    for m in range(M):
        block = [inv_idx[i] for i in range(len(inv_idx)) if labels[i] == m]
        print(f"  [{m}] {macro_names[m]}: {block}")
    print(f"  ---")
    print(f"  微观 EI_norm = {ei_micro:.4f}")
    print(f"  宏观 EI_norm = {ei_macro:.4f}")
    print(f"  因果涌现     = {ei_macro - ei_micro:+.4f}")
    print(f"  成块性误差 ε = {eps:.6f}")

    return {'ei_macro': ei_macro, 'emergence': ei_macro - ei_micro, 'eps': eps}


def main():
    print("=" * 60)
    print("  分组方案对比：默认义理 vs 网页主题框架")
    print("=" * 60)

    P, pi, idx, inv_idx = load_data()
    print(f"N = {P.shape[0]} 概念, 微观 EI_norm = {normalized_ei(P, pi):.4f}")

    # 方案 A：默认
    res_a = evaluate("A. 默认义理类别", SEMANTIC_PARTITION, MACRO_NAMES,
                     P, pi, idx, inv_idx)

    # 方案 B：网页框架
    res_b = evaluate("B. 网页主题框架（由体达用）", SEMANTIC_PARTITION_WEB, MACRO_NAMES_WEB,
                     P, pi, idx, inv_idx)

    # 汇总
    print("\n" + "=" * 60)
    print("  汇总对比")
    print("=" * 60)
    print(f"  {'方案':<16s} {'宏观EI':>8s} {'涌现':>8s} {'ε':>8s}")
    print("  " + "-" * 44)
    print(f"  {'A 默认':<16s} {res_a['ei_macro']:.4f}   {res_a['emergence']:+.4f}  {res_a['eps']:.6f}")
    print(f"  {'B 网页框架':<16s} {res_b['ei_macro']:.4f}   {res_b['emergence']:+.4f}  {res_b['eps']:.6f}")
    print(f"\n  结论：")
    print(f"    ε 变化: {res_a['eps']:.6f} → {res_b['eps']:.6f} "
          f"({(res_b['eps']-res_a['eps'])/res_a['eps']*100:+.1f}%)")
    print(f"    宏观 EI: {res_a['ei_macro']:.4f} → {res_b['ei_macro']:.4f} "
          f"({(res_b['ei_macro']-res_a['ei_macro']):+.4f})")
    if res_b['eps'] < res_a['eps']:
        print("    网页框架成块性更优（ε 更低，分组更自洽）")
    if res_b['emergence'] > res_a['emergence']:
        print("    网页框架因果涌现更强")


if __name__ == "__main__":
    main()