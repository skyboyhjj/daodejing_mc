# -*- coding: utf-8 -*-
"""
对比分析：默认义理分组 vs 网页主题框架分组

背景：用户询问能否借鉴 hui-skill.org《道德经》8 大板块主题
（道体论→辩证法→修身论→无为论→治国论 的"由体达用"认知层级）
来优化宏观态分组。

对比 3 种方案：
  A. 当前默认方案（义理类别导向，M=6）
  B. 网页上经认知层级方案（M=6，由体达用）—— SEMANTIC_PARTITION_WEB

评估指标：
  1. 成块性误差 ε（越小越自洽）
  2. 宏观 EI / 因果涌现
  3. 分组可解释性

运行：python compare_frameworks.py
"""

import os
import sys

# 【脚本位于 scripts/】项目根目录（core/）= 上一级；本脚本目录（main.py）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # core/
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
    semantic_macro_labels as _semantic_labels_core,
)
from main import (
    DAODEJING, build_full_sequence,
    SEMANTIC_PARTITION, MACRO_NAMES,
    SEMANTIC_PARTITION_WEB, MACRO_NAMES_WEB,
    SEMANTIC_PARTITION_M12, MACRO_NAMES_M12,
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
    # 用 core.pipeline 底层函数（接受 partition 参数），而非 main 的 mode 接口
    labels = _semantic_labels_core(idx, inv_idx, partition)
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
    print("  分组方案对比：默认义理 / 网页框架 / 细粒度 M=12")
    print("=" * 60)

    P, pi, idx, inv_idx = load_data()
    print(f"N = {P.shape[0]} 概念, 微观 EI_norm = {normalized_ei(P, pi):.4f}")

    # 方案 A：默认 m6
    res_a = evaluate("A. 默认义理类别 (M=6)", SEMANTIC_PARTITION, MACRO_NAMES,
                     P, pi, idx, inv_idx)

    # 方案 B：网页框架
    res_b = evaluate("B. 网页主题框架 (M=6)", SEMANTIC_PARTITION_WEB, MACRO_NAMES_WEB,
                     P, pi, idx, inv_idx)

    # 方案 C：细粒度 M=12
    res_c = evaluate("C. 细粒度子主题 (M=12)", SEMANTIC_PARTITION_M12, MACRO_NAMES_M12,
                     P, pi, idx, inv_idx)

    # 汇总
    print("\n" + "=" * 60)
    print("  汇总对比")
    print("=" * 60)
    print(f"  {'方案':<16s} {'M':>3s} {'宏观EI':>8s} {'涌现':>8s} {'ε':>8s}")
    print("  " + "-" * 48)
    print(f"  {'A 默认':<16s} {6:>3d} {res_a['ei_macro']:.4f}   {res_a['emergence']:+.4f}  {res_a['eps']:.6f}")
    print(f"  {'B 网页框架':<16s} {6:>3d} {res_b['ei_macro']:.4f}   {res_b['emergence']:+.4f}  {res_b['eps']:.6f}")
    print(f"  {'C 细粒度M12':<16s} {12:>3d} {res_c['ei_macro']:.4f}   {res_c['emergence']:+.4f}  {res_c['eps']:.6f}")
    print(f"\n  结论：")
    print(f"    M=12 宏观 EI: {res_a['ei_macro']:.4f} → {res_c['ei_macro']:.4f} "
          f"({(res_c['ei_macro']-res_a['ei_macro']):+.4f})")
    print(f"    M=12 因果涌现: {res_a['emergence']:+.4f} → {res_c['emergence']:+.4f}")
    if res_c['emergence'] > res_a['emergence']:
        print("    M=12 细粒度分组的因果涌现更强（更细子主题保留更多信息）")
    if res_c['eps'] > res_a['eps']:
        print(f"    M=12 成块性 ε 略升 ({res_a['eps']:.6f}→{res_c['eps']:.6f})，但可接受")


if __name__ == "__main__":
    main()
