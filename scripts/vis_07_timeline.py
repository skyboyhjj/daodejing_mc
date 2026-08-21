# -*- coding: utf-8 -*-
"""
T04: 概念时间线可视化（第 7 种可视化）
X 轴 = 81 章，Y 轴 = 概念，散点大小 ∝ 该章出现次数
颜色 = 所属宏观态（手工语义分组 M=6）
额外绘制每个概念的"首次出现 → 末次出现"跨度线

观察目标：
  - "道体论"概念（道/德/无/有...）是否集中在前半部
  - "治术"概念是否在后半部崛起
  - 每个概念在全书中的生命跨度

依赖: numpy, pandas, matplotlib
运行: python vis_07_timeline.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录（core/）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts 目录（main.py）
from main import (
    DAODEJING, build_full_sequence, build_transition_matrix,
    stationary_distribution, semantic_macro_labels, SEMANTIC_PARTITION,
    MACRO_NAMES, OUTPUT_DIR
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 数据准备
# ============================================================
def load_data():
    """加载概念序列与宏观分组"""
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)

    # 每个概念 → 宏观态
    labels = semantic_macro_labels(idx, inv_idx)
    concept_to_macro = {inv_idx[i]: int(labels[i]) for i in range(len(inv_idx))}

    # 每个概念 → 出现章节及次数
    concept_chapters = {}   # concept -> {chapter: count}
    for ch, seq in chapter_seqs.items():
        for c in seq:
            concept_chapters.setdefault(c, {})
            concept_chapters[c][ch] = concept_chapters[c].get(ch, 0) + 1

    return full_seq, chapter_seqs, pi, concept_to_macro, concept_chapters


def first_last_span(concept_chapters):
    """计算每个概念的首次/末次出现章节"""
    spans = {}
    for c, ch_map in concept_chapters.items():
        chapters = sorted(ch_map.keys())
        spans[c] = (chapters[0], chapters[-1], len(chapters))
    return spans


# ============================================================
# 绘图
# ============================================================
def plot_timeline(concept_to_macro, concept_chapters):
    """概念时间线散点图"""
    print("  [1/2] 绘制概念时间线...")
    concepts = sorted(concept_chapters.keys(),
                      key=lambda c: (concept_to_macro[c], first_last_span(concept_chapters)[c][0]))
    spans = first_last_span(concept_chapters)

    # 宏观态颜色
    macro_colors = plt.cm.Set2(np.linspace(0, 1, len(MACRO_NAMES)))
    color_by_macro = {m: macro_colors[m] for m in range(len(MACRO_NAMES))}

    fig, ax = plt.subplots(figsize=(20, 12))

    # 每个概念一行：先画跨度线，再画散点
    for y, c in enumerate(concepts):
        m = concept_to_macro[c]
        ch_first, ch_last, n_ch = spans[c]
        # 跨度线（首次 → 末次）
        ax.plot([ch_first, ch_last], [y, y], color=color_by_macro[m],
                alpha=0.35, lw=3, solid_capstyle='round', zorder=1)
        # 散点（每个出现章节）
        for ch, cnt in sorted(concept_chapters[c].items()):
            ax.scatter(ch, y, s=40 + cnt * 28, c=[color_by_macro[m]],
                       edgecolors='black', linewidths=0.4, alpha=0.9, zorder=2)

    # 分隔宏观态的参考线 + 名称标注
    macro_ranges = {}
    for y, c in enumerate(concepts):
        m = concept_to_macro[c]
        macro_ranges.setdefault(m, [y, y])
        macro_ranges[m][1] = y
    for m, (y0, y1) in macro_ranges.items():
        mid = (y0 + y1) / 2
        ax.text(82.5, mid, MACRO_NAMES[m], va='center', fontsize=11,
                fontweight='bold', color=color_by_macro[m])

    ax.set_xlim(0, 88)
    ax.set_ylim(-0.6, len(concepts) - 0.4)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=10)
    ax.set_xticks(range(1, 82, 4))
    ax.set_xticklabels([f'第{i}章' for i in range(1, 82, 4)],
                       rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('章节（第 1-81 章）', fontsize=13)
    ax.set_ylabel('概念', fontsize=13)
    ax.set_title('《道德经》概念时间线\n点大小 ∝ 该章出现次数 | 颜色 = 宏观义理 | 线 = 概念生命跨度',
                 fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.2, ls='--')

    # 图例（宏观态）
    legend_handles = [Patch(facecolor=color_by_macro[m], alpha=0.8, label=MACRO_NAMES[m])
                      for m in range(len(MACRO_NAMES))]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=11, ncol=2, framealpha=0.95)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'vis_07_timeline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {os.path.basename(path)}")
    return path


# ============================================================
# 量化分析
# ============================================================
def macro_span_stats(concept_to_macro, concept_chapters):
    """统计每个宏观态概念的首次出现章节分布"""
    spans = first_last_span(concept_chapters)
    stats = {}
    for m, name in enumerate(MACRO_NAMES):
        members = [c for c, mm in concept_to_macro.items() if mm == m]
        firsts = [spans[c][0] for c in members]
        lasts = [spans[c][1] for c in members]
        stats[name] = {
            '概念数': len(members),
            '平均首次出现章': np.mean(firsts),
            '平均末次出现章': np.mean(lasts),
            '平均生命跨度(章)': np.mean([spans[c][2] for c in members]),
        }
    return stats, spans


def print_stats(stats):
    """打印宏观态出现章节统计"""
    print("\n  [2/2] 宏观态出现章节统计:")
    print(f"  {'宏观态':<8s} {'概念数':>4s} {'平均首现章':>8s} {'平均末现章':>8s} {'平均跨度章':>8s}")
    print("  " + "-" * 44)
    for name, s in stats.items():
        print(f"  {name:<8s} {s['概念数']:>4d} {s['平均首次出现章']:>8.1f} "
              f"{s['平均末次出现章']:>8.1f} {s['平均生命跨度(章)']:>8.1f}")

    # 关键结论
    early = min(stats.items(), key=lambda kv: kv[1]['平均首次出现章'])
    late = max(stats.items(), key=lambda kv: kv[1]['平均首次出现章'])
    print(f"\n  最早出现的前部义理: {early[0]}（平均第 {early[1]['平均首次出现章']:.1f} 章）")
    print(f"  最晚出现的后部义理: {late[0]}（平均第 {late[1]['平均首次出现章']:.1f} 章）")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  T04: 概念时间线可视化")
    print("=" * 60)

    full_seq, chapter_seqs, pi, concept_to_macro, concept_chapters = load_data()
    print(f"  N = {len(concept_chapters)} 个概念, 81 章")

    path = plot_timeline(concept_to_macro, concept_chapters)

    stats, spans = macro_span_stats(concept_to_macro, concept_chapters)
    print_stats(stats)

    # 保存统计数据供其他分析使用
    df = pd.DataFrame([
        {'concept': c, 'macro_state': concept_to_macro[c],
         'macro_name': MACRO_NAMES[concept_to_macro[c]],
         'first_chapter': spans[c][0], 'last_chapter': spans[c][1],
         'n_chapters': spans[c][2]}
        for c in spans
    ])
    df.to_csv(os.path.join(OUTPUT_DIR, 'concept_timeline.csv'),
              index=False, encoding='utf-8-sig')
    print(f"\n  ✓ concept_timeline.csv 已保存")

    print(f"\n{'='*60}")
    print(f"  ✓ 概念时间线完成: {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
