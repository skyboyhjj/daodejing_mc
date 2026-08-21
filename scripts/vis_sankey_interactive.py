# -*- coding: utf-8 -*-
"""
T03: 桑基图升级 - plotly 交互式 HTML + 静态 PNG
- HTML：浏览器可交互（悬停查看数值、拖动节点重排）
- PNG：kaleido 导出（如已安装 kaleido + Chrome）
- matplotlib 静态版由 run_all_visualizations.py 生成（vis_04_sankey.png）

依赖: numpy, plotly, kaleido（可选，用于导出 PNG）
运行: python vis_sankey_interactive.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录（core/）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts 目录（main.py）
from main import (
    DAODEJING, build_full_sequence, build_transition_matrix,
    stationary_distribution, build_macro_transition,
    semantic_macro_labels, SEMANTIC_PARTITION, MACRO_NAMES, OUTPUT_DIR
)

# 【重构 T12】UTF-8 输出已由 main.py → core.env 统一配置，此处无需重复。


def main():
    print("=" * 60)
    print("  T03: 桑基图升级 - plotly 交互式 HTML")
    print("=" * 60)

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [跳过] plotly 不可用")
        return

    # 数据
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)
    N = P.shape[0]
    M = 6

    labels = semantic_macro_labels(idx, inv_idx)
    P_macro, Phi = build_macro_transition(P, labels, idx, inv_idx)

    # ---- 节点 ----
    micro_labels = [inv_idx[i] for i in range(N)]
    node_labels = micro_labels + list(MACRO_NAMES)
    node_colors = ['#5B9BD5'] * N + ['#ED7D31'] * M

    # ---- 微观→宏观 流（value 用真实平稳概率 π，hover 显示真实含义）----
    sources_micro, targets_micro, values_micro = [], [], []
    customdata_micro, hovertext_micro = [], []
    for i in range(N):
        j = int(labels[i])
        sources_micro.append(i)
        targets_micro.append(N + j)
        values_micro.append(float(pi[i]))  # 真实 π，便于 hover 展示
        customdata_micro.append([micro_labels[i], MACRO_NAMES[j], round(float(pi[i]), 4)])
        hovertext_micro.append(
            f"<b>{micro_labels[i]}</b> → <b>{MACRO_NAMES[j]}</b><br>"
            f"平稳概率 π = {pi[i]:.4f}<br>"
            f"(第 {int(labels[i])} 宏观态 · 手工语义分组)"
        )

    # ---- 宏观→宏观 流（value 用真实转移概率 P_macro，阈值 5%）----
    sources_macro, targets_macro, values_macro = [], [], []
    customdata_macro, hovertext_macro = [], []
    for i in range(M):
        for j in range(M):
            if P_macro[i, j] > 0.05:
                sources_macro.append(N + i)
                targets_macro.append(N + j)
                values_macro.append(float(P_macro[i, j]))  # 真实转移概率
                customdata_macro.append([MACRO_NAMES[i], MACRO_NAMES[j], round(float(P_macro[i, j]), 4)])
                hovertext_macro.append(
                    f"<b>{MACRO_NAMES[i]}</b> → <b>{MACRO_NAMES[j]}</b><br>"
                    f"转移概率 P'(>5%) = {P_macro[i, j]:.4f}"
                )

    # 视觉缩放：两组流单位不同，分别缩放到可读宽度，但保留真实数值用于 hover
    # 微观 π 均值 ~1/31≈0.032，宏观 P' ~0.1，统一乘 1000 让两条流宽度可比
    micro_scale, macro_scale = 1000.0, 1000.0
    values_all = [v * micro_scale for v in values_micro] + [v * macro_scale for v in values_macro]

    # 构建 Sankey
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=15, thickness=18,
            line=dict(color='black', width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources_micro + sources_macro,
            target=targets_micro + targets_macro,
            value=values_all,
            customdata=customdata_micro + customdata_macro,
            hovertemplate="%{customdata[0]} → %{customdata[1]}<br>"
                          "%{customdata[2]}<extra></extra>",
            color=(
                ['rgba(91,155,213,0.35)'] * len(sources_micro) +
                ['rgba(237,125,49,0.55)'] * len(sources_macro)
            ),
        ),
    )])

    # 标题与中文支持
    fig.update_layout(
        title_text=(
            "<b>《道德经》概念粗粒化桑基图（交互式）</b><br>"
            "左侧蓝柱 = 31 个微观概念（按平稳概率 π 加权）<br>"
            "中间流带 = 概念→宏观义理（手工语义分组 M=6）<br>"
            "右侧橙柱 = 6 个宏观义理（按平稳概率 π' 排序）<br>"
            "橙色流带 = 宏观态间转移概率 P'(>5%)<br>"
            "悬停任一流带可查看其真实概率值"
        ),
        font=dict(family="Microsoft YaHei, SimHei, Noto Sans CJK SC", size=12),
        width=1200, height=900,
    )

    out_html = os.path.join(OUTPUT_DIR, 'vis_04_sankey_interactive.html')
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"  ✓ {os.path.basename(out_html)}")

    # 尝试用 kaleido 导出静态图（可选，需要 Chrome）
    try:
        import plotly.io as pio
        import kaleido
        out_png = os.path.join(OUTPUT_DIR, 'vis_04_sankey_plotly.png')
        pio.write_image(fig, out_png, scale=2)
        print(f"  ✓ {os.path.basename(out_png)} (kaleido + Chrome)")
    except Exception as e:
        print(f"  [跳过静态 PNG] kaleido/Chrome 不可用: {type(e).__name__}")

    print(f"\n  在浏览器中打开 {os.path.basename(out_html)} 即可交互查看")


if __name__ == "__main__":
    main()
