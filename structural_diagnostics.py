# -*- coding: utf-8 -*-
"""
P2 T07 + T08: 概念网络结构诊断

T07: 时间可逆性检验 + 混合时间
  - F = diag(π) @ P 的对称性误差 ‖F - Fᵀ‖_F
  - 若可逆性高 → 概念流动是"循环"的
  - 若不可逆 → 有方向性（道→德→无为→自然）
  - 混合时间 τ_mix = 1/(1-|λ₂|)
  - τ_mix 小 → 读几章就能把握全书核心
  - τ_mix 大 → 需通读全书

T08: 随机游走中心性
  - PageRank（带阻尼因子的马尔科夫中心性）
  - 命中时间（Hitting Time）：从任一概念到达目标概念的平均步数
  - 覆盖时间（Cover Time）：从起点访问所有概念的预期步数（蒙特卡洛）
  - 识别"枢纽概念"（PageRank 高 + 命中时间短）

依赖: numpy, scipy, pandas, matplotlib, networkx
运行: python structural_diagnostics.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    DAODEJING, build_full_sequence, build_transition_matrix,
    stationary_distribution, semantic_macro_labels, SEMANTIC_PARTITION,
    MACRO_NAMES, OUTPUT_DIR
)

# 【重构 T12】环境配置（UTF-8 / 中文字体）抽到 core.env
from core.env import setup_env, CN_FONT
setup_env()

# 【重构 T12】结构动力学数学函数抽到 core.dynamics
from core.dynamics import (
    reversibility_check, mixing_time, pagerank,
    hitting_time_to, hitting_time_all, cover_time_mc,
)


# ============================================================
# T07 / T08 数学函数已抽到 core.dynamics（见文件顶部导入）
# ============================================================
# reversibility_check / mixing_time / pagerank / hitting_time_to /
# hitting_time_all / cover_time_mc 均从 core.dynamics 导入。
# 注意：core.dynamics.cover_time_mc 签名只接受 P（去掉了 idx/inv_idx 冗余参数）。


# ============================================================
# 报告生成
# ============================================================
def write_report(rev_result, mix_result, centrality_df, pr, hit_to_mean, hit_from_mean, inv_idx):
    """写入 output/reversibility.txt（按 TODO.md 指定的文件名）"""
    path = os.path.join(OUTPUT_DIR, 'reversibility.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  《道德经》概念网络结构诊断报告\n")
        f.write("  （T07 可逆性 + T08 中心性）\n")
        f.write("=" * 60 + "\n\n")

        # ---- T07 ----
        f.write("【T07】时间可逆性 + 混合时间\n")
        f.write("-" * 40 + "\n")
        f.write(f"稳态流矩阵 F = diag(π)@P  Frobenius 对称偏差: {rev_result['abs_err']:.6f}\n")
        f.write(f"相对偏差: {rev_result['rel_err']:.6f}\n")
        if rev_result['rel_err'] < 0.05:
            judgment = "→ 高度可逆，概念流动近似时间对称（循环论证）"
        elif rev_result['rel_err'] < 0.15:
            judgment = "→ 弱可逆，存在轻微方向性"
        else:
            judgment = "→ 强不可逆，概念流动有明显方向性"
        f.write(f"可逆性判断: {judgment}\n\n")

        f.write("混合时间（τ_mix = 1 / (1 - |λ₂|)）\n")
        f.write(f"  λ₁ (稳态)  = {mix_result['lambda_1']:.6f}\n")
        f.write(f"  λ₂ (次大)  = {mix_result['lambda_2']:.6f}\n")
        f.write(f"  谱间隙     = {mix_result['spectral_gap']:.6f}\n")
        f.write(f"  混合时间   = {mix_result['tau_mix_steps']:.2f} 步\n")
        if mix_result['tau_mix_steps'] < 10:
            mi_judgment = "→ 极快：连续读几章即可把握全书核心"
        elif mix_result['tau_mix_steps'] < 50:
            mi_judgment = "→ 较快：通读约 1/4 卷即可"
        else:
            mi_judgment = "→ 较慢：需通读全书才能把握整体"
        f.write(f"  混合时间判断: {mi_judgment}\n\n")

        # 流量最不对称的对
        f.write("流量最不对称的 5 对概念（i→j vs j→i 流量差）\n")
        f.write(f"  {'源→目标':<10s} {'F[i,j]':>10s} {'F[j,i]':>10s} {'差值':>10s}\n")
        for i, j, d in rev_result['asym_pairs']:
            Fi = rev_result['F'][i, j]
            Fj = rev_result['F'][j, i]
            f.write(f"  {inv_idx[i]:>3s}→{inv_idx[j]:<3s}  {Fi:>10.4f} {Fj:>10.4f} {d:>+10.4f}\n")

        f.write("\n")

        # ---- T08 ----
        f.write("【T08】随机游走中心性\n")
        f.write("-" * 40 + "\n")
        f.write("PageRank（带阻尼 α=0.85）：\n")
        pr_sorted = sorted(enumerate(pr), key=lambda x: -x[1])[:5]
        for rank, (i, score) in enumerate(pr_sorted, 1):
            f.write(f"  #{rank}  {inv_idx[i]:<4s}  PageRank = {score:.5f}\n")

        f.write("\n最易到达的枢纽概念（命中时间最短）：\n")
        h_to = sorted(enumerate(hit_to_mean), key=lambda x: x[1])[:5]
        for rank, (i, h) in enumerate(h_to, 1):
            f.write(f"  #{rank}  {inv_idx[i]:<4s}  平均命中时间 = {h:.2f} 步\n")

        f.write("\n最强扩散概念（从他出发的命中时间最短）：\n")
        h_from = sorted(enumerate(hit_from_mean), key=lambda x: x[1])[:5]
        for rank, (i, h) in enumerate(h_from, 1):
            f.write(f"  #{rank}  {inv_idx[i]:<4s}  平均出发命中时间 = {h:.2f} 步\n")

        f.write("\n完整的中心性指标见 output/centrality_rankings.csv\n")
        f.write("可视化见 output/vis_08_dynamics.png\n")

    print(f"  ✓ {os.path.basename(path)}")


def plot_diagnostics(rev_result, mix_result, centrality_df, pr, hit_to_mean, hit_from_mean, inv_idx):
    """绘制 T07+T08 诊断图（3 子图：流量不对称热力图、谱衰减、PageRank 排序）"""
    print("  [绘图] 诊断可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # (1) 流量不对称热力图：F - Fᵀ
    ax1 = axes[0, 0]
    asym = rev_result['F'] - rev_result['F'].T
    np.fill_diagonal(asym, 0)
    vmax = np.percentile(np.abs(asym), 99)
    im1 = ax1.imshow(asym, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax1.set_title(f'流量不对称 F - F.T\n可逆性偏差 = {rev_result["rel_err"]:.4f}',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(inv_idx)))
    ax1.set_yticks(range(len(inv_idx)))
    ax1.set_xticklabels(list(inv_idx.values()), rotation=90, fontsize=7)
    ax1.set_yticklabels(list(inv_idx.values()), fontsize=7)
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='π[i]P[i,j] - π[j]P[j,i]')

    # (2) 谱衰减
    ax2 = axes[0, 1]
    eigvals_top = mix_result['eigvals_top5']
    bar_pos = np.arange(len(eigvals_top))
    ax2.bar(bar_pos, eigvals_top, color='#2E75B6', edgecolor='white', linewidth=0.5)
    for i, v in enumerate(eigvals_top):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_xticks(bar_pos)
    ax2.set_xticklabels([f'λ{i+1}' for i in bar_pos], fontsize=11)
    ax2.set_ylabel('|λ_i|', fontsize=11)
    ax2.set_title(f'特征值谱（前 5）\n混合时间 τ_mix = {mix_result["tau_mix_steps"]:.1f} 步',
                  fontsize=12, fontweight='bold')
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1.15)

    # (3) PageRank 排序（横向条形图）
    ax3 = axes[1, 0]
    pr_sorted = sorted(enumerate(pr), key=lambda x: -x[1])
    concepts_pr = [inv_idx[i] for i, _ in pr_sorted]
    scores_pr = [s for _, s in pr_sorted]
    macro_labels = [MACRO_NAMES[semantic_macro_labels_idx(inv_idx[i])] for i, _ in pr_sorted]
    macro_colors = plt.cm.Set2(np.linspace(0, 1, len(MACRO_NAMES)))
    color_map = {MACRO_NAMES[m]: macro_colors[m] for m in range(len(MACRO_NAMES))}
    bar_colors = [color_map[ml] for ml in macro_labels]
    y_pos = np.arange(len(concepts_pr))
    ax3.barh(y_pos, scores_pr, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(concepts_pr, fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel('PageRank（阻尼 α=0.85）', fontsize=11)
    ax3.set_title('概念 PageRank 排名\n颜色 = 宏观义理', fontsize=12, fontweight='bold')
    for i, v in enumerate(scores_pr):
        ax3.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=8)
    ax3.grid(axis='x', alpha=0.3)

    # (4) 命中时间散点图：到概念时间 vs 从概念出发时间
    ax4 = axes[1, 1]
    scatter_colors = [color_map[MACRO_NAMES[semantic_macro_labels_idx(inv_idx[i])]]
                     for i in range(len(inv_idx))]
    ax4.scatter(hit_to_mean, hit_from_mean, c=scatter_colors, s=120, alpha=0.85,
                edgecolors='black', linewidths=0.5)
    for i in range(len(inv_idx)):
        ax4.annotate(inv_idx[i], (hit_to_mean[i], hit_from_mean[i]),
                     fontsize=8, ha='center', va='bottom', alpha=0.85)
    ax4.set_xlabel('平均命中时间 TO（到本概念）', fontsize=11)
    ax4.set_ylabel('平均命中时间 FROM（从本概念出发）', fontsize=11)
    ax4.set_title('命中时间散点图（左下角 = 双向枢纽）', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(np.mean(hit_from_mean), color='gray', ls=':', lw=0.7, alpha=0.6)
    ax4.axvline(np.mean(hit_to_mean), color='gray', ls=':', lw=0.7, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'vis_08_dynamics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {os.path.basename(path)}")


# 辅助：根据概念名查宏观态
def semantic_macro_labels_idx(concept):
    """从 main 的 SEMANTIC_PARTITION 查概念所属宏观态"""
    for m, block in enumerate(SEMANTIC_PARTITION):
        if concept in block:
            return m
    return -1


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  P2 T07 + T08: 概念网络结构诊断")
    print("=" * 60)

    # 加载数据
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)
    N = P.shape[0]

    # ---- T07 ----
    print("\n[1/4] 可逆性检验 ...")
    rev_result = reversibility_check(P, pi)
    print(f"  ‖F - Fᵀ‖_F  = {rev_result['abs_err']:.6f}（相对 {rev_result['rel_err']:.4f}）")
    if rev_result['rel_err'] < 0.05:
        print("  ✓ 高度可逆：概念流动近似时间对称（'道↔德'循环论证结构）")
    else:
        print(f"  ⚠ 弱可逆：相对偏差 {rev_result['rel_err']:.2%}，存在方向性")

    print("\n[2/4] 混合时间 ...")
    mix_result = mixing_time(P)
    print(f"  λ₂ = {mix_result['lambda_2']:.4f}, 谱间隙 = {mix_result['spectral_gap']:.4f}")
    print(f"  τ_mix = {mix_result['tau_mix_steps']:.2f} 步")
    if mix_result['tau_mix_steps'] < 15:
        print(f"  → 极快混合：约需读 5-10 章即可把握全书核心动力学")
    else:
        print(f"  → 需读约 {mix_result['tau_mix_steps']:.0f} 章才能稳态收敛")

    # ---- T08 ----
    print("\n[3/4] 中心性分析 ...")
    pr = pagerank(P, pi, alpha=0.85)
    print("  PageRank TOP 5:")
    pr_sorted = sorted(enumerate(pr), key=lambda x: -x[1])[:5]
    for rank, (i, s) in enumerate(pr_sorted, 1):
        print(f"    #{rank}  {inv_idx[i]:<4s}  PR = {s:.5f}")

    # 命中时间（双向）
    hit_to_all = np.zeros(N)
    hit_from_all = np.zeros(N)
    for i in range(N):
        hit_to_all += hitting_time_to(P, i)
        hit_from_all += hitting_time_to(P.T, i)  # from i = to i in P^T
    hit_to_mean = hit_to_all / (N - 1)
    hit_from_mean = hit_from_all / (N - 1)
    print("  命中时间 TOP 5（最短 → 中心枢纽）:")
    for rank, (i, h) in enumerate(sorted(enumerate(hit_to_mean), key=lambda x: x[1])[:5], 1):
        print(f"    #{rank}  {inv_idx[i]:<4s}  TO = {h:.2f} 步")

    # 覆盖时间
    print("\n[4/4] 覆盖时间（蒙特卡洛）...")
    cover_steps, avg_cover = cover_time_mc(P, n_starts=N)
    print(f"  覆盖时间（访问全部 31 概念）平均 {avg_cover:.1f} 步")
    print(f"  范围: {min(cover_steps)} - {max(cover_steps)} 步")

    # 保存中心性 CSV
    centrality_df = pd.DataFrame({
        'concept': list(inv_idx.values()),
        'macro_state': [semantic_macro_labels_idx(inv_idx[i]) for i in range(N)],
        'macro_name': [MACRO_NAMES[semantic_macro_labels_idx(inv_idx[i])] for i in range(N)],
        'pi': pi,
        'pagerank': pr,
        'hit_to_mean': hit_to_mean,
        'hit_from_mean': hit_from_mean,
        'cover_time': cover_steps,
    })
    centrality_df = centrality_df.sort_values('pagerank', ascending=False)
    centrality_df.to_csv(os.path.join(OUTPUT_DIR, 'centrality_rankings.csv'),
                          index=False, encoding='utf-8-sig')
    print(f"  ✓ centrality_rankings.csv")

    # 写报告 + 出图
    write_report(rev_result, mix_result, centrality_df, pr, hit_to_mean, hit_from_mean, inv_idx)
    plot_diagnostics(rev_result, mix_result, centrality_df, pr, hit_to_mean, hit_from_mean, inv_idx)

    print(f"\n{'='*60}")
    print(f"  ✓ T07 + T08 诊断完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
