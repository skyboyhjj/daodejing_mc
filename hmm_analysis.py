# -*- coding: utf-8 -*-
"""
P2 T06: HMM 软分配分析

对比硬划分（手工语义）vs HMM 软分配（隐马尔科夫模型）：
  - 隐状态数 M = 6（与宏观义理块数一致）
  - 观测 = 31 个概念的整数编码
  - 训练：Baum-Welch（hmmlearn.MultinomialHMM，多 seed 取最优）
  - 输出：
      A：M×M 隐态转移矩阵
      B：M×N 发射概率矩阵 P(概念 | 状态)
      γ：每个位置的隐态后验概率（用于软分配）

对比指标：
  - 硬划分 EI_norm（Φ 硬划分 → 宏观转移）
  - 软分配 EI_norm（Φ 软化 → 宏观转移）
  - 看软分配是否带来更高的因果涌现

依赖: numpy, pandas, matplotlib, hmmlearn
运行: python hmm_analysis.py
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
    stationary_distribution, build_macro_transition,
    effective_information, normalized_ei,
    semantic_macro_labels, SEMANTIC_PARTITION, MACRO_NAMES, OUTPUT_DIR
)

# 【重构 T12】环境配置（UTF-8 / 中文字体）抽到 core.env
from core.env import setup_env, CN_FONT
setup_env()


# ============================================================
# HMM 训练
# ============================================================
def train_hmm(X, M=6, N=31, hard_labels=None, n_seeds=10, n_iter=500, tol=1e-6,
              fix_emission=False):
    """多 seed 训练 HMM，取对数似然最高的模型。
    用硬划分作为智能初始化，避免状态坍缩到单一概念。
    hmmlearn 0.3+：单次分类观测用 CategoricalHMM。
    fix_emission=True：固定发射矩阵，只训练转移矩阵"""
    from hmmlearn.hmm import CategoricalHMM

    # 智能初始化：基于硬划分
    if hard_labels is not None:
        init_emission = np.zeros((M, N))
        for m in range(M):
            for j in range(N):
                init_emission[m, j] = 1.0 if hard_labels[j] == m else 1e-3
        init_emission = init_emission / init_emission.sum(axis=1, keepdims=True)
    else:
        init_emission = np.full((M, N), 1.0 / N)

    init_transmat = np.full((M, M), 1.0 / M)
    init_startprob = np.full(M, 1.0 / M)

    # 是否固定发射
    params = 't' if fix_emission else 'te'

    best_model = None
    best_ll = -np.inf
    for seed in range(n_seeds):
        m = CategoricalHMM(n_components=M, n_iter=n_iter, tol=tol,
                           random_state=seed, verbose=False,
                           params=params, init_params='')
        m.startprob_ = init_startprob.copy()
        m.transmat_ = init_transmat.copy()
        m.emissionprob_ = init_emission.copy()
        try:
            m.fit(X)
            ll = m.score(X)
            print(f"  seed={seed:2d}  log-likelihood = {ll:.2f}")
            if ll > best_ll:
                best_ll = ll
                best_model = m
        except Exception as e:
            print(f"  seed={seed:2d}  训练失败: {e}")
    print(f"\n  最优对数似然 = {best_ll:.2f}")
    return best_model, best_ll


def soft_macro_from_hmm(model, X, N, M):
    """从训练好的 HMM 构造软分配宏观态矩阵 P_soft"""
    # 后验 γ[t, i] = P(state_i | x_1:T)
    posteriors = model.predict_proba(X)  # (T, M)

    # 软 Φ：每个概念 j 的"在每个状态中的期望比例"
    soft_Phi = np.zeros((N, M))
    concept_counts = np.zeros(N)
    for t, x in enumerate(X[:, 0]):
        soft_Phi[x] += posteriors[t]
        concept_counts[x] += 1
    for j in range(N):
        if concept_counts[j] > 0:
            soft_Phi[j] /= concept_counts[j]
        else:
            soft_Phi[j] = np.ones(M) / M

    # 软宏观转移：先按"软Φ 转移矩阵"构造
    # 等价于：将每个观测位置 t 视为"软混合状态"的发射，求软混合之间的转移
    # 公式：P_soft = (Φ^T P) Φ 在归一化后
    # 这里我们用更直接的"软Φ 加权转移计数"
    P_soft_unnorm = soft_Phi.T @ soft_Phi
    # 保持行随机：归一化（隐态间联合出现频率 → 转移频率）
    # 严格转移是 E[Φ_当前 Φ_下一^T] / E[Φ_当前]
    # 由于序列是离散的，用经验估计：
    P_soft_count = np.zeros((M, M))
    P_soft_from = np.zeros(M)
    for t in range(len(X) - 1):
        cur = X[t, 0]
        nxt = X[t + 1, 0]
        gamma_cur = posteriors[t]  # (M,)
        gamma_nxt = posteriors[t + 1]
        # 该步对转移的贡献：γ_t[i] * γ_{t+1}[k] * P[cur, nxt]
        p_tn = 1.0  # 实际该步发生了的 transfer（已用 posteriors 编码）
        P_soft_count += np.outer(gamma_cur, gamma_nxt)
        P_soft_from += gamma_cur
    # 归一化为转移概率
    P_soft = np.zeros((M, M))
    for i in range(M):
        if P_soft_from[i] > 1e-12:
            P_soft[i] = P_soft_count[i] / P_soft_from[i]
        else:
            P_soft[i] = np.ones(M) / M

    # 平稳分布
    pi_soft = stationary_distribution(P_soft)

    return soft_Phi, P_soft, pi_soft, posteriors


# ============================================================
# 可视化
# ============================================================
def plot_hmm_results(B, A, hard_labels, soft_Phi, inv_idx, ei_compare):
    """绘制 HMM 结果"""
    print("  [绘图] HMM 软分配可视化...")
    M, N = B.shape
    concepts = list(inv_idx.values())

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # (1) 发射矩阵 B 热力图（M×N，行=状态，列=概念）
    ax1 = axes[0, 0]
    im1 = ax1.imshow(B, cmap='YlOrRd', aspect='auto', vmin=0, vmax=B.max())
    ax1.set_yticks(range(M))
    ax1.set_yticklabels([f'State {m}' for m in range(M)], fontsize=10)
    ax1.set_xticks(range(N))
    ax1.set_xticklabels(concepts, rotation=90, fontsize=7)
    ax1.set_title('HMM 发射概率 B[i,j] = P(概念_j | 状态_i)\n（行=隐状态，列=概念）',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='发射概率')

    # (2) 每个状态的 Top 概念
    ax2 = axes[0, 1]
    width = 0.13
    for m in range(M):
        top5 = np.argsort(B[m])[::-1][:5]
        names = [concepts[i] for i in top5]
        probs = [B[m, i] for i in top5]
        y_pos = np.arange(len(names)) + m * width
        ax2.barh(y_pos, probs, height=width, label=f'State {m}', alpha=0.85)
    ax2.set_yticks(np.arange(5) + (M - 1) * width / 2)
    ax2.set_yticklabels([f'Top {i+1}' for i in range(5)], fontsize=10)
    ax2.set_xlabel('发射概率', fontsize=11)
    ax2.set_title('每个隐态的 Top-5 关联概念', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9, ncol=2)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    # (3) 软 Φ 矩阵：每个概念在每个状态的期望概率
    ax3 = axes[1, 0]
    im3 = ax3.imshow(soft_Phi, cmap='Blues', aspect='auto', vmin=0, vmax=soft_Phi.max())
    ax3.set_xticks(range(M))
    ax3.set_xticklabels([f'S{m}' for m in range(M)], fontsize=10)
    ax3.set_yticks(range(N))
    ax3.set_yticklabels(concepts, fontsize=9)
    ax3.set_title('软分配 Φ[j,i] = 概念_j 在状态_i 中的期望占比\n（按出现位置的后验平均）',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='软分配概率')

    # (4) EI 对比
    ax4 = axes[1, 1]
    labels = ['微观基线', '硬划分\n（语义）', '软分配\n（HMM）']
    values = [ei_compare['micro'], ei_compare['hard'], ei_compare['soft']]
    colors = ['#5B9BD5', '#ED7D31', '#70AD47']
    bars = ax4.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5, width=0.55)
    for bar, v in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.001, f'{v:.4f}',
                 ha='center', fontsize=12, fontweight='bold')
    ax4.axhline(ei_compare['micro'], color='#5B9BD5', ls=':', lw=0.7, alpha=0.5)
    ax4.set_ylabel('归一化 EI', fontsize=12)
    ax4.set_title('硬划分 vs 软分配：因果涌现对比', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, max(values) * 1.25)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'vis_09_hmm.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {os.path.basename(path)}")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  P2 T06: HMM 软分配分析")
    print("=" * 60)

    # 加载数据
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)
    N = P.shape[0]
    M = 6

    # 硬划分标签
    hard_labels = semantic_macro_labels(idx, inv_idx)
    print(f"  N = {N} 概念, M = {M} 状态, T = {len(full_seq)} 观测")

    # 整数编码
    X = np.array([idx[c] for c in full_seq]).reshape(-1, 1)

    # 训练 HMM：固定发射矩阵为硬划分"软"版本（避免数据过于均匀导致状态坍缩），仅训练 A
    print("\n[1/4] 训练 HMM（固定发射 B = 硬划分的软化版本，仅训练 A）...")
    model, best_ll = train_hmm(X, M=M, N=N, hard_labels=hard_labels,
                                n_seeds=10, n_iter=500, fix_emission=True)
    if model is None:
        print("  训练失败：尝试自由发射...")
        model, best_ll = train_hmm(X, M=M, N=N, hard_labels=hard_labels,
                                    n_seeds=10, n_iter=500, fix_emission=False)

    # 自由发射 HMM 在不同 M 下的对数似然对比（诊断"是否能找到差异化结构"）
    print("\n[2/4] M=2..6 自由发射 HMM 扫描（观察结构）...")
    for M_test in [2, 3, 4, 5, 6]:
        _, ll = train_hmm(X, M=M_test, N=N, hard_labels=None, n_seeds=3, n_iter=150,
                          fix_emission=False)
        print(f"  → M={M_test} 自由发射最大 LL = {ll:.2f}")

    # 提取参数
    A = model.transmat_         # M×M 转移
    B = model.emissionprob_     # M×N 发射
    print(f"\n  A 形状: {A.shape},  B 形状: {B.shape}")

    # 软分配
    print("\n[2/3] 计算软分配...")
    soft_Phi, P_soft, pi_soft, posteriors = soft_macro_from_hmm(model, X, N, M)

    # EI 对比
    print("\n[3/3] EI 对比...")
    # 微观
    ei_micro_norm = normalized_ei(P, pi)
    # 硬划分
    P_hard, Phi_hard = build_macro_transition(P, hard_labels, idx, inv_idx)
    pi_hard = stationary_distribution(P_hard)
    ei_hard_norm = normalized_ei(P_hard, pi_hard)
    # 软分配
    ei_soft_norm = normalized_ei(P_soft, pi_soft)

    print(f"  微观 EI_norm       = {ei_micro_norm:.4f}")
    print(f"  硬划分 EI_norm     = {ei_hard_norm:.4f}")
    print(f"  软分配 EI_norm     = {ei_soft_norm:.4f}")
    print(f"  因果涌现（硬）     = {ei_hard_norm - ei_micro_norm:+.4f}")
    print(f"  因果涌现（软）     = {ei_soft_norm - ei_micro_norm:+.4f}")
    print(f"  软分配提升（vs 硬）= {ei_soft_norm - ei_hard_norm:+.4f}")

    # 打印每个状态的 Top 概念
    print("\n  隐态的核心概念（按发射概率）：")
    for m in range(M):
        top = np.argsort(B[m])[::-1][:5]
        names = [inv_idx[i] for i in top]
        print(f"    State {m}: {', '.join(names)} (B[0]={B[m].max():.3f})")

    # 与硬划分的吻合度
    soft_MLE = soft_Phi.argmax(axis=1)
    hard_state_for_soft = soft_MLE
    matches = sum(1 for i, sl in enumerate(soft_MLE) if hard_labels[i] == sl)
    print(f"\n  软分配 MLE 标签 vs 硬划分标签：{matches}/{N} 概念一致")

    # 保存
    np.save(os.path.join(OUTPUT_DIR, 'P_hmm.npy'), P_soft)
    np.save(os.path.join(OUTPUT_DIR, 'A_hmm.npy'), A)
    np.save(os.path.join(OUTPUT_DIR, 'B_hmm.npy'), B)
    np.save(os.path.join(OUTPUT_DIR, 'soft_Phi.npy'), soft_Phi)

    hmm_results = {
        'log_likelihood': float(best_ll),
        'A': A.tolist(),
        'B': B.tolist(),
        'concepts': list(inv_idx.values()),
        'hard_labels': hard_labels.tolist(),
        'soft_MLE_labels': soft_MLE.tolist(),
        'soft_Phi': soft_Phi.tolist(),
        'pi_hard': pi_hard.tolist(),
        'pi_soft': pi_soft.tolist(),
        'P_hard': P_hard.tolist(),
        'P_soft': P_soft.tolist(),
        'EI': {
            'micro': float(ei_micro_norm),
            'hard': float(ei_hard_norm),
            'soft': float(ei_soft_norm),
        }
    }
    with open(os.path.join(OUTPUT_DIR, 'hmm_results.json'), 'w', encoding='utf-8') as f:
        json.dump(hmm_results, f, ensure_ascii=False, indent=2)
    print(f"  ✓ hmm_results.json")

    # 可视化
    ei_compare = {'micro': ei_micro_norm, 'hard': ei_hard_norm, 'soft': ei_soft_norm}
    plot_hmm_results(B, A, hard_labels, soft_Phi, inv_idx, ei_compare)

    print(f"\n{'='*60}")
    print(f"  ✓ T06 HMM 软分配完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
