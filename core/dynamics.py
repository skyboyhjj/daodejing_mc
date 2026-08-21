# -*- coding: utf-8 -*-
"""
core/dynamics.py — 概念网络结构动力学分析函数

从 structural_diagnostics.py 抽取的纯计算函数，供 T07/T08 及后续分析复用：
  - 时间可逆性检验（稳态流矩阵对称偏差）
  - 混合时间（谱间隙）
  - 随机游走中心性（PageRank / 命中时间 / 覆盖时间）

只依赖 numpy，不含 matplotlib。
"""

import numpy as np


# ============================================================
# T07: 时间可逆性
# ============================================================
def reversibility_check(P, pi):
    """
    稳态流矩阵 F = diag(π)@P 的对称性偏差。
    若 π[i]*P[i,j] == π[j]*P[j,i]（即 F == Fᵀ），则马尔科夫链可逆（细致平衡）。
    返回 dict：abs_err, rel_err, F, asym_pairs（流量最不对称的 5 对）
    """
    F = np.diag(pi) @ P
    F_T = F.T
    err = np.linalg.norm(F - F_T, 'fro')
    rel_err = err / (np.linalg.norm(F, 'fro') + 1e-15)

    # 流量不对称最严重的对（i→j 流量远大于 j→i）
    asym = F - F_T
    np.fill_diagonal(asym, 0)
    n = F.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, asym[i, j]))  # asym[i,j] = F[i,j] - F[j,i]
    pairs.sort(key=lambda x: -abs(x[2]))

    return {
        'abs_err': float(err),
        'rel_err': float(rel_err),
        'F': F,
        'asym_pairs': pairs[:5],
    }


def mixing_time(P):
    """τ_mix = 1 / (1 - |λ₂|)
    λ₂ 是 P 的第二大的特征值绝对值（最大的恒为 1）。
    返回 dict：lambda_1, lambda_2, spectral_gap, tau_mix_steps, eigvals_top5
    """
    eigvals = np.linalg.eigvals(P)
    abs_eig = np.abs(eigvals)
    abs_eig_sorted = np.sort(abs_eig)[::-1]
    lambda2 = abs_eig_sorted[1] if len(abs_eig_sorted) > 1 else 0.0
    tau_mix = 1.0 / (1 - lambda2 + 1e-15)
    return {
        'lambda_1': float(abs_eig_sorted[0]),
        'lambda_2': float(lambda2),
        'spectral_gap': float(1 - lambda2),
        'tau_mix_steps': float(tau_mix),
        'eigvals_top5': abs_eig_sorted[:5].tolist(),
    }


# ============================================================
# T08: 随机游走中心性
# ============================================================
def pagerank(P, pi, alpha=0.85, max_iter=5000, tol=1e-12):
    """带阻尼因子的 PageRank（从 π 出发迭代）"""
    N = P.shape[0]
    pr = pi.copy()
    teleport = (1 - alpha) / N
    for _ in range(max_iter):
        pr_new = alpha * (pr @ P) + teleport
        if np.linalg.norm(pr_new - pr, 1) < tol:
            return pr_new
        pr = pr_new
    return pr


def hitting_time_to(P, target):
    """
    从所有状态到 target 的预期步数（命中时间）
    求解 h[i] = 1 + sum_j P[i,j] * h[j]，h[target]=0
    等价 (I - P_∖t) h_∖t = 1
    """
    N = P.shape[0]
    h = np.zeros(N)
    if N == 1:
        return h
    others = [j for j in range(N) if j != target]
    I_mat = np.eye(N - 1)
    P_sub = P[np.ix_(others, others)]
    ones = np.ones(N - 1)
    try:
        h_sub = np.linalg.solve(I_mat - P_sub, ones)
        for k, idx in enumerate(others):
            h[idx] = h_sub[k]
    except np.linalg.LinAlgError:
        h[:] = np.inf
    h[target] = 0.0
    return h


def hitting_time_all(P):
    """
    计算所有节点之间的命中时间矩阵 H。
    返回 (hit_to_mean, hit_from_mean)：
      hit_to_mean[i]   = 从任意节点到达 i 的平均步数（越短越中心）
      hit_from_mean[i] = 从 i 出发到达任意节点的平均步数（越短越具扩散性）
    """
    N = P.shape[0]
    hit_to_all = np.zeros(N)
    hit_from_all = np.zeros(N)
    for i in range(N):
        hit_to_all += hitting_time_to(P, i)
        hit_from_all += hitting_time_to(P.T, i)  # from i = to i in P^T
    hit_to_mean = hit_to_all / (N - 1)
    hit_from_mean = hit_from_all / (N - 1)
    return hit_to_mean, hit_from_mean


def cover_time_mc(P, n_starts=31, max_steps=20000, seed=42):
    """
    覆盖时间：从每个起点出发随机游走，访问全部 N 概念所需步数（蒙特卡洛估计）。
    返回 (cover_steps_list, avg_cover)
    """
    rng = np.random.default_rng(seed)
    N = P.shape[0]
    cover_steps = []
    for s in range(N):
        visited = {s}
        steps = 0
        cur = s
        while len(visited) < N and steps < max_steps:
            nxt = rng.choice(N, p=P[cur])
            visited.add(int(nxt))
            cur = int(nxt)
            steps += 1
        cover_steps.append(steps)
    avg_cover = float(np.mean(cover_steps))
    return cover_steps, avg_cover
