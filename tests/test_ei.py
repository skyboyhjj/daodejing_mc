# -*- coding: utf-8 -*-
"""
tests/test_ei.py — 有效信息 (EI) 单元测试

核心验证点（见 TODO.md T13）：
  1. 用已知解析结果验证 EI 计算
  2. 均匀转移矩阵 EI = 0（无因果效应）
  3. 确定性映射 EI 最大
  4. 归一化 EI = EI / log(N)
  5. 全书 EI 数值稳定（回归测试）

EI 定义：EI(P) = Σ_i π_i · D_KL(P_i || P_bar)，其中 P_bar = πP。
"""

import numpy as np
import pytest

from core.pipeline import (
    effective_information, normalized_ei, stationary_distribution,
)


# ============================================================
# 辅助：构造已知 EI 的矩阵
# ============================================================
def uniform_matrix(n):
    """n×n 均匀转移矩阵（每行都是 1/n），EI 应为 0"""
    return np.full((n, n), 1.0 / n)


def deterministic_permutation(n):
    """n×n 确定性置换矩阵（每行独热、列互不重叠），EI 应为最大 log(n)"""
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i + 1) % n] = 1.0
    return P


def identity_like(n):
    """n×n 恒等矩阵（每行指向自己），EI 也应最大"""
    return np.eye(n)


# ============================================================
# 1. 已知解析结果验证
# ============================================================
def test_uniform_matrix_ei_zero():
    """均匀矩阵：P 每行均一，无因果效应，EI 应为 0"""
    n = 4
    P = uniform_matrix(n)
    pi = stationary_distribution(P)  # 均匀矩阵平稳分布也均匀
    ei = effective_information(P, pi)
    assert abs(ei) < 1e-10, f"均匀矩阵 EI 应为 0，实际 {ei}"


def test_deterministic_permutation_ei_max():
    """确定性置换：EI 应为最大值 log(n)"""
    n = 3
    P = deterministic_permutation(n)
    pi = np.full(n, 1.0 / n)
    ei = effective_information(P, pi)
    # 确定性置换的 EI 理论上 = log(n)（均匀 π 下每行 KL = log(n)）
    assert abs(ei - np.log(n)) < 1e-6, f"置换矩阵 EI 应≈log({n})={np.log(n):.4f}，实际 {ei:.4f}"


def test_ei_nonnegative():
    """EI 应为非负（KL 散度非负）"""
    n = 3
    P = deterministic_permutation(n)
    pi = np.full(n, 1.0 / n)
    assert effective_information(P, pi) >= 0


def test_ei_single_state_zero():
    """单一状态矩阵：无不确定性，EI 应为 0"""
    P = np.array([[1.0]])
    ei = effective_information(P, np.array([1.0]))
    assert abs(ei) < 1e-10


# ============================================================
# 2. 归一化 EI
# ============================================================
def test_normalized_ei_zero_for_uniform():
    """均匀矩阵归一化 EI 应为 0"""
    n = 4
    P = uniform_matrix(n)
    pi = stationary_distribution(P)
    assert abs(normalized_ei(P, pi)) < 1e-10


def test_normalized_ei_one_for_deterministic():
    """确定性置换归一化 EI 应为 1"""
    n = 3
    P = deterministic_permutation(n)
    pi = np.full(n, 1.0 / n)
    assert abs(normalized_ei(P, pi) - 1.0) < 1e-6, "确定性矩阵归一化 EI 应为 1"


def test_normalized_ei_ratio():
    """归一化 EI 应等于 EI / log(N)"""
    n = 3
    P = deterministic_permutation(n)
    pi = np.full(n, 1.0 / n)
    ei = effective_information(P, pi)
    norm = normalized_ei(P, pi)
    assert abs(norm - ei / np.log(n)) < 1e-9


def test_normalized_ei_single_state():
    """单一状态归一化 EI 应为 0（避免除零）"""
    P = np.array([[1.0]])
    assert normalized_ei(P, np.array([1.0])) == 0.0


# ============================================================
# 3. EI 随确定性增加而增大
# ============================================================
def test_ei_increases_with_determinism():
    """确定性矩阵的 EI 应高于扰动后的矩阵"""
    n = 2
    P_det = np.array([[1.0, 0.0], [0.0, 1.0]])
    P_noisy = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = np.full(n, 0.5)
    ei_det = effective_information(P_det, pi)
    ei_noisy = effective_information(P_noisy, pi)
    assert ei_det > ei_noisy, "确定性矩阵 EI 应更大"


def test_ei_more_determinism_more_ei():
    """确定性程度越高，EI 越大"""
    n = 2
    P1 = np.array([[1.0, 0.0], [0.0, 1.0]])          # 完全确定
    P2 = np.array([[0.8, 0.2], [0.2, 0.8]])          # 高确定
    P3 = np.array([[0.5, 0.5], [0.5, 0.5]])          # 均匀
    pi = np.full(2, 0.5)
    e1 = effective_information(P1, pi)
    e2 = effective_information(P2, pi)
    e3 = effective_information(P3, pi)
    assert e1 > e2 > e3 >= 0, f"EI 应随确定性递减: {e1} > {e2} > {e3}"


# ============================================================
# 4. 全书 EI 回归测试（数值稳定）
# ============================================================
def test_full_text_ei_stable():
    """全书微观 EI 应为 0.1805 bits（回归基线）
    注：此值在 T13 修复 REVERSE_MAP bug 后可能微调，此处做数量级回归"""
    from main import build_full_sequence, DAODEJING
    from core.pipeline import build_transition_matrix, stationary_distribution
    full_seq, _ = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)
    ei = effective_information(P, pi)
    # 回归基线：数值在 0.10 ~ 0.25 之间（量级正确）
    assert 0.10 < ei < 0.25, f"全书 EI 应在合理范围，实际 {ei:.4f}"


def test_normalized_ei_between_zero_one():
    """全书归一化 EI 应在 [0, 1] 之间"""
    from main import build_full_sequence, DAODEJING
    from core.pipeline import build_transition_matrix, stationary_distribution
    full_seq, _ = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    pi = stationary_distribution(P)
    norm = normalized_ei(P, pi)
    assert 0.0 <= norm <= 1.0, f"归一化 EI 应在 [0,1]，实际 {norm:.4f}"
