# -*- coding: utf-8 -*-
"""
tests/test_transition_matrix.py — 转移矩阵单元测试

核心验证点（见 TODO.md T13）：
  1. 转移矩阵每行和 = 1（行随机矩阵）
  2. 矩阵为方阵 N×N
  3. 所有元素非负
  4. k=1 / k=2 构建正确
  5. idx / inv_idx 互逆
  6. 平滑（Laplace）不影响行和
"""

import numpy as np
import pytest

from core.pipeline import build_transition_matrix, stationary_distribution


# ============================================================
# 小样本固定数据（便于手算验证）
# ============================================================
FIXED_SEQ = ["道", "德", "道", "无为", "无", "道", "德"]
FIXED_UNIQUE = sorted(set(FIXED_SEQ))  # ['德', '无', '无为', '道']


# ============================================================
# 1. 行和 = 1（核心验证点）
# ============================================================
def test_k1_rows_sum_to_one():
    """k=1 转移矩阵每行和应为 1"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1, smoothing=1.0)
    row_sums = P.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(P.shape[0]), atol=1e-10)


def test_k2_rows_sum_to_one():
    """k=2 联合态转移矩阵每行和应为 1"""
    P2, C2, pair_idx, pair_list, idx2, inv_idx2 = build_transition_matrix(
        FIXED_SEQ, k=2, smoothing=1.0)
    row_sums = P2.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(P2.shape[0]), atol=1e-10)


def test_rows_sum_one_without_smoothing():
    """无平滑时行和仍为 1"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1, smoothing=0.0)
    row_sums = P.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(P.shape[0]), atol=1e-10)


# ============================================================
# 2. 方阵与非负性
# ============================================================
def test_k1_square_matrix():
    """k=1 矩阵应为方阵"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1)
    assert P.shape[0] == P.shape[1], f"矩阵应为方阵，实际 {P.shape}"


def test_k2_square_matrix():
    """k=2 矩阵应为方阵（联合态→联合态）"""
    P2, C2, pair_idx, pair_list, idx2, inv_idx2 = build_transition_matrix(
        FIXED_SEQ, k=2)
    assert P2.shape[0] == P2.shape[1], f"k=2 矩阵应为方阵，实际 {P2.shape}"


def test_all_elements_nonnegative():
    """所有元素应非负"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1)
    assert (P >= 0).all(), "转移矩阵不应有负元素"
    assert (C >= 0).all(), "计数矩阵不应有负元素"


def test_k1_size_matches_unique():
    """k=1 矩阵维度应等于唯一概念数"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1)
    assert P.shape == (len(FIXED_UNIQUE), len(FIXED_UNIQUE))


# ============================================================
# 3. idx / inv_idx 互逆
# ============================================================
def test_idx_inv_idx_inverse():
    """idx 与 inv_idx 应互逆"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1)
    for c, i in idx.items():
        assert inv_idx[i] == c, f"idx[{c}]={i} 但 inv_idx[{i}]={inv_idx[i]}"
    for i, c in inv_idx.items():
        assert idx[c] == i, f"inv_idx[{i}]={c} 但 idx[{c}]={idx[c]}"


def test_idx_covers_all_unique():
    """idx 应覆盖全部唯一概念"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1)
    assert set(idx.keys()) == set(FIXED_UNIQUE)


# ============================================================
# 4. 计数矩阵正确性（手算验证）
# ============================================================
def test_count_matrix_fixed_seq():
    """用固定序列手算验证计数矩阵"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1, smoothing=0.0)
    # FIXED_SEQ = [道, 德, 道, 无为, 无, 道, 德]
    # 转移对：道→德, 德→道, 道→无为, 无为→无, 无→道, 道→德
    assert C[idx["道"], idx["德"]] == 2  # 道→德 出现 2 次
    assert C[idx["道"], idx["无为"]] == 1  # 道→无为 出现 1 次
    assert C[idx["德"], idx["道"]] == 1  # 德→道 出现 1 次
    assert C[idx["无为"], idx["无"]] == 1  # 无为→无 出现 1 次
    assert C[idx["无"], idx["道"]] == 1  # 无→道 出现 1 次
    # 总转移次数 = len - 1 = 6
    assert C.sum() == len(FIXED_SEQ) - 1


# ============================================================
# 5. 平稳分布性质
# ============================================================
def test_stationary_distribution_properties():
    """平稳分布应满足 πP = π 且求和 = 1"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1)
    pi = stationary_distribution(P)
    # 求和为 1
    assert abs(pi.sum() - 1.0) < 1e-8, f"平稳分布求和应=1，实际 {pi.sum()}"
    # πP = π
    pi_new = pi @ P
    np.testing.assert_allclose(pi_new, pi, atol=1e-6)
    # 非负
    assert (pi >= 0).all()


def test_stationary_constant_for_regular_matrix():
    """正则矩阵的平稳分布应非退化"""
    P, C, idx, inv_idx = build_transition_matrix(FIXED_SEQ, k=1)
    pi = stationary_distribution(P)
    assert pi.shape == (len(FIXED_UNIQUE),)


# ============================================================
# 6. 全书转移矩阵
# ============================================================
def test_full_text_matrix_valid():
    """全书 81 章构建的转移矩阵应满足基本性质"""
    from main import build_full_sequence, DAODEJING
    full_seq, _ = build_full_sequence(DAODEJING)
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1)
    N = P.shape[0]
    assert N == 31, f"全书唯一概念应为 31，实际 {N}"
    np.testing.assert_allclose(P.sum(axis=1), np.ones(N), atol=1e-10)
    assert (P >= 0).all()
    assert P.shape == (31, 31)


def test_full_text_k2_shape():
    """全书 k=2 联合态矩阵应为方阵且行和=1"""
    from main import build_full_sequence, DAODEJING
    full_seq, _ = build_full_sequence(DAODEJING)
    P2, C2, pair_idx, pair_list, idx2, inv_idx2 = build_transition_matrix(full_seq, k=2)
    assert P2.shape[0] == P2.shape[1], "k=2 矩阵应为方阵"
    np.testing.assert_allclose(P2.sum(axis=1), np.ones(P2.shape[0]), atol=1e-10)
