# -*- coding: utf-8 -*-
"""
core/pipeline.py — 马尔科夫链建模核心算法

从 main.py 抽取的纯计算函数，与可视化/IO 解耦，供所有脚本复用：
  - 文本清洗与概念抽取
  - 转移矩阵构建（k=1 / k=2）
  - 平稳分布（幂迭代）
  - 有效信息 EI / 归一化 EI
  - 成块性检验
  - SVD 谱分解 + 粗粒化
  - 宏观转移矩阵构建
  - 语义分组映射

注意：本模块只含纯函数与 numpy/scipy 依赖，不含 matplotlib。
"""

import re
import numpy as np
from scipy.linalg import eig  # noqa: F401（部分脚本依赖 scipy eig）
from sklearn.cluster import KMeans


# ============================================================
# 文本清洗与概念抽取
# ============================================================
def clean_text(text):
    """
    去除所有标点和空白。
    只保留 CJK 汉字与数字字母（\\u4e00-\\u9fff 为 Unicode 中文字符区间），
    其余字符（标点、引号、空白等）一律删除。
    相比显式列举标点符号，此写法更稳健，且避免引号提前终止字符串字面量的问题。
    """
    return re.sub(r'[^\u4e00-\u9fff0-9A-Za-z]', '', text)


def extract_concepts(text, reverse_map, chapter_num=None):
    """
    最长优先匹配 + 单字回退。
    返回该章的概念序列列表。

    reverse_map: 变体 → 标准化概念 的映射（由 main.CONCEPT_DICT 构建）
    """
    text = clean_text(text)
    concepts = []
    i = 0
    while i < len(text):
        matched = False
        # 先尝试多字匹配（2-4字）
        for length in range(min(4, len(text) - i), 1, -1):
            chunk = text[i:i + length]
            if chunk in reverse_map:
                concepts.append(reverse_map[chunk])
                i += length
                matched = True
                break
        if not matched:
            # 单字匹配
            ch = text[i]
            if ch in reverse_map:
                concepts.append(reverse_map[ch])
            # 未识别的单字直接跳过
            i += 1
    return concepts


def build_full_sequence(chapter_texts, reverse_map, order='linear'):
    """
    构建全书概念序列。
    order='linear'：按1-81章顺序拼接
    order='sentence'：按句号切分，句内概念展开
    返回 (full_seq, chapter_seqs)
    """
    full_seq = []
    chapter_seqs = {}
    for ch_num in sorted(chapter_texts.keys()):
        seq = extract_concepts(chapter_texts[ch_num], reverse_map, ch_num)
        chapter_seqs[ch_num] = seq
        full_seq.extend(seq)
    return full_seq, chapter_seqs


# ============================================================
# 转移矩阵构建
# ============================================================
def build_transition_matrix(seq, k=1, smoothing=1.0):
    """
    构建 k 阶转移计数矩阵 + Laplace 平滑后的概率矩阵。
    返回: P (N×N), C (原始计数), idx (概念→索引), inv_idx (索引→概念)
    k=2 时返回 P2, C2, pair_idx, pair_list, idx, inv_idx
    """
    unique = sorted(list(set(seq)))
    n = len(unique)
    idx = {c: i for i, c in enumerate(unique)}
    inv_idx = {i: c for c, i in idx.items()}

    C = np.zeros((n, n), dtype=float)
    if k == 1:
        for t in range(len(seq) - 1):
            C[idx[seq[t]], idx[seq[t + 1]]] += 1
    elif k == 2:
        # 联合状态：(c_t, c_{t+1}) → (c_{t+1}, c_{t+2})
        # 注意：必须构建方阵（联合态→联合态），否则无法求平稳分布
        pair_idx = {}
        pair_list = []
        for t in range(len(seq) - 1):
            pair = (seq[t], seq[t + 1])
            if pair not in pair_idx:
                pair_idx[pair] = len(pair_list)
                pair_list.append(pair)
        n_pairs = len(pair_list)
        C2 = np.zeros((n_pairs, n_pairs), dtype=float)
        for t in range(len(seq) - 2):
            src = pair_idx[(seq[t], seq[t + 1])]
            dst = pair_idx[(seq[t + 1], seq[t + 2])]
            C2[src, dst] += 1
        # 平滑
        C2_smooth = C2 + smoothing
        P2 = C2_smooth / C2_smooth.sum(axis=1, keepdims=True)
        return P2, C2, pair_idx, pair_list, idx, inv_idx

    # k=1: 平滑 + 概率化
    C_smooth = C + smoothing
    row_sums = C_smooth.sum(axis=1, keepdims=True)
    P = C_smooth / row_sums
    return P, C, idx, inv_idx


# ============================================================
# 平稳分布（幂迭代法）
# ============================================================
def stationary_distribution(P, tol=1e-14, max_iter=100000):
    """πP = π, Σπ = 1"""
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            return pi_new
        pi = pi_new
    return pi


# ============================================================
# 有效信息 EI
# ============================================================
def effective_information(P, pi=None):
    """EI(P) = Σ_i π_i · D_KL(P_i || P_bar)"""
    if pi is None:
        pi = stationary_distribution(P)
    eps = 1e-15
    P_bar = pi @ P + eps
    P_safe = P + eps
    ei = 0.0
    for i in range(P.shape[0]):
        if pi[i] > 0:
            kl = np.sum(P_safe[i] * (np.log(P_safe[i]) - np.log(P_bar)))
            ei += pi[i] * kl
    return ei


def normalized_ei(P, pi=None):
    """归一化 EI = EI / log(N)"""
    if pi is None:
        pi = stationary_distribution(P)
    N = P.shape[0]
    if N <= 1:
        return 0.0
    ei = effective_information(P, pi)
    return ei / np.log(N)


# ============================================================
# 成块性检验
# ============================================================
def lumpability_error(P, partition, idx):
    """
    给定划分 partition（list of lists of concept names），
    计算最大成块性误差。
    """
    error = 0.0
    for block in partition:
        block_indices = [idx[c] for c in block if c in idx]
        if len(block_indices) < 2:
            continue
        for target_block in partition:
            target_indices = [idx[c] for c in target_block if c in idx]
            if not target_indices:
                continue
            probs = []
            for s in block_indices:
                p_to_j = sum(P[s, t] for t in target_indices)
                probs.append(p_to_j)
            if len(probs) > 1:
                var = np.var(probs)
                error = max(error, var)
    return error


# ============================================================
# SVD 谱分解 + 粗粒化
# ============================================================
def svd_coarse_grain(P, pi, num_macro_states=6, seed=42):
    """
    对稳态流矩阵 F = diag(pi) @ P 做 SVD，
    取前 K 个右奇异向量做 K-Means 聚类。
    返回: labels (N,), cluster_centers, explained_variance, s, embedding
    """
    F = np.diag(pi) @ P
    U, s, Vt = np.linalg.svd(F)

    K = num_macro_states
    # 取前 K 个右奇异向量作为嵌入
    embedding = Vt[:K, :].T  # N × K

    # K-Means 聚类
    km = KMeans(n_clusters=K, random_state=seed, n_init=20, max_iter=500)
    labels = km.fit_predict(embedding)

    explained = (s[:K] ** 2).sum() / (s ** 2).sum()

    return labels, km.cluster_centers_, explained, s, embedding


def build_macro_transition(P, labels, idx, inv_idx):
    """
    根据微观标签构建宏观转移矩阵 P_macro。
    Φ[i, j] = 1 如果微观态 i 属于宏观态 j。
    P_macro = Φ^T @ P @ Φ （归一化后）
    """
    N = P.shape[0]
    M = len(set(labels))
    Phi = np.zeros((N, M))
    for i in range(N):
        Phi[i, labels[i]] = 1.0

    P_macro_unnorm = Phi.T @ P @ Phi
    row_sums = P_macro_unnorm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P_macro = P_macro_unnorm / row_sums

    return P_macro, Phi


# ============================================================
# 语义分组映射
# ============================================================
def semantic_macro_labels(idx, inv_idx, semantic_partition):
    """
    根据手工语义分组为每个微观概念分配宏观标签（N 维向量）。
    供 build_outputs / export_visualization_data 等脚本复用，
    保证全链路使用与最终结论一致的"手工语义分组"。
    """
    N = len(inv_idx)
    labels = np.zeros(N, dtype=int)
    for m, block in enumerate(semantic_partition):
        for c in block:
            labels[idx[c]] = m
    return labels


# ============================================================
# 谱空间嵌入（SVD 坐标，用于散点图）
# ============================================================
def svd_embedding(pi, P, num_macro_states=6):
    """返回 (embedding, s_vals, explained_variance)
    F = diag(pi) @ P，取前 M 个右奇异向量作为嵌入坐标。"""
    F = np.diag(pi) @ P
    _, s_vals, Vt = np.linalg.svd(F)
    M = num_macro_states
    embedding = Vt[:M, :].T  # N×M
    explained = (s_vals[:M] ** 2).sum() / (s_vals ** 2).sum()
    return embedding, s_vals, explained
