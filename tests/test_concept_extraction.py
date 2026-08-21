# -*- coding: utf-8 -*-
"""
tests/test_concept_extraction.py — 概念抽取单元测试

核心验证点（见 TODO.md T13）：
  1. "无为"是完整概念，不被拆成 "无"+"为"
  2. 多字概念（如"无为""小国寡民""上善若水"）优先于单字匹配
  3. 单字概念回退正确
  4. 全书 81 章均能抽取到概念（无空章）
  5. 总观测数稳定（N=31 概念，T=849 观测）
"""

import pytest

from core.pipeline import clean_text, extract_concepts
from main import CONCEPT_DICT, REVERSE_MAP, DAODEJING


# ============================================================
# 1. "无为"不被拆成 "无"+"为"
# ============================================================
def test_wuwei_not_split():
    """'无为' 应整体识别为一个概念，而非拆成 '无' + '为'"""
    text = "无为而无不为"
    seq = extract_concepts(text, REVERSE_MAP)
    # "为" 不在概念词典单字中，因此 "无为" 匹配后不应再产生 "为"
    assert "无为" in seq, f"序列应包含 '无为'，实际: {seq}"
    # 不应出现单独的 '为'（非概念字）
    assert "为" not in seq, f"'为' 不是独立概念，不应出现在序列中，实际: {seq}"


def test_wuwei_matches_before_single_wu():
    """'无' 与 '无为' 共存时，多字 '无为' 应优先匹配"""
    text = "无为则无不治"
    seq = extract_concepts(text, REVERSE_MAP)
    # "无为" 应作为一个整体
    assert seq[0] == "无为", f"首个概念应为 '无为'，实际: {seq}"
    # 后面 "无不治" 中的 "无" 是单字概念
    assert "无" in seq, f"序列应包含单字 '无'，实际: {seq}"


# ============================================================
# 2. 多字概念优先于单字
# ============================================================
def test_multi_char_priority():
    """'小国寡民' 4 字应整体匹配，而非拆成单字"""
    text = "小国寡民，使有什伯之器"
    seq = extract_concepts(text, REVERSE_MAP)
    assert "小国寡民" in seq, f"应识别出 '小国寡民'，实际: {seq}"


def test_longest_match_wins():
    """'上善若水' 应匹配为 '水'（其变体），而非 '上'"""
    text = "上善若水。水善利万物而不争"
    seq = extract_concepts(text, REVERSE_MAP)
    # "上善若水" 映射到 "水"（见 CONCEPT_DICT["水"]）
    assert "水" in seq, f"应识别出 '水'，实际: {seq}"


def test_clean_text_strips_punctuation():
    """clean_text 应去除标点、引号、空白，只留汉字"""
    text = "道可道，非常道。名可名，非常名。"
    cleaned = clean_text(text)
    assert "，" not in cleaned
    assert "。" not in cleaned
    assert " " not in cleaned
    assert cleaned == "道可道非常道名可名非常名"


# ============================================================
# 3. 单字回退
# ============================================================
def test_single_char_fallback():
    """'道' 单字应被识别"""
    text = "道可道"
    seq = extract_concepts(text, REVERSE_MAP)
    assert seq == ["道", "道"], f"应识别出两个 '道'，实际: {seq}"


def test_unknown_char_skipped():
    """非概念字（如 '可'）应被跳过"""
    text = "道可道"
    seq = extract_concepts(text, REVERSE_MAP)
    assert all(c in REVERSE_MAP.values() for c in seq), "所有抽取概念应为标准化概念"
    assert "可" not in seq, f"'可' 非概念字，不应出现，实际: {seq}"


# ============================================================
# 4. 全书完整性
# ============================================================
def test_all_81_chapters_have_concepts():
    """81 章每章都应抽取到概念（无空章）"""
    from main import build_full_sequence
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    assert len(chapter_seqs) == 81, f"应有 81 章，实际 {len(chapter_seqs)}"
    empty = [ch for ch, seq in chapter_seqs.items() if len(seq) == 0]
    assert not empty, f"存在空章节: {empty}"


def test_total_observations_stable():
    """全书总概念观测数应为 849（稳定基线）"""
    from main import build_full_sequence
    full_seq, _ = build_full_sequence(DAODEJING)
    assert len(full_seq) == 849, f"总观测数应为 849，实际 {len(full_seq)}"


def test_unique_concepts_31():
    """全书唯一概念数应为 31"""
    from main import build_full_sequence
    full_seq, _ = build_full_sequence(DAODEJING)
    assert len(set(full_seq)) == 31, f"唯一概念数应为 31，实际 {len(set(full_seq))}"


# ============================================================
# 5. 概念词典自洽性
# ============================================================
def test_reverse_map_complete():
    """REVERSE_MAP 中每个变体都应映射到词典中的一个标准概念"""
    for variant, standard in REVERSE_MAP.items():
        assert standard in CONCEPT_DICT, f"变体 '{variant}' 映射到不存在的标准概念 '{standard}'"


def test_all_standard_concepts_are_words():
    """词典标准概念应非空"""
    assert all(len(c) > 0 for c in CONCEPT_DICT), "不应有空概念名"


def test_wuwei_is_distinct_concept():
    """'无为' 是独立标准概念，且其变体包含 '无为'"""
    assert "无为" in CONCEPT_DICT, "'无为' 应是独立概念"
    assert "无为" in CONCEPT_DICT["无为"], "'无为' 的变体列表应包含自身"
