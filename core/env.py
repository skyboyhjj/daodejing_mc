# -*- coding: utf-8 -*-
"""
core/env.py — 跨脚本共享的环境配置模块

抽取各脚本重复出现的三段样板代码：
  1. Windows 控制台 UTF-8 重配置（解决 GBK 无法输出 ✓ 等 Unicode 字符）
  2. 中文字体自动检测（get_cn_font，跨 Windows / Linux 平台）
  3. 项目根目录 / output 目录路径

用法：
  from core.env import setup_env, get_cn_font, CN_FONT, BASE_DIR, OUTPUT_DIR

  setup_env()          # 配置 UTF-8 + matplotlib 中文字体
  CN_FONT, OUTPUT_DIR  # 模块级便捷常量
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# ============================================================
# 1. Windows 控制台 UTF-8 重配置
# ============================================================
def setup_utf8_stdio():
    """Windows 控制台默认 GBK，无法输出 ✓/▶/╔ 等 Unicode 字符，统一改用 UTF-8。
    在无 reconfigure 属性（如 Jupyter/部分环境）时静默跳过。"""
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass


# ============================================================
# 2. 中文字体自动检测
# ============================================================
# 跨平台中文字体优先级：Windows 优先微软雅黑/黑体，Linux 用 Noto CJK（简体优先）
CN_FONT_CANDIDATES = [
    'Microsoft YaHei',      # Windows 微软雅黑
    'SimHei',               # Windows 黑体
    'Noto Sans CJK SC',     # 优先简体中文（项目为简体文本）
    'Noto Sans CJK JP',     # Linux Debian/Ubuntu 常装
    'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
    'WenQuanYi Zen Hei',
    'Arial Unicode MS',     # macOS
]


def get_cn_font(candidates=None):
    """自动检测可用的中文字体（兼容 Windows / Linux / macOS）。
    直接扫描已注册字体族名（fontManager.ttflist），不用 findfont 的 fallback，
    因为 findfont 会对缺失字体静默返回替代字体，掩盖真实可用性。"""
    if candidates is None:
        candidates = CN_FONT_CANDIDATES
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for f in candidates:
        if f in installed:
            return FontProperties(family=f)
    return FontProperties()


# ============================================================
# 3. 项目路径
# ============================================================
# 项目根目录 = 本模块（core/env.py）的上两级目录 = 脚本所在目录
# 原沙盒硬编码为 /data/workspace/daodejing_mc，改为自动检测以支持任意平台
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 便捷初始化
# ============================================================
def setup_env():
    """一键完成环境配置：UTF-8 输出 + matplotlib 中文字体。
    返回 (CN_FONT, OUTPUT_DIR)，供脚本直接使用。"""
    setup_utf8_stdio()
    font = get_cn_font()
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    print(f"  [字体] 使用: {font.get_name()}")
    return font, OUTPUT_DIR


# 模块加载时自动配置（保证 import core.env 即可用，无需显式调用）
setup_utf8_stdio()
CN_FONT = get_cn_font()
plt.rcParams['font.family'] = CN_FONT.get_name()
plt.rcParams['axes.unicode_minus'] = False
