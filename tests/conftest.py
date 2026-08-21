# -*- coding: utf-8 -*-
"""
pytest 配置：将项目根目录加入 sys.path，
使测试能 import core.* 与 main 模块。
"""

import os
import sys

# 项目根目录 = tests/ 的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
