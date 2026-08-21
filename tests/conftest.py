# -*- coding: utf-8 -*-
"""
pytest 配置：将项目根目录加入 sys.path，
使测试能 import core.* 与 main 模块。
"""

import os
import sys

# 项目根目录 = tests/ 的上一级；scripts/ 也加入（供 from main import ...）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
for _p in (PROJECT_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
