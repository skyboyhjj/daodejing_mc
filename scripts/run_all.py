# -*- coding: utf-8 -*-
"""
一键运行脚本：按顺序执行全部 pipeline
供 codeBuddy 接手后首次运行使用

执行顺序：
  1. main.py              → 主流程（清洗→P→π→SVD→粗粒化→保存）
  2. coarse_grain_v2.py   → 多方案粗粒化对比
  3. export_visualization_data.py → 导出 JSON/CSV
  4. build_outputs.py      → 构建仪表盘数据
  5. run_all_visualizations.py  → 6 种可视化
  6. generate_report.py    → Word 报告

用法：python run_all.py
"""

import subprocess
import sys
import os
import time

# Windows 控制台默认 GBK，无法输出 ▶╔ 等 Unicode 字符；统一改用 UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 脚本目录 = 本脚本所在目录（scripts/）；项目根目录 = scripts/ 的上一级
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
os.chdir(BASE_DIR)

SCRIPTS = [
    ('main.py',                      '主流程（清洗→转移矩阵→粗粒化）'),
    ('coarse_grain_v2.py',          '多方案粗粒化对比'),
    ('export_visualization_data.py', '导出可视化数据'),
    ('build_outputs.py',             '构建仪表盘数据'),
    ('run_all_visualizations.py',   '6 种可视化生成'),
    ('generate_report.py',           'Word 综合报告'),
]

def run_script(script_path, description):
    print(f"\n{'='*70}")
    print(f"  ▶ 运行: {os.path.basename(script_path)}")
    print(f"  ▷ 说明: {description}")
    print(f"{'='*70}\n")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        cwd=BASE_DIR
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n  ⚠️ 脚本异常退出 (returncode={result.returncode})，耗时 {elapsed:.1f}s")
        return False
    else:
        print(f"\n  ✓ 完成，耗时 {elapsed:.1f}s")
        return True

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   《道德经》马尔科夫链 — 一键运行全部 Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    results = []
    for script, desc in SCRIPTS:
        path = os.path.join(SCRIPTS_DIR, script)
        if not os.path.exists(path):
            print(f"\n  ⚠️ 文件不存在: {script}，跳过")
            results.append((script, 'SKIP'))
            continue
        ok = run_script(path, desc)
        results.append((script, 'OK' if ok else 'FAIL'))
    
    # 汇总
    print(f"\n{'='*70}")
    print("  运行汇总")
    print(f"{'='*70}")
    for script, status in results:
        icon = '✓' if status == 'OK' else ('⚠️' if status == 'SKIP' else '✗')
        print(f"  {icon} {script:<35s} [{status}]")
    
    # 列出 output 目录
    output_dir = os.path.join(BASE_DIR, 'output')
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        print(f"\n  output/ 目录 ({len(files)} 个文件):")
        for f in files:
            fpath = os.path.join(output_dir, f)
            size = os.path.getsize(fpath)
            print(f"    • {f:<40s} {size/1024:>8.1f} KB")
    
    # 列出根目录文档
    print(f"\n  项目文档:")
    for doc in ['docs/HANDOFF.md', 'docs/DESIGN_DOC_V2.md', 'scripts/run_all.py']:
        fpath = os.path.join(BASE_DIR, doc)
        if os.path.exists(fpath):
            print(f"    • {doc}")

if __name__ == "__main__":
    main()
