# 打包清单（Package Manifest）

> 项目：`《道德经》马尔科夫链粗粒化建模`
> 压缩包：`daodejing_mc_release.zip`（96 个文件，15.1 MB，压缩率 66.7%）
> 生成日期：2026-08-21

---

## 一、包含内容

| 类别 | 文件/目录 | 说明 |
|------|----------|------|
| **入口** | `run_all.py` | 一键运行完整 Pipeline（推荐入口） |
| **主流程** | `main.py` | 主流程：清洗→转移矩阵→EI→粗粒化→可视化 |
| **核心模块** | `core/` | 公共算法模块（T12 重构） |
| **分析脚本** | `coarse_grain_v2.py` 等 11 个 | 粗粒化对比 / 结构诊断 / HMM / 可视化等 |
| **单元测试** | `tests/` | 39 个测试（T13） |
| **文档** | `README.md` / `HANDOFF.md` / `DESIGN_DOC_V2.md` / `TODO.md` / `KNOWN_ISSUES.md` | 项目说明与交接文档 |
| **方法论** | `docs/methodology.md` | 数学方法系统说明（T14） |
| **配置** | `requirements.txt` / `.gitignore` | 依赖清单 / Git 忽略规则 |
| **许可证** | `LICENSE` | MIT 许可证 |
| **报告** | `道德经概念动力学分析报告.docx` | Word 综合报告 |
| **数据** | `daodejing_sample.txt` | 原文样例 |
| **产出** | `output/` | 62 个可视化与数据文件 |

---

## 二、目录结构

```
daodejing_mc/
├── README.md                  # 项目简介、安装、快速开始
├── HANDOFF.md                 # 交接文档
├── DESIGN_DOC_V2.md           # 设计方案
├── TODO.md                    # 任务清单
├── KNOWN_ISSUES.md            # 已知问题
├── LICENSE                    # MIT 许可证
├── requirements.txt           # 依赖清单
├── .gitignore                 # Git 忽略规则
├── run_all.py                 # 一键运行
├── main.py                    # 主流程
│
├── core/                      # 公共模块（T12 重构）
│   ├── __init__.py
│   ├── env.py                 # 环境配置（UTF-8/字体/路径）
│   ├── pipeline.py            # 核心算法（转移矩阵/EI/粗粒化）
│   └── dynamics.py            # 结构动力学（可逆性/中心性）
│
├── tests/                     # 单元测试（T13，39 个）
│   ├── conftest.py
│   ├── test_concept_extraction.py
│   ├── test_transition_matrix.py
│   └── test_ei.py
│
├── docs/
│   └── methodology.md         # 方法论说明（T14）
│
├── output/                    # 全部产出（62 个文件）
│   ├── P_matrix.npy / P_macro.npy / pi.npy / Phi.npy   # 核心矩阵
│   ├── dashboard_data.json / reversibility.txt          # 汇总数据/报告
│   ├── vis_01~vis_09*.png                               # 10 张可视化
│   ├── vis_04_sankey_interactive.html                   # 交互式桑基图
│   └── *.csv / *.json                                   # 数据导出
│
├── main.py 等 13 个分析脚本
└── daodejing_sample.txt       # 原文样例
```

---

## 三、未包含（已排除）的内容

| 排除项 | 原因 |
|--------|------|
| `__pycache__/`、`*.pyc` | Python 编译缓存，可自动生成 |
| `.pytest_cache/` | 测试缓存 |
| `~$*.docx` | WPS/Word 临时锁文件 |
| `create_archive.py` | 打包脚本自身（非项目功能） |
| `.git/`、`.idea/`、`.vscode/` | 版本控制/IDE 目录 |

> **说明**：项目中**不包含**任何密钥、密码、API token 或本地绝对路径。代码路径均基于 `os.path.dirname(__file__)` 自动检测，跨平台可运行。

---

## 四、GitHub 上传建议

1. **仓库结构**：直接将压缩包解压后的内容作为仓库根目录。
2. **`.gitignore`**：已配置好，将自动排除 `__pycache__`、`*.pyc`、`.pytest_cache` 等（`output/` 默认保留以展示成果，如需精简可自行调整）。
3. **首次运行**：
   ```bash
   pip install -r requirements.txt
   python run_all.py        # 一键运行
   python -m pytest tests/  # 运行单元测试
   ```
4. **README 展示**：`README.md` 已包含项目简介、核心发现图表、安装与快速开始，适合作为 GitHub 首页。

---

## 五、核心指标速览（复现基准）

```
N = 31 概念 / T = 849 观测 / M = 6 宏观态
微观 EI_norm = 0.0515 / 宏观 EI_norm = 0.0119 / 因果涌现 = -0.0397
成块性误差 ε = 0.005 / 混合时间 τ = 1.38 步
HMM 软分配 EI = 0.0539
```
