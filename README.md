# 《道德经》马尔科夫链粗粒化建模

将《道德经》81 章视为一个**马尔科夫链**，对概念转移进行建模，并通过**粗粒化（Coarse-Graining）**寻找隐藏的宏观义理结构，量化"道可道，非常道"的信息论本质。

## 项目简介

《道德经》五千言中，"道""德""无""有"等核心概念反复出现、彼此牵引。本项目用信息论 + 马尔科夫链的视角回答三个问题：

1. **概念如何流动？** —— 把全书 849 次概念观测建成转移矩阵 P，刻画"民→无→道"等概念间的转移概率。
2. **能否粗粒化？** —— 用 SVD 谱分解、层次聚类、手工语义分组将 31 个微观概念聚成 6 个宏观义理块，看是否涌现更强的因果结构。
3. **动力学是否自洽？** —— 用有效信息（EI）、成块性检验、混合时间、时间可逆性量化文本结构。

### 核心发现（实测）

> **粗粒化不涌现，但动力学极快混合。**

- 微观 EI = **0.0515**，宏观 EI（M=6）= **0.0119** → **因果涌现为负（-0.0397）**：粗粒化不增加可预测性，《道德经》的概念动力学本身是"扁平"的。
- 成块性误差 ε = **0.005**：分组在"流量守恒"意义上是自洽的，但不涌现信息。
- 混合时间 τ_mix = **1.38 步**：概念动力学极快收敛，约读 5 章即可把握全书核心。
- HMM 软分配 EI = **0.0539**（高于微观 0.0515）：软分配比硬划分更能恢复时间序信息。

这不是失败，而是对文本本质的**诚实刻画**——老子的"循环论证"（道→无→有→道）使转移矩阵接近均匀混合，信息天然偏低。

## 安装

### 环境要求

- Python 3.9+
- 建议使用虚拟环境

### 依赖安装

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn networkx
pip install python-docx       # Word 报告（T01 必需）
pip install hmmlearn plotly kaleido  # 可选：HMM 分析 / 交互式桑基图
pip install pytest            # 可选：运行单元测试
```

> **Windows 中文字体**：脚本自动检测 `Microsoft YaHei` / `SimHei`（Windows）、`Noto Sans CJK SC` / `Noto Sans CJK JP`（Linux，简体优先），无需手动配置。

## 快速开始

### 一键运行完整 Pipeline（推荐）

```bash
python scripts/run_all.py
```

依次执行：主流程 → 粗粒化对比 → 数据导出 → 仪表盘数据 → 6 种可视化 → Word 报告。全部产出写入 `output/`。

### 分步运行

```bash
# 1. 主流程：清洗 → 转移矩阵 → EI → 粗粒化 → 6 种可视化
python scripts/main.py

# 2. 多方案粗粒化对比（手工语义 vs Ward vs K-Means）
python scripts/coarse_grain_v2.py

# 3. 导出可视化数据（JSON/CSV）
python scripts/export_visualization_data.py
python scripts/build_outputs.py

# 4. 补充分析
python scripts/structural_diagnostics.py   # T07/T08：可逆性 + 混合时间 + 中心性
python scripts/hmm_analysis.py            # T06：HMM 软分配
python scripts/vis_07_timeline.py         # T04：概念时间线
python scripts/vis_sankey_interactive.py  # T03：plotly 交互式桑基图

# 5. 生成 Word 报告
python scripts/generate_report.py
```

### 运行单元测试

```bash
python -m pytest tests/ -v
```

39 个测试覆盖概念抽取、转移矩阵、EI 计算。

## 产出文件（`output/`）

### 核心数据

| 文件 | 说明 |
|------|------|
| `P_matrix.npy` / `P_macro.npy` | 微观 / 宏观转移矩阵 |
| `pi.npy` / `Phi.npy` | 平稳分布 / 粗粒化映射 |
| `dashboard_data.json` | 仪表盘汇总数据 |
| `reversibility.txt` | T07 可逆性 + 混合时间报告 |
| `centrality_rankings.csv` | T08 中心性排名 |
| `hmm_results.json` / `P_hmm.npy` | HMM 软分配结果 |

### 可视化（10 张 PNG + 1 交互 HTML）

| 文件 | 内容 |
|------|------|
| `vis_01_network.png` | 微观概念网络图（+ GML 供 Gephi） |
| `vis_02_heatmap*.png` | 转移矩阵聚类热力图 |
| `vis_03_spectral.png` | SVD 谱空间散点图 |
| `vis_04_sankey.png` / `.html` | 粗粒化桑基图（plotly 交互版） |
| `vis_05_theme_river.png` | 主题河流图（章节 × 宏观态） |
| `vis_06_emergence.png` | 因果涌现曲线 + 奇异值谱 |
| `vis_07_timeline.png` | 概念时间线（T04） |
| `vis_08_dynamics.png` | 结构诊断（T07+T08） |
| `vis_09_hmm.png` | HMM 软分配（T06） |

### 报告

- `道德经概念动力学分析报告.docx` — Word 综合报告（T01）

## 项目结构

```
daodejing_mc/
├── scripts/                    # 全部脚本（分析 / 可视化 / 报告）
│   ├── main.py                 #   主流程（清洗 → 转移矩阵 → EI → 粗粒化 → 可视化）
│   ├── run_all.py              #   一键运行全部 Pipeline
│   ├── coarse_grain_v2.py      #   多方案粗粒化对比（语义/Ward/KMeans）
│   ├── export_visualization_data.py  #   导出 JSON/CSV
│   ├── build_outputs.py        #   构建仪表盘数据
│   ├── run_all_visualizations.py     #   6 种可视化生成
│   ├── structural_diagnostics.py     #   T07/T08 可逆性 + 中心性
│   ├── hmm_analysis.py         #   T06 HMM 软分配
│   ├── vis_07_timeline.py      #   T04 概念时间线
│   ├── vis_sankey_interactive.py     #   T03 plotly 交互式桑基图
│   ├── generate_report.py      #   Word 综合报告
│   └── diagnose*.py / final_summary.py / test_clustering.py  # 诊断与辅助
├── core/                       # 公共模块（T12 重构）
│   ├── env.py                  #   环境配置（UTF-8/字体/路径）
│   ├── pipeline.py             #   核心算法（转移矩阵/EI/粗粒化）
│   └── dynamics.py             #   结构动力学（可逆性/中心性）
├── tests/                      # 单元测试（T13）
│   ├── conftest.py
│   ├── test_concept_extraction.py
│   ├── test_transition_matrix.py
│   └── test_ei.py
├── docs/                       # 项目文档
│   ├── HANDOFF.md              #   交接文档
│   ├── DESIGN_DOC_V2.md        #   设计方案
│   ├── methodology.md          #   方法论
│   ├── TODO.md                 #   任务清单
│   └── KNOWN_ISSUES.md         #   已知问题
├── daodejing_sample.txt        # 示例文本（3 章）
├── output/                     # 全部产出（运行脚本生成）
├── requirements.txt            # 依赖清单
├── LICENSE                     # 许可证
└── README.md                   # 本文档
```

### 脚本功能一览

| 脚本（均在 `scripts/` 下） | 功能 |
|------|------|
| `main.py` | 主流程：清洗 → 转移矩阵 → EI → 粗粒化 → 可视化 |
| `coarse_grain_v2.py` | 多方案粗粒化对比（语义/Ward/KMeans） |
| `structural_diagnostics.py` | T07 可逆性 + T08 中心性 |
| `hmm_analysis.py` | T06 HMM 软分配分析 |
| `run_all_visualizations.py` | 6 种可视化生成 |
| `vis_07_timeline.py` | T04 概念时间线 |
| `vis_sankey_interactive.py` | T03 plotly 交互式桑基图 |
| `generate_report.py` | Word 综合报告 |
| `build_outputs.py` / `export_visualization_data.py` | 数据导出 |
| `diagnose*.py` / `final_summary.py` / `test_clustering.py` | 诊断与辅助 |

## 方法论

详见 [`docs/methodology.md`](docs/methodology.md)。

## 许可证

仅供学术研究与文化分析使用。

## 引用

如使用本项目的分析结果，请注明基于《道德经》王弼通行本（第 10 章已校订为"爱国治民"）。
