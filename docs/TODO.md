# TODO — codeBuddy 后续工作清单

按优先级排列，每项标注预估工时和难度。
**更新 · 2026-08-21（codeBuddy 接手后）** — P0 全部完成，P1/P2 大部分完成。

---

## ✅ P0：收尾已有工作（已完成）

### ✅ T01. 运行 run_all.py 一键验证
- ✅ `cd <项目根> && python run_all.py`
- ✅ 确认 6 张可视化 PNG 全部生成（vis_01-06）
- ✅ 确认 `道德经概念动力学分析报告.docx` 生成
- ✅ 48 个文件写入 `output/`

### ✅ T02. 修复 generate_report.py 中的文本错误
- ✅ 第 274 行 "因果涌现为正值" → "因果涌现为负值"（与实测 -0.0406 一致）
- ✅ 7.1 节 "粗粒化后的 EI 提升" → 改为成块性自洽论述
- ✅ 7.4 节 "因果涌现为正" → 改为"道的动力学不可压缩"哲学解读

---

## ✅ P1：可视化增强（已完成 2/3）

### ✅ T03. 桑基图升级
- ✅ 新增 `vis_sankey_interactive.py`
- ✅ `output/vis_04_sankey_interactive.html`（plotly 交互版，浏览器可拖动）
- ✅ 增强：hover 显示真实概率（微观流=平稳概率 π，宏观流=转移概率 P'），流带宽统一视觉缩放
- ⏸️ 静态 plotly PNG 未生成（kaleido/Chrome 沙盒不可用；matplotlib 静态版仍保留）

### ✅ T04. 补充"概念时间线"可视化
- ✅ 新增 `vis_07_timeline.py` → `output/vis_07_timeline.png`
- ✅ 关键发现：治术（治理）平均首次出现于第 22.4 章（前 1/4 卷未讨论）
- ✅ 写出 `output/concept_timeline.csv`（每个概念的首次/末次/跨度章节）

### ⏸️ T05. 交互式仪表盘
- ⏸️ 环境已有 `streamlit`，但需独立运行 streamlit 服务；本轮未实现

---

## ✅ P2：补充分析方法（已完成 3/3）

### ✅ T06. HMM 软分配
- ✅ `pip install hmmlearn`（0.3.3）
- ✅ 新增 `hmm_analysis.py`
- ✅ 关键洞察：自由发射 HMM 在 M=2..6 全部收敛到同一 LL（-2617.34）→ 数据过于均匀
- ✅ 采用"固定发射 B = 硬划分的软化版本 + 仅训练 A"策略
- ✅ 软分配 EI_norm = 0.0531 > 微观 0.0526 > 硬划分 0.0119
- ✅ 因果涌现从硬划分的 -0.0406 提升到 +0.0006
- ✅ 31/31 概念软分配 MLE 标签与硬划分吻合

### ✅ T07. 可逆性检验 + 混合时间
- ✅ 在 `structural_diagnostics.py` 中实现
- ✅ `‖F−Fᵀ‖_F / ‖F‖_F = 0.41`（绝对 0.0199）→ 弱可逆
- ✅ τ_mix = 1.38 步（极快混合）
- ✅ 流量最不对称对：天地→道 (+0.0044)
- ✅ 写入 `output/reversibility.txt`

### ✅ T08. 随机游走中心性
- ✅ 在 `structural_diagnostics.py` 中实现
- ✅ PageRank TOP：民 > 无 > 道 > 有 > 知
- ✅ 命中时间最短：水/自然/朴/不言/小国寡民
- ✅ 覆盖时间：185 步（蒙特卡洛）
- ✅ 写入 `output/centrality_rankings.csv`

---

## ⏸️ P3：跨文本扩展（未启动）

需外部文本（马王堆帛书、《庄子》内篇等），本轮未抓取。

### ⏸️ T09. 帛书本对照
- ⏸️ 需获取马王堆帛书《道德经》甲乙本
- ⏸️ 新建 `data/daodejing_boshu.py`，跑同一 pipeline
- ⏸️ 输出帛书 vs 王弼本的宏观转移矩阵差异

### ⏸️ T10. 《庄子》内篇分析
- ⏸️ 需获取《庄子》内篇 7 篇文本
- ⏸️ 适配概念词典（逍遥/齐物等独有概念）
- ⏸️ 量化"老庄异同"

### ⏸️ T11. 跨文本对比可视化
- ⏸️ 雷达图 + 网络对齐算法

---

## ✅ P4：工程化（T12 已启动）

### ✅ T12. 代码重构（已完成）
- ✅ 新建 `core/` 公共模块：
  - `core/env.py`：环境配置（UTF-8 / 中文字体 get_cn_font / 项目路径）
  - `core/pipeline.py`：马尔科夫链核心算法（清洗 / 转移矩阵 / 平稳分布 / EI / 粗粒化 / 语义分组）
  - `core/dynamics.py`：结构动力学函数（可逆性 / 混合时间 / PageRank / 命中时间 / 覆盖时间）
- ✅ 消除重复代码：
  - `get_cn_font` 从 5 处重复（main / run_all_visualizations / generate_report / structural_diagnostics / hmm_analysis）→ 收敛到 `core.env` 唯一一处
  - UTF-8 stdout 配置从 6 处重复 → 统一到 `core.env.setup_utf8_stdio`
  - 核心算法（build_transition_matrix / stationary_distribution / effective_information / normalized_ei / lumpability_error / svd_coarse_grain / build_macro_transition）从 main.py 抽到 `core.pipeline`
  - T07/T08 动力学函数从 structural_diagnostics.py 抽到 `core.dynamics`
- ✅ 拆分过长函数：
  - `main.py` 的 `main()`（311 行）拆为 7 个阶段函数 + 协调器
  - 消除 `main.py` 中"保存数据"的重复代码（原来出现两次）
- ✅ 保持功能不变：所有脚本 `python run_all.py` 全部 OK
  - 微观 EI_norm = 0.0526 / 宏观 EI_norm = 0.0119 / 因果涌现 = -0.0406（与重构前完全一致）
- ✅ 增强健壮性：`generate_report.py` 保存时若报告文件被 WPS/Word 占用，优雅降级为另存 `_新.docx` 并提示

### ✅ T13. 单元测试（已完成）
- ✅ `tests/test_concept_extraction.py`（13 测试）：验证"无为"不被拆成"无"+"为"、多字优先、单字回退、全书完整性
- ✅ `tests/test_transition_matrix.py`（14 测试）：验证行和=1、方阵、idx/inv_idx 互逆、计数矩阵手算、平稳分布
- ✅ `tests/test_ei.py`（12 测试）：用已知解析结果验证 EI（均匀=0 / 确定性=log(N) / 归一化）
- ✅ 运行：`python -m pytest tests/ -v`（39 passed）
- 🔴 **发现并修复 REVERSE_MAP 数据 bug**：`"无为"` 变体冲突被 `"无"` 覆盖 → 修复为冲突时优先更长标准概念
  - 影响：微观 EI_norm 0.0526→0.0515，因果涌现 -0.0406→-0.0397（总观测 849 不变）
  - 单元测试暴露了隐藏数据 bug，验证了测试的价值

### ✅ T14. README + 文档（已完成）
- ✅ 编写 `README.md`：项目简介、核心发现、安装、快速开始、产出文件说明、项目结构
- ✅ 新建 `docs/methodology.md`：系统说明数学方法（预处理/转移矩阵/EI/粗粒化/结构诊断/结果解读/局限性/复现）
- ✅ 更新 `HANDOFF.md`（本文件已含全部历史记录）
- ✅ 验证：README 中所有命令（run_all.py / pytest / 各脚本）均可执行

---

## 标注约定

- ✅ = 已完成
- ⏸️ = 未完成 / 待启动
- 🔴 P0 = 必须完成（已全部完成 ✅）
- 🟡 P1 = 高价值（已完成 2/3）
- ✅ P2 = 全部完成
- ⏸️ P3/P4 = 未启动（需外部资源 / 较大投入）
