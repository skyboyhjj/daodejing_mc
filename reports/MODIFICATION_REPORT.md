# 最近修改与验证报告

> 报告日期：2026-08-22
> 范围：P1/P2 系列实验 + P2-B 评估 + 三个修复
> 项目：《道德经》马尔科夫链粗粒化建模（scripts/ + core/ 结构，与 GitHub 一致）

---

## 一、本轮新增实验脚本

| 脚本 | 对应任务 | 说明 |
|------|---------|------|
| `scripts/experiment_mc_w2v.py` | P1-A | MC×W2V 融合（DeepWalk 范式）+ 下游评估 |
| `scripts/experiment_w2v_gmm.py` | P1-B | W2V 嵌入上的 GMM 软聚类 → 宏观 EI |
| `scripts/experiment_w2v_mc_back.py` | P2-A | W2V→MC 反向融合（图传播）|
| `scripts/experiment_mc_w2v.py`（增强）| P1-A | 修复概念名提取 bug + 新增节点分类/链接预测 |

**依赖新增**：`gensim`（已安装 4.4.0，加入 requirements.txt）

---

## 二、实验结果与核心发现

### P1-A：MC×W2V 融合（DeepWalk 范式）

| 指标 | W2V | SVD | 结论 |
|------|-----|-----|------|
| Kendall τ | 0.1976 (p=1.92e-10) | — | 结构低相关 → 对应"正言若反" |
| 宏观态内相似度 | 0.7465 | — | — |
| 宏观态间相似度 | 0.7246 | — | — |
| 节点分类准确率 | 0.2000 | 0.2000 | 31 节点太小，统计功效不足 |
| **链接预测 AUC** | **0.6291** | 0.4455 | **W2V 显著优于 SVD** |

**关键发现**：W2V 局部共现结构 ≠ SVD 全局谱结构（τ=0.198）。但 W2V 在链接预测（局部转移预测）上更强。

### P1-B：W2V-GMM 软聚类 → 宏观 EI

| 指标 | 值 |
|------|-----|
| GMM 软分配 EI (M=6) | **0.0849** |
| HMM 软分配 EI | 0.0539 |
| GMM - HMM | +0.0310 |
| 后验熵 | 0.0000（软退化接近硬）|

**诚实发现（软效应隔离分析）**：
- 公式效应（硬划分改用软EI公式）= +0.0357
- **软效应（真实软 vs one-hot 硬）= +0.0237**（对 seed 不稳定）
- HMM 的软效应 = -0.0012（微负）
- **结论**："软聚类"提升 EI 的说法不成立——提升主要来自 EI 公式 + GMM 聚类结构，而非"软归属"

### P2-A：W2V→MC 反向融合（图传播）

| 方案 | ε | vs 手工语义 (0.00462) |
|------|---|---|
| 纯 SVD(16d) | **0.001697** | ↓ -63% 最优 |
| 图传播 α=0.0 | 0.002674 | ↓ -42% |
| 图传播 α=0.5 | 0.005013 | ↑ +9% |
| **纯 W2V (α=1.0)** | **0.006399** | ↑ +38% 最差 |

**TODO 假设被反驳**：W2V 语义先验未改善分组，反而纯 SVD/纯动力学最优。语义先验与转移动力学**冲突**（"正言若反"具体体现）。

### P2-B：web 模式完整评估

- **通过项 8 项**：main/export/build_outputs/vis_sankey/vis_timeline/compare_frameworks/hmm 全部 PASS + 错误处理健壮
- **未通过项 2 项**：coarse_graining.json 数据残留、run_all.py 不支持 --mode
- **待确认项 3 项**：数据覆盖设计、HMM web EI、时间线标注

---

## 三、三个修复（已完成并验证）

### 修复 1：coarse_graining.json 数据残留 ✅
- **方案**：模式专属文件名 `coarse_graining_{mode}.json` + 当前指针 `coarse_graining.json`（main.py 也更新）
- 涉及：`main.py`（save_core_outputs）、`export_visualization_data.py`
- **验证**：export web 后 mode=web，恢复 m6 后 mode=m6（无残留）

### 修复 2：run_all.py 支持 --mode ✅
- **方案**：argparse `--mode m6|m12|web`，支持模式参数的脚本自动传参
- 涉及：`run_all.py`
- **验证**：`run_all.py --mode web` 完整 pipeline 6 脚本全部 OK

### 修复 3：时间线 PNG 标注 mode ✅
- **方案**：标题显示分组模式 + 文件名 `vis_07_timeline_{mode}.png`
- 涉及：`vis_07_timeline.py`
- **验证**：`vis_07_timeline_web.png` 生成（标题含"分组模式: web"）

---

## 四、验证结果汇总

| 验证项 | 结果 |
|--------|------|
| 单元测试 | ✅ 39 passed |
| run_all.py 默认 m6 | ✅ 6 脚本全部 OK |
| run_all.py --mode web | ✅ 6 脚本全部 OK（修复后）|
| 错误处理（非法 --mode） | ✅ 优雅报错 returncode=2 |
| coarse_graining 数据一致性 | ✅ 修复后无残留 |
| 时间线 PNG 标注 | ✅ 修复后含模式标识 |
| lint | ✅ 无错误 |

---

## 五、发现的新问题（需后续处理）

| 问题 | 严重度 | 说明 |
|------|--------|------|
| `output/gmm_P_soft.npy` 空文件 | 低 | 初始为 0 字节（某次中断运行残留）。**已重跑修复**：现为 6×6、行和=1、416 字节，EI=0.0849 结果稳定 |
| 时间线旧文件 `vis_07_timeline.png` 仍存在 | 低 | 兼容保留，但可能导致混淆；如需清理可删除 |

---

## 六、本轮全部产出

### 实验报告（output/）
| 文件 | 内容 |
|------|------|
| `experiment_p1a_report.md` | P1-A DeepWalk 实验 |
| `experiment_p1b_report.md` | P1-B GMM 软聚类 + 诚实软效应分析 |
| `experiment_p2a_report.md` | P2-A 图传播（TODO 假设被反驳）|
| `experiment_p2b_web_eval.md` | P2-B web 模式评估 |
| `MODIFICATION_REPORT.md` | 本报告 |

### 可视化（output/）
| 文件 | 内容 |
|------|------|
| `vis_10_w2v_vs_svd.png` | P1-A 嵌入对比 |
| `vis_11_gmm_soft.png` | P1-B GMM EI 曲线 + 状态分布 |
| `vis_12_w2v_mc_back.png` | P2-A α 扫描 + 传播后嵌入 |

### 数据（output/）
- `w2v_vectors.npy`（31×16）、`w2v_similarity.csv`
- `gmm_soft_phi.npy`（31×6）、`svd_vs_w2v_comparison.csv`
- `w2v_experiment_results.json`、`w2v_gmm_experiment_results.json`、`w2v_mc_back_results.json`
- `coarse_graining_{m6,web}.json`（模式专属）

---

## 七、方法论总结

1. **W2V vs SVD 结构低相关（τ=0.198）**：局部共现 ≠ 全局谱结构，呼应"正言若反"
2. **"软聚类"提升 EI 不成立**：GMM/HMM 软效应均 ≈0，EI 提升来自公式 + 聚类结构
3. **W2V 语义先验与转移动力学冲突**：P2-A 证明纯 SVD/动力学最优
4. **诚实负面发现是重要科学贡献**：TODO P2-A 假设被数据反驳，避免了单向偏差
