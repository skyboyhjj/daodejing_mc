# P2-B 评估报告：web 模式完整评估

> 评估日期：2026-08-22
> 评估对象：`--mode web`（网页由体达用分组，M=6，6 个宏观态）
> 方法：逐项实际运行全部支持 web 模式的脚本，检查功能/流程/错误/性能/界面

## web 模式分组定义（实测确认）

```
道体论(认识道)：道 德 玄 象 自然 朴 无 有 天地
辩证法(道的规律)：反 辩证 刚强 柔弱 一
修身内化(内化功夫)：知足 知 欲 信 名 守 去 不言
无为论(方法论)：无为 水
治国论(外王之道)：圣人 侯王 治 小国寡民 民 兵
三宝(德之落实)：三宝
```
共 31 个概念，6 组，覆盖完整（已验证无遗漏、无重复）。

---

## 一、通过项 ✅

### 1. 主流程（main.py --mode web）— 通过
- **功能覆盖**：完整跑通清洗 → 转移矩阵 → EI → 粗粒化(web) → 可视化 → 保存 → k=2 对照
- **分组正确**：阶段四显示"手工语义粗粒化 (mode=web, M=6)"，6 组名称正确
- **性能**：21.6s
- **界面**：控制台输出清晰标注 mode=web

### 2. 数据导出（export_visualization_data.py --mode web）— 通过
- 正确写入 web 分组（macro_names = 道体论/辩证法/修身内化/无为论/治国论/三宝）
- 正确生成 concept_data.json / transition_matrix.csv / coarse_graining.json 等
- **性能**：3.6s

### 3. 仪表盘数据（build_outputs.py --mode web）— 通过
- 与 export 一致，生成 web 分组的 network/sankey/dashboard 数据
- **性能**：3.5s

### 4. 交互式桑基图（vis_sankey_interactive.py --mode web）— 通过
- 生成 vis_04_sankey_interactive.html（plotly）+ 静态 PNG
- **界面一致性**：HTML 标题包含"修身内化"等 web 分组名，正确标注
- **性能**：9.9s

### 5. 概念时间线（vis_07_timeline.py --mode web）— 通过
- 生成 vis_07_timeline.png + concept_timeline.csv
- **数据一致性**：CSV 的 macro_name 唯一值为 web 的 6 组 ✅
- **性能**：10.3s

### 6. 分组对比工具（compare_frameworks.py）— 通过
- 正确对比 m6 / web / m12 三方案
- web 的 ε=0.003834 是三方案中最优（成块性最自洽）
- **性能**：5.2s

### 7. HMM 软分配（hmm_analysis.py --mode web）— 通过
- 正确运行，web 分组下 HMM 软分配 EI=0.0737
- 硬划分 EI=0.0124，软分配涌现=+0.0222（正涌现）
- 31/31 概念软分配 MLE 标签与硬划分一致

### 8. 错误处理（非法参数）— 通过
- `--mode invalid` → 优雅报错（argparse: invalid choice），returncode=2
- 提示正确（choose from m6, m12, web）

---

## 二、未通过项 ❌

### 1. 数据残留/一致性（coarse_graining.json）— 未通过 ⚠️
- **现象**：运行默认 `main.py`（m6）后，`coarse_graining.json` 的 macro_names **仍是 web 分组**，不是 m6
- **根因**：`main.py` 主流程**不写入** coarse_graining.json；只有 `export_visualization_data.py` 写。所以 main.py 单独运行 m6 后，coarse_graining.json 残留上次 web 模式的写入
- **影响链**：`run_all_visualizations.py`（第54行）读取 coarse_graining.json → 若用户单独运行 main.py(m6) 后再运行可视化，会读到残留的 web 分组，导致**可视化与主流程结果不一致**
- **严重度**：中。按 run_all.py 顺序执行时（export 会覆盖）不受影响；单独运行 main.py 时存在

### 2. run_all.py 不支持 --mode — 未通过
- **现象**：`run_all.py` 没有 `--mode` 参数，无法一键切换 web 模式
- **影响**：用户想一键运行 web 模式的完整 pipeline 时，无法通过 run_all.py 实现，只能逐个脚本传 `--mode web`
- **严重度**：低-中。影响便利性，不阻塞功能

---

## 三、待确认项 ❓

### 1. 数据覆盖的最终一致性
- 确认点：如果用户先跑 web 模式（main/export web），再跑默认 m6 的完整 run_all.py，export 会覆盖 coarse_graining.json 为 m6。**需确认这是否符合预期**——即"最后一次运行的导出脚本决定分组"的设计是否可接受
- 建议：最好让 main.py 也写入 coarse_graining.json（或让输出文件名带模式后缀，如 coarse_graining_web.json），避免残留

### 2. HMM web 模式 EI=0.0737 的解读
- 确认点：web 分组下 HMM 软分配 EI=0.0737 > m6 的 0.0539，是否意味着 web 分组更适合软分配？需结合 P1-B 的"软效应≈0"发现进一步验证（可能主要来自公式效应而非分组）

### 3. 桑基图/时间线等可视化的 web 标注
- 桑基图 HTML 标题标注了 web（✅），但时间线 PNG 的标题**未显式标注 mode**（只显示"宏观义理"），需确认是否需要在图中标注模式以便区分 m6/web/m12 的图

---

## 四、性能汇总

| 脚本 | web 模式耗时 | 说明 |
|------|------------|------|
| main.py | 21.6s | 含全部 7 阶段 + 可视化 |
| export_visualization_data.py | 3.6s | 数据导出 |
| build_outputs.py | 3.5s | 仪表盘数据 |
| vis_sankey_interactive.py | 9.9s | plotly + kaleido |
| vis_07_timeline.py | 10.3s | 时间线 |
| compare_frameworks.py | 5.2s | 三方案对比 |

**性能整体良好**：6 个模块总计约 54s，无性能瓶颈。最耗时为 main.py（含完整 pipeline）。

---

## 五、总结

**web 模式功能完整、运行稳定**，6 个核心模块全部通过，错误处理健壮，性能良好。主要问题：
1. **coarse_graining.json 数据残留**（中严重度）——需要修复
2. **run_all.py 不支持 --mode**（低-中严重度）——影响一键切换便利性

两个待确认项涉及数据一致性设计和可视化标注，需用户确认是否符合预期。