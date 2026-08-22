# P2-B 评估报告：web 模式完整评估

> 首次评估日期：2026-08-22
> 复验更新日期：2026-08-22
> **复验说明**：本报告初版评估于三项修复（coarse_graining 数据残留 / run_all 支持 --mode / 时间线 mode 标注）实施之前，并据此定位了 2 个未通过项与 3 个待确认项。三项修复现已全部完成并被本次复验确认通过，故本版将原"未通过项"更新为"已修复并复验通过"，"待确认项"更新为"现状与结论"。
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

## 一、通过项 ✅（复验确认）

### 1. 主流程（main.py --mode web）— 通过
- **功能覆盖**：完整跑通清洗 → 转移矩阵 → EI → 粗粒化(web) → 可视化 → 保存 → k=2 对照
- **分组正确**：阶段四显示"手工语义粗粒化 (mode=web, M=6)"，6 组名称正确
- **复验指标**：微观归一化 EI=0.0515，宏观归一化 EI(M=6)=0.0124，因果涌现=-0.0391，成块性 ε=0.003834
- **界面**：控制台输出清晰标注 mode=web
- **性能**：约 30s（本次实测 29.7s，含全部 7 阶段 + 可视化）

### 2. 数据导出（export_visualization_data.py --mode web）— 通过
- 正确写入 web 分组（macro_names = 道体论/辩证法/修身内化/无为论/治国论/三宝）
- 正确生成 concept_data.json / transition_matrix.csv / coarse_graining.json 等

### 3. 仪表盘数据（build_outputs.py --mode web）— 通过
- 与 export 一致，生成 web 分组的 network/sankey/dashboard 数据

### 4. 交互式桑基图（vis_sankey_interactive.py --mode web）— 通过
- 生成 vis_04_sankey_interactive.html（plotly）+ 静态 PNG
- **界面一致性**：HTML 标题包含"修身内化"等 web 分组名与"分组模式: web, M=6"，正确标注

### 5. 概念时间线（vis_07_timeline.py --mode web）— 通过（修复 3 后）
- 生成 `vis_07_timeline_web.png` + `concept_timeline_web.csv`（文件名带模式后缀）
- **数据一致性**：CSV 的 macro_name 唯一值为 web 的 6 组 ✅
- **界面标注（修复 3）**：PNG 标题现在显示"《道德经》概念时间线（分组模式: web）"，可区分 m6/web/m12 的图

### 6. 分组对比工具（compare_frameworks.py）— 通过
- 正确对比 m6 / web / m12 三方案
- web 的 ε=0.003834 是三方案中最优（成块性最自洽）
- M=12 因果涌现较 m6 更强（-0.0397 → -0.0285）

### 7. HMM 软分配（hmm_analysis.py --mode web）— 通过（已支持 --mode）
- 正确运行，web 分组下 HMM 软分配 EI=0.0737
- 硬划分 EI=0.0124，软分配涌现=+0.0222（正涌现）
- 31/31 概念软分配 MLE 标签与硬划分一致

### 8. 错误处理（非法参数）— 通过
- `--mode invalid` → 优雅报错（argparse: invalid choice），returncode=2
- 提示正确（choose from m6, m12, web）

---

## 二、原未通过项 → 已修复并复验通过 ✅

### 1. 数据残留/一致性（coarse_graining.json）— 修复 1 后复验通过
- **原现象**：单独运行 `main.py`（m6）后，`coarse_graining.json` 的 macro_names 仍是上次 web 模式的写入（main.py 主流程原本不写该文件，只有 export 写）
- **修复方案（修复 1）**：采用模式专属文件名 `coarse_graining_{mode}.json` + 当前指针 `coarse_graining.json`；`main.py`（save_core_outputs）与 `export_visualization_data.py` 均同步更新
- **复验结果**：单独运行 `main.py --mode web` 后，目录同时存在 `coarse_graining_web.json` 与指针 `coarse_graining.json`，两者均指向 web 分组（实测确认）；恢复运行 m6 后指针随之切换为 m6，**无残留**

### 2. run_all.py 不支持 --mode — 修复 2 后复验通过
- **原现象**：`run_all.py` 没有 `--mode` 参数，无法一键切换 web 模式
- **修复方案（修复 2）**：argparse 新增 `--mode m6|m12|web`，脚本列表标记支持模式的脚本并自动传参
- **复验结果**：`run_all.py --mode web` 完整 pipeline 6 脚本全部 OK（本次实测全程 ~63s，退出码 0），`run_all.py --mode invalid` 优雅报错 returncode=2

---

## 三、原待确认项 → 现状与结论

### 1. 数据覆盖的最终一致性 — 已解决
- **原确认点**：最后一次运行的导出脚本决定分组的设计是否可接受
- **结论（修复 1）**：已改为"模式专属文件 + 当前指针"双写机制，各模式历史（`coarse_graining_{m6,web,m12}.json`）互不覆盖，指针 `coarse_graining.json` 始终指向最近一次运行的模式，数据一致性与可追溯性均满足

### 2. HMM web 模式 EI=0.0737 的解读 — 复验确认数值稳定，并结合 P1-B 结论
- **复验**：HMM `--mode web` 复测软分配 EI=0.0737，硬划分 EI=0.0124，软分配涌现=+0.0222，数值与初版一致
- **解读**：结合 P1-B 的隔离分析结论，web 分组下 EI 较高**大概率主要来自 EI 公式与分组结构本身**，而非"软归属"（P1-B 实测 GMM/HMM 的"软效应"均≈0，后验熵≈0，软分配实际接近硬）。web 分组成块性最优（ε=0.003834）但 EI 提升是否构成"更优分组"仍需谨慎，不宜据此断言 web 更适合软分配

### 3. 桑基图/时间线等可视化的 web 标注 — 已解决（修复 3）
- **原确认点**：时间线 PNG 标题未标注 mode，难以区分 m6/web/m12 的图
- **结论（修复 3）**：`vis_07_timeline` 标题现在显示分组模式，且文件名带模式后缀（`vis_07_timeline_{mode}.png`）；桑基图交互 HTML 原已标注 "分组模式: web, M=6"（本次复验确认）

---

## 四、性能汇总

| 脚本 | web 模式耗时 | 说明 |
|------|------------|------|
| main.py | ~30s（实测 29.7s） | 含全部 7 阶段 + 可视化 |
| export_visualization_data.py | ~3.6s | 数据导出 |
| build_outputs.py | ~3.5s | 仪表盘数据 |
| vis_sankey_interactive.py | ~9.9s | plotly + kaleido |
| vis_07_timeline.py | ~10.3s | 时间线 |
| compare_frameworks.py | ~5.2s | 三方案对比 |
| **run_all.py --mode web（汇总）** | **~63s（实测 62.8s）** | 一键全链路 6 脚本 |

**性能整体良好**：单项脚本数秒，一键全链路约 1 分钟，无性能瓶颈。最耗时为 main.py（含完整 pipeline）。耗时随机器负载有波动，上表为代表值。

---

## 五、总结

**web 模式功能完整、运行稳定**，6 个核心模块全部通过，错误处理健壮，性能良好。

本轮复验的关键变化（对应三项修复全部落地）：
1. **coarse_graining.json 数据残留** —— 已修复：模式专属文件 + 当前指针，复验确认无残留
2. **run_all.py 不支持 --mode** —— 已修复：一键 `--mode web` 全链路 6 脚本 OK
3. **时间线 PNG 未标注模式** —— 已修复：标题标注 + 文件名带模式后缀

结论：web 模式的分组、数据导出、仪表盘、可视化、HMM、一键运行与错误处理均通过复验，三项修复解决了初版评估发现的全部阻塞性问题。