# 已知问题与解决方案

## 1. 字体问题
**现象**：matplotlib 中文字体显示为方框
**原因**：不同环境可用中文字体不同——Linux 沙盒仅有 `Noto Sans CJK JP`（无 SC），Windows 本机有 `Microsoft YaHei` / `SimHei`（无 Noto）
**解决**：所有脚本已统一使用 `get_cn_font()` 自动检测函数，通过 `fontManager.ttflist` **实际扫描已注册字体**（不用 `findfont` 的 fallback，避免掩盖缺失），按优先级匹配 `Microsoft YaHei → SimHei → Noto Sans CJK JP → ...`
**验证**：Windows 下实际选用 `Microsoft YaHei`，10 张可视化 PNG 中文均正常显示（无方框）
**注意**：`vis_sankey_interactive.py` 的 plotly 也指定了 `Microsoft YaHei, SimHei, Noto Sans CJK JP` 字体族，跨平台兼容

## 2. Plotly/Kaleido（已解决 ✅）
**原现象**：沙盒环境中 kaleido 需要 Chrome，未安装，桑基图降级为 matplotlib 版
**现状（2026-08-21）**：已 `pip install kaleido`（含自动下载 Chrome），`vis_sankey_interactive.py` 成功导出：
- `vis_04_sankey_interactive.html`（15.7 KB，交互式）
- `vis_04_sankey_plotly.png`（761 KB，kaleido 高清静态版）
**保留**：`vis_04_sankey.png`（matplotlib 版，作为兼容备份）

## 3. python-docx（已解决 ✅）
**原现象**：`generate_report.py` 中 `from docx import Document` 可能失败
**现状**：`python-docx` 已安装，Word 报告 `道德经概念动力学分析报告.docx` 正常生成

## 4. SVD 第一奇异值主导
**现象**：SVD 嵌入近似一维（第一奇异值远大于其他）
**影响**：K-Means 在嵌入空间效果差（silhouette ~0.1-0.2）
**解决**：改用 Ward 层次聚类 + 手工语义分组
**根因**：《道德经》概念网络高度中心化（"道""无"是超级枢纽）

## 5. k=2 数据严重不足
**现象**：k=2 联合状态 39 个，密度仅 2.96%
**影响**：Laplace +1 平滑主导概率估计，EI 不可靠
**解决**：弃用 k=2，主分析用 k=1
**经验法则**：每个状态至少 20-50 次有效观测

## 6. 因果涌现为负值
**现象**：宏观 EI (0.0119) < 微观 EI (0.0526)
**直觉**：粗粒化后信息反而减少了
**解释**：不是 bug，是文本本质的诚实反映
**叙事转换**：从"发现涌现"改为"验证成块性自洽"

## 7. generate_report.py 文本与数据矛盾（已修复 ✅）
**位置**：第 274 行附近
**问题**：描述为"因果涌现为正值"，实测为负值（-0.0406）
**状态**：✅ 已修复（TODO.md T02）
**修复**：3 处文本改为"因果涌现为负值，说明微观序列已是最经济描述"，与实测数据一致

## 8. REVERSE_MAP 变体冲突 bug（已修复 ✅）
**现象**：字面"无为"被错误归入"无"概念（REVERSE_MAP["无为"]="无"）
**根因**：`CONCEPT_DICT["无"]` 和 `CONCEPT_DICT["无为"]` 都含变体 "无为"，构建反向映射时按字典键序遍历，"无" 覆盖了 "无为"
**发现**：由 T13 单元测试 `test_wuwei_matches_before_single_wu` 暴露
**修复**：`REVERSE_MAP` 构建时变体冲突优先选择更长标准概念（`len(std) > len(cur)` 才覆盖）
**影响**：概念频次更准确（"无" 87→78，"无为" 31→43，总观测 849 不变）；微观 EI_norm 0.0526→0.0515
**教训**：变体可能被多个标准概念收录，冲突处理需明确优先级

## 9. 概念词典覆盖不全
**现象**：低频概念（如"赤子""含德"）未被收录
**影响**：这些概念在转移矩阵中作为"孤立节点"
**后续**：可扩充词典至 50+ 概念，或用 TF-IDF 自动发现

## 10. 句子级概念抽取的信息损失
**现象**：一个句子含多个概念时，只取第一个
**影响**：转移矩阵偏向句首概念
**后续方案**：改用"集合转移"模型（句子=概念集合→下一句集合）
