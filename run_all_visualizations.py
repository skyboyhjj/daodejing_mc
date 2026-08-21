# -*- coding: utf-8 -*-
"""
《道德经》马尔科夫链 — 6 种可视化生成脚本
依赖: numpy, pandas, matplotlib, seaborn, scikit-learn, networkx, plotly
运行: python run_all_visualizations.py
"""

import numpy as np
import pandas as pd
import re
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx

# 确保能 import core 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 【重构 T12】环境配置（UTF-8 / 中文字体 / 路径）抽到 core.env
from core.env import setup_env, CN_FONT, OUTPUT_DIR

setup_env()

# 检测 plotly + kaleido 是否真的可用
use_plotly = False
try:
    from plotly.graph_objects import Sankey, Figure  # noqa
    import plotly.io as pio  # noqa
    import kaleido  # noqa
    use_plotly = True
except Exception:
    use_plotly = False

# ============================================================
# 数据加载
# ============================================================
def load_data():
    """从 output 目录加载所有数据"""
    P = np.load(os.path.join(OUTPUT_DIR, 'P_matrix.npy'))
    pi = np.load(os.path.join(OUTPUT_DIR, 'pi.npy'))
    P_macro = np.load(os.path.join(OUTPUT_DIR, 'P_macro.npy'))
    Phi = np.load(os.path.join(OUTPUT_DIR, 'Phi.npy'))
    
    with open(os.path.join(OUTPUT_DIR, 'concept_data.json'), 'r', encoding='utf-8') as f:
        concept_data = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'coarse_graining.json'), 'r', encoding='utf-8') as f:
        coarse_data = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'network_data.json'), 'r', encoding='utf-8') as f:
        network_data = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'spectral_data.json'), 'r', encoding='utf-8') as f:
        spectral_data = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'dashboard_data.json'), 'r', encoding='utf-8') as f:
        dashboard = json.load(f)
    
    return {
        'P': P, 'pi': pi, 'P_macro': P_macro, 'Phi': Phi,
        'concept_data': concept_data, 'coarse_data': coarse_data,
        'network_data': network_data, 'spectral_data': spectral_data,
        'dashboard': dashboard
    }

# ============================================================
# 可视化 1: 微观概念网络图
# ============================================================
def vis_concept_network(data):
    """概念网络图 + Gephi 兼容 GML 导出"""
    print("  [1/6] 微观概念网络图...")
    
    P = data['P']
    pi = data['pi']
    net = data['network_data']
    
    G = nx.DiGraph()
    for node in net['nodes']:
        G.add_node(node['id'], weight=node['pi'])
    for edge in net['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # 多布局对比
    pos_spring = nx.spring_layout(G, k=3.0, iterations=300, seed=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # 左图：弹簧布局
    ax1 = axes[0]
    node_sizes = [pi[list(G.nodes()).index(n)] * 10000 + 300 for n in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 12 for u, v in G.edges()]
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos_spring, node_size=node_sizes,
                           node_color='#5B9BD5', alpha=0.85,
                           edgecolors='#2E75B6', linewidths=1.5, ax=ax1)
    nx.draw_networkx_edges(G, pos_spring, width=edge_widths,
                           edge_color=edge_colors, edge_cmap=plt.cm.YlOrRd,
                           alpha=0.6, arrows=True, arrowsize=10, ax=ax1)
    nx.draw_networkx_labels(G, pos_spring, font_size=9,
                             font_family=CN_FONT.get_name(), ax=ax1)
    ax1.set_title('微观概念网络（弹簧布局）\n节点大小 ∝ π，边宽 ∝ 转移概率',
                  fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    # 右图：谱布局（用 Laplacian 特征向量）
    ax2 = axes[1]
    try:
        L = nx.normalized_laplacian_matrix(G).todense()
        eigvals, eigvecs = np.linalg.eig(np.array(L))
        idx_sorted = eigvals.argsort()
        # 取第2、3小特征向量（第1是常向量）
        pos_spec = {}
        nodes_list = list(G.nodes())
        for i, n in enumerate(nodes_list):
            pos_spec[n] = (float(eigvecs[i, idx_sorted[1]]),
                           float(eigvecs[i, idx_sorted[2]]))
    except:
        pos_spec = pos_spring
    
    nx.draw_networkx_nodes(G, pos_spec, node_size=node_sizes,
                           node_color='#70AD47', alpha=0.85,
                           edgecolors='#2E75B6', linewidths=1.5, ax=ax2)
    nx.draw_networkx_edges(G, pos_spec, width=edge_widths,
                           edge_color=edge_colors, edge_cmap=plt.cm.YlOrRd,
                           alpha=0.5, arrows=True, arrowsize=8, ax=ax2)
    nx.draw_networkx_labels(G, pos_spec, font_size=9,
                             font_family=CN_FONT.get_name(), ax=ax2)
    ax2.set_title('微观概念网络（谱布局）\n社区结构更清晰', fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vis_01_network.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 导出 GML
    nx.write_gml(G, os.path.join(OUTPUT_DIR, 'vis_01_network.gml'))
    print(f"    ✓ vis_01_network.png + .gml ({len(G.nodes())} nodes, {len(G.edges())} edges)")

# ============================================================
# 可视化 2: 转移矩阵聚类热力图
# ============================================================
def vis_heatmap(data):
    """聚类热力图"""
    print("  [2/6] 转移矩阵聚类热力图...")
    
    P = data['P']
    concepts = data['dashboard']['concept_labels']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 18))
    
    # 左图：原始顺序热力图
    ax1 = axes[0]
    im1 = ax1.imshow(P, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(concepts)))
    ax1.set_yticks(range(len(concepts)))
    ax1.set_xticklabels(concepts, rotation=90, fontsize=8)
    ax1.set_yticklabels(concepts, fontsize=8)
    ax1.set_title('转移矩阵 P（原始顺序）', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='P(i→j)')
    
    # 右图：层次聚类重排
    ax2 = axes[1]
    g = sns.clustermap(P, cmap='YlOrRd', xticklabels=concepts, yticklabels=concepts,
                        figsize=(10, 10), dendrogram_ratio=0.15,
                        cbar_pos=(0.02, 0.8, 0.03, 0.18))
    # clustermap 自己管理 figure，保存后关闭
    g.savefig(os.path.join(OUTPUT_DIR, 'vis_02_heatmap_clustered.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存左图
    plt.figure(fig.number)
    plt.savefig(os.path.join(OUTPUT_DIR, 'vis_02_heatmap_raw.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ vis_02_heatmap_raw.png + vis_02_heatmap_clustered.png")

# ============================================================
# 可视化 3: 谱空间散点图
# ============================================================
def vis_spectral_scatter(data):
    """SVD 谱空间 2D + 3D 散点图"""
    print("  [3/6] SVD 谱空间散点图...")
    
    sp = data['spectral_data']
    concepts = sp['concepts']
    labels = sp['labels']
    
    fig = plt.figure(figsize=(18, 8))
    
    # 2D
    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(sp['svd_1'], sp['svd_2'], c=labels, cmap='tab10',
                           s=200, alpha=0.85, edgecolors='black', linewidth=0.5)
    for i, name in enumerate(concepts):
        ax1.annotate(name, (sp['svd_1'][i], sp['svd_2'][i]),
                      fontsize=9, ha='center', va='bottom', alpha=0.9)
    ax1.axhline(0, color='gray', lw=0.3)
    ax1.axvline(0, color='gray', lw=0.3)
    ax1.set_xlabel('SVD-1', fontsize=12)
    ax1.set_ylabel('SVD-2', fontsize=12)
    ax1.set_title('谱空间 2D 投影（K-Means 着色）', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    
    # 3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(sp['svd_1'], sp['svd_2'], sp['svd_3'],
                c=labels, cmap='tab10', s=150, alpha=0.85, edgecolors='black', linewidth=0.3)
    for i, name in enumerate(concepts):
        ax2.text(sp['svd_1'][i], sp['svd_2'][i], sp['svd_3'][i],
                  name, fontsize=7, ha='center')
    ax2.set_xlabel('SVD-1', fontsize=10)
    ax2.set_ylabel('SVD-2', fontsize=10)
    ax2.set_zlabel('SVD-3', fontsize=10)
    ax2.set_title('谱空间 3D 投影', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vis_03_spectral.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ vis_03_spectral.png")

# ============================================================
# 可视化 4: 桑基图
# ============================================================
def vis_sankey(data):
    """桑基图：微观→宏观 投影 + 宏观转移"""
    print("  [4/6] 桑基图...")
    
    try:
        from plotly.graph_objects import Sankey, Figure
        import plotly.io as pio
        use_plotly = True
    except ImportError:
        use_plotly = False
    
    P = data['P']
    P_macro = data['P_macro']
    Phi = data['Phi']
    pi = data['pi']
    concepts = data['dashboard']['concept_labels']
    macro_names = data['dashboard']['macro_labels']
    
    N = len(concepts)
    M = len(macro_names)
    
    if use_plotly:
        # 微观→宏观流
        sources_micro = list(range(N))
        targets_micro = [N + int(Phi[i].argmax()) for i in range(N)]
        values_micro = [float(1.0)] * N
        
        # 宏观→宏观流（过滤弱边）
        sources_macro = []
        targets_macro = []
        values_macro = []
        for i in range(M):
            for j in range(M):
                if P_macro[i, j] > 0.05:
                    sources_macro.append(N + i)
                    targets_macro.append(N + j)
                    values_macro.append(float(P_macro[i, j]) * 10)
        
        node_labels = concepts + macro_names
        node_colors = ['#5B9BD5'] * N + ['#ED7D31'] * M
        
        fig = Figure(data=[Sankey(
            arrangement='snap',
            node=dict(label=node_labels, color=node_colors, pad=15, thickness=18,
                      line=dict(color='black', width=0.5)),
            link=dict(
                source=sources_micro + sources_macro,
                target=targets_micro + targets_macro,
                value=values_micro + values_macro,
                color=['rgba(91,155,213,0.25)'] * len(sources_micro) +
                      ['rgba(237,125,49,0.5)'] * len(sources_macro)
            )
        )])
        fig.update_layout(
            title_text="《道德经》概念粗粒化桑基图<br>蓝→橙：微观→宏观投影 | 橙→橙：宏观转移",
            font_size=12, width=1200, height=800
        )
        pio.write_image(fig, os.path.join(OUTPUT_DIR, 'vis_04_sankey.png'), scale=2)
        pio.write_html(fig, os.path.join(OUTPUT_DIR, 'vis_04_sankey_interactive.html'))
        print(f"    ✓ vis_04_sankey.png + vis_04_sankey_interactive.html")
    else:
        # Matplotlib 版
        fig, ax = plt.subplots(figsize=(10, 8))
        left_y = np.linspace(0.9, 0.1, N)
        right_y = np.linspace(0.8, 0.2, M)
        
        for i, name in enumerate(concepts):
            ax.plot(0.1, left_y[i], 'o', color='#5B9BD5', markersize=8)
            ax.text(0.08, left_y[i], name, ha='right', va='center', fontsize=9)
        
        for j, name in enumerate(macro_names):
            ax.plot(0.9, right_y[j], 's', color='#ED7D31', markersize=14)
            ax.text(0.92, right_y[j], name, ha='left', va='center', fontsize=11, fontweight='bold')
        
        labels_assign = [int(Phi[i].argmax()) for i in range(N)]
        for i in range(N):
            j = labels_assign[i]
            ax.plot([0.1, 0.9], [left_y[i], right_y[j]],
                    '-', color='#5B9BD5', alpha=0.12, linewidth=1.5)
        
        for i in range(M):
            for j in range(M):
                if P_macro[i, j] > 0.1:
                    ax.annotate('', xy=(0.92, right_y[j]),
                                xytext=(0.88, right_y[i]),
                                arrowprops=dict(arrowstyle='->', color='#ED7D31',
                                                lw=P_macro[i,j]*4, alpha=0.7))
        
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis('off')
        ax.set_title('《道德经》概念粗粒化流图', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'vis_04_sankey.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ vis_04_sankey.png (matplotlib 版)")

# ============================================================
# 可视化 5: 主题河流图
# ============================================================
def vis_theme_river(data):
    """主题河流图"""
    print("  [5/6] 主题河流图...")
    
    river = pd.read_csv(os.path.join(OUTPUT_DIR, 'theme_river.csv'))
    macro_names = data['dashboard']['macro_labels']
    M = len(macro_names)
    
    chapters = sorted(river['chapter'].unique())
    x = np.arange(len(chapters))
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # 构建数据矩阵
    data_matrix = np.zeros((len(chapters), M))
    for idx, ch in enumerate(chapters):
        row = river[river['chapter'] == ch]
        for _, r in row.iterrows():
            data_matrix[idx, int(r['macro_state'])] = r['density']
    
    colors = plt.cm.Set2(np.linspace(0, 1, M))
    ax.stackplot(x, [data_matrix[:, j] for j in range(M)],
                 labels=macro_names, colors=colors, alpha=0.85,
                 edgecolor='white', linewidth=0.3)
    
    # 标注关键章节
    key_chapters = {
        1: '道可道', 25: '道法自然', 37: '无为而无不为',
        40: '反者道之动', 42: '道生一', 57: '以正治国',
        64: '千里之行', 78: '正言若反', 81: '为而不争'
    }
    for ch, label in key_chapters.items():
        if ch in chapters:
            idx = chapters.index(ch)
            ax.axvline(x=idx, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax.text(idx, 1.02, f'第{ch}章\n{label}',
                    ha='center', fontsize=8, color='#333')
    
    ax.set_xticks(x[::5])
    ax.set_xticklabels([f'第{chapters[i]}章' for i in range(0, len(chapters), 5)],
                        rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('概念密度', fontsize=12)
    ax.set_title('《道德经》主题河流图\n宏观义理在81章中的兴衰起伏',
                  fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
    ax.set_xlim(0, len(chapters)-1)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vis_05_theme_river.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ vis_05_theme_river.png")

# ============================================================
# 可视化 6: 因果涌现曲线 + 奇异值谱
# ============================================================
def vis_ei_curve_and_spectrum(data):
    """因果涌现曲线 + 奇异值谱 + 宏观转移热力图"""
    print("  [6/6] 因果涌现曲线 + 奇异值谱...")
    
    P = data['P']
    pi = data['pi']
    P_macro = data['P_macro']
    macro_names = data['dashboard']['macro_labels']
    s_vals = data['spectral_data']['singular_values']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # (1) 因果涌现曲线
    ax1 = axes[0, 0]
    max_M = min(20, P.shape[0] - 1)
    M_values = list(range(2, max_M + 1))
    ei_norm_values = []
    ei_raw_values = []
    
    ei_micro = float(np.sum([pi[i] * np.sum((P[i]+1e-15) * (np.log(P[i]+1e-15) - np.log(pi@P+1e-15))) for i in range(P.shape[0])]))
    ei_micro_norm = ei_micro / np.log(P.shape[0])
    
    for M in M_values:
        try:
            embedding = data['spectral_data']
            # 简化：直接从已保存数据计算
            from sklearn.cluster import KMeans
            # 重新计算 SVD 嵌入
            F = np.diag(pi) @ P
            _, s, Vt = np.linalg.svd(F)
            emb = Vt[:M, :].T
            km = KMeans(n_clusters=M, random_state=42, n_init=10)
            labels_m = km.fit_predict(emb)
            
            # 构建宏观矩阵
            Phi_m = np.zeros((P.shape[0], M))
            for i in range(P.shape[0]):
                Phi_m[i, labels_m[i]] = 1.0
            P_m = Phi_m.T @ P @ Phi_m
            row_sums = P_m.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            P_m = P_m / row_sums
            
            pi_m = np.ones(M) / M
            for _ in range(10000):
                pi_new = pi_m @ P_m
                if np.linalg.norm(pi_new - pi_m, ord=1) < 1e-12:
                    break
                pi_m = pi_new
            
            eps = 1e-15
            ei_m = np.sum([pi_m[i] * np.sum((P_m[i]+eps) * (np.log(P_m[i]+eps) - np.log(pi_m@P_m+eps))) for i in range(M)])
            ei_m_norm = ei_m / max(np.log(M), 1e-10)
            ei_norm_values.append(ei_m_norm)
            ei_raw_values.append(ei_m)
        except:
            ei_norm_values.append(0)
            ei_raw_values.append(0)
    
    ax1.axhline(y=ei_micro_norm, color='red', linestyle='--', linewidth=1.5,
                label=f'微观基线 ({ei_micro_norm:.4f})')
    ax1.plot(M_values, ei_norm_values, 'o-', color='#2E75B6', linewidth=2, markersize=6)
    ax1.fill_between(M_values, ei_norm_values, alpha=0.1, color='#2E75B6')
    best_M = M_values[np.argmax(ei_norm_values)]
    best_ei = max(ei_norm_values)
    ax1.scatter([best_M], [best_ei], color='red', s=200, zorder=5, marker='*')
    ax1.annotate(f'最优 M={best_M}\nEI={best_ei:.4f}',
                xy=(best_M, best_ei),
                xytext=(best_M+1.5, best_ei*0.9),
                fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.set_xlabel('宏观状态数 M', fontsize=12)
    ax1.set_ylabel('归一化有效信息 Eff(P_M)', fontsize=12)
    ax1.set_title('因果涌现曲线', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # (2) 奇异值谱
    ax2 = axes[0, 1]
    s_arr = np.array(s_vals)
    s_norm = s_arr / s_arr[0] if s_arr[0] > 0 else s_arr
    ax2.bar(range(1, len(s_norm)+1), s_norm, color='#70AD47', edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('奇异值序号', fontsize=12)
    ax2.set_ylabel('归一化奇异值 σ_i/σ_1', fontsize=12)
    ax2.set_title('稳态流矩阵 F 的奇异值谱', fontsize=13, fontweight='bold')
    # 标注间隙
    if len(s_norm) > 1:
        gaps = [s_norm[i] - s_norm[i+1] for i in range(len(s_norm)-1)]
        best_gap = int(np.argmax(gaps)) + 1
        ax2.annotate(f'谱间隙 @K={best_gap}',
                    xy=(best_gap+0.5, s_norm[best_gap]),
                    fontsize=10, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # (3) 宏观转移矩阵热力图
    ax3 = axes[1, 0]
    im3 = ax3.imshow(P_macro, cmap='YlOrRd', vmin=0, vmax=1)
    ax3.set_xticks(range(len(macro_names)))
    ax3.set_yticks(range(len(macro_names)))
    ax3.set_xticklabels(macro_names, rotation=45, ha='right', fontsize=9)
    ax3.set_yticklabels(macro_names, fontsize=9)
    # 标注数值
    for i in range(P_macro.shape[0]):
        for j in range(P_macro.shape[1]):
            if P_macro[i, j] > 0.05:
                ax3.text(j, i, f'{P_macro[i,j]:.2f}', ha='center', va='center',
                          fontsize=9, fontweight='bold',
                          color='white' if P_macro[i,j] > 0.5 else 'black')
    ax3.set_title('宏观转移矩阵 P\'(M=6)', fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='P\'(i→j)')
    
    # (4) 平稳分布对比
    ax4 = axes[1, 1]
    pi_macro = np.ones(6) / 6
    P_tmp = P_macro.copy()
    for _ in range(10000):
        pi_new = pi_macro @ P_tmp
        if np.linalg.norm(pi_new - pi_macro, ord=1) < 1e-12:
            break
        pi_macro = pi_new
    
    y_pos = np.arange(len(macro_names))
    bars = ax4.barh(y_pos, pi_macro, color=plt.cm.Set2(np.linspace(0, 1, 6)),
                     edgecolor='white', linewidth=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(macro_names, fontsize=11)
    ax4.set_xlabel('宏观平稳概率 π\'', fontsize=12)
    ax4.set_title('宏观平稳分布（哪些义理是"重心"？）', fontsize=13, fontweight='bold')
    for i, v in enumerate(pi_macro):
        ax4.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
    ax4.set_xlim(0, max(pi_macro) * 1.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vis_06_emergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ vis_06_emergence.png (含4个子图)")
    print(f"    ✓ 最优 M = {best_M}, 最优 EI = {best_ei:.4f}")

# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("  《道德经》马尔科夫链 — 6 种可视化生成")
    print("=" * 70)
    
    data = load_data()
    
    vis_concept_network(data)       # ①
    vis_heatmap(data)              # ②
    vis_spectral_scatter(data)      # ③
    vis_sankey(data)               # ④
    vis_theme_river(data)           # ⑤
    vis_ei_curve_and_spectrum(data) # ⑥
    
    # 汇总
    print("\n" + "=" * 70)
    print("  ✓ 全部 6 种可视化完成！")
    print("=" * 70)
    
    files = sorted(os.listdir(OUTPUT_DIR))
    vis_files = [f for f in files if f.startswith('vis_')]
    other_files = [f for f in files if not f.startswith('vis_') and not f.startswith('sankey')]
    
    print(f"\n  可视化文件 ({len(vis_files)}):")
    for f in vis_files:
        fpath = os.path.join(OUTPUT_DIR, f)
        print(f"    • {f} ({os.path.getsize(fpath)/1024:.1f} KB)")
    
    print(f"\n  数据文件 ({len(other_files)}):")
    for f in other_files[:10]:
        fpath = os.path.join(OUTPUT_DIR, f)
        print(f"    • {f} ({os.path.getsize(fpath)/1024:.1f} KB)")
    if len(other_files) > 10:
        print(f"    ... 还有 {len(other_files)-10} 个文件")

if __name__ == "__main__":
    main()
