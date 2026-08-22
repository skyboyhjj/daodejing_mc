"""
experiment_mc_w2v.py
马尔科夫链 × Word2Vec 融合实验（P1-A：MC → W2V 单向融合 / DeepWalk 范式）

DeepWalk 核心流程：
  游走序列生成（随机游走）→ 语料构建 → Skip-gram 嵌入训练 → 下游评估

下游评估：
  1. Kendall τ：W2V 相似度 vs SVD 相似度（结构自洽性）
  2. 节点分类：用宏观态标签（M=6）训练 LogisticRegression，评估嵌入的语义可分性
  3. 链接预测：用转移概率作为边权，评估嵌入内积能否预测概念间转移强度

依赖：numpy, scipy, gensim, scikit-learn, matplotlib, scipy.stats
用法：python scripts/experiment_mc_w2v.py
"""
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.stats import kendalltau
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import json, os, sys

# 兼容直接运行与脚本式运行
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

# UTF-8 输出（Windows 控制台）
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 60)
print("实验：马尔科夫链 × Word2Vec 融合（P1-A / DeepWalk 范式）")
print("=" * 60)

# ============================================================
# 1. 加载数据（转移矩阵 + 平稳分布 + 概念/宏观标签）
# ============================================================
P = np.load(os.path.join(ROOT, "output/P_matrix.npy"))
pi = np.load(os.path.join(ROOT, "output/pi.npy"))
N = P.shape[0]

# 从 coarse_graining.json 提取概念名 + 宏观标签（groups 结构）
concepts = []
macro_labels = {}  # concept -> macro index
cg_path = os.path.join(ROOT, "output/coarse_graining.json")
if os.path.exists(cg_path):
    with open(cg_path, "r", encoding="utf-8") as f:
        cg = json.load(f)
    # groups 是 {宏观索引: {name, concepts:[{name,pi},...]}}
    groups = cg.get("groups", {})
    for m_str, g in groups.items():
        m = int(m_str)
        for item in g.get("concepts", []):
            concepts.append(item["name"])
            macro_labels[item["name"]] = m
    if not concepts:
        concepts = [f"C{i}" for i in range(N)]
else:
    concepts = [f"C{i}" for i in range(N)]

# 确保顺序与 P 一致（coarse_graining 可能乱序，按 P 的原始顺序重建）
# P 由 main.py 按 set(seq) 排序构建，概念顺序与 main 的 idx 一致。
# 此处用 groups 顺序（宏观0..5 内的概念）作为显示顺序即可，训练不受影响。
print(f"  加载转移矩阵 P: {N}×{N}")
print(f"  概念列表: {concepts[:8]}... (共{len(concepts)}个)")
print(f"  宏观态标签数: {len(set(macro_labels.values())) if macro_labels else 0}")

# ============================================================
# 2. DeepWalk：随机游走生成序列 → 语料
# ============================================================
print("\n[步骤1] 随机游走序列生成")
print("-" * 50)


def random_walk(P, start_idx, length=20):
    """从 start_idx 出发，按 P 为转移概率做长度 length 的随机游走"""
    walk = [start_idx]
    for _ in range(length - 1):
        probs = P[walk[-1]]
        if probs.sum() == 0:
            break
        nxt = np.random.choice(N, p=probs)
        walk.append(nxt)
    return [concepts[i] for i in walk]


np.random.seed(42)
num_walks, walk_length = 10000, 20
print(f"  生成 {num_walks} 条随机游走 (length={walk_length})...")
sentences = [random_walk(P, np.random.randint(0, N), walk_length)
             for _ in range(num_walks)]

seen = set()
for s in sentences:
    seen.update(s)
print(f"  游走中出现概念数: {len(seen)}/{N}")

# ============================================================
# 3. Skip-gram 嵌入训练（Word2Vec）
# ============================================================
print("\n[步骤2] Word2Vec (Skip-gram) 训练")
print("-" * 50)
print("  训练 Word2Vec (sg=1, window=3, dim=16, negative=5)...")
model = Word2Vec(sentences, vector_size=16, window=3, sg=1,
                 hs=0, negative=5, min_count=1, seed=42, workers=1)

# 提取概念向量
w2v = np.array([model.wv[c] for c in concepts])
np.save(os.path.join(ROOT, "output/w2v_vectors.npy"), w2v)
print(f"  W2V 向量 → output/w2v_vectors.npy {w2v.shape}")

w2v_sim = 1 - pairwise_distances(w2v, metric="cosine")
np.savetxt(os.path.join(ROOT, "output/w2v_similarity.csv"),
           w2v_sim, delimiter=",")
print("  W2V 相似度 → output/w2v_similarity.csv")

# ============================================================
# 4. SVD 谱嵌入（基准）
# ============================================================
print("\n[步骤3] SVD 谱嵌入（基准）")
print("-" * 50)
F = np.diag(pi) @ P
U, s, Vt = svd(F)
svd_vec = Vt[:2].T  # N×2
svd_sim = 1 - pairwise_distances(svd_vec, metric="cosine")
print("  SVD 前2个右奇异向量 → N×2 嵌入")

# ============================================================
# 5. 嵌入质量评估
# ============================================================
print("\n[步骤4] 嵌入质量评估")
print("-" * 50)

# 5.1 Kendall τ：W2V vs SVD 结构一致性
iu = np.triu_indices(N, k=1)
tau, pval = kendalltau(svd_sim[iu], w2v_sim[iu])
print(f"  Kendall τ = {tau:.4f}  (p = {pval:.2e})")
if tau > 0.6:
    print("  → 概念结构高度自洽 ✅（全局谱结构与局部随机游走一致）")
elif tau > 0.3:
    print("  → 概念结构中等的洽 ⚠️（大部分自洽，少数'表里不一'）")
else:
    print("  → 结构不一致，可能存在非线性跳跃（正言若反）🔍")

# 5.2 与宏观语义的一致性：宏观态内 W2V 相似度是否高于宏观态间
print("\n  宏观态内 vs 宏观态间的 W2V 相似度：")
if macro_labels:
    within, between = [], []
    for i in range(N):
        for j in range(i + 1, N):
            sim = w2v_sim[i, j]
            if macro_labels.get(concepts[i]) == macro_labels.get(concepts[j]):
                within.append(sim)
            else:
                between.append(sim)
    mean_within = np.mean(within) if within else 0
    mean_between = np.mean(between) if between else 0
    print(f"    宏观态内平均相似度: {mean_within:.4f}")
    print(f"    宏观态间平均相似度: {mean_between:.4f}")
    print(f"    （内-间差值: {mean_within - mean_between:+.4f}，>0 表示嵌入捕捉到宏观语义）")

# ============================================================
# 6. 下游任务：节点分类
# ============================================================
print("\n[步骤5] 下游任务 1：节点分类（宏观态标签）")
print("-" * 50)
if macro_labels:
    y = np.array([macro_labels.get(c, -1) for c in concepts])
    # 仅用有标签的概念（排除 -1）
    valid = y >= 0
    X = w2v[valid]
    yv = y[valid]
    # 分层划分
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, yv, test_size=0.3, random_state=42, stratify=yv)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    print(f"  训练样本: {len(X_tr)}, 测试样本: {len(X_te)}")
    print(f"  节点分类准确率 = {acc:.4f}")
    # 对比：SVD 嵌入上的分类
    X_svd = svd_vec[valid]
    clf_svd = LogisticRegression(max_iter=1000, random_state=42)
    X_tr_s, X_te_s, _, _ = train_test_split(
        X_svd, yv, test_size=0.3, random_state=42, stratify=yv)
    clf_svd.fit(X_tr_s, y_tr)
    acc_svd = clf_svd.score(X_te_s, y_te)
    print(f"  [对比] SVD 嵌入节点分类准确率 = {acc_svd:.4f}")
else:
    print("  无宏观标签，跳过节点分类")
    acc = acc_svd = None

# ============================================================
# 7. 下游任务：链接预测
# ============================================================
print("\n[步骤6] 下游任务 2：链接预测（转移强度）")
print("-" * 50)


def link_prediction(emb, P, concepts, seed=42):
    """用嵌入内积（cosine）预测转移概率，评估 AUC。
    正样本：转移概率高的概念对；负样本：低/零转移对。"""
    rng = np.random.default_rng(seed)
    # 概念对转移概率
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    p_ij = np.array([(P[i, j] + P[j, i]) / 2 for i, j in pairs])
    # 用中位数分为正负样本
    thr = np.percentile(p_ij, 70)
    y_link = (p_ij > thr).astype(int)
    # 嵌入相似度
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim_ij = np.array([np.dot(emb_norm[i], emb_norm[j]) for i, j in pairs])
    # AUC（随机采样平衡正负）
    pos_idx = np.where(y_link == 1)[0]
    neg_idx = np.where(y_link == 0)[0]
    n_pos = len(pos_idx)
    neg_sampled = rng.choice(neg_idx, min(n_pos * 3, len(neg_idx)), replace=False)
    sel = np.concatenate([pos_idx, neg_sampled])
    return roc_auc_score(y_link[sel], sim_ij[sel])


auc_w2v = link_prediction(w2v, P, concepts)
auc_svd = link_prediction(svd_vec, P, concepts)
print(f"  链接预测 AUC (W2V) = {auc_w2v:.4f}")
print(f"  链接预测 AUC (SVD) = {auc_svd:.4f}")

# ============================================================
# 8. 可视化
# ============================================================
print("\n[步骤7] 可视化")
print("-" * 50)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字体（复用 core.env）
    try:
        from core.env import CN_FONT
        plt.rcParams['font.family'] = CN_FONT.get_name()
    except Exception:
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # 左：SVD 谱嵌入
    ax[0].scatter(svd_vec[:, 0], svd_vec[:, 1], alpha=0.7)
    for i, c in enumerate(concepts):
        ax[0].annotate(c, (svd_vec[i, 0], svd_vec[i, 1]), fontsize=8)
    ax[0].set_title("SVD 谱空间嵌入")
    ax[0].set_xlabel("Dim1"); ax[0].set_ylabel("Dim2")
    ax[0].grid(True, alpha=0.3)
    # 右：W2V 嵌入（PCA 降维到2D）
    pca = PCA(n_components=2)
    w2v_2d = pca.fit_transform(w2v)
    ax[1].scatter(w2v_2d[:, 0], w2v_2d[:, 1], alpha=0.7, color="orange")
    for i, c in enumerate(concepts):
        ax[1].annotate(c, (w2v_2d[i, 0], w2v_2d[i, 1]), fontsize=8)
    ax[1].set_title("Word2Vec 嵌入（PCA 投影）")
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "output/vis_10_w2v_vs_svd.png"), dpi=150)
    plt.close()
    print("  可视化 → output/vis_10_w2v_vs_svd.png")
except Exception as e:
    print(f"  ⚠️ 可视化跳过: {e}")

# ============================================================
# 9. 保存对比数据
# ============================================================
comp = pd.DataFrame({
    "Concept": concepts,
    "SVD_Dim1": svd_vec[:, 0], "SVD_Dim2": svd_vec[:, 1],
})
pca2 = PCA(n_components=2)
w2v_2d_final = pca2.fit_transform(w2v)
comp["W2V_PC1"] = w2v_2d_final[:, 0]; comp["W2V_PC2"] = w2v_2d_final[:, 1]
if macro_labels:
    comp["Macro"] = [macro_labels.get(c, -1) for c in concepts]
comp.to_csv(os.path.join(ROOT, "output/svd_vs_w2v_comparison.csv"),
            index=False, encoding="utf-8")
print("  对比数据 → output/svd_vs_w2v_comparison.csv")

# 汇总评估结果
result = {
    "kendall_tau": float(tau),
    "kendall_p": float(pval),
    "node_classification_acc_w2v": acc,
    "node_classification_acc_svd": acc_svd,
    "link_prediction_auc_w2v": float(auc_w2v),
    "link_prediction_auc_svd": float(auc_svd),
    "num_walks": num_walks,
    "walk_length": walk_length,
}
with open(os.path.join(ROOT, "output/w2v_experiment_results.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\n  评估结果 → output/w2v_experiment_results.json")

print("\n" + "=" * 60)
print("P1-A 完成。核心产出：")
print("  1) output/w2v_vectors.npy")
print("  2) output/w2v_similarity.csv")
print("  3) output/vis_10_w2v_vs_svd.png")
print(f"  4) Kendall τ = {tau:.4f}")
print(f"  5) 节点分类准确率 (W2V) = {acc if acc is not None else 'N/A'}")
print(f"  6) 链接预测 AUC (W2V) = {auc_w2v:.4f}")
print("=" * 60)
