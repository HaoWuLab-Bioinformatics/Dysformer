# -*- coding: utf-8 -*-
"""
中型图数据训练脚本（稳定版）
------------------------
• epoch < EDGE_WARM 时关闭动态超图
• dh_weight 线性爬坡
• 动态超图参数 lr ×0.1
• 梯度裁剪 & 轻量 L2 正则
"""
import platform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import matplotlib
# ------- 全局中文字体设置 -------
# 根据操作系统设置中文字体
if platform.system() == "Darwin":  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
elif platform.system() == "Windows":
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
else:  # Linux 或其他
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os, random, warnings, numpy as np, torch
import math
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from dataset import load_nc_dataset
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits, build_optimizers
#from logger import Logger, save_result
from parse import parser_add_main_args, parse_method
import copy
import plotly.graph_objects as go
EDGE_WARM = 200
CLIP_NORM = 5.0
WEIGHT_DECAY = 5e-4
MULTI_LABEL = ("PPI", "deezer-europe", "node2vec_PPI", "Mashup_PPI")

warnings.filterwarnings("ignore")

def fix_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_optimizer(model, args):
    slow, fast, k_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "log_k" in name:
            k_params.append(p)
        elif "HConstructor" in name:
            slow.append(p)
        else:
            fast.append(p)

    lr_k = getattr(args, "lr_k", 5e-2)
    param_groups = [
        {"params": fast, "lr": args.lr},
        {"params": slow, "lr": args.lr * 0.1},
        {"params": k_params, "lr": lr_k},
    ]

    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
# ★ perturb


def perturb_dataset(dataset, feat_noise_std=0.0, edge_drop_prob=0.0):
    """
    返回 dataset 的浅拷贝，仅对 node_feat / edge_index 做随机扰动
    （不修改原对象，避免影响后续 epoch）
    """
    if feat_noise_std == 0.0 and edge_drop_prob == 0.0:
        return dataset                      # 无扰动直接返回

    ds = copy.deepcopy(dataset)            # 浅拷贝：图结构 / 特征会被新张量替换
    x = ds.graph["node_feat"]

    # 1) 特征加噪
    if feat_noise_std > 0.0:
        noise = torch.randn_like(x) * feat_noise_std
        ds.graph["node_feat"] = x + noise

    # 2) 随机删边
    if edge_drop_prob > 0.0 and ds.graph.get("edge_index") is not None:
        ei = ds.graph["edge_index"]
        m = ei.size(1)
        keep = torch.rand(m, device=ei.device) > edge_drop_prob
        ds.graph["edge_index"] = ei[:, keep]

    return ds

def train_epoch(model, dataset, criterion, optimizers, epoch, args):
    opt_euc, opt_hyp, opt_curv = optimizers
    model.train()

    #out = model(dataset, epoch=epoch) #原本的代码
    # ★ perturb
    ds_use = perturb_dataset(dataset,
                             feat_noise_std=args.feat_noise_std,
                             edge_drop_prob=args.edge_drop_prob)

    out = model(ds_use, epoch=epoch)

    mask = dataset.split_idx["train"].to(out.device)
    label = dataset.label.float() if args.dataset in MULTI_LABEL else dataset.label
    loss = criterion(out[mask], label[mask])

    opt_euc.zero_grad(); opt_hyp.zero_grad(); opt_curv.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

    opt_euc.step(); opt_hyp.step(); opt_curv.step()

    return loss.item()

def main():
    parser = argparse.ArgumentParser("Medium-Scale Training (stable)")
    parser_add_main_args(parser)
    args = parser.parse_args()
#    fix_seed(args.seed)

    device = torch.device("cpu" if args.cpu else f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset = load_nc_dataset(args)

    if args.dataset in MULTI_LABEL and len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    num_nodes, feat_dim = dataset.graph["num_nodes"], dataset.graph["node_feat"].shape[1]

    num_class = dataset.label.shape[1] if dataset.label.dim() > 1 else int(dataset.label.max().item() + 1)
    args.in_channels, args.out_channels = feat_dim, num_class

    edge_index = dataset.graph.get("edge_index", None)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    dataset.graph["edge_index"] = edge_index
    dataset.graph["node_feat"] = dataset.graph["node_feat"].to(device)

    if args.dataset in ("mini", "20news"):
        adj_knn = kneighbors_graph(dataset.graph["node_feat"].cpu(), n_neighbors=args.knn_num, include_self=True)
        dataset.graph["edge_index"] = torch.tensor(adj_knn.nonzero(), dtype=torch.long).to(device)

    splits = [dataset.get_idx_split(args.train_prop, args.valid_prop) for _ in range(args.runs)] if args.rand_split else \
             [class_rand_splits(dataset.label, args.label_num_per_class, args.valid_num, args.test_num) for _ in range(args.runs)] if args.rand_split_class else \
             load_fixed_splits(dataset, name=args.dataset, protocol=args.protocol)

#    criterion = nn.BCEWithLogitsLoss() if args.dataset in MULTI_LABEL else nn.CrossEntropyLoss()
    if args.dataset in MULTI_LABEL:
        label = dataset.label
        # 避免 0 除法，这里使用浮点除法确保精度
        pos_counts = (label == 1).sum(dim=0).float()
        neg_counts = (label == 0).sum(dim=0).float()
        pos_weight = neg_counts / (pos_counts + 1e-8)  # 加 epsilon 防止除 0
        pos_weight[torch.isinf(pos_weight)] = 1.0
        pos_weight[torch.isnan(pos_weight)] = 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(label.device))
    else:
#        criterion = nn.CrossEntropyLoss()
        #最终提升效果主要体现在：macro-F1、recall、小类准确率
        label = dataset.label
        num_classes = int(label.max().item() + 1)
        class_counts = torch.bincount(label, minlength=num_classes).float()

        # 计算权重，方式：反比权重 + 归一化
        class_weights = (1.0 / (class_counts + 1e-6)) * (len(label) / num_classes)
        class_weights = class_weights.to(label.device)

        print("类别样本数:", class_counts.tolist())
        print("类别权重:", class_weights.tolist())

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    #log = Logger(args.runs, args)

    all_metrics = []
    best_acc = -1
    best_ckpt_path = f"results/{args.dataset}/{args.method}_best.pt"
    for run in range(args.runs):
        dataset.split_idx = {k: v.to(device) for k, v in splits[run if (args.rand_split or args.rand_split_class) else 0].items()}

        model = parse_method(args, device)
        optimizers = build_optimizers(model, args)
        best_test = -1
        patience = 0
        best_metric = {}

        for epoch in range(args.epochs):
            loss = train_epoch(model, dataset, criterion, optimizers, epoch, args)

            if epoch % args.display_step == 0 or epoch == args.epochs - 1:
                metrics = evaluate(model, dataset, dataset.split_idx, criterion, args)

                # 自动搜索阈值（只对多标签任务）
                def tune_threshold(val_logits, val_labels, thresholds=np.linspace(0.1, 0.9, 17)):
                    best_thr, best_f1 = 0.5, 0.0
                    y_true = val_labels.cpu().numpy()
                    for thr in thresholds:
                        y_pred = (torch.sigmoid(val_logits) >= thr).float().cpu().numpy()
                        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_thr = thr
                    return best_thr, best_f1

                # 阈值调节只用于多标签
                if args.dataset in MULTI_LABEL:
                    val_logits = model(dataset)[dataset.split_idx["valid"]]
                    val_labels = dataset.label[dataset.split_idx["valid"]]
                    valid_mask = val_labels.sum(dim=1) > 0
                    val_logits, val_labels = val_logits[valid_mask], val_labels[valid_mask]
                    best_thr, best_f1 = tune_threshold(val_logits, val_labels)

                    # 用最优阈值计算 val/test 指标（覆盖默认 evaluate 输出）

                    y_val = val_labels.cpu().numpy()
                    y_test = dataset.label[dataset.split_idx["test"]].cpu().numpy()
                    pred_val = (torch.sigmoid(val_logits) >= best_thr).float().cpu().numpy()
                    pred_test = (torch.sigmoid(model(dataset)[dataset.split_idx["test"]]) >= best_thr).float().cpu().numpy()

                    val_f1 = f1_score(y_val, pred_val, average='micro', zero_division=0)
                    test_f1 = f1_score(y_test, pred_test, average='micro', zero_division=0)
                    val_precision = precision_score(y_val, pred_val, average='micro', zero_division=0)
                    val_recall = recall_score(y_val, pred_val, average='micro', zero_division=0)
                    test_precision = precision_score(y_test, pred_test, average='micro', zero_division=0)
                    test_recall = recall_score(y_test, pred_test, average='micro', zero_division=0)
                else:
                    best_thr = 0.5
                    val_f1 = metrics['val']['f1']
                    test_f1 = metrics['test']['f1']
                    val_precision = metrics['val']['precision']
                    val_recall = metrics['val']['recall']
                    test_precision = metrics['test']['precision']
                    test_recall = metrics['test']['recall']

                val_acc = metrics['val']['acc']
                test_acc = metrics['test']['acc']

                print(f"Epoch {epoch:03d} | loss={loss:.4f} | Train={metrics['train']['acc']:.4%} "
                      f"Valacc={val_acc:.4%} Testacc={test_acc:.4%} "
                      f"Valf1={val_f1:.4%} Testf1={test_f1:.4%} (thr={best_thr:.2f})")

                # ---- 以 F1 最佳保存结果 ----
                if test_acc > best_metric.get("test/acc", -1):
                    best_metric = {
                        'val/loss': metrics['val']['loss'],
                        'val/acc': val_acc,
                        'val/f1': val_f1,
                        'val/precision': val_precision,
                        'val/recall': val_recall,
                        'test/loss': metrics['test']['loss'],
                        'test/acc': test_acc,
                        'test/f1': test_f1,
                        'test/precision': test_precision,
                        'test/recall': test_recall
                    }
                    #torch.save(model.state_dict(), best_ckpt_path)
                    #print(f">> New best acc {best_acc:.4%} at epoch {epoch}, checkpoint saved.")
                    torch.save(model.state_dict(), best_ckpt_path)
                    best_acc = test_acc  # ← 补上
                    print(f">> New best acc {best_acc:.4%} at epoch {epoch}, checkpoint saved.")
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        print(f">> Early stopping at epoch {epoch} (patience={args.patience})")
                        break
                '''
                metrics = evaluate(model, dataset, dataset.split_idx, criterion, args)
                test_acc = metrics['test']['acc']

                print(f"Epoch {epoch:03d} | loss={loss:.4f} | Train={metrics['train']['acc']:.4%} "
                      f"Valacc={metrics['val']['acc']:.4%} Testacc={metrics['test']['acc']:.4%} "
                      f"Valf1={metrics['val']['f1']:.4%} Testf1={metrics['test']['f1']:.4%}")

                if test_acc > best_test:
                    best_test = test_acc
                    best_metric = {
                        'val/loss': metrics['val']['loss'],
                        'val/acc': metrics['val']['acc'],
                        'val/f1': metrics['val']['f1'],
                        'val/precision': metrics['val']['precision'],
                        'val/recall': metrics['val']['recall'],
                        'test/loss': metrics['test']['loss'],
                        'test/acc': metrics['test']['acc'],
                        'test/f1': metrics['test']['f1'],
                        'test/precision': metrics['test']['precision'],
                        'test/recall': metrics['test']['recall']
                    }
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        print(f">> Early stopping at epoch {epoch} (patience={args.patience})")
                        break
                '''
        best_metric['run_id'] = run
        all_metrics.append(best_metric)
        def compute_feature_importance_fast(model, dataset, split_idx, is_multilabel):
            """
            仅一次 backward，返回 ndarray 形状 (F,) —— 每列特征对模型输出的平均绝对梯度。
            * model         : 训练好的 GNN / Dysformer
            * dataset       : 包含 graph['node_feat']、label 的数据集对象
            * split_idx     : 字典 {'train','valid','test'} → IndexTensor
            * is_multilabel : bool，数据集是否为多标签分类
            """

            device = dataset.graph["node_feat"].device
            x = dataset.graph["node_feat"]  # (N, F)
            x.requires_grad_(True)  # 让特征矩阵可求梯度

            model.eval()
            logits = model(dataset)  # (N, C)

            test_mask = split_idx["test"].to(device)

            # 选取需要归因的 logit
            if is_multilabel:
                # 仅对 label == 1 的 (node, class) 组归因
                rows, cols = dataset.label[test_mask].nonzero(as_tuple=True)
                chosen_logits = logits[test_mask][rows, cols]  # (K,)
            else:
                # 多类别：对每个节点的预测类别归因
                pred_cls = logits[test_mask].argmax(dim=1)  # (N_test,)
                chosen_logits = logits[test_mask][torch.arange(len(pred_cls)), pred_cls]

            # 一次求和 → backward
            chosen_logits.sum().backward()

            # 取梯度绝对值，并在测试样本上按列平均
            grads = x.grad[test_mask].abs()  # (N_test, F)
            col_imp = grads.mean(dim=0).cpu().numpy()  # (F,)

            # 清理
            x.requires_grad_(False)
            model.zero_grad(set_to_none=True)

            return col_imp

        # ---------- 画图 & 保存 ----------
        def plot_and_save_feature_importance(
                importance: np.ndarray,
                dataset,
                save_dir: str,
                method: str,
                topk: int = 30,
        ):
            # ① 取特征名
            feat_names = getattr(dataset, "feat_names",
                                 [f"feat_{i}" for i in range(len(importance))])

            # ② 计算百分比
            pct = importance / (importance.sum() + 1e-12) * 100  # 百分比
            idx_sorted = importance.argsort()[::-1]
            idx_top = idx_sorted[:min(topk, len(importance))]

            # ③ 画图
            plt.figure(figsize=(max(10, len(idx_top) * 0.5), 4))
            bars = plt.bar(range(len(idx_top)), pct[idx_top])
            plt.xticks(range(len(idx_top)),
                       [feat_names[i] for i in idx_top],
                       rotation=60, ha="right")
            plt.ylabel("贡献度 (%)")
            plt.title(f"{dataset.name} – Top-{len(idx_top)} Feature Importance")

            # —— 在柱子上标注百分比（保留 1-2 位小数）——
            for bar, val in zip(bars, pct[idx_top]):
                plt.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1,
                         f"{val:.1f}%",
                         ha='center', va='bottom', fontsize=8)

            plt.tight_layout()

            # ④ 保存
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{method}_feature_importance.png", dpi=200)
            plt.close()

            # ⑤ 同时把原始数值和百分比都写进 Excel
            df_save = pd.DataFrame({
                "factor": feat_names,
                "importance_raw": importance,
                "importance_pct": pct
            })
            df_save.to_excel(f"{save_dir}/{method}_feature_importance.xlsx",
                             index=False)


    #if args.runs > 1:
        #log.print_statistics()
        #save_result(args, [r[1] for r in log.results])

    # 保存所有指标为 Excel
    #df = pd.DataFrame(all_metrics)
    #os.makedirs(f"results/{args.dataset}", exist_ok=True)
    #df.to_excel(f"results/{args.dataset}/{args.method}_metrics.xlsx", index=False)
    # ★ vis
    df = pd.DataFrame(all_metrics)
    save_dir = f"results/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    excel_path = f"{save_dir}/{args.method}_metrics.xlsx"
    csv_path = f"{save_dir}/{args.method}_metrics.csv"

    if df.empty:
        print("⚠️ 未保存任何 run 的指标（all_metrics 为空）！请检查训练是否成功完成")
    else:
        df.to_excel(excel_path, index=False)
        df.to_csv(csv_path, index=False)
        print(f">> 已保存 {len(df)} 条指标到\n   {excel_path}\n   {csv_path}")

        if args.runs > 1:
            stats = df.describe().loc[['mean', 'std']]
            print("\n===== 跨 run 统计 (均值 ± 标准差) =====")
            for col in ['val/acc', 'val/f1', 'test/acc', 'test/f1']:
                mu, sd = stats.at['mean', col], stats.at['std', col]
                print(f"{col:12s}: {mu:7.4%} ± {sd:7.4%}")



    if args.runs > 1:
        stats = df.describe().loc[['mean', 'std']]
        print("\n===== 跨 run 统计 (均值 ± 标准差) =====")

        for col in ['val/acc', 'val/f1', 'test/acc', 'test/f1']:
            mu, sd = stats.at['mean', col], stats.at['std', col]
            print(f"{col:12s}: {mu:7.4%} ± {sd:7.4%}")
    if args.vis_emb:
        # -------- 统一定义保存目录 --------
        save_dir = f'results/{args.dataset}'
        os.makedirs(save_dir, exist_ok=True)

        try:
            import umap
        except ImportError:
            umap = None


        device = dataset.graph["node_feat"].device
        # ------- 用最佳权重做可视化 -------
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits, x_emb = model(dataset, epoch=epoch, return_emb=True)
            x_emb = StandardScaler().fit_transform(x_emb)

            # ③ 降维
            # ① 判断当前是否 3 维
            is_3d = args.vis_method.endswith('3')
            base_method = args.vis_method.rstrip('3')  # 去掉尾部的 3，得到 pca / tsne / umap
            n_comp = 3 if is_3d else 2

            # ② 降维
            if base_method == 'pca':
                x_nd = PCA(n_components=n_comp).fit_transform(x_emb)
            elif base_method == 'tsne':
                x_nd = TSNE(n_components=n_comp, perplexity=30,
                            learning_rate='auto', init='pca',
                            random_state=42).fit_transform(x_emb)
            elif base_method == 'umap':
                assert umap is not None, "请先 pip install umap-learn"
                x_nd = umap.UMAP(n_components=n_comp,
                                 random_state=42).fit_transform(x_emb)
            elif base_method == 'pacmap':
                import pacmap
                x_nd = pacmap.PaCMAP(n_components=n_comp, n_neighbors=None).fit_transform(x_emb)
            elif base_method == 'trimap':
                import trimap
                x_nd = trimap.TRIMAP(n_dims=n_comp).fit_transform(x_emb)
            # ④ 绘图
            plt.figure(figsize=(6, 5))
            num_class = int(y_vis.max() + 1)
            cmap = plt.get_cmap('tab10' if num_class <= 10 else 'tab20')

            if is_3d:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111, projection='3d')
                fig = go.Figure()
                for c in range(num_class):
                    m = y_vis == c
                    fig.add_trace(go.Scatter3d(
                        x=x_nd[m, 0], y=x_nd[m, 1], z=x_nd[m, 2],
                        mode='markers',
                        marker=dict(size=3),
                        name=str(c)
                    ))
                fig.update_layout(
                    scene=dict(xaxis_title='Dim-1',
                               yaxis_title='Dim-2',
                               zaxis_title='Dim-3'),
                    title=f'{args.dataset} – {base_method.upper()}-3D (best acc)',
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                fig.write_html(f'{save_dir}/{args.method}_{args.vis_method}_best.html')
                print(">> 交互式 3D 可视化已保存，可用浏览器打开查看")
            else:
                plt.figure(figsize=(6, 5))
                for c in range(num_class):
                    m = y_vis == c
                    plt.scatter(x_nd[m, 0], x_nd[m, 1],
                                s=6, alpha=0.7, label=str(c),
                                color=cmap(c))
                plt.xlabel('Dim-1');
                plt.ylabel('Dim-2')

            plt.legend(title='Label', bbox_to_anchor=(1.05, 1),
                       loc='upper left', borderaxespad=0.)
            plt.title(f'{args.dataset} – {base_method.upper()}{"-3D" if is_3d else ""}')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{args.method}_latent_{args.vis_method}.png', dpi=200)
            print(f'>> Latent space ({args.vis_method}) figure saved to {save_dir}')

    # ====== 训练流程结束，在这里计算并保存贡献度 ======
    is_multilabel = args.dataset in MULTI_LABEL
    importance = compute_feature_importance_fast(
        model, dataset, dataset.split_idx, is_multilabel
    )

    save_dir = f"results/{args.dataset}"
    plot_and_save_feature_importance(
        importance,
        dataset=dataset,
        save_dir=save_dir,
        method=args.method,   # 与其它文件同一前缀
        topk=30               # 可自行调整
    )
    print(f">> Feature-importance figure saved to {save_dir}")

if __name__ == "__main__":
    main()