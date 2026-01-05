# -*- coding: utf-8 -*-
"""
中型图数据训练脚本（稳定版）
------------------------
• epoch < EDGE_WARM 时关闭动态超图
• dh_weight 线性爬坡
• 动态超图参数 lr ×0.1
• 梯度裁剪 & 轻量 L2 正则
"""
import json
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

from dataset import load_nc_dataset
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits, build_optimizers
#from logger import Logger, save_result
from parse import parser_add_main_args, parse_method
import copy
import plotly.graph_objects as go
import torch

# 保存原始的 torch.load
_orig_torch_load = torch.load

def _torch_load_unsafe(*args, **kwargs):
    # 如果外部没传 weights_only，就默认设为 False
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

# 全局替换 torch.load
torch.load = _torch_load_unsafe

EDGE_WARM = 200
CLIP_NORM = 5.0
WEIGHT_DECAY = 5e-4
MULTI_LABEL = ("PPI", "deezer-europe", "node2vec_PPI", "Mashup_PPI")
# main_new2.py 开头或 train_epoch 入口
import torch, warnings

def add_nan_debug_hooks(model):
    def _make_hook(name):
        def _hook(grad):
            if grad is None:
                return
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"[NaN/Inf grad] at param: {name}")
        return _hook

    for n, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(_make_hook(n))

def check_tensor(name, x):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"[NaN/Inf tensor] {name} has non-finite values")

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

    lr_k = getattr(args, "lr_k", 1e-3)
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

    out = model(dataset, epoch=epoch)
    if not torch.isfinite(out).all():
        raise RuntimeError("[NaN/Inf] logits contains non-finite values")

    mask = dataset.split_idx["train"].to(out.device)
    labels = dataset.label

    # 多标签：过滤全零标签样本（避免把“未知”当全负）
    if args.dataset in MULTI_LABEL:
        train_mask = mask.clone()
        no_label_nodes = (labels.sum(dim=1) == 0)
        train_mask[no_label_nodes] = False
        loss = criterion(out[train_mask], labels[train_mask].float())
    else:
        # 单标签（如 Cora）：确保 label 合法且 one-hot 不是必须（CE 用整型类别）
        if labels.dtype != torch.long:
            labels = labels.long()
        loss = criterion(out[mask], labels[mask])

    if not torch.isfinite(loss).all():

        bad_idx = mask.nonzero(as_tuple=True)[0]
        raise RuntimeError(f"[NaN/Inf] loss is non-finite at epoch {epoch}, train_batch_size={bad_idx.numel()}")

    opt_euc.zero_grad(); opt_hyp.zero_grad(); opt_curv.zero_grad()

    # 反向传播（异常时更快定位）
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

    opt_euc.step(); opt_hyp.step(); opt_curv.step()
    return float(loss.detach().cpu())



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


def main():

    parser = argparse.ArgumentParser("Medium-Scale Training (stable)")
    parser_add_main_args(parser)          # 这⾥面已包含 --device
    args = parser.parse_args()

    # ★ 如果想让进程的“默认 GPU”就是你传的那块卡，必须手动 set_device
    if args.device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    device = torch.device(
        "cpu" if args.cpu else f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )


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
        pos_counts = (label == 1).sum(dim=0).float()
        neg_counts = (label == 0).sum(dim=0).float()
        pos_weight = neg_counts / (pos_counts + 1e-8)
        pos_weight[torch.isinf(pos_weight)] = 1.0
        pos_weight[torch.isnan(pos_weight)] = 1.0
        # **Cap extremely large pos_weight to avoid huge gradients:**
        pos_weight = torch.clamp(pos_weight, max=50.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(label.device))
    else:
        label = dataset.label
        num_classes = int(label.max().item() + 1)
        class_counts = torch.bincount(label, minlength=num_classes).float()
        class_weights = (1.0 / (class_counts + 1e-6)) * (len(label) / num_classes)
        class_weights = class_weights.to(label.device)
        print("类别样本数:", class_counts.tolist())
        print("类别权重:", class_weights.tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    all_metrics = []
    best_acc = -1
    save_dir = f"results/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    excel_path = f"{save_dir}/{args.method}_metrics.xlsx"
    csv_path = f"{save_dir}/{args.method}_metrics.csv"
    best_ckpt_path = f"results/{args.dataset}/{args.method}_best.pt"
    for run in range(args.runs):
        dataset.split_idx = {k: v.to(device) for k, v in splits[run if (args.rand_split or args.rand_split_class) else 0].items()}

        model = parse_method(args, device)
        print(">>> use_dhyper =", getattr(args, "use_dhyper", None),
              "| model =", type(model).__name__, flush=True)

        optimizers = build_optimizers(model, args)
        best_test = -1
        patience = 0
        best_metric = {}

        for epoch in range(args.epochs):
            loss = train_epoch(model, dataset, criterion, optimizers, epoch, args)

            if epoch % args.display_step == 0 or epoch == args.epochs - 1:
                metrics = evaluate(model, dataset, dataset.split_idx, criterion, args)

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

                if args.dataset in MULTI_LABEL:
                    val_logits = model(dataset)[dataset.split_idx["valid"]]
                    val_labels = dataset.label[dataset.split_idx["valid"]]
                    valid_mask = val_labels.sum(dim=1) > 0
                    val_logits, val_labels = val_logits[valid_mask], val_labels[valid_mask]
                    best_thr, best_f1 = tune_threshold(val_logits, val_labels)

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
                    torch.save(model.state_dict(), best_ckpt_path)
                    best_acc = test_acc
                    print(f">> New best acc {best_acc:.4%} at epoch {epoch}, checkpoint saved.")
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        print(f">> Early stopping at epoch {epoch} (patience={args.patience})")
                        break

        best_metric['run_id'] = run
        # (A) 打印本 run 的最终 best_metric（就是最后一次被保存到 ckpt 的那份）
        print("\n===== 本次 run 的最终 best_metric =====")
        print(json.dumps(best_metric, indent=2, ensure_ascii=False))
        print("=====================================\n")

        # (B) 即时落盘：把本 run 的最佳指标写入 CSV & Excel
        df_run = pd.DataFrame([best_metric])

        # 1) CSV 追加写（首行含表头，后续不重复）
        if os.path.exists(csv_path):
            df_run.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_run.to_csv(csv_path, index=False)

        # 2) Excel：读旧表合并后重写，避免表头/格式问题
        if os.path.exists(excel_path):
            try:
                df_old = pd.read_excel(excel_path)
                df_new = pd.concat([df_old, df_run], ignore_index=True)
            except Exception:
                df_new = df_run
        else:
            df_new = df_run
        df_new.to_excel(excel_path, index=False)

        print(f">> 本次 run({run}) 指标已写入：\n   {excel_path}\n   {csv_path}")
        all_metrics.append(best_metric)
        # === 新增：每个 run 结束后立刻写入 Excel/CSV ===
        # 将本次 run 的最佳指标落盘
        df_run = pd.DataFrame([best_metric])

        # 1) 写 CSV（支持原生追加写，但为避免表头重复，做存在性判断）
        if os.path.exists(csv_path):
            df_run.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_run.to_csv(csv_path, index=False)

        # 2) 写 Excel：为稳妥起见，采用“读旧表 + 合并 + 重写”的方式
        if os.path.exists(excel_path):
            try:
                df_old = pd.read_excel(excel_path)
                df_new = pd.concat([df_old, df_run], ignore_index=True)
            except Exception:
                # 若旧表损坏或版本不兼容，退化为仅写当前 run
                df_new = df_run
        else:
            df_new = df_run
        df_new.to_excel(excel_path, index=False)

        print(f">> 本次 run({run}) 指标已写入：\n   {excel_path}\n   {csv_path}")

    df = pd.DataFrame(all_metrics)

    print("use_dhyper =", args.use_dhyper)
    print("model =", type(model))
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



    if args.vis_emb:
        # -------- 统一定义保存目录 --------
        save_dir = f'results/{args.dataset}'
        os.makedirs(save_dir, exist_ok=True)

        try:
            import umap
        except ImportError:
            umap = None

        device = dataset.graph["node_feat"].device
        # === 替换开始 ===
        ckpt = torch.load(best_ckpt_path, map_location=device)
        key_mu = 'dhgnn_conv.HConstructor.edges_mu'
        key_ls = 'dhgnn_conv.HConstructor.edges_logsigma'

        # 只有在 HypFormer 且启用了 DHGNN 分支时才需要对齐
        need_dhyper = hasattr(model, 'dhgnn_conv') and model.dhgnn_conv is not None and hasattr(model.dhgnn_conv,
                                                                                                'HConstructor')

        if need_dhyper and key_mu in ckpt and key_ls in ckpt:
            E_ckpt = ckpt[key_mu].shape[0]
            E_target = getattr(args, 'num_edges', E_ckpt)

            # 1) 先按 ckpt 的超边数重建一次模型并加载（保留已学到的原型）
            args_backup_num_edges = args.num_edges
            args.num_edges = E_ckpt
            model = parse_method(args, device)
            model.load_state_dict(ckpt, strict=True)
            print(f'✅ 按 ckpt 超边数 {E_ckpt} 行成功加载参数')

            # 2) 如需扩到目标行数（例如你现在配置是 312）
            if E_target > E_ckpt:
                hc = model.dhgnn_conv.HConstructor
                hc.num_edges = E_target
                with torch.no_grad():
                    hc._expand_parameters()  # 你类里已实现：复制最后一行扩展
                print(f'✅ 已把超边参数从 {E_ckpt} 扩展到 {E_target} 行')

            # 还原 args 以免后续逻辑用到
            args.num_edges = args_backup_num_edges
        else:
            # 没启用 DHGNN 或 ckpt 不含这些键，正常加载
            model.load_state_dict(ckpt, strict=False)
            print('ℹ️ 未检测到 DHGNN 超边参数冲突，使用宽松加载（strict=False）')

        model.eval()
        # === 替换结束 ===

        # ==== 抽取“改变后”嵌入：使用 forward hook ====
        latent_list = []

        def hook(module, inp, out):
            latent_list.append(out.detach())

        # 根据你的模型结构，这里假设目标层是最后一层 trans_conv 中的 convs[-1]
        # 如需调整，请换成你想观察的层
        assert hasattr(model, "trans_conv"), "模型中未找到 trans_conv，请检查结构"
        target_layer = model.trans_conv.convs[-1]
        handle = target_layer.register_forward_hook(hook)

        with torch.no_grad():
            _ = model(dataset)  # 触发 forward，hook 捕获中间输出

        handle.remove()  # 清除 hook

        x_emb = latent_list[0].cpu().numpy()  # 提取嵌入

        # ==== 后续处理保持一致 ====
        if 0 < args.vis_sample < 1:
            n = x_emb.shape[0]
            idx = np.random.choice(n, int(n * args.vis_sample), replace=False)
            x_emb = x_emb[idx]
            y_vis = dataset.label.cpu().numpy()[idx]
        else:
            y_vis = dataset.label.cpu().numpy()

        is_3d = args.vis_method.endswith('3')
        base_method = args.vis_method.rstrip('3')
        n_comp = 3 if is_3d else 2

        if base_method == 'pca':
            x_nd = PCA(n_components=n_comp).fit_transform(x_emb)
        elif base_method == 'tsne':
            # sklearn 0.24 里不要用 learning_rate='auto'，用数值即可
            # init 用 'random' 更稳，不依赖 PCA
            x_nd = TSNE(
                n_components=n_comp,
                perplexity=30,
                learning_rate=200.0,  # ⭐ 改成数值
                init='random',  # ⭐ 改成 random，避免额外依赖
                random_state=42
            ).fit_transform(x_emb)

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

        plt.figure(figsize=(6, 5))
        num_class = int(y_vis.max() + 1)
        cmap = plt.get_cmap('tab10' if num_class <= 10 else 'tab20')

        # ==== 可视化（替换你原来的 is_3d 分支及其后全局的 legend/title/savefig）====
        if is_3d:
            # 1) 静态 PNG：用 matplotlib/mplot3d 画点
            fig_png = plt.figure(figsize=(7, 6))
            ax = fig_png.add_subplot(111, projection='3d')
            for c in range(num_class):
                m = (y_vis == c)
                ax.scatter(x_nd[m, 0], x_nd[m, 1], x_nd[m, 2],
                           s=6, alpha=0.7, label=str(c))
            ax.set_xlabel('Dim-1');
            ax.set_ylabel('Dim-2');
            ax.set_zlabel('Dim-3')
            ax.legend(title='Label', bbox_to_anchor=(1.05, 1),
                      loc='upper left', borderaxespad=0.)
            plt.title(f'{args.dataset} – {base_method.upper()}-3D (changed emb)')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{args.method}_latent_{args.vis_method}_changed.png', dpi=200)
            plt.close(fig_png)

            # 2) 交互式 HTML：用 Plotly，注意不要覆盖上面的 fig 变量
            fig_html = go.Figure()
            for c in range(num_class):
                m = (y_vis == c)
                fig_html.add_trace(go.Scatter3d(
                    x=x_nd[m, 0], y=x_nd[m, 1], z=x_nd[m, 2],
                    mode='markers',
                    marker=dict(size=3),
                    name=str(c)
                ))
            fig_html.update_layout(
                scene=dict(xaxis_title='Dim-1', yaxis_title='Dim-2', zaxis_title='Dim-3'),
                title=f'{args.dataset} – {base_method.upper()}-3D (changed emb)',
                margin=dict(l=0, r=0, b=0, t=40)
            )
            fig_html.write_html(f'{save_dir}/{args.method}_{args.vis_method}_changed.html')
            print(">> 交互式 3D 可视化已保存，可用浏览器打开查看")
            print(f'>> Latent space ({args.vis_method}) with changed embedding saved to {save_dir}')


        else:

            # 2D 分支保持 matplotlib 散点图，并在分支内完成 legend/title/savefig

            plt.figure(figsize=(6, 5))

            for c in range(num_class):
                m = (y_vis == c)

                plt.scatter(x_nd[m, 0], x_nd[m, 1], s=6, alpha=0.7, label=str(c))

            plt.xlabel('Dim-1');

            plt.ylabel('Dim-2')

            plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            plt.title(f'{args.dataset} – {base_method.upper()} (changed emb)')

            plt.tight_layout()

            plt.savefig(f'{save_dir}/{args.method}_latent_{args.vis_method}_changed.png', dpi=200)

            plt.close()

            # (★) 把 print 语句移到这里

            print(f'>> Latent space ({args.vis_method}) with changed embedding saved to {save_dir}')


    is_multilabel = args.dataset in MULTI_LABEL
    importance = compute_feature_importance_fast(
        model, dataset, dataset.split_idx, is_multilabel
    )

    plot_and_save_feature_importance(
        importance,
        dataset=dataset,
        save_dir=save_dir,
        method=args.method,
        topk=30
    )
    print(f">> Feature-importance figure saved to {save_dir}")



if __name__ == "__main__":
    main()