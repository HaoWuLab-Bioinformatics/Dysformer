# -*- coding: utf-8 -*-
"""
中型图数据训练脚本（稳定版）
------------------------
• epoch < EDGE_WARM 时关闭动态超图
• dh_weight 线性爬坡
• 动态超图参数 lr ×0.1
• 梯度裁剪 & 轻量 L2 正则
"""

import argparse, os, random, warnings, numpy as np, torch

import math
import torch.nn as nn, torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph

from dataset       import load_nc_dataset
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits, build_optimizers
from logger        import Logger
from parse         import parser_add_main_args, parse_method

# ----------- 常量 -----------
EDGE_WARM    = 100        # epoch 冷启动
DH_RAMP      = 100        # dh_weight 爬坡期
CLIP_NORM    = 5.0
WEIGHT_DECAY = 5e-4
MULTI_LABEL  = ("PPI", "deezer-europe", "node2vec_PPI","Mashup_PPI")
SINGLE_LABEL = ("cora", "citeseer", "pubmed", "airport", "disease",
                "node2vec_PPI", "Mashup_PPI", "alzheimers", "Clin_Term_COOC",
                "diabet", "diabetuci", "jiaolvyiyu")

warnings.filterwarnings("ignore")


# ======================================================================
def fix_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ======================================================================
def setup_optimizer(model, args):
    """
    为不同类型参数设置不同学习率:
    • fast          : 普通层，lr = args.lr
    • slow          : “HConstructor” 动态超图，lr = 0.1 × args.lr
    • k_params      : TrainableLorentz.log_k，可学习曲率，lr = args.lr_k (默认 5e‑2)
    """
    slow, fast, k_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "log_k" in name:              # 可学习曲率
            k_params.append(p)
        elif "HConstructor" in name:     # 动态超图
            slow.append(p)
        else:
            fast.append(p)

    lr_k = getattr(args, "lr_k", 5e-2)   # 若命令行未提供，默认 0.05
    param_groups = [
        {"params": fast, "lr": args.lr},
        {"params": slow, "lr": args.lr * 0.1},
        {"params": k_params, "lr": lr_k},
    ]

    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

# ======================================================================
def train_epoch(model, dataset, criterion,
                optimizers,      # <- (opt_euc, opt_hyp, opt_curv)
                epoch, args):

    opt_euc, opt_hyp, opt_curv = optimizers
    model.train()

    # ---------- forward ----------
    out   = model(dataset, epoch=epoch)
    mask  = dataset.split_idx["train"].to(out.device)
    label = dataset.label.float() if args.dataset in MULTI_LABEL else dataset.label
    loss  = criterion(out[mask], label[mask])

    # ---------- backward ----------
    opt_euc.zero_grad();  opt_hyp.zero_grad();  opt_curv.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    # ---------- step ----------
    opt_euc.step();  opt_hyp.step();  opt_curv.step()

    return loss.item()



# ======================================================================
def main():
    parser = argparse.ArgumentParser("Medium‑Scale Training (stable)")
    parser_add_main_args(parser)
    args = parser.parse_args()
    print("====" * 20, "\n", args, "\n", "====" * 20)

    device = torch.device("cpu" if args.cpu else f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(">> Using", device)

    # 载入数据
    dataset = load_nc_dataset(args)
    if args.dataset in MULTI_LABEL and len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    # 统计
    num_nodes, feat_dim = dataset.graph["num_nodes"], dataset.graph["node_feat"].shape[1]

    def _infer_num_class(label: torch.Tensor) -> int:
        """
        返回数据集的类别数:
        • 单标签:  max(label)+1
        • 多标签:  label.shape[1]
        """
        return int(label.shape[1]) if label.dim() > 1 else int(label.max().item() + 1)

    # -----------------------
    num_class = _infer_num_class(dataset.label)

    args.in_channels, args.out_channels = feat_dim, num_class
#    dataset.graph["edge_index"] = dataset.graph["edge_index"].to(device)
    edge_index = dataset.graph.get("edge_index", None)

    # ① 如果 loader 本身没有图结构，但你打算用 K‑NN，就暂存为 None；
    #    稍后会在 “KNN 图（可选）” 那个分支里被覆盖。
    # ② 否则（确实有边），再搬到 device。
    if edge_index is not None:
        edge_index = edge_index.to(device)
    dataset.graph["edge_index"] = edge_index  # 先放回去，保持后续接口一致
    dataset.graph["node_feat"] = dataset.graph["node_feat"].to(device)

    print(f">> num nodes {num_nodes} | num classes {num_class} | feats {feat_dim}")

    # KNN 图（可选）
    if args.dataset in ("mini", "20news"):
        adj_knn = kneighbors_graph(dataset.graph["node_feat"].cpu(), n_neighbors=args.knn_num, include_self=True)
        dataset.graph["edge_index"] = torch.tensor(adj_knn.nonzero(), dtype=torch.long).to(device)

    # 划分
    if args.rand_split:
        splits = [dataset.get_idx_split(args.train_prop, args.valid_prop) for _ in range(args.runs)]
    elif args.rand_split_class:
        splits = [class_rand_splits(dataset.label, args.label_num_per_class, args.valid_num, args.test_num) for _ in range(args.runs)]
    else:
        splits = load_fixed_splits(dataset, name=args.dataset, protocol=args.protocol)
    print(">> splits ready")

    # 损失
    criterion = nn.BCEWithLogitsLoss() if args.dataset in MULTI_LABEL else nn.CrossEntropyLoss()
    log = Logger(args.runs, args)

    # ================= run =================
    for run in range(args.runs):
        print(f"🔥Run {run + 1}/{args.runs}")

        dataset.split_idx = {
            k: v.to(device) for k, v in splits[
                run if (args.rand_split or args.rand_split_class) else 0
            ].items()
        }

        model = parse_method(args, device)  # 创建模型
        # -------- 新：三类 optimizer --------
        # ======================================
        # 创建 optimizers
        opt_euc, opt_hyp, opt_curv = build_optimizers(model, args)

        best_val, patience = -1, 0

        for epoch in range(args.epochs):
            # ---------- 训练一步 ----------
            loss = train_epoch(model, dataset, criterion,
                               (opt_euc, opt_hyp, opt_curv),  # 依次执行 step
                               epoch, args)

            # ---------- 每 display_step 评估 ----------
            if epoch % args.display_step == 0 or epoch == args.epochs - 1:
                res = evaluate(
                    model,
                    dataset,
                    dataset.split_idx,
                    criterion,  # ★ 第 4 个参数：loss
                    args  # ★ 第 5 个参数：args
                )

                log.add_result(run, res[:-1])  # res = (train, val, test, val_loss, logits)
                k_in = model.manifold_in.k.item()
                k_hidden = model.manifold_hidden.k.item()
                k_out = model.manifold_out.k.item()
                print(f"E{epoch:03d} "
                      f"| loss={loss:.4f} "
                      f"| Tr={res[0] * 100:.2f}% Va={res[1] * 100:.2f}% Te={res[2] * 100:.2f}% "
                      f"| k_in={k_in:.4f} k_hidden={k_hidden:.4f} k_out={k_out:.4f}")

                # 提前停止
                if res[1] > best_val:
                    best_val, patience = res[1], 0
                else:
                    patience += 1
                if patience >= args.patience:
                    break

        log.print_statistics(run)
        del model, opt_euc, opt_hyp, opt_curv
        # ======================================

    if args.runs > 1:
        print("========   Overall   ========")
        print(log.print_statistics())
    print("use_dhyper =", args.use_dhyper)
    print("model =", type(model))


if __name__ == "__main__":
    main()
