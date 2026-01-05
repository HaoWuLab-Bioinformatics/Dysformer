# -*- coding: utf-8 -*-
"""
Medium‑scale graph training (stable, one‑row Excel logging)
"""

import argparse, os, random, warnings, numpy as np, torch, math, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score, precision_score, recall_score

from dataset       import load_nc_dataset
from data_utils    import class_rand_splits, eval_acc, evaluate, load_fixed_splits, build_optimizers
from logger        import Logger
from parse         import parser_add_main_args, parse_method

# ---------------- 常量 ----------------
EDGE_WARM, DH_RAMP = 150, 150
CLIP_NORM, WEIGHT_DECAY = 5.0, 5e-4
MULTI_LABEL = ("PPI","deezer-europe","node2vec_PPI","Mashup_PPI")
SINGLE_LABEL = ("cora","citeseer","pubmed","airport","disease",
                "node2vec_PPI","Mashup_PPI","alzheimers","Clin_Term_COOC",
                "diabet","diabetuci","jiaolvyiyu")

warnings.filterwarnings("ignore")
def setup_optimizer(model, args):
    """
    旧版策略：
      - 普通参数（fast）：        lr = args.lr
      - 动态超图参数（HConstructor）： lr = args.lr * 0.1
      - 可学习曲率参数（log_k）：    lr = args.lr_k (默认 0.05)
    返回一个 torch.optim.Optimizer 实例。
    """
    fast_params, slow_params, k_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "log_k" in name:
            k_params.append(p)
        elif "HConstructor" in name:
            slow_params.append(p)
        else:
            fast_params.append(p)

    lr_k = getattr(args, "lr_k", 5e-2)
    param_groups = [
        {"params": fast_params, "lr": args.lr},
        {"params": slow_params, "lr": args.lr * 0.1},
        {"params": k_params,   "lr": lr_k},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

# -------- utils --------
def fix_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_metrics_to_excel(path, row_dict):
    """追加一行(best metrics)到 Excel"""
    if os.path.exists(path):
        df = pd.read_excel(path)
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([row_dict])
    df.to_excel(path, index=False)

# -------- 训练一个 epoch --------
def train_epoch(model, dataset, criterion, optimizer, epoch, args):
#    opt_euc, opt_hyp, opt_curv = optimizers
    model.train()

    # ----- 冷启动 & 线性爬坡 -----

    if epoch < EDGE_WARM:
        model.use_dhyper = False
        model.dh_weight  = 0.0
    else:
        model.use_dhyper = bool(args.use_dhyper)
        ramp = min(1.0, (epoch - EDGE_WARM) / DH_RAMP)
        model.dh_weight = args.dh_weight * ramp

    # forward
    out = model(dataset, epoch=epoch)   # 现在 forward 接收 epoch
    mask = dataset.split_idx["train"].to(out.device)
    loss = criterion(out[mask], dataset.label.squeeze()[mask])

    # backward + step
#    opt_euc.zero_grad(); opt_hyp.zero_grad(); opt_curv.zero_grad()
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
#    opt_euc.step(); opt_hyp.step(); opt_curv.step()
    optimizer.step()
    return loss.item()


# -------- 评估，返回 10 指标 --------
def full_metrics(model, dataset, split_idx, criterion, args):
    model.eval()
    with torch.no_grad():
        out = model(dataset)

    multilabel = args.dataset in MULTI_LABEL
    if multilabel:
        val_loss = criterion(out[split_idx["valid"]],
                             dataset.label.float()[split_idx["valid"]])
        test_loss = criterion(out[split_idx["test"]],
                              dataset.label.float()[split_idx["test"]])
        y_val, y_test = dataset.label.float(), dataset.label.float()
    else:
        out = F.log_softmax(out, dim=1)
        # ★★ 这里删掉 squeeze(1) ★★
        val_loss = criterion(out[split_idx["valid"]],
                             dataset.label[split_idx["valid"]])
        test_loss = criterion(out[split_idx["test"]],
                              dataset.label[split_idx["test"]])
        y_val, y_test = dataset.label, dataset.label
    logits_val, logits_test = out, out


    def _calc(y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        if multilabel:
            y_true = (y_true > -0.5).astype(int)
            y_pred = (y_pred > 0).astype(int)
            avg = 'micro'
        else:
            y_true = y_true.squeeze()
            y_pred = y_pred.argmax(axis=-1)
            avg = 'macro'
        acc = (y_true == y_pred).mean()
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
        return acc, f1, prec, rec

    val_acc, val_f1, val_prec, val_rec   = _calc(y_val[split_idx["valid"]], logits_val[split_idx["valid"]])
    test_acc, test_f1, test_prec, test_rec = _calc(y_test[split_idx["test"]], logits_test[split_idx["test"]])

    return (val_loss.item(), val_acc, val_f1, val_prec, val_rec,
            test_loss.item(), test_acc, test_f1, test_prec, test_rec)

# ----------------- main -----------------


def main():
    parser = argparse.ArgumentParser("Medium‑Scale Training (stable)")
    parser_add_main_args(parser)
    args = parser.parse_args()
    print("====" * 20, "\n", args, "\n", "====" * 20)

    device = torch.device("cpu" if args.cpu else f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(">> Using", device)
    fix_seed(args.seed)

    # ---------- load data ----------
    dataset = load_nc_dataset(args)
    if args.dataset in MULTI_LABEL and len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    num_nodes = dataset.graph["num_nodes"]
    feat_dim  = dataset.graph["node_feat"].shape[1]
    num_class = dataset.label.shape[1] if dataset.label.dim() > 1 else int(dataset.label.max() + 1)
    args.in_channels, args.out_channels = feat_dim, num_class

    # 如果数据集中没有 edge_index，就关闭图分支
    if "edge_index" not in dataset.graph or dataset.graph["edge_index"] is None:
        args.use_graph = False
        print(">> No edge_index found — disabling graph branch")

    # 只有在 edge_index 非 None 时才 .to(device)
    edge_index = dataset.graph.get("edge_index", None)
    if edge_index is not None:
        dataset.graph["edge_index"] = edge_index.to(device)
    dataset.graph["node_feat"] = dataset.graph["node_feat"].to(device)

    print(f">> num nodes {num_nodes} | num classes {num_class} | feats {feat_dim}")

    # -------- splits --------
    if args.rand_split:
        splits = [dataset.get_idx_split(args.train_prop, args.valid_prop) for _ in range(args.runs)]
    elif args.rand_split_class:
        splits = [class_rand_splits(dataset.label,
                                    args.label_num_per_class,
                                    args.valid_num,
                                    args.test_num)
                  for _ in range(args.runs)]
    else:
        splits = load_fixed_splits(dataset, name=args.dataset, protocol=args.protocol)
    print(">> splits ready")

    # -------- criterion & logger --------
    criterion = nn.BCEWithLogitsLoss() if args.dataset in MULTI_LABEL else nn.CrossEntropyLoss()
    log = Logger(args.runs, args)
    excel_path = f'results/{args.dataset}_{args.method}_metrics.xlsx'
    os.makedirs('results', exist_ok=True)

    # ------------- runs -------------
    for run in range(args.runs):
        print(f"🔥 Run {run + 1}/{args.runs}")
        # 设置 split_idx
        idx = 0 if (args.rand_split or args.rand_split_class) else run
        dataset.split_idx = {k: v.to(device) for k, v in splits[idx].items()}

        # 初始化模型和优化器
        model = parse_method(args, device)
#        opt_euc, opt_hyp, opt_curv = build_optimizers(model, args)
        optimizer = setup_optimizer(model, args)
        best_val_f1 = -1
        best_metrics = None
        patience = 0

        for epoch in range(args.epochs):
#            loss = train_epoch(model, dataset, criterion,
#                               (opt_euc, opt_hyp, opt_curv), epoch, args)
            loss = train_epoch(model, dataset, criterion, optimizer, epoch, args)
            if epoch % args.display_step == 0 or epoch == args.epochs - 1:
                metrics = full_metrics(model, dataset, dataset.split_idx, criterion, args)
                val_acc, val_f1, test_acc = metrics[1], metrics[2], metrics[6]

                # 打印当前曲率（如果有的话）
                k_in = getattr(model, "manifold_in", None)
                k_hd = getattr(model, "manifold_hidden", None)
                k_ot = getattr(model, "manifold_out", None)
                if k_in is not None:
                    k_in, k_hd, k_ot = k_in.k.item(), k_hd.k.item(), k_ot.k.item()

                print(f"E{epoch:03d} | loss={loss:.4f} | "
                      f"VaAcc={val_acc * 100:.2f}% VaF1={val_f1 * 100:.2f}% "
                      f"TeAcc={test_acc * 100:.2f}% | "
                      f"k_in={k_in:.4f} k_hidden={k_hd:.4f} k_out={k_ot:.4f}")

                # 更新早停逻辑
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_metrics = metrics
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        print(f">> Early stopping at epoch {epoch}")
                        break

        # ----- run 结束：记录结果 -----
        if best_metrics is not None:
            val_loss, val_acc, val_f1, val_prec, val_rec, \
            test_loss, test_acc, test_f1, test_prec, test_rec = best_metrics
            log.add_result(run, (val_acc, test_acc))
            row = dict(run=run + 1,
                       val_loss=val_loss, val_acc=val_acc, val_f1=val_f1,
                       val_precision=val_prec, val_recall=val_rec,
                       test_loss=test_loss, test_acc=test_acc,
                       test_f1=test_f1, test_precision=test_prec,
                       test_recall=test_rec)
            save_metrics_to_excel(excel_path, row)
            log.print_statistics(run)
        else:
            print(f">> Warning: run {run + 1} did not record any valid metrics.")

#        del model, opt_euc, opt_hyp, opt_curv
    del model, optimizer
    if args.runs > 1:
        print("========   Overall   ========")
        log.print_statistics()

if __name__ == "__main__":
    main()

