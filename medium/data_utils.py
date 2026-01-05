import os
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.metrics import f1_score, roc_auc_score
from torch_sparse import SparseTensor
# 你可以在前面增加这个，全局设置多标签的数据集有哪些
MULTI_LABEL_DATASETS = ('PPI', 'deezer-europe', 'node2vec_PPI','Mashup_PPI')

def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """
    随机划分 train/valid/test。
    • 单标签：label 为 (N,)，取值 -1 或类别 id
    • 多标签：label 为 (N,C)，取值 {-1,0,1}
    """
    if ignore_negative:
        if label.dim() == 1:
            labeled_nodes = (label != -1).nonzero(as_tuple=True)[0]
        else:                       # 多标签：只要该节点存在一个非 -1 标签即视为已标注
            labeled_nodes = ((label != -1).any(dim=1)).nonzero(as_tuple=True)[0]
    else:
        labeled_nodes = torch.arange(label.shape[0], device=label.device)

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.randperm(n, device=label.device)

    train_idx = labeled_nodes[perm[:train_num]]
    valid_idx = labeled_nodes[perm[train_num: train_num + valid_num]]
    test_idx  = labeled_nodes[perm[train_num + valid_num:]]

    return train_idx, valid_idx, test_idx



def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    """use all remaining data points as test data, so test_num will not be used"""
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0]).to(label.device)
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = (
        non_train_idx[:valid_num],
        non_train_idx[valid_num: valid_num + test_num],
    )
    print(f"train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")
    split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    return split_idx


def normalize_feat(mx):
    """Row-normalize np or sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

'''
def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)
'''

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score
import torch

def multilabel_micro_f1(y_true: torch.Tensor,
                        logits: torch.Tensor,
                        thr: float = 0.5) -> float:
    """
    y_true : (N,C) ∈ {-1,0,1}
    logits : (N,C) 未过 sigmoid
    """
    mask = y_true != -1
    y_true = y_true.clone().float()
    y_true[~mask] = 0
    y_pred = (torch.sigmoid(logits) >= thr).float()

    y_true = y_true[mask].cpu().numpy().astype(int)
    y_pred = y_pred[mask].cpu().numpy().astype(int)
    return f1_score(y_true, y_pred, average="micro")




def eval_acc(y_true, y_pred):
    """
    • 单标签：整体 accuracy
    • 多标签：逐列 accuracy 取平均（忽略 -1）
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # --- 单标签 ---
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        if y_pred.ndim == 2:
            y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)

    # --- 多标签 ---
    acc_list = []
    y_pred = (y_pred > 0.5).astype(np.float32)

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] != -1
        if not is_labeled.any():
            continue
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(correct.mean())

    return float(np.mean(acc_list)) if acc_list else 0.0


'''
@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        if args.method == "fast_transgnn" or args.method == "glcn":
            out, _ = model(dataset)
        else:
            out = model(dataset)

    train_acc = eval_func(dataset.label[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(dataset.label[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(dataset.label[split_idx["test"]], out[split_idx["test"]])
    if args.dataset in (
            "yelp-chi",
            "deezer-europe",
            "twitch-e",
            "fb100",
            "ogbn-proteins",
    ):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(
            out[split_idx["valid"]],
            true_label.squeeze(1)[split_idx["valid"]].to(torch.float),
        )
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx["valid"]], dataset.label.squeeze(1)[split_idx["valid"]]
        )

    return train_acc, valid_acc, test_acc, valid_loss, out
'''



'''
@torch.no_grad()
def evaluate(model, dataset, split_idx, criterion, args, result=None):
    """
    统一评估函数：
    • 多标签 → 返回 micro‑F1 + precision/recall
    • 单标签 → 返回 acc + f1 + precision + recall
    """
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset)

    y_true = dataset.label
    y_pred = out

    def get_metrics(y_true, y_pred, mask, multilabel=False):
        y_true = y_true[mask].cpu()
        y_pred = y_pred[mask].cpu()

        if multilabel:
            y_pred_bin = (torch.sigmoid(y_pred) >= 0.5).float()
            f1 = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
            precision = precision_score(y_true, y_pred_bin, average='micro', zero_division=0)
            recall = recall_score(y_true, y_pred_bin, average='micro', zero_division=0)
            acc = (y_true == y_pred_bin).float().mean().item()
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float()).item()

        else:
            y_true = y_true.long()
            y_pred_class = y_pred.argmax(dim=-1)
            f1 = f1_score(y_true, y_pred_class, average='macro', zero_division=0)
            precision = precision_score(y_true, y_pred_class, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred_class, average='macro', zero_division=0)
            acc = accuracy_score(y_true, y_pred_class)
            loss = criterion(y_pred, y_true).item()

        return {
            "loss": loss,
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    multilabel = args.dataset in ("PPI", "deezer-europe", "node2vec_PPI", "Mashup_PPI")

    metrics = {
        "train": get_metrics(y_true, y_pred, split_idx["train"], multilabel),
        "val":   get_metrics(y_true, y_pred, split_idx["valid"], multilabel),
        "test":  get_metrics(y_true, y_pred, split_idx["test"],  multilabel)
    }

    return metrics  # 一个包含所有指标的字典
'''
#会导致准确率下降
@torch.no_grad()
def evaluate(model, dataset, split_idx, criterion, args, result=None):
    """
    统一评估函数：
    • 多标签 → 返回 micro‑F1 + precision/recall
    • 单标签 → 返回 acc + f1 + precision + recall
    """
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset)

    y_true = dataset.label
    y_pred = out

    def get_metrics(y_true, y_pred, mask_idx, multilabel=False):
        # ----------------------
        # 只保留有真实标签的节点
        # ----------------------
        y_true_masked = y_true[mask_idx]
        y_pred_masked = y_pred[mask_idx]

        # 多标签：过滤掉标签全为 0 的节点（视为无标签）
        if multilabel:
            valid_mask = (y_true_masked.sum(dim=1) > 0)
            y_true_masked = y_true_masked[valid_mask]
            y_pred_masked = y_pred_masked[valid_mask]

            if y_true_masked.size(0) == 0:
                return {"loss": 0, "acc": 0, "f1": 0, "precision": 0, "recall": 0}

            y_pred_bin = (torch.sigmoid(y_pred_masked) >= 0.5).float()
            f1 = f1_score(y_true_masked.cpu(), y_pred_bin.cpu(), average='micro', zero_division=0)
            precision = precision_score(y_true_masked.cpu(), y_pred_bin.cpu(), average='micro', zero_division=0)
            recall = recall_score(y_true_masked.cpu(), y_pred_bin.cpu(), average='micro', zero_division=0)
            acc = (y_true_masked == y_pred_bin).float().mean().item()
            loss = F.binary_cross_entropy_with_logits(y_pred_masked, y_true_masked.float()).item()
        else:
            y_true_masked = y_true_masked.long()
            y_pred_class = y_pred_masked.argmax(dim=-1)
            f1 = f1_score(y_true_masked.cpu(), y_pred_class.cpu(), average='macro', zero_division=0)
            precision = precision_score(y_true_masked.cpu(), y_pred_class.cpu(), average='macro', zero_division=0)
            recall = recall_score(y_true_masked.cpu(), y_pred_class.cpu(), average='macro', zero_division=0)
            acc = accuracy_score(y_true_masked.cpu(), y_pred_class.cpu())
            loss = criterion(y_pred_masked, y_true_masked).item()

        return {
            "loss": loss,
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    multilabel = args.dataset in ("PPI", "deezer-europe", "node2vec_PPI", "Mashup_PPI")

    metrics = {
        "train": get_metrics(y_true, y_pred, split_idx["train"], multilabel),
        "val":   get_metrics(y_true, y_pred, split_idx["valid"], multilabel),
        "test":  get_metrics(y_true, y_pred, split_idx["test"],  multilabel)
    }

    return metrics

def load_fixed_splits(dataset, name, protocol):
    splits_lst = []
    if name in ["cora", "citeseer", "pubmed", "airport", "disease","PPI"] and protocol == "semi":
        splits = {}
        splits["train"] = torch.as_tensor(dataset.train_idx)
        splits["valid"] = torch.as_tensor(dataset.valid_idx)
        splits["test"] = torch.as_tensor(dataset.test_idx)
        splits_lst.append(splits)
    elif name in ["chameleon", "squirrel"]:
        file_path = f"../../data/wiki_new/{name}/{name}_filtered.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # (10, N), 10 splits
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        for i in range(10):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"] = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)

    elif name in ["film"]:
        for i in range(10):
            splits_file_path = (
                    "../../data/geom-gcn/{}/{}".format(name, name)
                    + "_split_0.6_0.2_"
                    + str(i)
                    + ".npz"
            )
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits["train"] = torch.BoolTensor(splits_file["train_mask"])
                splits["valid"] = torch.BoolTensor(splits_file["val_mask"])
                splits["test"] = torch.BoolTensor(splits_file["test_mask"])
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst

'''
def split_data(labels, val_prop, test_prop, seed=1234):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg
'''


def split_data(labels, val_prop, test_prop):
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)

    if len(labels.shape) == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
        # 单标签多分类任务
        np.random.shuffle(all_idx)
        num_val = int(val_prop * nb_nodes)
        num_test = int(test_prop * nb_nodes)
        idx_val = all_idx[:num_val]
        idx_test = all_idx[num_val:num_val + num_test]
        idx_train = all_idx[num_val + num_test:]
        return idx_val.tolist(), idx_test.tolist(), idx_train.tolist()

    else:
        # 多标签多分类任务
        pos_idx = labels.nonzero()[0]
        neg_idx = (1. - labels).nonzero()[0]
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        nb_pos_neg = min(len(pos_idx), len(neg_idx))
        nb_val = round(val_prop * nb_pos_neg)
        nb_test = round(test_prop * nb_pos_neg)

        idx_val_pos = pos_idx[:nb_val]
        idx_test_pos = pos_idx[nb_val:nb_val + nb_test]
        idx_train_pos = pos_idx[nb_val + nb_test:]

        idx_val_neg = neg_idx[:nb_val]
        idx_test_neg = neg_idx[nb_val:nb_val + nb_test]
        idx_train_neg = neg_idx[nb_val + nb_test:]

        idx_val = np.concatenate([idx_val_pos, idx_val_neg])
        idx_test = np.concatenate([idx_test_pos, idx_test_neg])
        idx_train = np.concatenate([idx_train_pos, idx_train_neg])

        return idx_val.tolist(), idx_test.tolist(), idx_train.tolist()


# medium/optimizer.py  （或放在 utils/optim.py，随你）
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam
import torch


# ===========================================
#  替换原来的 build_optimizers
# ===========================================
'''
def build_optimizers(model, args):
    """
    返回:
        opt_euc  ——  普通 Euclidean 参数的 Adam
        opt_hyp  ——  Hyperbolic ManifoldParameter 的 RiemannianAdam
        opt_curv ——  可学习曲率(log_k / _logc) 的 Adam
    """
    # --------------------------------------------------
    # 0) geoopt 可能在某些环境里没安装；做一个兜底
    try:
        import geoopt
        from geoopt import ManifoldParameter
        _has_geoopt = True
    except ImportError:          # 允许只跑欧氏/曲率分支
        _has_geoopt = False
        class ManifoldParameter(nn.Parameter):   # 占位，保证 isinstance 不报错
            pass

    # --------------------------------------------------
    # 1) Euclidean 参数（既不是 ManifoldParameter，也不是曲率参数）
    euc_params = [p for n, p in model.named_parameters()
                  if ('_logc' not in n) and ('log_k' not in n)
                  and (not isinstance(p, ManifoldParameter))
                  and p.requires_grad]

    opt_euc = torch.optim.Adam(
        euc_params,
        lr          = args.lr,
        weight_decay= args.weight_decay
    )

    # --------------------------------------------------
    # 2) Hyperbolic 参数（geoopt.ManifoldParameter）
    if _has_geoopt:
        hyp_params = [p for p in model.parameters()
                      if isinstance(p, ManifoldParameter) and p.requires_grad]
        opt_hyp = geoopt.optim.RiemannianAdam(
            hyp_params,
            lr          = args.hyp_lr,
            weight_decay= args.hyp_weight_decay,
            stabilize   = 50
        ) if hyp_params else None
    else:
        opt_hyp = None   # 没装 geoopt 就跳过

    # --------------------------------------------------
    # 3) 曲率参数（log_k 兼容旧版，_logc 兼容新版）
    curv_params = [p for n, p in model.named_parameters()
                   if (('log_k' in n) or ('_logc' in n)) and p.requires_grad]

    opt_curv = torch.optim.Adam(
        curv_params,
        lr = getattr(args, "c_lr", 0.1)
    ) if curv_params else None

    return opt_euc, opt_hyp, opt_curv

'''

import torch

def build_optimizers(model, args):
    """
    返回:
        opt_euc  —— Euclidean 参数的 Adam/AdamW
        opt_hyp  —— Hyperbolic ManifoldParameter 的 RiemannianAdam（若 geoopt 可用）
        opt_curv —— 曲率参数(log_k / _logc 等) 的 Adam

    关键保证：即使没有 hyp_params / curv_params，也不会返回 None，
    而是返回 NoOpOptimizer，从而不会在 train_epoch 的
        opt_euc.zero_grad(); opt_hyp.zero_grad(); opt_curv.zero_grad()
    这一行崩溃。
    """

    # ---------------------------
    # 0) No-op Optimizer：顶替空优化器
    # ---------------------------
    class NoOpOptimizer:
        def zero_grad(self, *args, **kwargs):
            return None
        def step(self, *args, **kwargs):
            return None
        def state_dict(self):
            return {}
        def load_state_dict(self, state_dict):
            return None

    # ---------------------------
    # 1) geoopt 兼容：没装也能跑
    # ---------------------------
    try:
        import geoopt
        from geoopt import ManifoldParameter
        from geoopt.optim import RiemannianAdam
        _has_geoopt = True
    except Exception:
        _has_geoopt = False
        ManifoldParameter = None
        RiemannianAdam = None

    # ---------------------------
    # 2) 工具：参数去重（防止重复被多个 optimizer 管）
    # ---------------------------
    def _dedup(params):
        seen = set()
        out = []
        for p in params:
            if p is None:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
        return out

    # ---------------------------
    # 3) 分组：euc / hyp / curv
    # ---------------------------
    euc_params = []
    hyp_params = []
    curv_params = []

    for name, p in model.named_parameters():
        if p is None or (not p.requires_grad):
            continue

        # ---- 曲率参数识别（你原逻辑：log_k 或 _logc）----
        # 注意：这里不会把 trans_conv.W_k 这种权重误判为曲率
        # 因为我们只匹配明确的字段名片段
        if ("log_k" in name) or ("_logc" in name):
            curv_params.append(p)
            continue

        # ---- 双曲 manifold 参数（geoopt.ManifoldParameter）----
        if _has_geoopt and isinstance(p, ManifoldParameter):
            hyp_params.append(p)
            continue

        # ---- 其余归为欧氏参数 ----
        euc_params.append(p)

    euc_params = _dedup(euc_params)
    hyp_params = _dedup(hyp_params)
    curv_params = _dedup(curv_params)

    # 防止同一个参数被多个 optimizer 同时管理
    curv_ids = {id(p) for p in curv_params}
    hyp_ids = {id(p) for p in hyp_params}
    euc_params = [p for p in euc_params if (id(p) not in curv_ids) and (id(p) not in hyp_ids)]
    hyp_params = [p for p in hyp_params if id(p) not in curv_ids]

    # ---------------------------
    # 4) 创建三个优化器（永不返回 None）
    # ---------------------------
    lr = float(getattr(args, "lr", 1e-3))
    weight_decay = float(getattr(args, "weight_decay", 0.0))

    # 你原代码里用 args.hyp_lr / args.hyp_weight_decay / args.c_lr
    hyp_lr = float(getattr(args, "hyp_lr", lr))
    hyp_weight_decay = float(getattr(args, "hyp_weight_decay", 0.0))
    c_lr = float(getattr(args, "c_lr", 0.1))

    # opt_euc：有参数就正常 Adam；否则 NoOp
    if len(euc_params) > 0:
        # 你也可以改成 AdamW，但保持与原项目一致即可
        opt_euc = torch.optim.Adam(euc_params, lr=lr, weight_decay=weight_decay)
    else:
        opt_euc = NoOpOptimizer()

    # opt_hyp：geoopt 可用且有 hyp params 才用 RiemannianAdam，否则 NoOp
    if _has_geoopt and (len(hyp_params) > 0):
        opt_hyp = RiemannianAdam(
            hyp_params,
            lr=hyp_lr,
            weight_decay=hyp_weight_decay,
            stabilize=50
        )
    else:
        opt_hyp = NoOpOptimizer()

    # opt_curv：有曲率参数才用 Adam，否则 NoOp
    if len(curv_params) > 0:
        opt_curv = torch.optim.Adam(curv_params, lr=c_lr)
    else:
        opt_curv = NoOpOptimizer()

    # ---------------------------
    # 5) 可选：打印检查（你可以注释掉）
    # ---------------------------
    try:
        print("opt_euc:", type(opt_euc).__name__, "nparams_euc:", sum(p.numel() for p in euc_params))
        print("opt_hyp:", type(opt_hyp).__name__, "nparams_hyp:", sum(p.numel() for p in hyp_params))
        print("opt_curv:", type(opt_curv).__name__, "nparams_curv:", sum(p.numel() for p in curv_params))
    except Exception:
        pass

    return opt_euc, opt_hyp, opt_curv
