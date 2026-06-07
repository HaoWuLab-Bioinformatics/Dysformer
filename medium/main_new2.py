# -*- coding: utf-8 -*-
"""
中型图数据训练脚本（稳定版 + IG解释版）
----------------------------------
• epoch < EDGE_WARM 时关闭动态超图
• dh_weight 线性爬坡
• 动态超图参数 lr ×0.1
• 梯度裁剪 & 轻量 L2 正则
• Integrated Gradients 特征归因
• 动态超边参数根据 best checkpoint 自动对齐 shape
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import time
import json
import platform
import argparse
import os
import random
import warnings
import math
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import kneighbors_graph

from captum.attr import IntegratedGradients

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from dataset import load_nc_dataset
from data_utils import (
    class_rand_splits,
    eval_acc,
    evaluate,
    load_fixed_splits,
    build_optimizers,
)
from parse import parser_add_main_args, parse_method


# =========================
# 全局中文字体设置
# =========================
if platform.system() == "Darwin":
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
elif platform.system() == "Windows":
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]

matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# torch.load 兼容
# =========================
_orig_torch_load = torch.load

def _torch_load_unsafe(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_unsafe


EDGE_WARM = 200
CLIP_NORM = 5.0
WEIGHT_DECAY = 5e-4
MULTI_LABEL = ("PPI", "deezer-europe", "node2vec_PPI", "Mashup_PPI")


# =========================
# 基础工具函数
# =========================
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def perturb_dataset(dataset, feat_noise_std=0.0, edge_drop_prob=0.0):
    """
    返回 dataset 的拷贝，仅对 node_feat / edge_index 做随机扰动。
    不修改原对象，避免影响后续 epoch。
    """
    if feat_noise_std == 0.0 and edge_drop_prob == 0.0:
        return dataset

    ds = copy.deepcopy(dataset)
    x = ds.graph["node_feat"]

    if feat_noise_std > 0.0:
        noise = torch.randn_like(x) * feat_noise_std
        ds.graph["node_feat"] = x + noise

    if edge_drop_prob > 0.0 and ds.graph.get("edge_index") is not None:
        ei = ds.graph["edge_index"]
        m = ei.size(1)
        keep = torch.rand(m, device=ei.device) > edge_drop_prob
        ds.graph["edge_index"] = ei[:, keep]

    return ds


# =========================
# 动态 checkpoint 加载工具
# =========================
def get_module_by_name(model, module_name):
    """
    支持根据类似 'ss_blocks.0' / 'dhgnn_conv.HConstructor' 的路径获取模块。
    """
    module = model

    for name in module_name.split("."):
        if name.isdigit():
            module = module[int(name)]
        else:
            module = getattr(module, name)

    return module


def resize_dynamic_params_to_ckpt(
        model,
        ckpt,
        dynamic_param_names=("edges_mu", "edges_logsigma"),
):
    """
    根据 best checkpoint 中动态超边参数的 shape，
    自动调整当前模型对应参数的 shape。

    这样不会跳过 ss_blocks.0.edges_mu / edges_logsigma，
    而是先把当前模型参数 shape 改成 checkpoint 的 shape，再完整加载。
    """

    model_state = model.state_dict()

    for key, value in ckpt.items():
        if not any(key.endswith(pname) for pname in dynamic_param_names):
            continue

        if key not in model_state:
            print(f"⚠️ ckpt 中存在动态参数，但当前模型没有: {key}")
            continue

        current_shape = tuple(model_state[key].shape)
        ckpt_shape = tuple(value.shape)

        if current_shape == ckpt_shape:
            continue

        module_name, param_name = key.rsplit(".", 1)
        module = get_module_by_name(model, module_name)
        old_param = getattr(module, param_name)

        new_param = torch.nn.Parameter(
            torch.empty(
                ckpt_shape,
                device=old_param.device,
                dtype=old_param.dtype,
            ),
            requires_grad=old_param.requires_grad,
        )

        setattr(module, param_name, new_param)

        if hasattr(module, "num_edges"):
            module.num_edges = ckpt_shape[0]

        print(f"✅ 动态参数已对齐: {key}: {current_shape} -> {ckpt_shape}")

    return model


def load_best_checkpoint_with_dynamic_resize(model, ckpt_path, device, strict=False):
    """
    加载 best checkpoint。
    对动态超边参数自动按 checkpoint shape 对齐，避免 shape mismatch。
    """

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model = resize_dynamic_params_to_ckpt(model, ckpt)

    missing, unexpected = model.load_state_dict(ckpt, strict=strict)

    print(">> best checkpoint 加载完成")
    print(">> missing keys:", missing)
    print(">> unexpected keys:", unexpected)

    model.eval()
    return model


# =========================
# 训练一个 epoch
# =========================
def train_epoch(model, dataset, criterion, optimizers, epoch, args):
    opt_euc, opt_hyp, opt_curv = optimizers

    model.train()

    out = model(dataset, epoch=epoch)

    if not torch.isfinite(out).all():
        raise RuntimeError("[NaN/Inf] logits contains non-finite values")

    mask = dataset.split_idx["train"].to(out.device)
    labels = dataset.label

    if args.dataset in MULTI_LABEL:
        train_mask = mask.clone()
        no_label_nodes = labels.sum(dim=1) == 0
        train_mask[no_label_nodes] = False

        loss = criterion(out[train_mask], labels[train_mask].float())
    else:
        if labels.dtype != torch.long:
            labels = labels.long()

        loss = criterion(out[mask], labels[mask])

    if not torch.isfinite(loss).all():
        bad_idx = mask.nonzero(as_tuple=True)[0]
        raise RuntimeError(
            f"[NaN/Inf] loss is non-finite at epoch {epoch}, "
            f"train_batch_size={bad_idx.numel()}"
        )

    opt_euc.zero_grad()
    opt_hyp.zero_grad()
    opt_curv.zero_grad()

    with torch.autograd.set_detect_anomaly(True):
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

    opt_euc.step()
    opt_hyp.step()
    opt_curv.step()

    return float(loss.detach().cpu())


# =========================
# Integrated Gradients 特征重要性
# =========================
def compute_feature_importance_fast(
        model,
        dataset,
        split_idx,
        is_multilabel,
        n_steps=64,
        baseline_type="mean",
        target_class=None,
        node_scope="test",
        internal_batch_size=4,
):
    """
    Integrated Gradients 版本的特征重要性计算。

    返回:
        ndarray, shape = (F,)
        每个输入特征的平均绝对 IG attribution。

    参数:
    - n_steps:
        IG 积分步数，建议 64 / 128。
    - baseline_type:
        "zero" 或 "mean"。
        表格类特征一般建议用 "mean"。
    - target_class:
        None：单标签任务解释模型预测类别。
        int ：固定解释某一类别，例如焦虑阳性类别为 1。
    - node_scope:
        "test"：只统计测试节点自身 attribution。
        "all"：统计所有节点。
        "active"：统计 attribution 非零节点。
    """

    device = dataset.graph["node_feat"].device

    model.eval()
    model.zero_grad(set_to_none=True)

    original_x = dataset.graph["node_feat"].detach()
    test_idx = split_idx["test"].to(device)

    # 固定 IG 解释目标类别
    with torch.no_grad():
        dataset.graph["node_feat"] = original_x
        logits0 = model(dataset)

        if is_multilabel:
            if target_class is None:
                label_test = dataset.label[test_idx]
                rows, cols = label_test.nonzero(as_tuple=True)

                if rows.numel() == 0:
                    raise RuntimeError(
                        "[IG] 测试集中没有正标签样本，无法按真实正标签计算多标签 IG。"
                    )
            else:
                rows, cols = None, None
                target_class = int(target_class)

        else:
            test_logits0 = logits0[test_idx]

            if target_class is None:
                fixed_cls = test_logits0.argmax(dim=1)
            else:
                fixed_cls = torch.full(
                    (test_logits0.shape[0],),
                    int(target_class),
                    dtype=torch.long,
                    device=device,
                )

    def forward_func(x_batch):
        """
        Captum 会把积分路径上的多个输入拼成 batch。
        x_batch 可能是 (B, N, F)，也可能是 (N, F)。
        """

        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(0)

        outputs = []

        for b in range(x_batch.size(0)):
            dataset.graph["node_feat"] = x_batch[b]
            logits = model(dataset)

            if is_multilabel:
                if target_class is None:
                    selected_logits = logits[test_idx][rows, cols]
                else:
                    selected_logits = logits[test_idx, target_class]
            else:
                test_logits = logits[test_idx]
                selected_logits = test_logits[
                    torch.arange(fixed_cls.numel(), device=device),
                    fixed_cls,
                ]

            outputs.append(selected_logits.sum())

        return torch.stack(outputs)

    ig_input = original_x.unsqueeze(0).clone().detach().requires_grad_(True)

    if baseline_type == "zero":
        baselines = torch.zeros_like(ig_input)
    elif baseline_type == "mean":
        mean_x = original_x.mean(dim=0, keepdim=True).expand_as(original_x)
        baselines = mean_x.unsqueeze(0).clone().detach()
    else:
        raise ValueError("baseline_type 只能是 'zero' 或 'mean'")

    ig = IntegratedGradients(forward_func)

    try:
        attributions, delta = ig.attribute(
            inputs=ig_input,
            baselines=baselines,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )
    finally:
        dataset.graph["node_feat"] = original_x

    attributions = attributions.squeeze(0).detach()

    if node_scope == "test":
        node_attr = attributions[test_idx]
    elif node_scope == "all":
        node_attr = attributions
    elif node_scope == "active":
        active_nodes = attributions.abs().sum(dim=1) > 0
        node_attr = attributions[active_nodes]
    else:
        raise ValueError("node_scope 只能是 'test', 'all', 或 'active'")

    col_imp = node_attr.abs().mean(dim=0).cpu().numpy()

    model.zero_grad(set_to_none=True)

    print(
        f">> IG attribution completed | "
        f"n_steps={n_steps}, baseline={baseline_type}, "
        f"target_class={target_class}, node_scope={node_scope}, "
        f"convergence_delta_mean={delta.abs().mean().item():.6f}"
    )

    return col_imp


# =========================
# 特征重要性画图
# =========================
def plot_and_save_feature_importance(
        importance: np.ndarray,
        dataset,
        save_dir: str,
        method: str,
        topk: int = 30,
):
    feat_names = getattr(
        dataset,
        "feat_names",
        [f"feat_{i}" for i in range(len(importance))],
    )

    pct = importance / (importance.sum() + 1e-12) * 100

    idx_sorted = importance.argsort()[::-1]
    idx_top = idx_sorted[:min(topk, len(importance))]

    plt.figure(figsize=(max(10, len(idx_top) * 0.5), 4))

    bars = plt.bar(range(len(idx_top)), pct[idx_top])

    plt.xticks(
        range(len(idx_top)),
        [feat_names[i] for i in idx_top],
        rotation=60,
        ha="right",
    )

    plt.ylabel("IG归因占比 (%)")
    plt.title(f"{dataset.name} – Top-{len(idx_top)} IG Feature Attribution")

    for bar, val in zip(bars, pct[idx_top]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        f"{save_dir}/{method}_feature_importance.png",
        dpi=200,
    )
    plt.close()

    df_save = pd.DataFrame({
        "factor": feat_names,
        "ig_attribution_raw": importance,
        "ig_attribution_pct": pct,
    })

    df_save.to_excel(
        f"{save_dir}/{method}_feature_importance.xlsx",
        index=False,
    )


# =========================
# 保存单个 run 的指标
# =========================
def append_run_metrics(best_metric, excel_path, csv_path):
    df_run = pd.DataFrame([best_metric])

    if os.path.exists(csv_path):
        df_run.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_run.to_csv(csv_path, index=False)

    if os.path.exists(excel_path):
        try:
            df_old = pd.read_excel(excel_path)
            df_new = pd.concat([df_old, df_run], ignore_index=True)
        except Exception:
            df_new = df_run
    else:
        df_new = df_run

    df_new.to_excel(excel_path, index=False)
def select_best_embedding_layer_by_val(
        model,
        dataset,
        split_idx,
        metric="f1_macro",
        max_iter=2000,
):
    """
    提取 model.trans_conv.convs 每一层的输出 embedding，
    用 train split 训练线性探针，
    用 valid split 选择效果最好的一层。

    返回:
        best_layer_id: 最佳层编号
        best_emb: 最佳层 embedding, ndarray, shape=(N, D)
        layer_scores: 每层验证集得分
    """

    assert hasattr(model, "trans_conv"), "模型中未找到 trans_conv"
    assert hasattr(model.trans_conv, "convs"), "模型中未找到 trans_conv.convs"

    model.eval()

    layer_outputs = []
    handles = []

    def make_hook(layer_id):
        def hook(module, inp, out):
            # 有些层可能输出 tuple/list，这里取第一个 tensor
            if isinstance(out, (tuple, list)):
                out_tensor = out[0]
            else:
                out_tensor = out

            layer_outputs.append((layer_id, out_tensor.detach()))
        return hook

    # 给每一层都注册 hook
    for layer_id, layer in enumerate(model.trans_conv.convs):
        handles.append(layer.register_forward_hook(make_hook(layer_id)))

    with torch.no_grad():
        _ = model(dataset)

    for h in handles:
        h.remove()

    if len(layer_outputs) == 0:
        raise RuntimeError("没有捕获到任何层的 embedding，请检查 model.trans_conv.convs。")

    # 按层编号排序，避免 hook 顺序问题
    layer_outputs = sorted(layer_outputs, key=lambda x: x[0])

    y = dataset.label.detach().cpu()

    # 多分类标签应为 1D
    if y.dim() > 1:
        y = y.argmax(dim=1)

    y_np = y.numpy()

    train_idx = split_idx["train"].detach().cpu().numpy()
    valid_idx = split_idx["valid"].detach().cpu().numpy()

    best_layer_id = None
    best_score = -1.0
    best_emb = None
    layer_scores = {}

    for layer_id, emb_tensor in layer_outputs:
        emb = emb_tensor.detach().cpu().numpy()

        x_train = emb[train_idx]
        y_train = y_np[train_idx]

        x_valid = emb[valid_idx]
        y_valid = y_np[valid_idx]

        clf = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            n_jobs=-1,
        )

        clf.fit(x_train, y_train)
        pred_valid = clf.predict(x_valid)

        if metric == "acc":
            score = accuracy_score(y_valid, pred_valid)
        elif metric == "f1_macro":
            score = f1_score(y_valid, pred_valid, average="macro", zero_division=0)
        elif metric == "f1_micro":
            score = f1_score(y_valid, pred_valid, average="micro", zero_division=0)
        else:
            raise ValueError("metric 只能是 'acc', 'f1_macro', 或 'f1_micro'")

        layer_scores[layer_id] = score

        print(f">> Layer {layer_id} linear-probe val {metric}: {score:.4%}")

        if score > best_score:
            best_score = score
            best_layer_id = layer_id
            best_emb = emb

    print(
        f">> Best embedding layer = {best_layer_id}, "
        f"val {metric} = {best_score:.4%}"
    )

    return best_layer_id, best_emb, layer_scores


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser("Medium-Scale Training (stable + IG)")
    parser_add_main_args(parser)
    args = parser.parse_args()

    if args.device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    device = torch.device(
        "cpu"
        if args.cpu
        else f"cuda:{args.device}"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = load_nc_dataset(args)

    if args.dataset in MULTI_LABEL and len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.label = dataset.label.to(device)

    num_nodes = dataset.graph["num_nodes"]
    feat_dim = dataset.graph["node_feat"].shape[1]

    num_class = (
        dataset.label.shape[1]
        if dataset.label.dim() > 1
        else int(dataset.label.max().item() + 1)
    )

    args.in_channels = feat_dim
    args.out_channels = num_class

    edge_index = dataset.graph.get("edge_index", None)
    if edge_index is not None:
        edge_index = edge_index.to(device)

    dataset.graph["edge_index"] = edge_index
    dataset.graph["node_feat"] = dataset.graph["node_feat"].to(device)

    if args.dataset in ("mini", "20news"):
        adj_knn = kneighbors_graph(
            dataset.graph["node_feat"].cpu(),
            n_neighbors=args.knn_num,
            include_self=True,
        )
        dataset.graph["edge_index"] = torch.tensor(
            adj_knn.nonzero(),
            dtype=torch.long,
        ).to(device)

    if args.rand_split:
        splits = [
            dataset.get_idx_split(args.train_prop, args.valid_prop)
            for _ in range(args.runs)
        ]
    elif args.rand_split_class:
        splits = [
            class_rand_splits(
                dataset.label,
                args.label_num_per_class,
                args.valid_num,
                args.test_num,
            )
            for _ in range(args.runs)
        ]
    else:
        splits = load_fixed_splits(
            dataset,
            name=args.dataset,
            protocol=args.protocol,
        )

    # =========================
    # loss
    # =========================
    if args.dataset in MULTI_LABEL:
        label = dataset.label

        pos_counts = (label == 1).sum(dim=0).float()
        neg_counts = (label == 0).sum(dim=0).float()

        pos_weight = neg_counts / (pos_counts + 1e-8)

        pos_weight[torch.isinf(pos_weight)] = 1.0
        pos_weight[torch.isnan(pos_weight)] = 1.0
        pos_weight = torch.clamp(pos_weight, max=50.0)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(label.device)
        )

    else:
        label = dataset.label

        num_classes = int(label.max().item() + 1)
        class_counts = torch.bincount(label, minlength=num_classes).float()

        class_weights = (1.0 / (class_counts + 1e-6)) * (
            len(label) / num_classes
        )
        class_weights = class_weights.to(label.device)

        print("类别样本数:", class_counts.tolist())
        print("类别权重:", class_weights.tolist())

        criterion = nn.CrossEntropyLoss(weight=class_weights)

    all_metrics = []

    save_dir = f"results/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    excel_path = f"{save_dir}/{args.method}_metrics.xlsx"
    csv_path = f"{save_dir}/{args.method}_metrics.csv"
    best_ckpt_path = f"{save_dir}/{args.method}_best.pt"

    model = None

    # =========================
    # runs
    # =========================
    for run in range(args.runs):
        split_id = run if (args.rand_split or args.rand_split_class) else 0

        dataset.split_idx = {
            k: v.to(device)
            for k, v in splits[split_id].items()
        }

        model = parse_method(args, device)

        print(
            ">>> use_dhyper =",
            getattr(args, "use_dhyper", None),
            "| model =",
            type(model).__name__,
            flush=True,
        )

        optimizers = build_optimizers(model, args)

        patience = 0
        best_metric = {}

        for epoch in range(args.epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            epoch_start = time.time()

            loss = train_epoch(
                model,
                dataset,
                criterion,
                optimizers,
                epoch,
                args,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            epoch_time = time.time() - epoch_start

            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(device) / 1024 ** 2
                mem_reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
                max_mem_alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                max_mem_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2
            else:
                mem_alloc = 0.0
                mem_reserved = 0.0
                max_mem_alloc = 0.0
                max_mem_reserved = 0.0

            if epoch % args.display_step == 0 or epoch == args.epochs - 1:
                metrics = evaluate(
                    model,
                    dataset,
                    dataset.split_idx,
                    criterion,
                    args,
                )

                def tune_threshold(
                        val_logits,
                        val_labels,
                        thresholds=np.linspace(0.1, 0.9, 17),
                ):
                    best_thr, best_f1 = 0.5, 0.0
                    y_true = val_labels.cpu().numpy()

                    for thr in thresholds:
                        y_pred = (
                            torch.sigmoid(val_logits) >= thr
                        ).float().cpu().numpy()

                        f1 = f1_score(
                            y_true,
                            y_pred,
                            average="micro",
                            zero_division=0,
                        )

                        if f1 > best_f1:
                            best_f1 = f1
                            best_thr = thr

                    return best_thr, best_f1

                if args.dataset in MULTI_LABEL:
                    with torch.no_grad():
                        val_logits = model(dataset)[dataset.split_idx["valid"]]
                        val_labels = dataset.label[dataset.split_idx["valid"]]

                    valid_mask = val_labels.sum(dim=1) > 0
                    val_logits = val_logits[valid_mask]
                    val_labels = val_labels[valid_mask]

                    best_thr, best_f1 = tune_threshold(
                        val_logits,
                        val_labels,
                    )

                    with torch.no_grad():
                        test_logits = model(dataset)[dataset.split_idx["test"]]

                    y_val = val_labels.cpu().numpy()
                    y_test = dataset.label[
                        dataset.split_idx["test"]
                    ].cpu().numpy()

                    pred_val = (
                        torch.sigmoid(val_logits) >= best_thr
                    ).float().cpu().numpy()

                    pred_test = (
                        torch.sigmoid(test_logits) >= best_thr
                    ).float().cpu().numpy()

                    val_f1 = f1_score(
                        y_val,
                        pred_val,
                        average="micro",
                        zero_division=0,
                    )
                    test_f1 = f1_score(
                        y_test,
                        pred_test,
                        average="micro",
                        zero_division=0,
                    )

                    val_precision = precision_score(
                        y_val,
                        pred_val,
                        average="micro",
                        zero_division=0,
                    )
                    val_recall = recall_score(
                        y_val,
                        pred_val,
                        average="micro",
                        zero_division=0,
                    )
                    test_precision = precision_score(
                        y_test,
                        pred_test,
                        average="micro",
                        zero_division=0,
                    )
                    test_recall = recall_score(
                        y_test,
                        pred_test,
                        average="micro",
                        zero_division=0,
                    )

                else:
                    best_thr = 0.5

                    val_f1 = metrics["val"]["f1"]
                    test_f1 = metrics["test"]["f1"]

                    val_precision = metrics["val"]["precision"]
                    val_recall = metrics["val"]["recall"]

                    test_precision = metrics["test"]["precision"]
                    test_recall = metrics["test"]["recall"]

                val_acc = metrics["val"]["acc"]
                test_acc = metrics["test"]["acc"]

                print(
                    f"Epoch {epoch:03d} | "
                    f"loss={loss:.4f} | "
                    f"Train={metrics['train']['acc']:.4%} "
                    f"Valacc={val_acc:.4%} Testacc={test_acc:.4%} "
                    f"Valf1={val_f1:.4%} Testf1={test_f1:.4%} "
                    f"(thr={best_thr:.2f}) | "
                    f"time={epoch_time:.2f}s | "
                    f"GPU alloc={mem_alloc:.1f}MB | "
                    f"reserved={mem_reserved:.1f}MB | "
                    f"peak_alloc={max_mem_alloc:.1f}MB | "
                    f"peak_reserved={max_mem_reserved:.1f}MB"
                )

                # ======================================================
                # 注意：
                # 这里保留你原来的逻辑：用 test_acc 选择 best。
                # 论文更规范的做法是改成 val_acc 或 val_f1。
                # 如果要用 val_f1，改成：
                # if val_f1 > best_metric.get("val/f1", -1):
                # ======================================================
                if test_acc > best_metric.get("test/acc", -1):
                    best_metric = {
                        "val/loss": metrics["val"]["loss"],
                        "val/acc": val_acc,
                        "val/f1": val_f1,
                        "val/precision": val_precision,
                        "val/recall": val_recall,
                        "test/loss": metrics["test"]["loss"],
                        "test/acc": test_acc,
                        "test/f1": test_f1,
                        "test/precision": test_precision,
                        "test/recall": test_recall,
                        "best_epoch": epoch,
                    }

                    torch.save(model.state_dict(), best_ckpt_path)

                    print(
                        f">> New best acc {test_acc:.4%} "
                        f"at epoch {epoch}, checkpoint saved."
                    )

                    patience = 0

                else:
                    patience += 1

                    if patience >= args.patience:
                        print(
                            f">> Early stopping at epoch {epoch} "
                            f"(patience={args.patience})"
                        )
                        break

        best_metric["run_id"] = run

        print("\n===== 本次 run 的最终 best_metric =====")
        print(json.dumps(best_metric, indent=2, ensure_ascii=False))
        print("=====================================\n")

        append_run_metrics(best_metric, excel_path, csv_path)

        print(
            f">> 本次 run({run}) 指标已写入：\n"
            f"   {excel_path}\n"
            f"   {csv_path}"
        )

        all_metrics.append(best_metric)

    # =========================
    # 汇总指标
    # =========================
    df = pd.DataFrame(all_metrics)

    print("use_dhyper =", args.use_dhyper)
    print("model =", type(model))

    if df.empty:
        print("⚠️ 未保存任何 run 的指标，请检查训练是否成功完成。")
    else:
        df.to_excel(excel_path, index=False)
        df.to_csv(csv_path, index=False)

        print(
            f">> 已保存 {len(df)} 条指标到\n"
            f"   {excel_path}\n"
            f"   {csv_path}"
        )

        if args.runs > 1:
            stats = df.describe().loc[["mean", "std"]]

            print("\n===== 跨 run 统计 (均值 ± 标准差) =====")

            for col in ["val/acc", "val/f1", "test/acc", "test/f1"]:
                mu = stats.at["mean", col]
                sd = stats.at["std", col]
                print(f"{col:12s}: {mu:7.4%} ± {sd:7.4%}")

    # =========================================================
    # 训练结束后：重新加载 best checkpoint
    # 这一点很关键：
    # 不用最后 epoch 的 model，而是用 best epoch 的 checkpoint。
    # 同时自动对齐动态超边参数 shape。
    # =========================================================
    model = load_best_checkpoint_with_dynamic_resize(
        model=model,
        ckpt_path=best_ckpt_path,
        device=device,
        strict=False,
    )

    print("✅ 已加载 best checkpoint，用 best epoch 模型进行可视化和 IG 归因")

    # =========================
    # embedding 可视化
    # =========================
    if args.vis_emb:
        save_dir = f"results/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)

        try:
            import umap
        except ImportError:
            umap = None

        # 自动选择验证集效果最好的 embedding 层
        best_layer_id, x_emb, layer_scores = select_best_embedding_layer_by_val(
            model=model,
            dataset=dataset,
            split_idx=dataset.split_idx,
            metric="f1_macro",  # 多分类推荐 f1_macro；如果类别均衡，也可以用 "acc"
            max_iter=2000,
        )

        print(f">> 使用验证集效果最好的第 {best_layer_id} 层 embedding 进行可视化")

        if 0 < args.vis_sample < 1:
            n = x_emb.shape[0]
            idx = np.random.choice(n, int(n * args.vis_sample), replace=False)

            x_emb = x_emb[idx]
            y_vis = dataset.label.cpu().numpy()[idx]
        else:
            y_vis = dataset.label.cpu().numpy()

        if y_vis.ndim > 1:
            y_vis = y_vis.argmax(axis=1)

        is_3d = args.vis_method.endswith("3")
        base_method = args.vis_method.rstrip("3")
        n_comp = 3 if is_3d else 2

        if base_method == "pca":
            x_nd = PCA(n_components=n_comp).fit_transform(x_emb)

        elif base_method == "tsne":
            x_nd = TSNE(
                n_components=n_comp,
                perplexity=30,
                learning_rate=200.0,
                init="random",
                random_state=42,
            ).fit_transform(x_emb)

        elif base_method == "umap":
            assert umap is not None, "请先 pip install umap-learn"

            x_nd = umap.UMAP(
                n_components=n_comp,
                random_state=42,
            ).fit_transform(x_emb)

        elif base_method == "pacmap":
            import pacmap

            x_nd = pacmap.PaCMAP(
                n_components=n_comp,
                n_neighbors=None,
            ).fit_transform(x_emb)

        elif base_method == "trimap":
            import trimap

            x_nd = trimap.TRIMAP(
                n_dims=n_comp,
            ).fit_transform(x_emb)

        else:
            raise ValueError(f"未知 vis_method: {args.vis_method}")

        num_class_vis = int(y_vis.max() + 1)

        if is_3d:
            fig_png = plt.figure(figsize=(7, 6))
            ax = fig_png.add_subplot(111, projection="3d")

            for c in range(num_class_vis):
                m = y_vis == c
                ax.scatter(
                    x_nd[m, 0],
                    x_nd[m, 1],
                    x_nd[m, 2],
                    s=6,
                    alpha=0.7,
                    label=str(c),
                )

            ax.set_xlabel("Dim-1")
            ax.set_ylabel("Dim-2")
            ax.set_zlabel("Dim-3")

            ax.legend(
                title="Label",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )

            plt.title(
                f"{args.dataset} – {base_method.upper()}-3D "
                f"(best checkpoint emb)"
            )
            plt.tight_layout()

            plt.savefig(
                f"{save_dir}/{args.method}_latent_{args.vis_method}_changed.png",
                dpi=200,
            )
            plt.close(fig_png)

            fig_html = go.Figure()

            for c in range(num_class_vis):
                m = y_vis == c

                fig_html.add_trace(
                    go.Scatter3d(
                        x=x_nd[m, 0],
                        y=x_nd[m, 1],
                        z=x_nd[m, 2],
                        mode="markers",
                        marker=dict(size=3),
                        name=str(c),
                    )
                )

            fig_html.update_layout(
                scene=dict(
                    xaxis_title="Dim-1",
                    yaxis_title="Dim-2",
                    zaxis_title="Dim-3",
                ),
                title=(
                    f"{args.dataset} – {base_method.upper()}-3D "
                    f"(best checkpoint emb)"
                ),
                margin=dict(l=0, r=0, b=0, t=40),
            )

            fig_html.write_html(
                f"{save_dir}/{args.method}_{args.vis_method}_changed.html"
            )

            print(">> 交互式 3D 可视化已保存，可用浏览器打开查看")
            print(
                f">> Latent space ({args.vis_method}) with best checkpoint "
                f"embedding saved to {save_dir}"
            )

        else:
            plt.figure(figsize=(6, 5))

            for c in range(num_class_vis):
                m = y_vis == c

                plt.scatter(
                    x_nd[m, 0],
                    x_nd[m, 1],
                    s=6,
                    alpha=0.7,
                    label=str(c),
                )

            plt.xlabel("Dim-1")
            plt.ylabel("Dim-2")

            plt.legend(
                title="Label",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )

            plt.title(
                f"{args.dataset} – {base_method.upper()} "
                f"(best checkpoint emb)"
            )
            plt.tight_layout()

            plt.savefig(
                f"{save_dir}/{args.method}_latent_{args.vis_method}_changed.png",
                dpi=200,
            )
            plt.close()

            print(
                f">> Latent space ({args.vis_method}) with best checkpoint "
                f"embedding saved to {save_dir}"
            )

    # =========================
    # Integrated Gradients 特征归因
    # =========================
    is_multilabel = args.dataset in MULTI_LABEL

    # 如果是二分类，默认解释类别 1。
    # 如果你的焦虑标签不是 1，请手动改这里。
    if (not is_multilabel) and num_class == 2:
        ig_target_class = 1
    else:
        ig_target_class = None

    importance = compute_feature_importance_fast(
        model=model,
        dataset=dataset,
        split_idx=dataset.split_idx,
        is_multilabel=is_multilabel,
        n_steps=64,
        baseline_type="mean",
        target_class=ig_target_class,
        node_scope="test",
        internal_batch_size=4,
    )

    plot_and_save_feature_importance(
        importance=importance,
        dataset=dataset,
        save_dir=save_dir,
        method=args.method,
        topk=30,
    )

    print(f">> IG feature-importance figure saved to {save_dir}")


if __name__ == "__main__":
    main()
