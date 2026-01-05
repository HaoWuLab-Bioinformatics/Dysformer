import torch
import torch.nn.functional as F

from torch_geometric.utils import subgraph

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from sklearn.metrics import f1_score, precision_score, recall_score

# ----------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------
def _tensor_to_numpy(y):
    """确保转换到 CPU numpy，一维标签"""
    if y.is_cuda:
        y = y.cpu()
    return y.squeeze().numpy()


def _classification_metrics(y_true, logits, average='macro'):
    """
    计算 4 个分类指标（acc / f1 / precision / recall）
    y_true : Tensor  [n]
    logits : Tensor  [n, num_classes]
    """
    pred = logits.argmax(dim=1)
    y_true = _tensor_to_numpy(y_true)
    pred = _tensor_to_numpy(pred)

    acc = (y_true == pred).mean()
    f1 = f1_score(y_true, pred, average=average, zero_division=0)
    prec = precision_score(y_true, pred, average=average, zero_division=0)
    rec = recall_score(y_true, pred, average=average, zero_division=0)
    return acc, f1, prec, rec


# ----------------------------------------------------------------------
# evaluate  (适用于小图)
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion,
             args, degrees, threshold, device="cpu", result=None):
    """
    返回 10 指标：
        val_loss, val_acc, val_f1, val_precision, val_recall,
        test_loss, test_acc, test_f1, test_precision, test_recall
    """
    if result is None:
        model.eval()
        logits = model(
            dataset.graph['node_feat'].to(device),
            dataset.graph['edge_index'].to(device)
        )
    else:
        logits = result.to(device)

    # ----- 索引 -----
    y_val = dataset.label[split_idx['valid']].to(device).squeeze()
    y_test = dataset.label[split_idx['test']].to(device).squeeze()

    # ----- Loss -----
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e',
                        'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            y_oh = F.one_hot(dataset.label,
                             dataset.label.max() + 1).squeeze(1).to(device)
        else:
            y_oh = dataset.label.to(device)
        val_loss = criterion(logits[split_idx['valid']], y_oh[split_idx['valid']].float())
        test_loss = criterion(logits[split_idx['test']],  y_oh[split_idx['test']].float())
    else:
        logp = F.log_softmax(logits, dim=1)
        val_loss = criterion(logp[split_idx['valid']], y_val)
        test_loss = criterion(logp[split_idx['test']],  y_test)

    # ----- 分类指标 -----
    val_acc, val_f1, val_prec, val_rec = _classification_metrics(
        y_val, logits[split_idx['valid']]
    )
    test_acc, test_f1, test_prec, test_rec = _classification_metrics(
        y_test, logits[split_idx['test']]
    )

    return (
        val_loss.item(), val_acc, val_f1, val_prec, val_rec,
        test_loss.item(), test_acc, test_f1, test_prec, test_rec
    )


# ----------------------------------------------------------------------
# evaluate_large  (适用于大图，在 GPU / CPU 指定设备下推理)
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_large(model, dataset, split_idx, eval_func, criterion,
                   args, degrees, threshold, device="cpu",
                   result=None):
    """
    与 evaluate 相同，但把数据/模型都放到 device 上
    """
    model = model.to(device)
    dataset.label = dataset.label.to(device)

    if result is None:
        model.eval()
        logits = model(
            dataset.graph['node_feat'].to(device),
            dataset.graph['edge_index'].to(device)
        )
    else:
        logits = result.to(device)

    y_val = dataset.label[split_idx['valid']].squeeze()
    y_test = dataset.label[split_idx['test']].squeeze()

    # ----- Loss -----
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e',
                        'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            y_oh = F.one_hot(dataset.label,
                             dataset.label.max() + 1).squeeze(1)
        else:
            y_oh = dataset.label
        val_loss = criterion(logits[split_idx['valid']],
                             y_oh[split_idx['valid']].float())
        test_loss = criterion(logits[split_idx['test']],
                              y_oh[split_idx['test']].float())
    else:
        logp = F.log_softmax(logits, dim=1)
        val_loss = criterion(logp[split_idx['valid']], y_val)
        test_loss = criterion(logp[split_idx['test']], y_test)

    # ----- 分类指标 -----
    val_acc, val_f1, val_prec, val_rec = _classification_metrics(
        y_val, logits[split_idx['valid']]
    )
    test_acc, test_f1, test_prec, test_rec = _classification_metrics(
        y_test, logits[split_idx['test']]
    )

    return (
        val_loss.item(), val_acc, val_f1, val_prec, val_rec,
        test_loss.item(), test_acc, test_f1, test_prec, test_rec
    )


# ----------------------------------------------------------------------
# evaluate_batch  (如原逻辑，未改动指标结构)
# ----------------------------------------------------------------------
def evaluate_batch(model, dataset, split_idx, args, device,
                   n, true_label):
    """
    若仍需分批评估大图，可按需保留/改写
    （这里只维持原先返回长度/用途，以免别处引用）
    """
    num_batch = n // args.batch_size + 1
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    train_mask = torch.zeros(n, dtype=torch.bool);   train_mask[split_idx['train']] = True
    valid_mask = torch.zeros(n, dtype=torch.bool);   valid_mask[split_idx['valid']] = True
    test_mask  = torch.zeros(n, dtype=torch.bool);   test_mask[split_idx['test']]  = True

    model.to(device);  model.eval()

    idx = torch.randperm(n)
    train_total = train_correct = 0
    valid_total = valid_correct = 0
    test_total  = test_correct  = 0

    with torch.no_grad():
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size : (i+1)*args.batch_size]
            x_i = x[idx_i].to(device)
            edge_i, _ = subgraph(idx_i, edge_index,
                                 num_nodes=n, relabel_nodes=True)
            edge_i = edge_i.to(device)
            y_i = true_label[idx_i].to(device)

            out_i = model(x_i, edge_i)

            cur_tt, cur_tc = eval_acc(y_i[train_mask[idx_i]],
                                       out_i[train_mask[idx_i]])
            train_total   += cur_tt;    train_correct += cur_tc
            cur_vt, cur_vc = eval_acc(y_i[valid_mask[idx_i]],
                                       out_i[valid_mask[idx_i]])
            valid_total   += cur_vt;    valid_correct += cur_vc
            cur_te, cur_tc = eval_acc(y_i[test_mask[idx_i]],
                                       out_i[test_mask[idx_i]])
            test_total    += cur_te;   test_correct  += cur_tc

    train_acc = train_correct / train_total
    valid_acc = valid_correct / valid_total
    test_acc  = test_correct  / test_total
    return train_acc, valid_acc, test_acc, 0, None


# ----------------------------------------------------------------------
# 其它工具函数
# ----------------------------------------------------------------------
def eval_acc(true, pred):
    """
    兼容您旧版 eval_acc：返回 (total_num, correct_num)
    """
    pred = pred.argmax(dim=1, keepdim=True)
    correct = (true == pred).sum()
    return true.shape[0], correct.item()


if __name__ == '__main__':
    # 简单 self-test
    x = torch.arange(4).unsqueeze(1)
    y = torch.tensor([
        [3, 0, 0, 0],
        [3, 2, 1.5, 2.8],
        [0, 0, 2, 1],
        [0, 0, 1, 3]
    ])
    tot, cor = eval_acc(x, y)
    print('total', tot, 'correct', cor)
