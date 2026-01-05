import argparse
import os
import random
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter

import wandb

from logger import Logger
from dataset import load_dataset
from data_utils import (normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1,
                        to_sparse_tensor, load_fixed_splits, adj_mul, get_gpu_memory_map,
                        count_parameters, compute_degrees)
from eval import evaluate, evaluate_large
from parse import parse_method, parser_add_main_args
from manifolds import Optimizer

warnings.filterwarnings('ignore')



def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    return parser.parse_args()


def get_device(use_cpu, device_id):
    if use_cpu:
        print('>> Using CPU')
        return torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print('>> Using GPU')
        return device


def load_and_preprocess_data(args, device):
    print(f'>> Loading dataset {args.dataset}')
    if 'DHHT_data_dir' in os.environ:
        args.data_dir = os.environ['DHHT_data_dir']
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    return dataset


def get_data_splits(args, dataset):
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                         for _ in range(args.runs)]
        print('>> Using random split')
    elif args.rand_split_class:
        split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                         for _ in range(args.runs)]
        print('>> Using random class split')
    elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
        split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]
        print('>> Using fixed split')
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)
        print('>> Using fixed split')

    return split_idx_lst


def print_dataset_info(dataset):
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    print(f">> Dataset {dataset.name} | num nodes {n} | num edges {e} | num node feats {d} | num classes {c}")
    return n, c, d


def compute_and_print_degrees(dataset):
    degrees = compute_degrees(dataset.graph['edge_index'], dataset.graph['num_nodes'])
    print(f">> Total degree is {degrees.sum()}")
    print(f">> Degree shape is {degrees.shape}")

    print(f">> Highest degree is {degrees.max().item()}")
    print(f">> Lowest degree is {degrees.min().item()}")

    sorted_degrees, _ = torch.sort(degrees)
    percentile_index = int(len(sorted_degrees) * 0.8)
    threshold = sorted_degrees[percentile_index]
    print(f'>> Mean degree: {degrees.float().mean().item():.2f}')
    print(f'>> Std degree: {degrees.float().std().item():.2f}')
    print(f'>> Number of nodes with degree 0: {(degrees == 0).sum().item()}')
    print(f'>> Threshold: {threshold:.2f}')

    less_than_degree = (degrees <= threshold).sum().item()
    greater_than_degree = (degrees > threshold).sum().item()
    print(
        f">> Number of nodes with degree less than {threshold}: {less_than_degree}, it accounts for {less_than_degree / degrees.shape[0]:.2f}")
    print(
        f">> Number of nodes with degree greater than {threshold}: {greater_than_degree}, it accounts for {greater_than_degree / degrees.shape[0]:.2f}")

    return degrees, threshold


def preprocess_graph(dataset, args, n):
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(args.device), \
    dataset.graph['node_feat'].to(args.device)
    return dataset


def initialize_wandb(args, run):
    if args.wandb_name == '0':
        now = datetime.now()
        timestamp = now.strftime("%m%d-%H%M")
        args.wandb_name = timestamp
        if args.use_wandb:
            wandb.init(project=f'HyperbolicFormer({args.dataset})', config=vars(args),
                   name=f'{args.dataset}-Params-{args.wandb_name}-run-{run}')



def select_loss_function(args):
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        criterion = nn.BCEWithLogitsLoss()
        print(f'>> Using BCEWithLogitsLoss for {args.dataset}')
    else:
        criterion = nn.NLLLoss()
        print(f'>> Using NLLLoss for {args.dataset}')
    return criterion


def select_eval_function(args):
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
        print('>> Using ROC-AUC metric')
    elif args.metric == 'f1':
        eval_func = eval_f1
        print('>> Using F1 metric')
    else:
        eval_func = eval_acc
        print('>> Using Accuracy metric')
    return eval_func


def train_and_evaluate(args, dataset, split_idx_lst, device, criterion, eval_func):
    n, c, d = print_dataset_info(dataset)
    degrees, threshold = compute_and_print_degrees(dataset)
    degrees = degrees.to(device)                         # degree 放 GPU
    preprocess_graph(dataset, args, n)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        initialize_wandb(args, run)
        split_idx = split_idx_lst[0] if args.dataset in ['cora', 'citeseer', 'pubmed'] \
                     and args.protocol == 'semi' else split_idx_lst[run]
        train_idx = split_idx['train'].to(device)

        model = parse_method(args, c, d)
        model.reset_parameters()
        optimizer = Optimizer(model, args)

        # ----------- epoch 训练循环 -----------
        for epoch in range(args.epochs):
            train_start = time.time()
            loss = train_one_epoch(epoch, args, dataset, device,
                                   model, optimizer, criterion, train_idx)
            if epoch % args.eval_step == 0:
                evaluate_and_log(epoch, args, model, dataset, split_idx,
                                 eval_func, criterion, degrees, threshold,
                                 device, logger, run, loss)
        # ========== 单个 run 训练结束 ==========
        logger.print_statistics(run)

        # 🔹🔹🔹新增 5 行：每个 run 结束就追加写入 Excel🔹🔹🔹
        if args.save_result:
            excel_path = f"results/{args.dataset}_runs.xlsx"
            # 按 val_acc 选最佳 epoch；如要按 test_acc 改成 select_by='test_acc'
            logger.append_run_to_excel(run, excel_path, select_by='val_acc')

        if args.use_wandb:
            wandb.finish()

    # （可保留）训练全部完成后，再导出一个汇总文件
    if args.save_result:
        final_excel = f"results/{args.dataset}_summary.xlsx"
        logger.save_to_excel(final_excel)

def train_one_epoch(epoch, args, dataset, device, model, optimizer, criterion, train_idx):
    model.to(device)
    model.train()
    optimizer.zero_grad()

    out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])#修改
    true_label = get_true_label(dataset, args)
    loss = compute_loss(out, criterion, train_idx, true_label, args)
    loss.backward()
    optimizer.step()

    return loss


def get_true_label(dataset, args):
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            return F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    return dataset.label


def compute_loss(out, criterion, train_idx, true_label, args):
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        return criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        return criterion(out[train_idx], true_label.squeeze(1)[train_idx])

'''
def evaluate_and_log(epoch, args, model, dataset, split_idx, eval_func, criterion, degrees, threshold, device, logger,
                     run, loss):
    result = evaluate_large(model, dataset, split_idx, eval_func, criterion, args, degrees, threshold, device=device)
    logger.add_result(run, result)
    if epoch % args.display_step == 0:
        display_evaluation_results(epoch, result, split_idx, degrees, threshold, loss)
    if args.use_wandb:
        wandb.log({"run": run, "epoch": epoch, "loss": loss.item(), "train_acc": result[0], "val_acc": result[1],
                   "test_acc": result[2], "val_loss": result[3]})
'''
def evaluate_and_log(epoch, args, model, dataset, split_idx,
                     eval_func, criterion, degrees, threshold,
                     device, logger, run, loss):

    (val_loss, val_acc, val_f1, val_prec, val_rec,
     test_loss, test_acc, test_f1, test_prec, test_rec) = evaluate_large(
         model, dataset, split_idx, eval_func, criterion,
         args, degrees, threshold, device=device
     )

    logger.add_result(run, (
        val_loss, val_acc, val_f1, val_prec, val_rec,
        test_loss, test_acc, test_f1, test_prec, test_rec
    ))

    if epoch % args.display_step == 0:
        print(
            f'👉 Epoch {epoch:03d} | '
            f'val_loss {val_loss:.4f} val_acc {val_acc:.4f} '
            f'test_loss {test_loss:.4f} test_acc {test_acc:.4f}'
        )

    if args.use_wandb:
        wandb.log({
            "run": run, "epoch": epoch, "loss": loss.item(),
            "val/loss": val_loss, "val/acc": val_acc,
            "val/f1": val_f1, "val/precision": val_prec, "val/recall": val_rec,
            "test/loss": test_loss, "test/acc": test_acc,
            "test/f1": test_f1, "test/precision": test_prec, "test/recall": test_rec
        })

def display_evaluation_results(epoch, result, split_idx, degrees, threshold, loss):
    degrees_in_test = degrees[split_idx['test']]
    top_indices = split_idx['test'][degrees_in_test > threshold]
    bottom_indices = split_idx['test'][degrees_in_test <= threshold]

    max_top_acc = top_indices.shape[0] / split_idx['test'].shape[0]
    max_bottom_acc = bottom_indices.shape[0] / split_idx['test'].shape[0]
    # print_str = (f'👉Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, '
    #              f'Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%, '
    #              f'Top: {100 * result[5]:.2f} | {100 * max_top_acc:.2f}%, '
    #              f'Bottom: {100 * result[6]:.2f} | {100 * max_bottom_acc:.2f}%')
    print_str = (f'👉Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, '
                 f'Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%')
    print(print_str)


def main():
    args = parse_args()
    print(args)

    fix_seed(args.seed)
    device = get_device(args.cpu, args.device)
    dataset = load_and_preprocess_data(args, device)
    split_idx_lst = get_data_splits(args, dataset)

    criterion = select_loss_function(args)
    eval_func = select_eval_function(args)

    train_and_evaluate(args, dataset, split_idx_lst, device, criterion, eval_func)


if __name__ == "__main__":
    main()
