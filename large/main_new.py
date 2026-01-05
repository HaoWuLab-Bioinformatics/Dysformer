import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
import numpy as np
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


###############################################################################
# LINK
###############################################################################

class LINK(nn.Module):
    """ logistic regression on adjacency matrix """

    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index):
        """
        x: [N, D] (unused here, just for interface consistency)
        edge_index: [2, E]，表示 row -> col
        """
        N = x.shape[0]
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            # 用正常的 row=row, col=col
            A_sp = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
            A = A_sp.to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        # 直接对 A 做 logistic 回归
        logits = self.W(A)
        return logits


###############################################################################
# MLP
###############################################################################

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            # 只有一层线性层，相当于 logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        # MLP 不用 edge_index，保持接口一致
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


###############################################################################
# SGC
###############################################################################

class SGC(nn.Module):
    """
    直接调用 PyG 自带的 SGConv
    """

    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=False)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x


###############################################################################
# SGCMem
###############################################################################

class SGCMem(nn.Module):
    """
    自己实现的 SGC，手动做 k-hop 邻接乘法
    """

    def __init__(self, in_channels, out_channels, hops, use_bn=False):
        super(SGCMem, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.hops = hops
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(in_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.use_bn:
            self.bn.reset_parameters()

    def forward(self, x, edge_index):
        n = x.shape[0]
        if self.use_bn:
            x = self.bn(x)

        # gcn_norm 帮助做 D^-1/2 A D^-1/2 形式
        edge_weight = None
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, n, False, dtype=x.dtype)
        row, col = edge_index
        # 用 row=row, col=col，不翻转
        adj = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(n, n))

        # 做 k 次 A x
        for _ in range(self.hops):
            x = matmul(adj, x)

        x = self.lin(x)
        return x


###############################################################################
# SGC2
###############################################################################

class SGC2(nn.Module):
    """
    用 MLP 替代原先的单线性层
    """

    def __init__(self, in_channels, hidden_channels, out_channels, hops, num_layers, dropout, use_bn=False):
        super(SGC2, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.hops = hops
        self.dropout = dropout
        self.use_bn = use_bn

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        n = x.shape[0]
        edge_weight = None
        # 先做标准化
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, n, False, dtype=x.dtype)
        row, col = edge_index
        adj = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(n, n))

        # k-hop 传播
        for _ in range(self.hops):
            x = matmul(adj, x)

        # MLP
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x


###############################################################################
# GCN
###############################################################################

class GCN(nn.Module):
    """
    使用 PyG 自带的 GCNConv
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


###############################################################################
# SIGN
###############################################################################

class SIGN(nn.Module):
    """
    原本代码里手动计算度数并做多次邻接乘法，然后拼接
    这里统一为 row=row, col=col，不再翻转
    """

    def __init__(self, in_channels, hidden_channels, out_channels, hops, num_layers, dropout, use_bn=False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * (hops + 1), hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.num_layers = num_layers
        self.hops = hops
        self.use_bn = use_bn

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        N = x.shape[0]
        row, col = edge_index
        # 手动做 GCN-like 归一化
        d = degree(row, N).float()
        d_inv_sqrt = 1. / d.sqrt()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0.
        # 对每条边 (row->col) 分配的权重 = 1/sqrt(d[row]* d[col])
        value = d_inv_sqrt[row] * d_inv_sqrt[col]
        adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))

        # 多次相乘
        x_list = [x]
        out_x = x
        for _ in range(self.hops):
            out_x = matmul(adj, out_x)
            x_list.append(out_x)

        # 拼接
        x_concat = torch.cat(x_list, dim=1)

        # MLP 部分
        for i, lin in enumerate(self.lins[:-1]):
            x_concat = lin(x_concat)
            if self.use_bn:
                x_concat = self.bns[i](x_concat)
            x_concat = F.relu(x_concat)
            x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        x_concat = self.lins[-1](x_concat)
        return x_concat


###############################################################################
# GAT
###############################################################################

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, use_bn=False, heads=2, out_heads=1):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, dropout=dropout, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, dropout=dropout, heads=out_heads, concat=False))

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


###############################################################################
# MultiLP
###############################################################################

class MultiLP(nn.Module):
    """
    label propagation, with possibly multiple hops
    """

    def __init__(self, out_channels, alpha, hops, num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.hops = hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin

    def forward(self, x, edge_index, label, train_idx):
        n = x.shape[0]
        edge_weight = None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, n, False)
            row, col = edge_index
            # 构造正常的 adj
            adj_t = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, edge_weight, n, False)
            edge_weight = None
            adj_t = edge_index

        # 初始化label向量
        y = torch.zeros((n, self.out_channels), device=x.device)
        if label.shape[1] == 1:
            # 单标签 -> one hot
            y[train_idx] = F.one_hot(label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            # 多任务二分类
            y = torch.zeros((n, 2 * self.out_channels), device=x.device)
            for task in range(label.shape[1]):
                y[train_idx, 2 * task:2 * task + 2] = F.one_hot(label[train_idx, task], 2).to(y)
        else:
            # 多维标签
            y[train_idx] = label[train_idx].to(y.dtype)

        result = y.clone()
        for _ in range(self.num_iters):
            # 每次迭代里做 hops 次 A x
            temp = result
            for _h in range(self.hops):
                temp = matmul(adj_t, temp)
            temp = temp * self.alpha
            result = temp + (1 - self.alpha) * y

        if self.mult_bin:
            # 每个task取二分类里正类别的分数
            output = torch.zeros((n, self.out_channels), device=result.device)
            for task in range(label.shape[1]):
                output[:, task] = result[:, 2 * task + 1]
            result = output

        return result


###############################################################################
# MixHop
###############################################################################

class MixHopLayer(nn.Module):
    """ Our MixHop layer """

    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for _ in range(self.hops + 1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        # adj_t: SparseTensor
        # 计算 0~hops 次方的聚合
        xs = []
        for j in range(self.hops + 1):
            x_j = self.lins[j](x)
            # 做 j 次邻接乘法
            for _ in range(j):
                x_j = matmul(adj_t, x_j)
            xs.append(x_j)
        return torch.cat(xs, dim=1)


class MixHop(nn.Module):
    """
    包含多层 MixHopLayer，每层都做 0..hops 次方聚合并拼接
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))

        for _ in range(num_layers - 2):
            self.convs.append(MixHopLayer(hidden_channels * (hops + 1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))

        self.convs.append(MixHopLayer(hidden_channels * (hops + 1), out_channels, hops=hops))
        self.final_project = nn.Linear(out_channels * (hops + 1), out_channels)
        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, edge_index):
        n = x.shape[0]
        edge_weight = None
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, n, False, dtype=x.dtype)
        row, col = edge_index
        # 用 row=row, col=col
        adj_t = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(n, n))

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.final_project(x)
        return x


###############################################################################
# GCNJK
###############################################################################

class GCNJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else:  # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x


###############################################################################
# GATJK
###############################################################################

class GATJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False))

        self.dropout = dropout
        self.activation = F.elu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels * heads, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * heads * num_layers, out_channels)
        else:  # max or lstm
            self.final_project = nn.Linear(hidden_channels * heads, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x


###############################################################################
# H2GCNConv (仅展示原本逻辑，未改动过多)
###############################################################################

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """

    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        """
        adj_t: 一阶邻接
        adj_t2: 二阶邻接
        """
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


###############################################################################
# APPNP_Net
###############################################################################

class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1):
        super(APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x


###############################################################################
# GPRGNN
###############################################################################

class GPR_prop(MessagePassing):
    """
    GPRGNN propagation class, from original repo https://github.com/jianhao2016/GPRGNN
    """

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = 0.0 * np.ones(K + 1)
            # 直接把 alpha 当成 index？
            # 原作者可能做了 alpha=int(hop)
            # 这里保留原逻辑
            TEMP[int(self.alpha)] = 1.0
        elif Init == 'PPR':
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP, dtype=torch.float))

    def reset_parameters(self):
        # 根据默认 PPR 初始化
        nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GPRGNN(nn.Module):
    """GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN"""

    def __init__(self, in_channels, hidden_channels, out_channels, Init='PPR', dprate=.5, dropout=.5,
                 K=10, alpha=.1, Gamma=None, ppnp='GPR_prop'):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


###############################################################################
# GraphConvLayer & GraphConv
###############################################################################

class GraphConvLayer(nn.Module):
    """
    原先使用 row=col, col=row 方式，这里统一改回正常的(row->col)。
    也要注意在 forward() 里度数的计算相匹配。
    """

    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()
        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index

        # GCN-like对称归一化: 1 / sqrt(deg[row] * deg[col])
        deg_out = degree(row, N).float()  # row的度数
        deg_out[deg_out < 1] = 1.  # 防止除0
        deg_out_sqrt = deg_out.sqrt()
        val = 1. / (deg_out_sqrt[row] * deg_out_sqrt[col])

        val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(N, N))

        x = matmul(adj, x)  # A * x

        if self.use_init:
            x = torch.cat([x, x0], 1)
        if self.use_weight:
            x = self.W(x)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5,
                 use_bn=True, use_residual=True, use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        # 先做一层线性变换
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)

        # 多层 GraphConvLayer
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
            layer_.append(x)

        return x
