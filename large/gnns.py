import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import (
    GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
)
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
import torch_sparse
import numpy as np

###############################################################################
# LINK
###############################################################################
class LINK(nn.Module):
    """基于邻接矩阵的简单 logistic regression (只做示例)"""

    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index):
        """
        x: [N, D] —— 这里并未真正使用 x，只是兼容接口。
        edge_index: [2, E], 表示 row->col (0-based)
        """
        N = x.shape[0]
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()

        # 对整个邻接矩阵做 logistic regression
        logits = self.W(A)
        return logits


###############################################################################
# MLP
###############################################################################
class MLP(nn.Module):
    """
    一个简单多层感知机示例
    （和图无关，所以 forward 不怎么处理 edge_index）
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # 只有一层，则是线性回归
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
        # 普通 MLP 并不需要 edge_index，仅为接口保留
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


###############################################################################
# SGC (PyG 自带 SGConv)
###############################################################################
class SGC(nn.Module):
    """
    调用 PyG 的 SGConv（可以指定 hops 次邻接乘法）
    """
    def __init__(self, in_channels, out_channels, hops):
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=False)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


###############################################################################
# SGCMem (手动实现 k 次邻接乘法)
###############################################################################
class SGCMem(nn.Module):
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
        edge_weight = None

        # 做标准 GCN 归一化, 不翻转 row/col
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype
            )
            row, col = edge_index
            adj_t = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, n, False, dtype=x.dtype)
            adj_t = edge_index

        if self.use_bn:
            x = self.bn(x)

        # 连续做 hops 次邻接乘法
        for _ in range(self.hops):
            x = matmul(adj_t, x)

        # 最后过一个线性映射
        x = self.lin(x)
        return x


###############################################################################
# SGC2 (用 MLP 替代单线性层)
###############################################################################
class SGC2(nn.Module):
    """
    先做 k-hop 邻接乘法，再 MLP
    """
    def __init__(self, in_channels, hidden_channels, out_channels, hops,
                 num_layers, dropout, use_bn=False):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

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
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, n, False, dtype=x.dtype)
        row, col = edge_index
        adj_t = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(n, n))

        # k-hop 邻接乘法
        for _ in range(self.hops):
            x = matmul(adj_t, x)

        # 再做 MLP
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


###############################################################################
# GCN (PyG GCNConv)
###############################################################################
class GCN(nn.Module):
    """
    多层 GCNConv
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem)
        )

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
    多次邻接乘法 (A^0, A^1, ... A^hops) 并拼接，再 MLP
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 hops, num_layers, dropout, use_bn=False):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 先把 (hops+1) 次拼接后特征输入到第一层
        self.lins.append(nn.Linear(in_channels * (hops+1), hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

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

        # 1/sqrt(deg[row]*deg[col]) 归一化
        d = degree(row, N).float()
        d_inv_sqrt = 1. / d.sqrt()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0.
        value = d_inv_sqrt[row] * d_inv_sqrt[col]

        adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))

        # A^0 x, A^1 x, ... A^hops x
        out_x = x
        embeddings = [x]  # A^0 x
        for _ in range(self.hops):
            out_x = matmul(adj, out_x)
            embeddings.append(out_x)

        # 拼接后送 MLP
        x_concat = torch.cat(embeddings, dim=1)
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
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, use_bn=False,
                 heads=2, out_heads=1):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, dropout=dropout, heads=heads, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, dropout=dropout,
                    heads=out_heads, concat=False)
        )

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
    Label Propagation 示例，可多次迭代、支持多标签/二分类等
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
            adj_t = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, n, False)
            adj_t = edge_index

        # 初始化 label
        y = torch.zeros((n, self.out_channels), device=x.device)
        if label.shape[1] == 1:
            # 单列标签 -> one_hot
            y[train_idx] = F.one_hot(label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            # 多任务二分类
            y = torch.zeros((n, 2*self.out_channels), device=x.device)
            for task in range(label.shape[1]):
                y[train_idx, 2*task:2*task+2] = F.one_hot(label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = label[train_idx].to(y.dtype)

        result = y.clone()
        for _ in range(self.num_iters):
            # 每轮迭代再做 hops 次邻接乘法
            for _h in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1 - self.alpha) * y

        if self.mult_bin:
            # 多任务二分类，需要映射回1维
            output = torch.zeros((n, self.out_channels), device=result.device)
            for task in range(label.shape[1]):
                output[:, task] = result[:, 2*task+1]
            result = output

        return result


###############################################################################
# MixHop
###############################################################################
class MixHopLayer(nn.Module):
    """MixHopLayer: 计算 [A^0 x, A^1 x, ..., A^hops x] 并拼接"""
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for _ in range(hops + 1):
            self.lins.append(nn.Linear(in_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = []
        for j, lin in enumerate(self.lins):
            x_j = lin(x)
            # 做 j 次邻接乘法
            for _ in range(j):
                x_j = matmul(adj_t, x_j)
            xs.append(x_j)
        return torch.cat(xs, dim=1)


class MixHop(nn.Module):
    """
    多层 MixHop，每层都做 0..hops 次方聚合并拼接
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, hops=2):
        super(MixHop, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))
        self.bns.append(nn.BatchNorm1d(hidden_channels * (hops+1)))

        for _ in range(num_layers - 2):
            self.convs.append(MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))

        self.convs.append(MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))
        self.final_project = nn.Linear(out_channels * (hops+1), out_channels)

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
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, n, False, dtype=x.dtype)
            adj_t = edge_index

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
    """
    结合 JumpingKnowledge 的 GCN
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem)
        )

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else:  # 'max' or 'lstm'
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
    """
    结合 JumpingKnowledge 的 GAT
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)
        )

        self.dropout = dropout
        self.activation = F.elu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels * heads, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * heads * num_layers, out_channels)
        else:  # 'max' or 'lstm'
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
# H2GCNConv (仅示例)
###############################################################################
class H2GCNConv(nn.Module):
    """ 做两阶邻居聚合 """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        # adj_t, adj_t2 均为 SparseTensor
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


###############################################################################
# APPNP_Net
###############################################################################
class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, K=10, alpha=0.1):
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
# GPRGNN (含 GPR_prop)
###############################################################################
class GPR_prop(MessagePassing):
    """
    GPRGNN 的传播逻辑
    """
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = 0.0 * np.ones(K+1)
            TEMP[int(alpha)] = 1.0
        elif Init == 'PPR':
            TEMP = alpha * (1 - alpha)**np.arange(K+1)
            TEMP[-1] = (1 - alpha)**K
        elif Init == 'NPPR':
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP, dtype=torch.float))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha * (1 - self.alpha)**k
        self.temp.data[-1] = (1 - self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(edge_index, edge_weight,
                                        x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, edge_weight,
                                  x.size(0), dtype=x.dtype)
            norm = None

        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GPRGNN(nn.Module):
    """GPRGNN，来自官方实现"""
    def __init__(self, in_channels, hidden_channels, out_channels, Init='PPR',
                 dprate=0.5, dropout=0.5, K=10, alpha=0.1, Gamma=None, ppnp='GPR_prop'):
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
    自定义图卷积层。按 GCN-like 归一化, 并可选 use_init 做残差拼接
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
        # (row, col) = edge_index
        row, col = edge_index
        # 1 / sqrt(deg[row] * deg[col])
        d = degree(row, N).float()
        d_inv_sqrt = 1.0 / d.sqrt()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
        val = d_inv_sqrt[row] * d_inv_sqrt[col]

        adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(N, N))
        x = matmul(adj, x)

        if self.use_init:
            x = torch.cat([x, x0], dim=1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)
        return x


class GraphConv(nn.Module):
    """
    多层 GraphConvLayer，可选 residual, BN, etc.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        # 先做一层线性把 in_channels -> hidden_channels
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers):
            self.convs.append(GraphConvLayer(
                hidden_channels, hidden_channels, use_weight, use_init
            ))
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

        # 第一层线性
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        # 之后多层 GraphConvLayer
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])  # 传入初始x0做拼接
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]

        return x
