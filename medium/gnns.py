import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
import scipy.sparse
import numpy as np
import torch_sparse
from typing import Optional



class LINK(nn.Module):
    """ logistic regression on adjacency matrix """

    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index):
        N = x.shape[0]
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return logits


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
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
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=False)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x


class SGCMem(nn.Module):
    def __init__(self, in_channels, out_channels, hops, use_bn=False):
        """ self-implementation of SGC
        """
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
        # x = self.lin(x)
        n = x.shape[0]
        edge_weight = None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        if self.use_bn:
            x = self.bn(x)

        for _ in range(self.hops):
            x = matmul(adj_t, x)

        x = self.lin(x)
        return x


class SGC2(nn.Module):
    '''
    Use MLP instead of a single linear layer.
    '''

    def __init__(self, in_channels, hidden_channels, out_channels, hops, num_layers, dropout, use_bn=False):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
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
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, n, False,
            dtype=x.dtype)
        row, col = edge_index
        adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))

        for _ in range(self.hops):
            x = matmul(adj_t, x)

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem))

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


class SIGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hops, num_layers, dropout, use_bn=False):
        super().__init__()
        # print(f'in_channel:{in_channels}')
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * (hops + 1), hidden_channels))
        self.bns = torch.nn.ModuleList()
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
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))

        embedding = [x]
        for _ in range(self.hops):
            x = torch_sparse.matmul(adj, x)
            embedding.append(x)

        x = torch.cat(embedding, dim=1)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, use_bn=False, heads=2, out_heads=1):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True))

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


class MultiLP(nn.Module):
    """ label propagation, with possibly multiple hops of the adjacency """

    def __init__(self, out_channels, alpha, hops, num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.hops = hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin  # handle multiple binary tasks

    def forward(self, x, edge_index, label, train_idx):
        n = x.shape[0]
        edge_weight = None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False)
            edge_weight = None
            adj_t = edge_index

        y = torch.zeros((n, self.out_channels)).to(adj_t.device())
        if label.shape[1] == 1:
            # make one hot
            y[train_idx] = F.one_hot(label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            y = torch.zeros((n, 2 * self.out_channels)).to(adj_t.device())
            for task in range(label.shape[1]):
                y[train_idx, 2 * task:2 * task + 2] = F.one_hot(label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = label[train_idx].to(y.dtype)
        result = y.clone()
        for _ in range(self.num_iters):
            for _ in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1 - self.alpha) * y

        if self.mult_bin:
            output = torch.zeros((n, self.out_channels)).to(result.device)
            for task in range(label.shape[1]):
                output[:, task] = result[:, 2 * task + 1]
            result = output

        return result


class MixHopLayer(nn.Module):
    """ Our MixHop layer """

    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops + 1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x)]
        for j in range(1, self.hops + 1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)


class MixHop(nn.Module):
    """ our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels * (hops + 1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))

        self.convs.append(
            MixHopLayer(hidden_channels * (hops + 1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
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
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        return x


class GCNJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

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


class GATJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.dropout = dropout
        self.activation = F.elu  # note: uses elu

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


class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """

    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


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


class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(nn.Module):
    """GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN"""

    def __init__(self, in_channels, hidden_channels, out_channels, Init='PPR', dprate=.5, dropout=.5, K=10, alpha=.1,
                 Gamma=None, ppnp='GPR_prop'):
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



# ======== GraphConvLayer（纯 GPU）========
class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_weight=True, use_init=False,
                 add_self_loops=True, make_undirected=True):
        super().__init__()
        self.use_weight = use_weight
        self.use_init = use_init
        self.add_self_loops = add_self_loops
        self.make_undirected = make_undirected

        if use_weight:
            self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
            self.bias   = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias",  None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_weight:
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

    # ---- GPU-safe 构图 ----
    @staticmethod
    def build_adj_gpu(
            edge_index: torch.Tensor,
            N: int,
            edge_weight: Optional[torch.Tensor] = None,
            make_undirected: bool = True,
            add_self_loops: bool = True
    ) -> SparseTensor:
        """
        edge_index 必须已在 GPU；返回 GPU SparseTensor
        """
        if edge_index.dim() != 2 or 2 not in edge_index.shape:
            raise ValueError("edge_index shape 应为 (2,E) 或 (E,2)")

        # 统一为 (2,E)
        if edge_index.size(0) != 2:
            edge_index = edge_index.t()

        # 保证 long/contiguous/在 GPU
        edge_index = edge_index.long().contiguous()
        row, col = edge_index[0], edge_index[1]

        # —— 越界裁剪（GPU）——
        mask = (row >= 0) & (row < N) & (col >= 0) & (col < N)
        if mask.sum() == 0:
            return SparseTensor.sparse_empty(N, N, device=row.device)

        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        row = row.index_select(0, idx)
        col = col.index_select(0, idx)
        if edge_weight is not None:
            edge_weight = edge_weight.to(row.device)
            edge_weight = edge_weight.index_select(0, idx)

        # 无向化
        if make_undirected:
            row, col = torch.cat([row, col]), torch.cat([col, row])
            if edge_weight is not None:
                edge_weight = torch.cat([edge_weight, edge_weight])

        # 自环
        if add_self_loops:
            loop = torch.arange(N, device=row.device, dtype=row.dtype)
            row = torch.cat([row, loop])
            col = torch.cat([col, loop])
            if edge_weight is not None:
                edge_weight = torch.cat(
                    [edge_weight, torch.ones(N, device=row.device, dtype=edge_weight.dtype)]
                )

        # 断言安全
        assert int(row.max()) < N and int(col.max()) < N, "仍有越界索引！"

        edge_idx = torch.stack([row, col], dim=0)
        if edge_weight is None:
            adj = SparseTensor.from_edge_index(
                edge_idx,
                sparse_sizes=(N, N)
            ).coalesce()
        else:
            adj = SparseTensor.from_edge_index(
                edge_idx,
                edge_weight,
                sparse_sizes=(N, N)
            ).coalesce()
        return adj

    def forward(self, x, edge_index, x_init=None, edge_weight=None):
        N, device = x.size(0), x.device
        adj = self.build_adj_gpu(edge_index.to(device), N,
                                 edge_weight=edge_weight,
                                 make_undirected=self.make_undirected,
                                 add_self_loops=self.add_self_loops)

        # GCN 归一化
        deg = adj.sum(dim=1)                        # (N,)
        deg_inv_sqrt = (deg + 1e-12).pow(-0.5)
        x = x * deg_inv_sqrt.unsqueeze(-1)
        x = adj.matmul(x)
        x = x * deg_inv_sqrt.unsqueeze(-1)

        if self.use_weight:
            x = x @ self.weight + self.bias

        if self.use_init and (x_init is not None):
            x = x + x_init
        return x

# ======== GraphConv（逐时间步，纯 GPU）========
class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, args):
        super().__init__()
        self.dropout      = args.gnn_dropout
        self.num_layers   = args.gnn_num_layers
        self.use_bn       = args.gnn_use_bn
        self.use_residual = args.gnn_use_residual
        self.use_act      = args.gnn_use_act
        self.activation   = F.relu

        self.fcs = nn.ModuleList([nn.Linear(in_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])

        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(GraphConvLayer(hidden_channels, hidden_channels,
                                             use_weight=args.gnn_use_weight,
                                             use_init=args.gnn_use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        if x.dim() == 2:     # (N,F)
            x0 = self.fcs[0](x)
            if self.use_bn: x0 = self.bns[0](x0)
            if self.use_act: x0 = self.activation(x0)
            x0 = F.dropout(x0, p=self.dropout, training=self.training)

            h = x0
            for i, conv in enumerate(self.convs):
                h = conv(h, edge_index, x_init=x0)
                if self.use_bn:  h = self.bns[i+1](h)
                if self.use_act: h = self.activation(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.use_residual: h = h + x0
            return h

        elif x.dim() == 3:   # (T,N,F)
            T, N, Fdim = x.shape
            x_flat = x.reshape(T*N, Fdim)
            x0 = self.fcs[0](x_flat)
            if self.use_bn: x0 = self.bns[0](x0)
            if self.use_act: x0 = self.activation(x0)
            x0 = F.dropout(x0, p=self.dropout, training=self.training)
            x0 = x0.view(T, N, -1)

            h = x0
            for i, conv in enumerate(self.convs):
                outs = [conv(h[t], edge_index, x_init=x0[t]) for t in range(T)]
                h    = torch.stack(outs, 0)          # (T,N,H)
                if self.use_bn:
                    h = h.view(T*N, -1)
                    h = self.bns[i+1](h)
                    h = h.view(T, N, -1)
                if self.use_act: h = self.activation(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.use_residual: h = h + x0
            return h

        else:
            raise ValueError("x 维度应为 2 或 3")



class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, save_mem=True, args=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = F.relu
        self.dropout = args.gnn_dropout
        self.num_layers = args.gnn_num_layers
        self.dropout = args.gnn_dropout
        self.use_bn = args.gnn_use_bn

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(self.in_channels, self.hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GCNConv(self.hidden_channels, self.hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(self.hidden_channels))

        self.convs.append(
            GCNConv(self.hidden_channels, self.out_channels, cached=not save_mem))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
