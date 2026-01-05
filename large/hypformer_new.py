import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GATConv

# 以下 import 需你本地有对应文件:
# - manifolds/layer.py: HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization, HypCLS
# - manifolds/lorentz.py: Lorentz
# - gnns.py: GraphConv (若你想对比普通图卷积)
from manifolds.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization, HypCLS
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
from gnns import GraphConv

# 如果显存足够，可以不手动清，否则可以手动清
torch.cuda.empty_cache()

# 扩展显存策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

##############################################################################
#               1) 老版的超图 / Hyperbolic Layers  (可选保留)
##############################################################################

def get_dim_act_curv(args):
    """
    老版本中的一个工具函数，用来生成维度、激活函数和曲率等。
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HypLinearOld(nn.Module):
    """
    老版 Hyperbolic linear layer (Mobius mat-vec).
    仅保留以做对比，可根据需要删除。
    """
    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinearOld, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias

        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res


class HypActOld(nn.Module):
    """
    老版 Hyperbolic activation layer.
    """
    def __init__(self, manifold, c_in, c_out, act):
        super(HypActOld, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        x_logmap0 = self.manifold.logmap0(x, c=self.c_in)
        xt = self.act(x_logmap0)
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        xt = self.manifold.expmap0(xt, c=self.c_out)
        return self.manifold.proj(xt, c=self.c_out)


class HypAggOld(nn.Module):
    """
    老版 Hyperbolic aggregation layer
    """
    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAggOld, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        # 如果需要可再加 self.att = DenseAtt(...) 等

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output


class HNNLayer(nn.Module):
    """
    老版: Hyperbolic neural networks layer
    """
    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinearOld(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypActOld(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    老版: Hyperbolic graph convolution layer
    """
    def __init__(self, manifold, in_features, out_features, c_in, c_out,
                 dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinearOld(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAggOld(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypActOld(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        return h, adj


##############################################################################
#        2) 动态超图模块(新) : FasterHConstructor, FasterHGNN_conv
##############################################################################

class FasterHConstructor(nn.Module):
    """
    动态超图构造器:
     - 在计算 H = inputs@edges^T 时，采用 mini-batch 方式分块
    """
    def __init__(self, num_edges, f_dim, iters=1, eps=1e-8,
                 hidden_dim=128, topk_n=8, topk_e=8, adjust_freq=10,
                 chunk_size=50000):
        """
        :param chunk_size: 每次处理多少节点的mini-batch
        """
        super().__init__()
        self.num_edges = num_edges
        self.f_dim = f_dim
        self.iters = iters
        self.eps = eps
        self.topk_n = topk_n
        self.topk_e = topk_e
        self.adjust_freq = adjust_freq
        self.forward_count = 0
        self.chunk_size = chunk_size  # mini-batch分块大小

        # 可训练的超边初始化
        self.edges_mu = nn.Parameter(torch.randn(1, f_dim))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))
        nn.init.xavier_uniform_(self.edges_logsigma)

        # Q, K, V
        self.to_q = nn.Linear(f_dim, f_dim)
        self.to_k = nn.Linear(f_dim, f_dim)
        self.to_v = nn.Linear(f_dim, f_dim)

        # 用 GRUCell 或 MLP 更新 edges
        hidden_dim = max(f_dim, hidden_dim)
        self.gru = nn.GRUCell(f_dim, f_dim)
        self.mlp = nn.Sequential(
            nn.Linear(f_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, f_dim),
        )

        # LN
        self.norm_input = nn.LayerNorm(f_dim)
        self.norm_edges = nn.LayerNorm(f_dim)

    def mask_attn(self, attn, k):
        indices = torch.topk(attn, k).indices
        mask = torch.zeros_like(attn).bool()
        row_idx = torch.arange(attn.size(0), device=attn.device).unsqueeze(-1)
        mask[row_idx, indices] = True
        return attn * mask

    def ajust_edges(self, s_level, args):
        if self.adjust_freq <= 0:
            return
        if self.forward_count % self.adjust_freq != 0:
            return
        if args.stage != 'train':
            return
        if s_level > args.up_bound:
            self.num_edges += 1
        elif s_level < args.low_bound:
            self.num_edges -= 1
            self.num_edges = max(self.num_edges, args.min_num_edges)

    def forward(self, inputs, args):
        self.forward_count += 1
        device = inputs.device
        N = inputs.size(0)

        # Step0: 采样 edges
        mu = self.edges_mu.expand(self.num_edges, -1)
        sigma = self.edges_logsigma.exp().expand(self.num_edges, -1)
        edges = mu + sigma * torch.randn(mu.shape, device=device)

        # Step1: LN
        inputs = self.norm_input(inputs)

        # Step2: K, V
        k = F.relu(self.to_k(inputs))
        v = F.relu(self.to_v(inputs))

        # 迭代 iters 次
        for _ in range(self.iters):
            edges = self.norm_edges(edges)
            q = F.relu(self.to_q(edges))

            # 注意力 => shape=[S,N]
            attn = q @ k.T
            attn = attn * (self.f_dim ** -0.5) + self.eps
            attn = F.softmax(attn, dim=1)
            attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-9)
            attn = self.mask_attn(attn, self.topk_n)

            updates = attn @ v
            edges = self.gru(updates, edges)

            # 计算 saturation level
            # => 这里如果一次性: H_tmp = inputs @ edges.T => [N,S]
            # 我们改成mini-batch
            chunk_size = self.chunk_size
            n_chunks = (N + chunk_size - 1)//chunk_size
            empty_count = 0
            for i in range(n_chunks):
                start = i*chunk_size
                end = min(start+chunk_size, N)
                partial_inputs = inputs[start:end]
                partial_H = partial_inputs @ edges.T  # [chunk, S]
                partial_H = F.softmax(partial_H, dim=1)
                partial_H = self.mask_attn(partial_H, self.topk_e)

                cc = (partial_H>0).float()
                de = cc.sum(dim=0)       # [S]
                empty_count += (de==0).sum().item()

            # empty_count是所有块加和
            s_level = 1 - empty_count/(self.num_edges*n_chunks)

            self.ajust_edges(s_level, args)

        # 最终要返回 H => [N,S]
        # 这里做真正的mini-batch合并
        allH = []
        chunk_size = self.chunk_size
        n_chunks = (N + chunk_size - 1)//chunk_size
        for i in range(n_chunks):
            start = i*chunk_size
            end = min(start+chunk_size, N)
            partial_inputs = inputs[start:end]             # [chunk, f_dim]
            partialH = partial_inputs @ edges.T           # => [chunk, S]
            allH.append(partialH)
        H = torch.cat(allH, dim=0)  # => [N,S]

        return edges, H, H  # raw_attn = H


class FasterHGNN_conv(nn.Module):
    """
    动态超图卷积 + mini-batch
    """
    def __init__(self, in_ft, out_ft, num_edges, bias=True, iters=1,
                 topk_n=8, topk_e=8, adjust_freq=1, chunk_size=50000):
        super().__init__()
        self.HConstructor = FasterHConstructor(
            num_edges=num_edges,
            f_dim=in_ft,
            iters=iters,
            topk_n=topk_n,
            topk_e=topk_e,
            adjust_freq=adjust_freq,
            chunk_size=chunk_size,   # <-- 传递分块大小
        )
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft)) if bias else None

        self.linear_in = nn.Linear(in_ft, out_ft, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.linear_in.weight)

    def forward(self, x, args):
        edges, H, raw_attn = self.HConstructor(x, args)

        edges_mapped = edges @ self.weight
        if self.bias is not None:
            edges_mapped += self.bias

        nodes = H @ edges_mapped
        x_mapped = self.linear_in(x)
        x_out = x_mapped + nodes
        return x_out, H, raw_attn


##############################################################################
#   3) 多头注意力 (可选)
##############################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, q: int, v: int, h: int,
                 device: str, mask: bool=False, dropout: float=0.1):
        super(MultiHeadAttention, self).__init__()
        self.W_q = nn.Linear(d_model, q*h)
        self.W_k = nn.Linear(d_model, q*h)
        self.W_v = nn.Linear(d_model, v*h)
        self.W_o = nn.Linear(v*h, d_model)
        self.device = device
        self._h = h
        self._q = q
        self.mask = mask
        self.dropout = nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score
        if self.mask and stage=='train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask>0, score,
                                torch.Tensor([-2**32+1]).expand_as(score[0]).to(self.device))
        score = F.softmax(score, dim=-1)
        attention = torch.matmul(score, V)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        self_attention = self.W_o(attention_heads)
        return self_attention, self.score


##############################################################################
#   4) (可选) TransConvLayer, TransConv
##############################################################################

class TransConvLayer(nn.Module):
    """
    如果你还想在双曲空间做多头注意力，这里只是个占位示例
    """
    def __init__(self, manifold, in_channels, out_channels,
                 num_heads, use_weight=True, args=None):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = args.attention_type if args else 'full'

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))

        # 其它实现省略

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None,
                output_attn=False):
        # 你可以在这里实现你真正的多头逻辑
        return query_input  # 占位


class TransConv(nn.Module):
    """
    如果你需要双曲Transformer分支，也可以参考本示例
    """
    def __init__(self, manifold_in, manifold_hidden, manifold_out,
                 in_channels, hidden_channels, num_layers=1, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True,
                 use_act=True, args=None):
        super().__init__()
        # 占位
        self.convs = nn.ModuleList()

    def forward(self, x_input):
        # 占位，真正实现留给你
        return x_input

    def get_attentions(self, x):
        # 占位
        return None


##############################################################################
#   5) 常规 GCN / GAT / 及其它 (可对比，不是必需)
##############################################################################

class GCN(nn.Module):
    """
    普通GCN，用于对比
    """
    def __init__(self, args, layer_number=2):
        super(GCN, self).__init__()
        in_dim = args.in_dim
        hid_dim = args.hid_dim
        out_dim = args.out_dim

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        for i in range(1, layer_number):
            self.convs.append(GCNConv(hid_dim, hid_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, data, args):
        x = data['fts']
        edge_index = data['edge_index']
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        out = self.classifier(x)
        return out, x, None, None


class GAT(nn.Module):
    """
    普通GAT，用于对比
    """
    def __init__(self, args, layer_number=2):
        super(GAT, self).__init__()
        in_dim = args.in_dim
        hid_dim = args.hid_dim
        out_dim = args.out_dim

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hid_dim))
        for i in range(1, layer_number):
            self.convs.append(GATConv(hid_dim, hid_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, data, args):
        x = data['fts']
        edge_index = data['edge_index']
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        out = self.classifier(x)
        return out, x, None, None


##############################################################################
#   6) HGNN_classifier: 使用新的 FasterHGNN_conv 做动态超图
##############################################################################

class HGNN_classifier(nn.Module):
    """
    演示用动态超图网络（FasterHGNN_conv）来做分类：
      - backbone: linear 或 GCN
      - 多层 FasterHGNN_conv
      - classifier
    """
    def __init__(self, args, dropout=0.5):
        super(HGNN_classifier, self).__init__()
        in_dim = args.in_dim
        hid_dim = args.hid_dim
        out_dim = args.out_dim
        num_edges = args.num_edges
        self.conv_number = args.conv_number  # 堆叠多少层动态超图Conv
        self.dropout = dropout

        # 1) backbone: linear
        self.linear_backbone = nn.ModuleList()
        self.linear_backbone.append(nn.Linear(in_dim, hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))

        # 2) 也可以试 gcn_backbone
        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(in_dim, hid_dim))
        self.gcn_backbone.append(GCNConv(hid_dim, hid_dim))

        # 3) 多层 dynamic hypergraph conv
        self.convs = nn.ModuleList()
        self.transfers = nn.ModuleList()
        for i in range(self.conv_number):
            self.convs.append(FasterHGNN_conv(
                in_ft=hid_dim, out_ft=hid_dim,
                num_edges=num_edges,
                iters=args.iters,   # 迭代次数
                topk_n=args.k_n,
                topk_e=args.k_e,
                adjust_freq=args.adjust_freq,
            ))
            self.transfers.append(nn.Linear(hid_dim, hid_dim))

        # 4) classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_number * hid_dim, out_dim),
        )

    def forward(self, data, args):
        # data 可能是 x or 包含 data['fts'], data['edge_index'] 等
        x = data
        if args.backbone == 'linear':
            # 线性 backbone
            x = F.relu(self.linear_backbone[0](x))
            x = F.relu(self.linear_backbone[1](x))
            x = self.linear_backbone[2](x)
        elif args.backbone == 'gcn':
            # gcn backbone
            x = data['fts']
            edge_index = data['edge_index']
            x = self.gcn_backbone[0](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.gcn_backbone[1](x, edge_index)

        # 堆叠 conv_number 层 FasterHGNN_conv
        tmp = []
        H_list, H_raw_list = [], []
        for i in range(self.conv_number):
            x, H, h_raw = self.convs[i](x, args)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if args.transfer == 1:
                x = F.relu(self.transfers[i](x))
            tmp.append(x)
            H_list.append(H)
            H_raw_list.append(h_raw)

        # 拼接所有层输出
        x_cat = torch.cat(tmp, dim=1)
        out = self.classifier(x_cat)
        return out, x_cat, H_list, H_raw_list


##############################################################################
#   7) 其他 (DHGNN_conv 等，如果要保留老代码，就放这)
##############################################################################

class DHGNN_conv(nn.Module):
    """
    若你想保留老版本DHGNN_conv，可写这里
    """
    def __init__(self, in_ft, out_ft, num_edges, bias=True):
        super(DHGNN_conv, self).__init__()
        # 占位
        pass

    def forward(self, x, args):
        return x, None, None


##############################################################################
#   8) HypFormer: 同时支持「动态超图卷积」+「Hyp解码」
##############################################################################

class Dysformer(nn.Module):
    """
    演示一个“动态超图 + 双曲空间”主干模型：
      - 输入欧式 x => FasterHGNN_conv => Minkowski => Hyp解码
      - 也可与 TransConv / GraphConv 等组合
    """
    def __init__(
            self, in_channels, hidden_channels, out_channels,
            trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
            trans_use_weight=True, trans_use_act=True,
            gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
            gnn_use_residual=True, gnn_use_act=True,
            use_graph=True, graph_weight=0.5, aggregate='add',  decoder_type='euc',adjust_freq=1,
            k_n=1000000,k_e=1000000,num_edges=10000000,epochs=1000,stage='train',chunk_size=50000,iters=1,
            args=None
    ):
        super().__init__()

        # 1) 定义 manifold
        self.manifold_in = Lorentz(k=float(args.k_in))      # 例： k_in=1.0
        self.manifold_hidden = Lorentz(k=float(args.k_out)) # 例： k_out=2.0
        self.manifold_out = Lorentz(k=float(args.k_out))

        self.decoder_type = decoder_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_edges = num_edges
        self.hcon_iters = iters


        # 2) 动态超图卷积: FasterHGNN_conv
        self.dhyper_conv = FasterHGNN_conv(
            in_ft=in_channels,
            out_ft=hidden_channels,
            num_edges=num_edges,
            iters=iters,
            topk_n=k_n,
            topk_e=k_e,
            adjust_freq=adjust_freq,
            chunk_size = chunk_size,
        )
        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act, args)
        self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn, gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)

        self.dropout = trans_dropout

        # 3) 解码：欧式 or 超曲
        if self.decoder_type == 'euc':
            self.decode = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.decoder_type == 'hyp':
            # Minkowski => HypCLS
            # 先把 out_features 也可以设 hidden_channels => out_channels
            self.decode = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, args):
        """
        x: [N, in_channels], in Euclidean space
        """
        # 1) 动态超图卷积 => [N, hidden_channels]
        x_out, H, raw_attn = self.dhyper_conv(x, args)
        x_out = F.relu(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)

        # 2) 如果要把 x_out 映射到 Minkowski => 需要 HypLinear(..., x_manifold='euc')
        #    这里示例: 仅仅 logmap0 => 维度不会变，会出现 shape mismatch
        #    一种做法: 直接在 decode 里写 x_manifold='euc' => decodeHyp
        if self.decoder_type == 'hyp':
            # decode(...) 里指定 x_manifold='euc' => 把欧式 -> Minkowski -> ...
            out = self.decode(x_out, x_manifold='euc')  # => [N, out_channels+1] or out_channels
        else:
            # euc decode
            out = self.decode(x_out)  # => [N, out_channels]
        return out

    def reset_parameters(self):
        """
        重置模型的所有可训练参数，用于和外部的 model.reset_parameters() 对接
        """
        # 例如：重置动态超图卷积的参数
        self.dhyper_conv.reset_parameters()

        # 如果你还有其他层 (比如 self.decode 是 nn.Linear 或 HypCLS),
        # 也可以把它们也 reset，比如:
        if hasattr(self.decode, 'reset_parameters'):
            self.decode.reset_parameters()
        else:
            # 或者自己写初始化
            if isinstance(self.decode, nn.Linear):
                nn.init.xavier_uniform_(self.decode.weight)
                if self.decode.bias is not None:
                    nn.init.constant_(self.decode.bias, 0)

##############################################################################
#  示例：这样就包含了“动态超图卷积+双曲解码”的完整逻辑
##############################################################################

