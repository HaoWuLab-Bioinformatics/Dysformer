# =============================================================================
#  1) 统一 Imports
# =============================================================================
import math
import os
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# ---- 如果需要用到 torch_geometric 之类的依赖，就保持引入 ----
from torch_geometric.nn import GCNConv, GATConv


# ---- 原本第一段里 hypergraph.py、CBAM 等，如果项目中需要就保留 ----
# from hypergraph import Hypergraph
# from model.attention.CBAM import CBAMBlock
# from model.attention.MobileViTv2Attention import MobileViTv2Attention

# ---- 这部分是合并核心，与第二段 "manifolds.layer" 重名函数冲突注意 ----
# from manifolds.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization, HypCLS
# from manifolds.lorentz import Lorentz
# from geoopt import ManifoldParameter
# （上面这几个 import，如果确实需要，就在环境中提供相应文件/模块）

# =============================================================================
#  2) 先放置原始【第一段】中的主要类，并改名以避免冲突
# =============================================================================

def get_dim_act_curv(args):
    """
    原先第一段中的工具函数
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
    【第一段版本】的 Hyperbolic linear layer (Mobius mat-vec).
    为了避免和 manifolds/layer.py 中的 HypLinear 重名，这里改名为 HypLinearOld。
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


class HypActOld(Module):
    """
    【第一段版本】的 Hyperbolic activation layer.
    改名为 HypActOld 以区分 manifolds/layer 中的 HypActivation
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypActOld, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        x_logmap0 = self.manifold.logmap0(x, c=self.c_in)
        xt = self.act(x_logmap0)  # activation in tangent space
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        xt = self.manifold.expmap0(xt, c=self.c_out)
        return self.manifold.proj(xt, c=self.c_out)


class HypAggOld(Module):
    """
    【第一段版本】的 Hyperbolic aggregation layer.
    改名 HypAggOld 以与自身 aggregator 区别
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAggOld, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        # self.att = DenseAtt(...) 如果需要可以补充

    def forward(self, x, adj):
        # 这里是老版本的聚合逻辑
        x_tangent = self.manifold.logmap0(x, c=self.c)
        # 省略若干 ...
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output


class HNNLayer(nn.Module):
    """
    第一段: Hyperbolic neural networks layer.
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
    第一段: Hyperbolic graph convolution layer.
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


# =============================================================================
#  3) 动态超图相关的类(HConstructor, HGNN_conv, HGNN_classifier等)
# =============================================================================
class HConstructor(nn.Module):
    """
    动态超图构造器，用于生成或更新超边特征
    """

    def __init__(self, num_edges, f_dim, iters=1, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_edges = num_edges
        self.edges = None
        self.iters = iters
        self.eps = eps
        self.scale = f_dim ** -0.5

        self.edges_mu = nn.Parameter(torch.randn(1, f_dim))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))
        nn.init.xavier_uniform_(self.edges_logsigma)

        self.to_q = nn.Linear(f_dim, f_dim)
        self.to_k = nn.Linear(f_dim, f_dim)
        self.to_v = nn.Linear(f_dim, f_dim)

        self.gru = nn.GRUCell(f_dim, f_dim)
        hidden_dim = max(f_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(f_dim + f_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, f_dim)
        )

        self.norm_input = nn.LayerNorm(f_dim)
        self.norm_edgs = nn.LayerNorm(f_dim)
        self.norm_pre_ff = nn.LayerNorm(f_dim)

    def mask_attn(self, attn, k):
        """
        动态mask操作：只保留 topk 的注意力，模拟稀疏超图
        """
        indices = torch.topk(attn, k).indices
        mask = torch.zeros(attn.shape).bool().to(attn.device)
        for i in range(attn.shape[0]):
            mask[i][indices[i]] = 1
        return attn.mul(mask)

    def ajust_edges(self, s_level, args):
        """
        根据超图饱和度 s_level 动态调整边数
        """
        if args.stage != 'train':
            return
        if s_level > args.up_bound:
            self.num_edges = self.num_edges + 1
        elif s_level < args.low_bound:
            self.num_edges = self.num_edges - 1
            self.num_edges = max(self.num_edges, args.min_num_edges)
        else:
            return

    def forward(self, inputs, args):
        """
        inputs: [N, f_dim], 结点特征
        """
        n, d = inputs.shape
        device = inputs.device
        n_s = self.num_edges

        # 生成 / 重参数化 超边特征
        mu = self.edges_mu.expand(n_s, -1)
        sigma = self.edges_logsigma.exp().expand(n_s, -1)
        edges = mu + sigma * torch.randn(mu.shape, device=device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k = F.relu(k)
        v = F.relu(v)

        for _ in range(self.iters):
            edges = self.norm_edgs(edges)
            # 查询 Q
            q = self.to_q(edges)
            q = F.relu(q)

            # 与结点特征做注意力
            dots = torch.einsum('ni,mi->nm', q, k) * self.scale  # shape [n_s, N]
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)
            attn = self.mask_attn(attn, args.k_n)  # 动态mask

            # 超边特征更新
            updates = torch.einsum('nm,md->nd', attn, v)
            edges = torch.cat((edges, updates), dim=1)
            edges = self.mlp(edges)

            # 反向更新结点归属
            q2 = self.to_q(inputs)
            k2 = self.to_k(edges)
            k2 = F.relu(k2)
            v2 = F.relu(k2)  # 这里原本是一致用 v 吗？可根据需求改
            dots_v = torch.einsum('ni,mi->nm', q2, k2) * self.scale
            attn_v = dots_v.softmax(dim=1)
            attn_v = self.mask_attn(attn_v, args.k_e)
            H = attn_v

            # 计算边饱和度
            cc = H.ceil().abs()
            de = cc.sum(dim=0)
            empty = (de == 0).sum()
            s_level = 1 - empty / n_s
            self.ajust_edges(s_level, args)

            if args.verbose:
                print("Num edges is: {}; Saturation level is: {}".format(self.num_edges, s_level))

        self.edges = edges

        return edges, H, dots


class HGNN_conv(nn.Module):
    """
    HGNN卷积，对应自适应生成的超图做一次卷积聚合
    """

    def __init__(self, in_ft, out_ft, num_edges, bias=True):
        super(HGNN_conv, self).__init__()
        self.HConstructor = HConstructor(num_edges, in_ft)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, args):
        edges, H, H_raw = self.HConstructor(x, args)
        edges = edges.matmul(self.weight)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        x = x + nodes
        return x, H, H_raw


class HGNN_classifier(nn.Module):
    """
    动态超图网络上的示例分类器
    """

    def __init__(self, args, dropout=0.5):
        super(HGNN_classifier, self).__init__()
        in_dim = args.in_dim
        hid_dim = args.hid_dim
        out_dim = args.out_dim
        num_edges = args.num_edges
        self.conv_number = args.conv_number
        self.dropout = dropout

        # backbone
        self.linear_backbone = nn.ModuleList()
        self.linear_backbone.append(nn.Linear(in_dim, hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))

        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(in_dim, hid_dim))
        self.gcn_backbone.append(GCNConv(hid_dim, hid_dim))

        # HGNN conv
        self.convs = nn.ModuleList()
        self.transfers = nn.ModuleList()
        for i in range(self.conv_number):
            self.convs.append(HGNN_conv(hid_dim, hid_dim, num_edges))
            self.transfers.append(nn.Linear(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_number * hid_dim, out_dim),
        )

    def forward(self, data, args):
        # 这里 data 可能只是 x (shape: [N, in_dim]) 或者包含更多
        x = data
        # 根据 backbone 选择
        if args.backbone == 'linear':
            x = F.relu(self.linear_backbone[0](x))
            x = F.relu(self.linear_backbone[1](x))
            x = self.linear_backbone[2](x)
        elif args.backbone == 'gcn':
            # 需要 data={ 'fts':..., 'edge_index':... } 之类
            x = data['fts']
            edge_index = data['edge_index']
            x = self.gcn_backbone[0](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.gcn_backbone[1](x, edge_index)

        tmp = []
        H = []
        H_raw = []
        for i in range(self.conv_number):
            x, h, h_raw = self.convs[i](x, args)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            if args.transfer == 1:
                x = F.relu(self.transfers[i](x))
            tmp.append(x)
            H.append(h)
            H_raw.append(h_raw)

        x = torch.cat(tmp, dim=1)
        out = self.classifier(x)
        return out, x, H, H_raw


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


class DHGNN_conv(nn.Module):
    """
    动态超图卷积，与 HGNN_conv 类似，保留给用户做实验
    """

    def __init__(self, in_ft, out_ft, num_edges, bias=True):
        super(DHGNN_conv, self).__init__()
        self.HConstructor = HConstructor(num_edges, in_ft)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, args):
        edges, H, H_raw = self.HConstructor(x, args)
        edges = edges.matmul(self.weight)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        x = x + nodes
        return x, H, H_raw


# =============================================================================
#  4) 多头注意力 / 其他注意力模块（如 MobileViTv2Attention）可选保留
# =============================================================================
class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.W_q = nn.Linear(d_model, q * h)
        self.W_k = nn.Linear(d_model, q * h)
        self.W_v = nn.Linear(d_model, v * h)
        self.W_o = nn.Linear(v * h, d_model)
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
        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score,
                                torch.Tensor([-2 ** 32 + 1]).expand_as(score[0]).to(self.device))
        score = F.softmax(score, dim=-1)
        attention = torch.matmul(score, V)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        self_attention = self.W_o(attention_heads)
        return self_attention, self.score


# =============================================================================
#  5) 示例：HyperDHGNNConvolution / HyperbolicHGNNConv
#     （用户可结合自身需求进一步改写）
# =============================================================================

class HyperDHGNNConvolution(nn.Module):
    """
    用一个超图+双曲的聚合示例
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout,
                 act, use_bias, use_att, local_agg, args):
        super(HyperDHGNNConvolution, self).__init__()
        self.linear = HypLinearOld(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAggOld(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypActOld(manifold, c_in, c_out, act)
        self.args = args

        # 这里可以嵌入你想要的 Dynamic HGNN 部分
        self.classifier = HGNN_classifier(args)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        # 也可以插入注意力处理
        # h = ...
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)

        # 如果想用 classifier 做输出
        out, x_rep, H, H_raw = self.classifier(h, self.args)
        return h, adj


class HyperbolicHGNNConv(nn.Module):
    """
    类似示例
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out,
                 dropout, act, use_bias, use_att, local_agg, use_bn: bool = True):
        super(HyperbolicHGNNConv, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.use_att = use_att
        self.local_agg = local_agg
        self.dropout = dropout

        # 改用老版HypLinearOld或者 manifolds/layer.HypLinear 均可，需要注意API
        self.hyp_hid = HypLinearOld(manifold, in_features, 512, c_in, dropout=dropout, use_bias=use_bias)
        self.hyp_linear = HypLinearOld(manifold, 512, out_features, c_in, dropout=dropout, use_bias=use_bias)
        self.hyp_agg = HypAggOld(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypActOld(manifold, c_in, c_out, act)

        # 如果要BN，可以自行加
        # self.bn = nn.BatchNorm1d(512) if use_bn else None

    def forward(self, input):
        x, adj = input
        h = self.hyp_hid(x)
        h = self.hyp_agg(h, adj)
        h = self.hyp_act(h)
        h = F.dropout(h, self.dropout)
        h = self.hyp_linear(h)
        h = self.hyp_agg(h, adj)
        h = self.hyp_act(h)
        return h, adj


# =============================================================================
#  6) 下面放置【第二段】中的超曲面Transformer代码(TransConvLayer, TransConv, HypFormer)
#     注意我们需要 import manifolds.lorentz.Lorentz 等所需依赖
# =============================================================================

# ---- 如果已在别处 import，请在上面保留，这里省略 import 只贴核心类 ----

from manifolds.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization, HypCLS
from manifolds.lorentz import Lorentz


# from geoopt import ManifoldParameter   # 如果要用就保留


class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = args.attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.power_k = args.power_k
        self.trans_heads_concat = args.trans_heads_concat

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def full_attention(self, qs, ks, vs, output_attn=False):
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))  # [H, N, N]
        att_weight = att_weight / self.scale + self.bias  # [H, N, N]

        att_weight = nn.Softmax(dim=-1)(att_weight)
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)  # [N, H, D]
        att_output = att_output.transpose(0, 1)  # [N, H, D]
        att_output = self.manifold.mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        v = hyp_vs[..., 1:]
        phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)
        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))
        denominator = denominator.unsqueeze(-1)
        attn_output = numerator / (denominator + 1e-6)

        # 线性映射
        vss = self.v_map_mlp(v)
        attn_output = attn_output + vss

        # 多头聚合
        if self.trans_heads_concat:
            # 这里如果要 concat，需要定义 self.final_linear
            # Demo: self.final_linear = nn.Linear(self.num_heads*out_channels, out_channels)
            raise NotImplementedError("Please define self.final_linear for heads concat.")
        else:
            attn_output = attn_output.mean(dim=1)

        attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

        if output_attn:
            return attn_output, attn_output
        else:
            return attn_output

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)  # [N, H, D]
        key = torch.stack(k_list, dim=1)  # [N, H, D]
        value = torch.stack(v_list, dim=1)  # [N, H, D]

        if output_attn:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(query, key, value, output_attn)
            elif self.attention_type == 'full':
                attention_output, attn = self.full_attention(query, key, value, output_attn)
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(query, key, value)
            elif self.attention_type == 'full':
                attention_output = self.full_attention(query, key, value)
            else:
                raise NotImplementedError
            attn = None

        if output_attn:
            return attention_output, attn
        else:
            return attention_output


class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out,
                 in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True,
                 use_act=True, args=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第一层：Eucl -> Hyperbolic
        self.fcs.append(HypLinear(self.manifold_in, self.in_channels,
                                  self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in, self.in_channels,
                                             self.hidden_channels, self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device=args.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels,
                               num_heads=self.num_heads, use_weight=self.use_weight, args=args)
            )
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        # 最后一层
        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels,
                                  self.hidden_channels, self.manifold_out))

    def forward(self, x_input):
        layer_ = []
        # Eucl -> hyperbolic
        x = self.fcs[0](x_input, x_manifold='euc')

        # 是否加位置编码
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        # 叠加 TransConvLayer
        for i, conv in enumerate(self.convs):
            new_x = conv(x, x)  # 自注意力
            if self.residual:
                new_x = self.manifold_hidden.mid_point(torch.stack((new_x, layer_[i]), dim=1))
            if self.use_bn:
                new_x = self.bns[i + 1](new_x)
            x = new_x
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)


def aggregate(*args, **kwargs):
    """
    如果在HypFormer中需要graph聚合，可在此处实现
    """
    pass


class Dysformer(nn.Module):
    """
    超曲面Transformer主干
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5,
                 trans_use_bn=True, trans_use_residual=True, trans_use_weight=True,
                 trans_use_act=True, args=None):
        super().__init__()
        # 定义 manifold
        self.manifold_in = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.manifold_out = Lorentz(k=float(args.k_out))
        self.decoder_type = args.decoder_type

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # 构建 Transformer
        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out,
                                    in_channels, hidden_channels, trans_num_layers, trans_num_heads,
                                    trans_dropout, trans_use_bn, trans_use_residual,
                                    trans_use_weight, trans_use_act, args)

        # 如果需要图聚合的话，提供函数
        self.aggregate = aggregate

        # 根据解码类型设置decoder
        if self.decoder_type == 'euc':
            self.decode_trans = nn.Linear(self.hidden_channels, self.out_channels)
            self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.decoder_type == 'hyp':
            self.decode_graph = HypLinear(self.manifold_out,
                                          self.hidden_channels, self.hidden_channels)
            self.decode_trans = HypCLS(self.manifold_out,
                                       self.hidden_channels, self.out_channels)
        else:
            raise NotImplementedError

    def forward(self, x):
        # 1) 先经过超曲面transformer
        x1 = self.trans_conv(x)
        # 2) 再做decoder
        if self.decoder_type == 'euc':
            # 需要先把 hyperbolic -> Eucl
            x = self.decode_trans(self.manifold_out.logmap0(x1)[..., 1:])
        elif self.decoder_type == 'hyp':
            x = self.decode_trans(x1)
        else:
            raise NotImplementedError
        return x

    def get_attentions(self, x):
        return self.trans_conv.get_attentions(x)

    def reset_parameters(self):
        # 如果还有其他图卷积之类需要reset，可在此实现
        pass


# =============================================================================
#  7) 使用方式说明
# =============================================================================
"""
整合后，你可以：
1. 使用 HGNN_classifier / HGNN_conv 等类做你的动态超图网络。
2. 使用 HypFormer 做你的超曲面Transformer。
3. 或者在 HypFormer 里，把输入先丢给 HGNN_conv 得到新的特征后，再接 TransConv 做后续。
   例如：

   def forward(self, x):
       # x shape: [N, in_dim]
       x, _, _ = self.hgnn_conv(x, args)    # 先动态超图一把
       x1 = self.trans_conv(x)
       ...
       return x

根据项目需求自行改写即可。
"""

