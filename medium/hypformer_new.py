import math
import os
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv, GATConv

##############################################################################
#               1) 老版的超图 / Hyperbolic Layers  (同你原代码)
##############################################################################

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
    老版 Hyperbolic linear layer (Mobius mat-vec).
    改名为 HypLinearOld 以避免和 manifolds/layer.py 中的HypLinear冲突
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


class HypAggOld(Module):
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
        # self.att = DenseAtt(...)  # 如果需要可自行加

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
#        2) 动态超图模块: HConstructor, HGNN_conv, etc. (同你原代码)
##############################################################################

class HConstructor(nn.Module):
    """动态超图构造器"""

    def __init__(self, num_edges, f_dim, iters=1, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_edges = num_edges
        self.edges = None
        self.iters = iters
        self.eps = eps
        self.scale = f_dim ** -0.5

        # 3) 更小的初始 σ
        self.edges_mu = nn.Parameter(torch.randn(1, f_dim))
        self.edges_logsigma = nn.Parameter(torch.full((1, f_dim), -.0))

        self.to_q = nn.Linear(f_dim, f_dim)
        self.to_k = nn.Linear(f_dim, f_dim)
        self.to_v = nn.Linear(f_dim, f_dim)

        self.gru = nn.GRUCell(f_dim, f_dim)
        hidden_dim = max(f_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(f_dim + f_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, f_dim),
        )

        self.norm_input = nn.LayerNorm(f_dim)
        self.norm_edgs = nn.LayerNorm(f_dim)
        self.norm_pre_ff = nn.LayerNorm(f_dim)

    # ---------------- 改动 1：mask 后重新归一化 ----------------
    def mask_attn(self, attn: torch.Tensor, k: int, dim: int = 1):
        """保留每行前 k 个最大值，超出行长自动截断，并重新归一化。"""
        k = min(k, attn.size(dim))
        if k <= 0:
            return attn
        topk = torch.topk(attn, k, dim=dim)
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask.scatter_(dim, topk.indices, True)
        attn = attn.masked_fill(~mask, 0.0)
        return attn / (attn.sum(dim=dim, keepdim=True) + 1e-9)
    # ---------------- 改动 2：边数自适应冷启动 ----------------
    # ---------------- 边数自适应（加入冷启动硬门控） ----------------
    def _adjust_edges(self, s_level: float, args):
        """
        根据饱和度 s_level 调整 self.num_edges
        · 仅当 epoch ≥ args.edge_warm 且 args.use_dynamic == True 时才生效
        """
        cur_epoch   = getattr(args, "epoch", 0)
        edge_warm   = getattr(args, "edge_warm", 0)  # 冷启动轮数
        if cur_epoch < edge_warm or not getattr(args, "use_dynamic", True):
            return                                    # 冷启动阶段直接退出

        inc = 0
        if s_level > args.up_bound:                       # 饱和 → 加边
            self.num_edges += 1; inc = +1
        elif s_level < args.low_bound:                    # 稀疏 → 减边
            self.num_edges = max(self.num_edges - 1, args.min_num_edges)
            inc = -1
        else:
            return

        # 仅在“增边”时扩充参数
        if inc > 0 and self.num_edges > self.edges_mu.size(0):
            with torch.no_grad():
                self._expand_parameters()


    def _expand_parameters(self):
        cur_n, feat = self.edges_mu.shape
        new_rows = self.num_edges - cur_n
        if new_rows <= 0:
            return
        mu_last = self.edges_mu.data[-1:].repeat(new_rows, 1)
        ls_last = self.edges_logsigma.data[-1:].repeat(new_rows, 1)
        self.edges_mu       = nn.Parameter(torch.cat([self.edges_mu.data,   mu_last], dim=0))
        self.edges_logsigma = nn.Parameter(torch.cat([self.edges_logsigma.data, ls_last], dim=0))

    def forward(self, inputs, args):
        """inputs: [N, f_dim]"""
        n, d = inputs.shape
        device = inputs.device
        n_s = self.num_edges

        mu = self.edges_mu[: n_s]
        sigma = self.edges_logsigma.exp()[: n_s]
        edges = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k, v = F.relu(k), F.relu(v)

        for _ in range(self.iters):
            edges = self.norm_edgs(edges)
            q = F.relu(self.to_q(edges))

            dots = torch.einsum("ni,mi->nm", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = self.mask_attn(attn, args.k_n)  # 已在内部归一化

            updates = torch.einsum("nm,md->nd", attn, v)
            edges = torch.cat((edges, updates), dim=1)
            edges = self.mlp(edges)

            # ----- 反向注意力，用于计算结点→超边 H -----
            q2 = self.to_q(inputs)
            k2 = F.relu(self.to_k(edges))
            dots_v = torch.einsum("ni,mi->nm", q2, k2) * self.scale
            attn_v = self.mask_attn(dots_v.softmax(dim=1), args.k_e)
            H = attn_v

            # 饱和度
            de = (H.ceil().abs()).sum(dim=0)
            empty = (de == 0).sum()
            s_level = 1 - empty / n_s
            self._adjust_edges(s_level, args)

        self.edges = edges
        return edges, H, dots


class HGNN_conv(nn.Module):
    """带度归一化的 HGNN 卷积"""

    def __init__(self, in_ft, out_ft, num_edges, bias=True):
        super().__init__()
        self.HConstructor = HConstructor(num_edges, in_ft)
        self.linear_in = nn.Linear(in_ft, out_ft, bias=False)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.linear_in.weight)

    def _degree_norm(self, H: torch.Tensor, eps: float = 1e-9):
        """对称归一化 D_v^{-1/2} H D_e^{-1}，始终返回 (N,E)。"""
        Dv = H.sum(dim=1, keepdim=True) + eps   # (N,1)
        De = H.sum(dim=0, keepdim=True) + eps   # (1,E)
        H_norm = H / De                         # 右乘 D_e^{-1}
        return H_norm / Dv.sqrt()               # 左乘 D_v^{-1/2}

    def forward(self, x, args):
        edges, H, H_raw = self.HConstructor(x, args)

        # 超边特征
        edges = edges @ self.weight
        if self.bias is not None:
            edges = edges + self.bias

        # 度归一化信息流
        H_norm = self._degree_norm(H)
        nodes = H_norm @ edges

        # 残差 + 映射
        x_out = self.linear_in(x) + nodes
        return x_out, H, H_raw



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
        # data 可能是 x or dict
        x = data
        if args.backbone == 'linear':
            x = F.relu(self.linear_backbone[0](x))
            x = F.relu(self.linear_backbone[1](x))
            x = self.linear_backbone[2](x)
        elif args.backbone == 'gcn':
            x = data['fts']
            edge_index = data['edge_index']
            x = self.gcn_backbone[0](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.gcn_backbone[1](x, edge_index)

        tmp = []
        H_list, H_raw_list = [], []
        for i in range(self.conv_number):
            x, H, h_raw = self.convs[i](x, args)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            if args.transfer == 1:
                x = F.relu(self.transfers[i](x))
            tmp.append(x)
            H_list.append(H)
            H_raw_list.append(h_raw)

        x_cat = torch.cat(tmp, dim=1)
        out = self.classifier(x_cat)
        return out, x_cat, H_list, H_raw_list


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
    动态超图卷积，与 HGNN_conv 类似
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
        edges = edges @ self.weight
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H @ edges
        x = x + nodes
        return x, H, H_raw


##############################################################################
#   3) 多头注意力 / MobileViTv2Attention  (同你原)
##############################################################################

class MultiHeadAttention(Module):
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
#  4) 改造后的TransConvLayer, TransConv (保留Minkowski维), HypFormer
##############################################################################

# 如果还没import
# from manifolds.hyp_layer import HypLinear, HypLayerNorm, ...
# from manifolds.lorentz import Lorentz

class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels,
                 num_heads, use_weight=True, args=None):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels    # 这里可以是 Minkowski维(例如 257)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = args.attention_type

        # 多头 Q,K
        from manifolds.hyp_layer import HypLinear  # 根据你的项目结构
        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for _ in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for _ in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(self.in_channels, self.out_channels, bias=True)
        self.power_k = args.power_k
        self.trans_heads_concat = args.trans_heads_concat

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def full_attention(self, qs, ks, vs, output_attn=False):
        # [N,H,D] => cinner => [H,N,N]
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0,1), ks.transpose(0,1))
        att_weight = att_weight / self.scale + self.bias
        att_weight = nn.Softmax(dim=-1)(att_weight)
        att_output = self.manifold.mid_point(vs.transpose(0,1), att_weight)  # => [N,H,D]
        att_output = att_output.transpose(0,1)
        att_output = self.manifold.mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
        # Minkowski => shape [N,H,D],  D = out_channels+1 if Minkowski
        # 只取后 D-1 维
        qs = hyp_qs[...,1:]
        ks = hyp_ks[...,1:]
        v  = hyp_vs[...,1:]

        phi_qs = (F.relu(qs)+1e-6)/(self.norm_scale.abs()+1e-6)
        phi_ks = (F.relu(ks)+1e-6)/(self.norm_scale.abs()+1e-6)

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks,v)
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs,k_transpose_v)
        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))
        denominator = denominator.unsqueeze(-1)
        attn_output = numerator/(denominator+1e-6)

        vss = self.v_map_mlp(v)
        attn_output = attn_output + vss

        if self.trans_heads_concat:
            raise NotImplementedError("Concat not implemented yet.")
        else:
            attn_output = attn_output.mean(dim=1)

        # Minkowski time: [N, D-1] => => [N,D]
        attn_output_time = ((attn_output**2).sum(dim=-1,keepdims=True)+self.manifold.k)**0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

        if output_attn:
            return attn_output, attn_output
        else:
            return attn_output

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None,
                output_attn=False):
        # query,key,value
        q_list, k_list, v_list = [], [], []
        for _ in range(self.num_heads):
            q_list.append(self.Wq[_](query_input))
            k_list.append(self.Wk[_](source_input))
            if self.use_weight:
                v_list.append(self.Wv[_](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)
        key   = torch.stack(k_list, dim=1)
        value = torch.stack(v_list, dim=1)

        if output_attn:
            if self.attention_type=='linear_focused':
                attention_output, attn = self.linear_focus_attention(query, key, value, True)
            elif self.attention_type=='full':
                attention_output, attn = self.full_attention(query, key, value, True)
            else:
                raise NotImplementedError
        else:
            if self.attention_type=='linear_focused':
                attention_output = self.linear_focus_attention(query, key, value, False)
                attn = None
            elif self.attention_type=='full':
                attention_output = self.full_attention(query, key, value, False)
                attn = None
            else:
                raise NotImplementedError

        if output_attn:
            return attention_output, attn
        else:
            return attention_output


class TransConv(nn.Module):
    """
    改造后的TransConv, 允许 Minkowski维 (in_channels+1).
    """
    def __init__(self, manifold_in, manifold_hidden, manifold_out,
                 in_channels, hidden_channels, num_layers=1, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True,
                 use_act=True, args=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.in_channels = in_channels       # 例如 256
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight

        # Modules
        from manifolds.hyp_layer import HypLinear, HypLayerNorm, HypActivation, HypDropout

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第一层： HypLinear( in_features = in_channels )
        # 如果 Minkowski => in_channels+1,
        # 你可以自定义.
        # 下例：假设外部已把 x_input=[N, in_channels+1].
        self.fcs.append(HypLinear(self.manifold_in,
                                  self.in_channels,
                                  self.hidden_channels,
                                  self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in,
                                             self.in_channels,
                                             self.hidden_channels,
                                             self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device=args.device)

        # 多层 conv
        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden,
                               self.hidden_channels,
                               self.hidden_channels,
                               num_heads=self.num_heads,
                               use_weight=self.use_weight,
                               args=args)
            )
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        # 最后一层
        self.fcs.append(HypLinear(self.manifold_hidden,
                                  self.hidden_channels,
                                  self.hidden_channels,
                                  self.manifold_out))

    def forward(self, x_input):
        layer_ = []

        # ========== 如果 Minkowski => x_input.shape[1] == in_channels+1, 继续即可 ==========
        # 如果 x_input=[N, in_channels], 也直接用.
        # ---------- 第1层：Eucl->Hyp -----------
        x = self.fcs[0](x_input, x_manifold='euc')

        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon*x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        # ---------- 多层 TransConvLayer -----------
        for i, conv in enumerate(self.convs):
            new_x = conv(x, x)
            if self.residual:
                new_x = self.manifold_hidden.mid_point(torch.stack((new_x, layer_[i]), dim=1))
            if self.use_bn:
                new_x = self.bns[i+1](new_x)
            x = new_x
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x):
        # 如果要输出注意力
        layer_, attentions = [], []
        # 第1层
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = F.relu(x)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i+1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)


def aggregate(*args, **kwargs):
    pass
def euc2lorentz(x_euc: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """
    将欧氏向量映射到 k‑Lorentz 流形的原点邻域。
    输入  : (N, D)
    输出  : (N, D+1)  —  [time | space]
    """
    x2 = (x_euc ** 2).sum(dim=-1, keepdim=True)              # ‖x‖²
    time = torch.sqrt(x2 + k)                                # √(‖x‖² + k)
    return torch.cat([time, x_euc], dim=-1)                  # (N, D+1)


##############################################################################
#          5) HypFormer, 保留Minkowski维度, 只在最后才解码
##############################################################################

from manifolds.lorentz import Lorentz
from manifolds.hyp_layer import HypLinear, HypCLS, HypLayerNorm, HypActivation, HypDropout
from gnns import GraphConv, GCN  # 如果有


# -*- coding: utf-8 -*-
"""
Dysformer (稳定版)
-----------------
• 保留 Lorentz/Minkowski 坐标
• epoch < edge_warm 时关闭动态超图（use_dhyper=False，dh_weight=0）
• edge_warm ≤ epoch < edge_warm+dh_ramp 时线性爬坡 dh_weight
"""


from manifolds.hyp_layer import TrainableLorentz


# ====================== hypformer_new.py ======================
class Dysformer(nn.Module):
    """
    超曲面 Transformer 主干（Lorentz 流形）
    支持：
      • Transformer 分支（TransConv）
      • 可选 GraphConv 分支
      • 可选动态超图分支 (HGNN_conv)
    全部在超曲面上解码。
    """
    # -------------------------------
    def __init__(self, args):
        super().__init__()
        self.args = args

        # --------- manifolds ---------
        # --------- manifolds ---------
        self.manifold_in = TrainableLorentz(c_init=args.k_in, c_max=args.c_max)
        self.manifold_hidden = TrainableLorentz(c_init=args.k_hidden, c_max=args.c_max)
        self.manifold_out = TrainableLorentz(c_init=args.k_out, c_max=args.c_max)

        # --------- channels ----------
        self.in_channels     = args.in_channels      # Euclidean feat dim
        self.hidden_channels = args.hidden_channels  # Hyperbolic hidden dim
        self.out_channels    = args.out_channels     # 类别数

        # --------- 分支开关 ----------
        self.use_graph   = args.use_graph
        self.graph_weight= args.graph_weight
        self.use_dhyper  = args.use_dhyper          # 动态超图
        self.dh_weight   = args.dh_weight

        # --------- 输入映射 ----------
        self.input_map = nn.Linear(self.in_channels,
                                   self.hidden_channels,
                                   bias=False)

        # --------- Transformer 主分支 ----------
        self.trans_conv = TransConv(
            manifold_in     = self.manifold_in,
            manifold_hidden = self.manifold_hidden,
            manifold_out    = self.manifold_out,
            in_channels     = self.hidden_channels,
            hidden_channels = self.hidden_channels,
            args            = args
        )

        # --------- Graph 分支（可选） ----------
        self.graph_conv = GraphConv(self.hidden_channels,
                                    self.hidden_channels,
                                    args=args) if self.use_graph else None

        # --------- 动态超图分支（可选） ----------
        self.dhgnn_conv = HGNN_conv(self.hidden_channels,
                                    self.hidden_channels,
                                    args.num_edges) if self.use_dhyper else None

        # --------- 超曲面解码器 ----------
        # Transformer logits
        self.decode_trans = HypCLS(
            self.manifold_out,
            self.hidden_channels,
            self.out_channels
        )
        # Graph logits （同样在 Lorentz 上解码）
        self.decode_graph = HypCLS(
            self.manifold_out,
            self.hidden_channels,
            self.out_channels
        )

    # -------------------------------
    def forward(self, dataset, *, epoch: int = 0):
        """
        前向计算：
          - 只有在 use_graph=True 且 dataset.graph['edge_index'] 不为 None 时才使用图分支
          - 动态超图和 Transformer 分支按需融合
        """
        node_feat = dataset.graph['node_feat']  # (N, F)
        device = node_feat.device

        # --------- 安全获取 edge_index ----------
        if self.use_graph and 'edge_index' in dataset.graph and dataset.graph['edge_index'] is not None:
            edge_index = dataset.graph['edge_index'].to(device)
        else:
            edge_index = None

        # --------- 动态超图分支（Cold‑start 由 train_epoch 设置）----------
        x_euc = self.input_map(node_feat)  # (N, H)
        if self.use_dhyper and self.dhgnn_conv is not None:
            x_dh, _, _ = self.dhgnn_conv(x_euc, args=self.args)
            x_euc = (1.0 - self.dh_weight) * x_euc + self.dh_weight * x_dh

        # --------- Transformer 分支 ----------
        x_tr = self.trans_conv(x_euc)  # (N, H+1)

        # --------- Graph 分支（可选） ----------
        if edge_index is not None and self.graph_conv is not None:
            x_gc_euc = self.graph_conv(x_euc, edge_index)  # (N, H)
            x_gc_hyp = euc2lorentz(x_gc_euc, k=self.manifold_out.k)
        else:
            x_gc_hyp = None

        # --------- 超曲面解码 ----------
        tr_logits = self.decode_trans(x_tr)  # (N, C)
        if x_gc_hyp is not None:
            gc_logits = self.decode_graph(x_gc_hyp)  # (N, C)
            out = (1.0 - self.graph_weight) * tr_logits \
                  + self.graph_weight * gc_logits
        else:
            out = tr_logits

        return out

    # -------------------------------
    def get_attentions(self, x_euc):
        """返回 Transformer 中间层注意力 (L, N, N)"""
        return self.trans_conv.get_attentions(x_euc)

    # -------------------------------
    def reset_parameters(self):
        # 需要时可补充
        pass