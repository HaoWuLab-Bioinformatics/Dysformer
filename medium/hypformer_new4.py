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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from manifolds.hyp_layer import TrainableLorentz, HypCLS
from gnns import GraphConv


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

class SwiGLU(nn.Module):
    """SwiGLU ≈ LLaMA‐2 激活"""
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid_dim * 2)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return F.silu(a) * b

class HyperbolicGraphWavelet(nn.Module):
    """
    Hyperbolic Graph Wavelet Transform (Spectral Stream).
    Extracts local high-frequency information.
    """

    def __init__(self, in_dim, out_dim, order=3, tau=1.0):
        super().__init__()
        self.order = order
        self.tau = tau
        self.coeffs = nn.Parameter(torch.Tensor(order + 1))
        nn.init.uniform_(self.coeffs, -0.1, 0.1)
        self.linear = nn.Linear(in_dim, out_dim)
        self.math = PoincareMath(c=1.0)

    def forward(self, x, c_in):
        # Update math with correct curvature tensor
        self.math = PoincareMath(c=c_in)

        # [cite_start]1. Build Laplacian (Simplified k-NN) [cite: 188-195]
        # Calculate pairwise distances
        dist_mat = self.math.dist(x.unsqueeze(0), x.unsqueeze(1))
        # Heat kernel
        adj = torch.exp(- (dist_mat ** 2) / self.tau)

        # Sparsify (Top-K)
        k = min(10, x.size(0))
        _, idx = torch.topk(adj, k=k, dim=-1)
        # Create mask
        mask = torch.zeros_like(adj).scatter_(-1, idx, 1.0)
        adj = adj * mask  # Out-of-place multiplication

        # Normalize
        deg = adj.sum(dim=1).clamp(min=1e-5).pow(-0.5)
        deg_mat = torch.diag(deg)
        L = torch.eye(x.size(0), device=x.device) - deg_mat @ adj @ deg_mat

        # 2. Chebyshev Approximation
        x_tan = self.math.logmap0(x)
        Lx = torch.matmul(L, x_tan)
        Tx_0, Tx_1 = x_tan, Lx

        out = self.coeffs[0] * Tx_0 + self.coeffs[1] * Tx_1

        for k in range(2, self.order + 1):
            Tx_2 = 2 * torch.matmul(L, Tx_1) - Tx_0
            # --- 关键修改 (Key Fix) ---
            # 原代码: out += self.coeffs[k] * Tx_2  (In-place error!)
            # 修改后: out = out + ...             (Safe new tensor)
            out = out + self.coeffs[k] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return self.linear(out)

class DysformerLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_edges):
        super().__init__()
        self.num_edges = num_edges

        # Spatial Stream
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.sat_gate = SaturationGate(phi=5.0)
        self.W_spatial = nn.Linear(in_dim, out_dim)

        # Spectral Stream
        self.wavelet = HyperbolicGraphWavelet(in_dim, out_dim)

        # Fusion
        self.ggu = GeometricGatingUnit(out_dim)

    def forward(self, x, c_in, c_out):
        # Pass curvature tensors directly
        math_in = PoincareMath(c=c_in)
        math_out = PoincareMath(c=c_out)

        # --- 1. Spatial Stream (Dynamic Hypergraph) [cite: 114] ---
        dist_mat = math_in.dist(x.unsqueeze(0), x.unsqueeze(1))
        attn_score = -self.beta * dist_mat + self.gamma  # [cite: 212]

        # Masking
        k = min(self.num_edges, x.size(0))
        _, nn_idx = torch.topk(dist_mat, k=k, dim=-1, largest=False)
        mask = torch.zeros_like(attn_score).scatter_(-1, nn_idx, 1.0)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)

        # Dynamic Structure
        H = self.sat_gate(attn_score)

        # Aggregation
        he_feats = math_in.einstein_midpoint(x, H)

        # Back Projection
        he_tan = math_in.logmap0(he_feats)
        he_trans = self.W_spatial(he_tan)
        spatial_out_tan = torch.matmul(H, he_trans)

        # --- 2. Spectral Stream (Wavelet) [cite: 185] ---
        spectral_out_tan = self.wavelet(x, c_in)

        # --- 3. Fusion [cite: 250] ---
        fused_tan = self.ggu(spatial_out_tan, spectral_out_tan)

        return math_out.expmap0(fused_tan)

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
        # ——【时间维支持】——————————————
        restored_shape = None
        if x.dim() == 3:
            T, N, D = x.shape
            x = x.reshape(T * N, D)
            restored_shape = (T, N)

        edges, H, H_raw = self.HConstructor(x, args)

        # 超边特征
        edges = edges @ self.weight
        if self.bias is not None:
            edges = edges + self.bias

        # ——【乘法前维度对齐】———————————
        E = min(H.size(1), edges.size(0))
        if H.size(1) != E:
            H = H[:, :E]
        if edges.size(0) != E:
            edges = edges[:E]

        # 度归一化信息流
        H_norm = self._degree_norm(H)
        nodes = H_norm @ edges

        # 残差 + 映射
        x_out = self.linear_in(x) + nodes

        # ——【还原时间维】——————————————
        if restored_shape is not None:
            T, N = restored_shape
            x_out = x_out.view(T, N, -1)

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
        # ——【时间维支持】——————————————
        # 接受形如 (T, N, D) 或 (N, D)
        restored_shape = None
        if x.dim() == 3:
            T, N, D = x.shape
            x = x.reshape(T * N, D)
            restored_shape = (T, N)

        # 构图 + 超边特征
        edges, H, H_raw = self.HConstructor(x, args)
        edges = edges @ self.weight
        if self.bias is not None:
            edges = edges + self.bias

        # ——【乘法前维度对齐】———————————
        # 保证 H.shape[1] == edges.shape[0]
        E = min(H.size(1), edges.size(0))
        if H.size(1) != E:
            H = H[:, :E]
        if edges.size(0) != E:
            edges = edges[:E]

        # E→V
        nodes = H @ edges
        x = x + nodes

        # ——【还原时间维】——————————————
        if restored_shape is not None:
            T, N = restored_shape
            x = x.view(T, N, -1)

        return x, H, H_raw

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
        attn_output_time = torch.sqrt(
            torch.clamp((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k, min=1e-6))
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
class PoincareMath:
    """数值稳定的 Poincare Ball 基础算子。"""

    def __init__(self, c=1.0, eps: float = 1e-6):
        self.c = c if isinstance(c, torch.Tensor) else torch.tensor(float(c))
        self.eps = float(eps)

    def set_c(self, c):
        self.c = c if isinstance(c, torch.Tensor) else torch.tensor(float(c))

    def _sqrt_c(self):
        return torch.sqrt(torch.clamp(self.c, min=self.eps))

    def proj(self, x: torch.Tensor, safe_margin: float = 1e-3) -> torch.Tensor:
        sqrt_c = self._sqrt_c().to(x.device, x.dtype)
        maxnorm = (1.0 - safe_margin) / sqrt_c
        # 【修复】使用安全的 norm 计算：先截断，再开方
        norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True).clamp_min(self.eps))
        cond = norm > maxnorm
        scale = maxnorm / norm
        return torch.where(cond, x * scale, x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c.to(x.device, x.dtype) if isinstance(self.c, torch.Tensor) else torch.tensor(self.c, device=x.device, dtype=x.dtype)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
        out = num / den.clamp_min(self.eps)
        return self.proj(out)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        c = self.c.to(v.device, v.dtype) if isinstance(self.c, torch.Tensor) else torch.tensor(self.c, device=v.device, dtype=v.dtype)
        sqrt_c = torch.sqrt(torch.clamp(c, min=self.eps))
        # 【修复】
        v_norm = torch.sqrt((v * v).sum(dim=-1, keepdim=True).clamp_min(self.eps))
        out = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        return self.proj(out)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        y = self.proj(y)
        c = self.c.to(y.device, y.dtype) if isinstance(self.c, torch.Tensor) else torch.tensor(self.c, device=y.device, dtype=y.dtype)
        sqrt_c = torch.sqrt(torch.clamp(c, min=self.eps))
        # 【修复】
        y_norm = torch.sqrt((y * y).sum(dim=-1, keepdim=True).clamp_min(self.eps))
        arg = torch.clamp(sqrt_c * y_norm, min=0.0, max=1.0 - self.eps)
        return torch.atanh(arg) * y / (sqrt_c * y_norm)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = self.mobius_add(-x, y)
        c = self.c.to(diff.device, diff.dtype) if isinstance(self.c, torch.Tensor) else torch.tensor(self.c, device=diff.device, dtype=diff.dtype)
        sqrt_c = torch.sqrt(torch.clamp(c, min=self.eps))
        # 【修复】
        diff_norm = torch.sqrt((diff * diff).sum(dim=-1).clamp_min(self.eps))
        arg = torch.clamp(sqrt_c * diff_norm, min=0.0, max=1.0 - self.eps)
        return 2.0 * torch.atanh(arg) / sqrt_c

    def einstein_midpoint(self, x_ball: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x_ball = self.proj(x_ball)
        c = self.c.to(x_ball.device, x_ball.dtype) if isinstance(self.c, torch.Tensor) else torch.tensor(self.c, device=x_ball.device, dtype=x_ball.dtype)
        x2 = (x_ball * x_ball).sum(dim=-1, keepdim=True)

        x_klein = 2.0 * x_ball / (1.0 + c * x2).clamp_min(self.eps)
        gamma = (1.0 + c * x2) / (1.0 - c * x2).clamp_min(self.eps)

        w_gamma = weights.unsqueeze(-1) * gamma.unsqueeze(1)
        num = (w_gamma * x_klein.unsqueeze(1)).sum(dim=0)
        den = w_gamma.sum(dim=0).clamp_min(self.eps)
        midpoint_klein = num / den

        k2 = (midpoint_klein * midpoint_klein).sum(dim=-1, keepdim=True)
        max_k_norm = (1.0 - self.eps) / torch.sqrt(torch.clamp(c, min=self.eps))
        max_k2 = max_k_norm ** 2
        cond = k2 > max_k2
        if cond.any():
            midpoint_klein = torch.where(
                cond,
                # 【修复核心 Bug】先 clamp_min 再 sqrt！
                midpoint_klein * (max_k_norm / torch.sqrt(k2.clamp_min(self.eps))),
                midpoint_klein,
            )
            k2 = torch.where(cond, max_k2, k2)

        midpoint_ball = midpoint_klein / (1.0 + torch.sqrt((1.0 - c * k2).clamp_min(self.eps)))
        return self.proj(midpoint_ball)

class SaturationGate(nn.Module):
    """论文里 saturation gate 的一个可运行实现。"""

    def __init__(self, phi: float = 5.0, tau_mode: str = "mean"):
        super().__init__()
        self.phi = float(phi)
        self.tau_mode = str(tau_mode)

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        if self.tau_mode == "median":
            tau = score.median(dim=-1, keepdim=True).values
        else:
            tau = score.mean(dim=-1, keepdim=True)
        gate = torch.tanh(self.phi * F.relu(score - tau))
        prob = torch.sigmoid(score) * gate
        return prob

class GeometricGatingUnit(nn.Module):
    """空间流 / 频谱流的几何门控融合。"""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, h_spatial: torch.Tensor, h_spectral: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.gate(torch.cat([h_spatial, h_spectral], dim=-1)))
        return z * h_spatial + (1.0 - z) * h_spectral
class AcoshSafe(torch.autograd.Function):
    """
    带有梯度截断的安全反双曲余弦函数，专门用于防止双曲图神经网络中的梯度爆炸。
    """
    @staticmethod
    def forward(ctx, x, eps=1e-6):
        ctx.eps = eps
        # 限制最小值，防止前向计算时内部出现负数或 0 导致 NaN
        x_clamp = torch.clamp(x, min=1.0 + eps)
        ctx.save_for_backward(x_clamp)
        return torch.log(x_clamp + torch.sqrt(x_clamp - 1.0) * torch.sqrt(x_clamp + 1.0))

    @staticmethod
    def backward(ctx, grad_output):
        x_clamp, = ctx.saved_tensors
        # 反向传播时给分母加一个平滑项，防止除以 0
        denom = torch.sqrt(torch.clamp(x_clamp * x_clamp - 1.0, min=ctx.eps))
        grad_x = grad_output / denom
        # 硬截断梯度，如果还是爆炸，可以把 100.0 调小一点，比如 10.0
        grad_x = torch.clamp(grad_x, -100.0, 100.0)
        # 因为 forward 有两个参数 (x, eps)，所以 backward 需要返回两个梯度
        # eps 是常数，不需要梯度，所以返回 None
        return grad_x, None
class HConstructor(nn.Module):
    """
    Dysformer 风格动态双曲超图构造器。

    结构：
        节点自注意力 -> 几何亲和力 + saturation gate 构边
        -> Einstein midpoint 节点到超边聚合
        -> 超边自注意力
        -> HGWT 频谱流 + GGU 融合

    兼容性：
        return_node_feat=False  -> 返回 (edges, H, H_raw)
        return_node_feat=True   -> 返回 (x_out, edges, H, H_raw)
    """

    def __init__(
        self,
        num_edges: int,
        f_dim: int,
        iters: int = 1,
        eps: float = 1e-6,
        hidden_dim: int = 256,
        *,
        learnable_k: bool = True,
        k_init: float = 1.0,
        act: str = "gelu",
        adj_cache: bool = True,
        sat_beta: float = 5.0,
        sat_tau_mode: str = "mean",
    ):
        super().__init__()
        self.num_edges = int(num_edges)
        self.f_dim = int(f_dim)
        self.iters = int(iters)
        self.eps = float(eps)
        self._need_expand = False
        self._need_shrink = False

        self.edges_mu = nn.Parameter(torch.randn(max(1, self.num_edges), self.f_dim) * 0.02)
        self.edges_logsigma = nn.Parameter(torch.zeros(max(1, self.num_edges), self.f_dim))

        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer("k", torch.tensor(float(k_init)), persistent=False)

        self.pmath = PoincareMath(c=1.0, eps=self.eps)

        self.v_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.v_k = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.v_v = nn.Linear(self.f_dim, self.f_dim, bias=False)

        self.ve_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.ve_k = nn.Linear(self.f_dim, self.f_dim, bias=False)

        self.e_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.e_k = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.e_v = nn.Linear(self.f_dim, self.f_dim, bias=False)

        self.norm_x = nn.LayerNorm(self.f_dim)
        self.norm_e = nn.LayerNorm(self.f_dim)

        act = act.lower()
        if act == "relu":
            act_layer = nn.ReLU(inplace=False)
        else:
            act_layer = nn.GELU()

        hid = max(self.f_dim, int(hidden_dim))
        self.edge_update = nn.Sequential(
            nn.Linear(2 * self.f_dim, hid),
            act_layer,
            nn.Linear(hid, self.f_dim),
        )
        self.nodes_fuse = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.node_proj = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.ggu = GeometricGatingUnit(self.f_dim)

        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.sat_gate = SaturationGate(phi=sat_beta, tau_mode=sat_tau_mode)

        self.h = 4
        self._cache_adj_enabled = bool(adj_cache)
        self._cached_H_pattern = None
        self._cached_Av = None
        self._cached_Ae = None

    @staticmethod
    def _get(args, name, default):
        return getattr(args, name, default)

    def _positive_k(self) -> torch.Tensor:
        return F.softplus(self.k) + self.eps

    def mask_attn(self, attn: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
        if k <= 0:
            return attn
        k = min(k, attn.size(dim))
        topk = torch.topk(attn, k, dim=dim)
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask = mask.scatter(dim, topk.indices, True)
        attn = attn.masked_fill(~mask, 0.0)
        return attn / (attn.sum(dim=dim, keepdim=True) + 1e-9)

    def _mark_expand(self):
        self._need_expand = True

    def _mark_shrink(self):
        self._need_shrink = True

    @torch.no_grad()
    def _expand_parameters(self):
        cur_n = self.edges_mu.size(0)
        if self.num_edges <= cur_n:
            self._need_expand = False
            return
        add_rows = self.num_edges - cur_n
        mu_pad = self.edges_mu[-1:].repeat(add_rows, 1).detach().clone()
        ls_pad = self.edges_logsigma[-1:].repeat(add_rows, 1).detach().clone()
        self.edges_mu = nn.Parameter(torch.cat([self.edges_mu.detach(), mu_pad], dim=0))
        self.edges_logsigma = nn.Parameter(torch.cat([self.edges_logsigma.detach(), ls_pad], dim=0))
        self._need_expand = False

    @torch.no_grad()
    def _shrink_parameters(self):
        cur_n = self.edges_mu.size(0)
        if self.num_edges >= cur_n:
            self._need_shrink = False
            return
        self.edges_mu = nn.Parameter(self.edges_mu[: self.num_edges].detach().clone())
        self.edges_logsigma = nn.Parameter(self.edges_logsigma[: self.num_edges].detach().clone())
        self._need_shrink = False

    def _adjust_edges(self, s_level: float, args) -> None:
        cur_epoch = self._get(args, "epoch", 0)
        edge_warm = self._get(args, "edge_warm", 0)
        if cur_epoch < edge_warm or not self._get(args, "use_dynamic", True):
            return
        up = float(self._get(args, "up_bound", 0.95))
        low = float(self._get(args, "low_bound", 0.60))
        min_e = int(self._get(args, "min_num_edges", 4))

        if s_level > up:
            self.num_edges += 1
            if self.num_edges > self.edges_mu.size(0):
                self._mark_expand()
        elif s_level < low:
            new_num = max(self.num_edges - 1, min_e)
            if new_num < self.num_edges:
                self.num_edges = new_num
                self._mark_shrink()


    @staticmethod
    def euc2lorentz(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
        # 【修复】增加 torch.clamp
        t = torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True) + k, min=1e-6))
        return torch.cat([t, x], dim=-1)

    def hyperbolic_score(self, Q: torch.Tensor, K: torch.Tensor, k: float) -> torch.Tensor:
        Hh, Nq, Dh = Q.shape
        Nk = K.shape[1]
        Ql = self.euc2lorentz(Q.reshape(-1, Dh), k).reshape(Hh, Nq, Dh + 1)
        Kl = self.euc2lorentz(K.reshape(-1, Dh), k).reshape(Hh, Nk, Dh + 1)

        tQ, sQ = Ql[..., :1], Ql[..., 1:]
        tK, sK = Kl[..., :1], Kl[..., 1:]

        # 计算洛伦兹内积
        lor = -(tQ @ tK.transpose(-1, -2)).squeeze(-3) + (sQ @ sK.transpose(-1, -2))

        cosh_d = -lor / max(float(k), self.eps)

        # 核心修改：使用 .apply() 调用全局的 AcoshSafe，并传入 self.eps
        d = AcoshSafe.apply(cosh_d, self.eps)

        return -(d ** 2)

    def _split_heads(self, x: torch.Tensor, h: int) -> torch.Tensor:
        return x.view(x.size(0), h, -1).transpose(0, 1).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1).contiguous().view(x.size(1), -1)

    @staticmethod
    def _norm_adj(A: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        deg = A.sum(dim=-1) + eps
        inv_sqrt = torch.pow(deg, -0.5)
        return (A * inv_sqrt.unsqueeze(-1)) * inv_sqrt.unsqueeze(-2)

    @staticmethod
    def _degree_norm(H: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        Dv = H.sum(dim=1, keepdim=True) + eps
        De = H.sum(dim=0, keepdim=True) + eps
        return (H / De) / torch.sqrt(Dv)

    def _heat_wavelet(
        self,
        A_norm: torch.Tensor,
        X: torch.Tensor,
        scales=(0.25, 0.5, 1.0),
        order: int = 3,
    ) -> torch.Tensor:
        I = torch.eye(A_norm.size(0), device=X.device, dtype=X.dtype)
        L = I - A_norm
        outs = []
        for s in scales:
            Y = X - s * (L @ X)
            if order >= 2:
                Y = Y + (s ** 2) / 2.0 * (L @ (L @ X))
            if order >= 3:
                Y = Y - (s ** 3) / 6.0 * (L @ (L @ (L @ X)))
            outs.append(Y)
        return torch.stack(outs, dim=-2).sum(dim=-2)

    def _memory_efficient_attn_2d(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        use_hyper: bool = False,
        k_curv: float = 1.0,
        k_chunk_size: int = 256,
        work_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        Hh, Nq, Dh = Q.shape
        Nk = K.shape[1]
        device = Q.device
        compute_dtype = torch.float32 if work_dtype in (torch.float16, torch.bfloat16) else work_dtype
        Qw, Kw, Vw = Q.to(compute_dtype), K.to(compute_dtype), V.to(compute_dtype)

        m = torch.full((Hh, Nq, 1), -float("inf"), dtype=compute_dtype, device=device)
        s = torch.zeros((Hh, Nq, 1), dtype=compute_dtype, device=device)
        out = torch.zeros((Hh, Nq, Dh), dtype=compute_dtype, device=device)

        ks = 0
        while ks < Nk:
            ke = min(ks + int(k_chunk_size), Nk)
            Kb, Vb = Kw[:, ks:ke], Vw[:, ks:ke]
            scores = (
                self.hyperbolic_score(Qw, Kb, k_curv)
                if use_hyper else
                torch.matmul(Qw, Kb.transpose(-1, -2)) / math.sqrt(Dh)
            )
            m_block = scores.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_block)
            exp_prev = torch.exp(m - m_new)
            s = s * exp_prev
            out = out * exp_prev
            exp_scores = torch.exp(scores - m_new)
            s = s + exp_scores.sum(dim=-1, keepdim=True)
            out = out + torch.matmul(exp_scores, Vb)
            m = m_new
            ks = ke

        return (out / (s + 1e-9)).to(Q.dtype)

    def forward(self, inputs: torch.Tensor, args, return_node_feat: bool = False):
        if self._need_expand:
            self._expand_parameters()
        if self._need_shrink:
            self._shrink_parameters()

        x = self.norm_x(inputs)
        N, D = x.shape
        heads = int(self._get(args, "attn_heads", 4))
        k_e = int(self._get(args, "k_e", 16))
        use_h = bool(self._get(args, "use_hyper", True))
        scales = self._get(args, "wavelet_scales", (0.25, 0.5, 1.0))
        order = int(self._get(args, "wavelet_order", 3))
        renorm_incidence = bool(self._get(args, "renorm_incidence", False))
        iters = int(self._get(args, "ss_iters", self.iters))

        k_curv = self._positive_k()
        k_val = k_curv.detach().item()
        self.pmath.set_c(k_curv)

        assert D % heads == 0, f"feature dim {D} must be divisible by heads {heads}"
        E = int(self.num_edges)
        sigma = self.edges_logsigma.exp()[:E]
        edges = self.edges_mu[:E] + sigma * torch.randn_like(sigma)

        H = None
        H_raw = None
        x_out = x

        for _ in range(iters):
            # 1) 节点自注意力
            Qv = self._split_heads(self.v_q(x_out), heads)
            Kv = self._split_heads(self.v_k(x_out), heads)
            Vv = self._split_heads(self.v_v(x_out), heads)
            x_sa = self.node_proj(
                self._merge_heads(
                    self._memory_efficient_attn_2d(Qv, Kv, Vv, use_hyper=use_h, k_curv=k_val)
                )
            ) + x_out

            # 2) V -> E: 几何亲和力 + saturation gate
            q_tan = self.ve_q(x_sa)
            e_tan = self.ve_k(self.norm_e(edges))
            q_ball = self.pmath.expmap0(q_tan)
            e_ball = self.pmath.expmap0(e_tan)
            dist_ne = self.pmath.dist(q_ball.unsqueeze(1), e_ball.unsqueeze(0))
            H_raw = -self.beta.abs() * dist_ne + self.gamma
            H_prob = self.sat_gate(H_raw)
            H = self.mask_attn(H_prob, k=k_e, dim=-1)
            if renorm_incidence:
                H = H / (H.sum(dim=-1, keepdim=True) + 1e-9)

            # 2.5) Einstein midpoint 聚合到超边
            e_from_v_ball = self.pmath.einstein_midpoint(q_ball, H)
            E_from_V = self.pmath.logmap0(e_from_v_ball)

            # 3) 超边自注意力
            norm_edges = self.norm_e(edges)
            Qe = self._split_heads(self.e_q(norm_edges), heads)
            Ke = self._split_heads(self.e_k(norm_edges), heads)
            Ve = self._split_heads(self.e_v(norm_edges), heads)
            E_self = self._merge_heads(
                self._memory_efficient_attn_2d(Qe, Ke, Ve, use_hyper=use_h, k_curv=k_val)
            )
            edges = self.norm_e(self.edge_update(torch.cat([E_from_V, E_self], dim=-1)) + edges)

            # 4) E -> V 空间流
            Hn = self._degree_norm(H)
            x_spatial = Hn @ edges

            # 5) HGWT 频谱流
            H_pattern = H > 0
            if (
                self._cache_adj_enabled
                and self._cached_H_pattern is not None
                and torch.equal(H_pattern, self._cached_H_pattern)
            ):
                Av, Ae = self._cached_Av, self._cached_Ae
            else:
                Av = self._norm_adj(H @ H.transpose(0, 1))
                Ae = self._norm_adj(H.transpose(0, 1) @ H)
                if self._cache_adj_enabled:
                    self._cached_H_pattern = H_pattern.detach().clone()
                    self._cached_Av = Av.detach().clone()
                    self._cached_Ae = Ae.detach().clone()

            x_spectral = self._heat_wavelet(Av, x_sa, scales=scales, order=order)
            e_spectral = self._heat_wavelet(Ae, edges, scales=scales, order=order)

            # 6) GGU 融合
            x_fused = self.ggu(x_spatial, x_spectral)
            x_out = self.norm_x(self.nodes_fuse(x_fused) + x_sa)
            edges = self.norm_e(edges + e_spectral)

            # 7) 动态调节超边数
            with torch.no_grad():
                empty = (H.sum(dim=0) == 0).sum()
                s_level = 1.0 - empty.float() / max(H.size(1), 1)
                self._adjust_edges(float(s_level), args)

            if self.num_edges > edges.size(0):
                pad_rows = self.num_edges - edges.size(0)
                pad = self.edges_mu[edges.size(0): edges.size(0) + pad_rows]
                edges = torch.cat([edges, pad.to(edges.dtype)], dim=0)

        if return_node_feat:
            return x_out, edges, H, H_raw
        return edges, H, H_raw

def euc2lorentz(x: torch.Tensor, k=1.0) -> torch.Tensor:
    if isinstance(k, torch.Tensor):
        k = k.to(x.device, x.dtype)
    else:
        k = torch.tensor(float(k), device=x.device, dtype=x.dtype)
    # 【修复】增加 torch.clamp，防止内部出现负数或 0
    t = torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True) + k, min=1e-6))
    return torch.cat([t, x], dim=-1)

class DysFormer(nn.Module):


    def __init__(self, args):
        super().__init__()
        self.args = args

        self.manifold_in = TrainableLorentz(c_init=args.k_in, c_max=args.c_max)
        self.manifold_hidden = TrainableLorentz(c_init=args.k_hidden, c_max=args.c_max)
        self.manifold_out = TrainableLorentz(c_init=args.k_out, c_max=args.c_max)

        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.out_channels

        self.use_graph = bool(getattr(args, "use_graph", False))
        self.use_dhyper = bool(getattr(args, "use_dhyper", True))
        self.ss_layers = int(getattr(args, "ss_layers", 1))

        self.input_map = nn.Linear(self.in_channels, self.hidden_channels, bias=False)
        self.input_norm = nn.LayerNorm(self.hidden_channels)

        self.ss_blocks = nn.ModuleList()
        if self.use_dhyper:
            for _ in range(self.ss_layers):
                self.ss_blocks.append(
                    HConstructor(
                        num_edges=getattr(args, "num_edges", 16),
                        f_dim=self.hidden_channels,
                        iters=getattr(args, "ss_iters", 1),
                        hidden_dim=max(self.hidden_channels, int(getattr(args, "ss_hidden_dim", self.hidden_channels))),
                        learnable_k=True,
                        k_init=float(getattr(args, "k_hidden", 1.0)),
                        act=getattr(args, "ss_act", "gelu"),
                        adj_cache=bool(getattr(args, "ss_adj_cache", True)),
                        sat_beta=float(getattr(args, "sat_beta", 5.0)),
                        sat_tau_mode=str(getattr(args, "sat_tau_mode", "mean")),
                    )
                )

        dh_init = float(getattr(args, "dh_weight", 0.5))
        dh_init = min(max(dh_init, 1e-4), 1.0 - 1e-4)
        self._w_dh_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(math.log(dh_init / (1.0 - dh_init)), dtype=torch.float32))
            for _ in range(self.ss_layers)
        ])

        self.trans_conv = TransConv(
            manifold_in=self.manifold_in,
            manifold_hidden=self.manifold_hidden,
            manifold_out=self.manifold_out,
            in_channels=self.hidden_channels,
            hidden_channels=self.hidden_channels,
            args=args,
        )

        self.graph_conv = GraphConv(self.hidden_channels, self.hidden_channels, args=args) if self.use_graph else None

        self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        self.decode_graph = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)

        graph_init = float(getattr(args, "graph_weight", 0.5))
        graph_init = min(max(graph_init, 1e-4), 1.0 - 1e-4)
        self._w_graph_logit = nn.Parameter(
            torch.tensor(math.log(graph_init / (1.0 - graph_init)), dtype=torch.float32)
        )

        self.last_H_list = []
        self.last_H_raw_list = []

    @staticmethod
    def _repeat_edge_index_over_time(edge_index, T: int, N: int, device, dtype=torch.long):
        row = edge_index[0].to(device=device, dtype=dtype)
        col = edge_index[1].to(device=device, dtype=dtype)
        offsets = (torch.arange(T, device=device, dtype=dtype).unsqueeze(1) * N)
        row_t = row.unsqueeze(0) + offsets
        col_t = col.unsqueeze(0) + offsets
        return torch.stack([row_t.reshape(-1), col_t.reshape(-1)], dim=0)

    @staticmethod
    def _sanitize_edge_index_for_size(edge_index, N_target: int):
        row, col = edge_index[0], edge_index[1]
        mask = (row >= 0) & (row < N_target) & (col >= 0) & (col < N_target)
        if mask.all():
            return edge_index, 1.0
        return torch.stack([row[mask], col[mask]], dim=0), mask.float().mean().item()

    def forward(self, dataset, *, epoch: int = 0):
        node_feat = dataset.graph["node_feat"]
        device = node_feat.device
        self.args.epoch = epoch

        x_euc = self.input_norm(self.input_map(node_feat))
        H_list, H_raw_list = [], []

        if self.use_dhyper and len(self.ss_blocks) > 0:
            for i, block in enumerate(self.ss_blocks):
                x_ss, _, H, H_raw = block(x_euc, self.args, return_node_feat=True)
                alpha = torch.sigmoid(self._w_dh_logits[i])
                x_euc = (1.0 - alpha) * x_euc + alpha * x_ss
                H_list.append(H)
                H_raw_list.append(H_raw)

        self.last_H_list = H_list
        self.last_H_raw_list = H_raw_list

        x_tr = self.trans_conv(x_euc)
        tr_logits = self.decode_trans(x_tr)

        edge_index = None
        if self.use_graph and "edge_index" in dataset.graph and dataset.graph["edge_index"] is not None:
            edge_index = dataset.graph["edge_index"].to(device)

        if edge_index is not None and self.graph_conv is not None:
            ei0 = edge_index.long()
            if x_euc.dim() == 2:
                ei_use, _ = self._sanitize_edge_index_for_size(ei0, x_euc.size(0))
                x_gc_euc = self.graph_conv(x_euc, ei_use)
                x_gc_hyp = euc2lorentz(x_gc_euc, k=self.manifold_out.k)
            elif x_euc.dim() == 3:
                T, N_step, Hdim = x_euc.shape
                x_gc_in = x_euc.reshape(T * N_step, Hdim)
                ei_time = self._repeat_edge_index_over_time(ei0, T=T, N=N_step, device=x_euc.device)
                ei_use, _ = self._sanitize_edge_index_for_size(ei_time, x_gc_in.size(0))
                x_gc_flat = self.graph_conv(x_gc_in, ei_use)
                x_gc_hyp = euc2lorentz(x_gc_flat, k=self.manifold_out.k)
            else:
                raise ValueError(f"x_euc should be 2D or 3D, got {tuple(x_euc.shape)}")

            gc_logits = self.decode_graph(x_gc_hyp)
            w_graph = torch.sigmoid(self._w_graph_logit)
            out = (1.0 - w_graph) * tr_logits + w_graph * gc_logits
        else:
            out = tr_logits

        if getattr(self.args, "return_emb", False):
            return out, x_tr, H_list, H_raw_list
        return out

    def get_attentions(self, x_euc: torch.Tensor):
        return self.trans_conv.get_attentions(x_euc)

    def get_fusion_weights(self):
        with torch.no_grad():
            return {
                "w_dh": [torch.sigmoid(w).item() for w in self._w_dh_logits],
                "w_graph": torch.sigmoid(self._w_graph_logit).item(),
                "k_ss": [F.softplus(block.k).item() for block in self.ss_blocks] if len(self.ss_blocks) > 0 else [],
            }
