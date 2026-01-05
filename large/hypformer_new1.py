import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from manifolds.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization, HypCLS
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
from gnns import GraphConv
import torch
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
###############################################################################
#                             HypLinear
###############################################################################

###############################################################################
#                          HypLayerNorm (示例)
###############################################################################
# 这里不修改 HypLayerNorm 的实现，只要在调用时，传入正确的维度即可。

###############################################################################
#                       超图构造器：HConstructor (双曲空间)
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class HConstructor(nn.Module):
    """
    将中间的 MLP 用普通的 nn.Linear(2*d->hidden->d)，
    对 Q/K/V 仍用 HypLinear(d->d)，这样不会出现时间维错位的问题。
    """

    def __init__(self, manifold, num_edges, f_dim, c, iters=1, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.manifold = manifold
        self.num_edges = num_edges
        self.edges = None
        self.iters = iters
        self.eps = eps
        self.c = c

        # 双曲空间下，每个点最终(d+1)维，但这里的 f_dim=100 是欧几里得部分
        self.in_features = f_dim + 1
        self.hidden_dim = hidden_dim
        self.scale = f_dim ** -0.5

        # LayerNorm 只对欧几里得部分做
        self.norm_input = HypLayerNorm(manifold, f_dim)
        self.norm_edges = HypLayerNorm(manifold, f_dim)

        # 超边初始化
        self.edges_mu = nn.Parameter(torch.randn(1, f_dim))  # (1, d)
        self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))
        init.xavier_uniform_(self.edges_logsigma)

        # Q/K/V：HypLinear(d->d)
        self.to_q = HypLinear(self.manifold, f_dim, f_dim)
        self.to_k = HypLinear(self.manifold, f_dim, f_dim)
        self.to_v = HypLinear(self.manifold, f_dim, f_dim)

        # MLP：纯欧几里得线性层 (2*d->hidden->d)
        self.mlp_euc = nn.Sequential(
            nn.Linear(2 * f_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, f_dim)
        )

    def mask_attn(self, attn, k=10):
        """仅保留每行 topk 的注意力，其余清零"""
        indices = torch.topk(attn, k).indices
        mask = torch.zeros_like(attn, dtype=torch.bool)
        for i in range(attn.shape[0]):
            mask[i][indices[i]] = True
        return attn * mask

    def adjust_edges(self, s_level, args):
        """根据饱和度动态增减超边数量，也可根据需要省略"""
        if s_level > 0.95:
            self.num_edges += 1
        elif s_level < 0.9:
            self.num_edges -= 1
            self.num_edges = max(self.num_edges, 64)

    def forward(self, inputs, args):
        """
        inputs: (N, f_dim) 的欧几里得输入
        输出:    edges, H, dots
        """
        device = inputs.device
        n, d = inputs.shape  # d=100

        # 1) 拼时间维 => (N, d+1)，再 expmap0
        inputs_time = torch.sqrt((inputs ** 2).sum(dim=-1, keepdim=True) + self.manifold.k)
        inputs_hyp = torch.cat([inputs_time, inputs], dim=-1)  # (N, d+1)
        inputs_hyp = self.manifold.expmap0(inputs_hyp)        # (N, d+1)

        # 没有 edges 就随机初始化
        if self.edges is None:
            mu = self.edges_mu.expand(self.num_edges, -1)   # (num_edges, d)
            sigma = self.edges_logsigma.exp().expand(self.num_edges, -1)
            e_euc = mu + sigma * torch.randn(mu.shape, device=device)
            e_time = torch.sqrt((e_euc**2).sum(dim=-1, keepdim=True) + self.manifold.k)
            edges = torch.cat([e_time, e_euc], dim=-1)
            edges = self.manifold.expmap0(edges)
        else:
            edges = self.edges

        # LayerNorm 只对欧几里得部分
        inputs_hyp = self.norm_input(inputs_hyp)
        edges = self.norm_edges(edges)

        # 2) 计算 K/V (只对欧几里得部分线性)
        k = self.to_k(inputs, x_manifold='euc')  # => (N, d+1)
        v = self.to_v(inputs, x_manifold='euc')
        k_euc = torch.tanh(k[..., 1:])  # => (N, d)
        v_euc = torch.tanh(v[..., 1:])

        # 多次迭代更新 edges
        for _ in range(self.iters):
            # Q
            q = self.to_q(edges[..., 1:], x_manifold='euc')  # => (num_edges, d+1)
            q_euc = torch.tanh(q[..., 1:])                   # => (num_edges, d)

            # (E, d) * (N, d) => (E, N) 点积
            dots = torch.einsum('ed,nd->en', q_euc, k_euc) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)
            attn = self.mask_attn(attn, k=10)

            # 加权聚合
            updates = torch.einsum('en,nd->ed', attn, v_euc)  # => (E, d)

            # 拼接 (edges[...,1:], updates) => (E, 2d)
            cat_euc = torch.cat([edges[..., 1:], updates], dim=-1)
            new_euc = self.mlp_euc(cat_euc)  # => (E, d)

            # 拼回时间维
            new_time = torch.sqrt((new_euc**2).sum(dim=-1, keepdim=True) + self.manifold.k)
            new_edges = torch.cat([new_time, new_euc], dim=-1)  # => (E, d+1)
            edges = new_edges  # 如果想再 expmap0，可以加：edges = self.manifold.expmap0(new_edges)

            # ===== 动态调整超边(如果要禁用可直接删) =====
            # 重新算 attention
            k2 = self.to_k(edges[..., 1:], x_manifold='euc')    # => (E, d+1)
            k2_euc = torch.tanh(k2[..., 1:])
            # 这里和之前的 bug 不同：对 inputs 直接传入
            q2 = self.to_q(inputs, x_manifold='euc')            # => (N, d+1)
            q2_euc = torch.tanh(q2[..., 1:])                    # => (N, d)

            dots2 = torch.einsum('nd,ed->ne', q2_euc, k2_euc) * self.scale
            attn_v = dots2.softmax(dim=1)
            attn_v = self.mask_attn(attn_v, k=10)
            H = attn_v  # => (N, E)

            # 饱和度
            cc = H.ceil().abs()
            de = cc.sum(dim=0)
            empty = (de == 0).sum()  # 多少条超边为空
            s_level = 1 - empty / self.num_edges
            self.adjust_edges(s_level, args)
            print(f"Num edges: {self.num_edges}; Saturation: {s_level}")

        self.edges = edges
        # dots 仅在第一次算注意力时产生，如果你想返回，也可以直接返回 None
        return edges, H, dots


###############################################################################
#               （保持不变）TransConvLayer 与 TransConv
###############################################################################
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

        if self.trans_heads_concat:
            self.final_linear = nn.Linear(out_channels * self.num_heads, out_channels, bias=True)

    def full_attention(self, qs, ks, vs, output_attn=False):
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))  # [H, N, N]
        att_weight = att_weight / self.scale + self.bias
        att_weight = nn.Softmax(dim=-1)(att_weight)  # [H, N, N]
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)  # [N, H, D]
        att_output = att_output.transpose(0, 1)  # [N, H, D]
        att_output = self.manifold.mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        v = hyp_vs[..., 1:]
        phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)  # [H, D, D]
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, D]
        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks)).unsqueeze(-1)
        attn_output = numerator / (denominator + 1e-6)

        vss = self.v_map_mlp(v)
        attn_output = attn_output + vss

        if self.trans_heads_concat:
            attn_output = self.final_linear(attn_output.reshape(-1, self.num_heads * self.out_channels))
        else:
            attn_output = attn_output.mean(dim=1)

        attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

        if output_attn:
            attention = torch.einsum('nhd,mhd->nmh', phi_qs, phi_ks)
            attention = attention / (denominator.unsqueeze(1) + 1e-6)
            attention = attention.mean(dim=-1)  # 平均多个头
            return attn_output, attention
        else:
            return attn_output

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        q_list, k_list, v_list = [], [], []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)  # [N, H, D+1]
        key = torch.stack(k_list, dim=1)    # [N, H, D+1]
        value = torch.stack(v_list, dim=1) # [N, H, D+1]

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

        if output_attn:
            return attention_output, attn
        else:
            return attention_output


class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, hidden_channels,
                 num_layers=2, num_heads=1, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_act=True, args=None):
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

        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in, self.in_channels, self.hidden_channels,
                                             self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device=args.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels,
                               num_heads=self.num_heads, use_weight=self.use_weight, args=args))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels, self.hidden_channels, self.manifold_out))

    def forward(self, x_input):
        layer_ = []

        # Euclidean -> 第一次线性变换
        x = self.fcs[0](x_input, x_manifold='euc')
        # 加位置编码
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        # 多层 TransConv
        for i, conv in enumerate(self.convs):
            x_new = conv(x, x)
            if self.residual:
                x_new = self.manifold_hidden.mid_point(torch.stack((x_new, layer_[i]), dim=1))
            if self.use_bn:
                x_new = self.bns[i + 1](x_new)
            x = x_new
            layer_.append(x_new)

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
        return torch.stack(attentions, dim=0)  # [num_layers, N, N]


###############################################################################
#                       一个占位的GNN类 (GraphConv)
###############################################################################
class GraphConv(nn.Module):
    """仅做演示占位。可以换成你实际的 GNN 代码。"""
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers=1,
                 dropout=0.5,
                 use_bn=True,
                 use_residual=True,
                 use_weight=True,
                 use_init=True,
                 use_act=True):
        super().__init__()
        pass

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index):
        # 简化处理: 返回 x
        return x


###############################################################################
#                       HypFormer 主体
###############################################################################
class Dysformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5,
                 trans_use_bn=True, trans_use_residual=True, trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.5, aggregate='add',
                 args=None):
        super().__init__()
        from manifolds.lorentz import Lorentz

        self.manifold_in = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.manifold_out = Lorentz(k=float(args.k_out))

        self.in_channels = in_channels          # 100
        self.hidden_channels = hidden_channels  # 32
        self.out_channels = out_channels
        self.decoder_type = args.decoder_type
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        # Transformer 分支
        self.trans_conv = TransConv(
            self.manifold_in, self.manifold_hidden, self.manifold_out,
            in_channels, hidden_channels,
            trans_num_layers, trans_num_heads,
            trans_dropout, trans_use_bn, trans_use_residual,
            trans_use_weight, trans_use_act, args
        )

        # GNN 分支（占位）
        self.graph_conv = GraphConv(
            in_channels, hidden_channels,
            gnn_num_layers, gnn_dropout,
            gnn_use_bn, gnn_use_residual,
            gnn_use_weight, gnn_use_init, gnn_use_act
        )

        # 在 GNN 之后，把 (N,100) -> (N,32)
        self.gnn_fc = nn.Linear(in_channels, hidden_channels, bias=True)

        # 解码器
        # 注意: HypLinear(..., hidden_channels, hidden_channels)
        #       实际上需要输入 (N, hidden_channels+1)，因此我们会在 forward 里手动拼时间维
        if self.decoder_type == 'euc':
            self.decode_trans = nn.Linear(hidden_channels, out_channels)
            self.decode_graph = nn.Linear(hidden_channels, out_channels)
        elif self.decoder_type == 'hyp':
            self.decode_graph = HypLinear(self.manifold_out, hidden_channels, hidden_channels)
            self.decode_trans = HypCLS(self.manifold_out, hidden_channels, out_channels)
        else:
            raise NotImplementedError

        # 动态超图构造器(双曲空间版本)
        self.hyper_constructor = HConstructor(
            manifold=self.manifold_in,
            num_edges=64,
            f_dim=in_channels,  # 输入欧几里得维度=100
            c=float(args.k_in),
            iters=1,
            eps=1e-8,
            hidden_dim=128
        )

    def forward(self, x, edge_index):
        # 1) 构造动态超图 => 返回 edges, H, ...
        edges, H, dots = self.hyper_constructor(x, args=None)
        edge_index_dynamic = torch.nonzero(H > 0, as_tuple=False).T  # (2,E_new)

        # 2) Transformer 分支 => (N, hidden_channels+1)
        x1 = self.trans_conv(x)

        # 3) 可选 GNN 分支
        if self.use_graph:
            # (a) 原先 GraphConv: (N,100) -> (N,100)
            x2 = self.graph_conv(x, edge_index_dynamic)
            # (b) 先用 gnn_fc: (N,100) -> (N,32)
            x2_euc = self.gnn_fc(x2)  # => (N,32)
            # (c) 这一步极重要：decode_graph 是 HypLinear，需要输入 (N,32+1)
            x2_time = torch.sqrt((x2_euc ** 2).sum(dim=-1, keepdim=True) + self.manifold_out.k)
            x2_hyp = torch.cat([x2_time, x2_euc], dim=-1)  # => (N,33)

            if self.decoder_type == 'euc':
                # transformer 分支是 (N, hidden_channels+1) => 先转欧几里得
                x1_euc = self.manifold_out.logmap0(x1)[..., 1:]  # (N,32)
                # graph 分支本身是 (N,32)
                # decode_graph 若是普通 nn.Linear 就行，但你这里是 HypLinear => (冲突)？
                # 如果 decode_graph 是 nn.Linear(32->out_channels)，就可直接
                x_graph = self.decode_graph(x2_euc)  # 占位: (N,out_channels)
                x_trans = self.decode_trans(x1_euc)  # (N,out_channels)
                x = (1 - self.graph_weight) * x_trans + self.graph_weight * x_graph

            elif self.decoder_type == 'hyp':
                # 用 decode_graph(x2_hyp) => (N, hidden_channels+1)
                z_graph_hyp = self.decode_graph(x2_hyp)
                # 与 x1 (N, hidden_channels+1) 做融合
                z_hyp = torch.stack([
                    (1 - self.graph_weight) * x1,
                     self.graph_weight  * z_graph_hyp
                ], dim=1)
                z = self.manifold_out.mid_point(z_hyp)
                # 最终送给 decode_trans
                x = self.decode_trans(z)

            else:
                raise NotImplementedError

        else:
            # 不走图分支
            if self.decoder_type == 'euc':
                x = self.decode_trans(self.manifold_out.logmap0(x1)[..., 1:])
            elif self.decoder_type == 'hyp':
                x = self.decode_trans(x1)
            else:
                raise NotImplementedError

        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer_num, N, N]
        return attns

    def reset_parameters(self):
        if self.use_graph:
            self.graph_conv.reset_parameters()
        self.gnn_fc.reset_parameters()
