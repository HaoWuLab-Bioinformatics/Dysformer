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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import torch
from torch import nn
import torch.nn.functional as F


# ------------------------------
#  激活函数封装
# ------------------------------
class SwiGLU(nn.Module):
    """SwiGLU ≈ LLaMA‐2 激活"""
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid_dim * 2)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return F.silu(a) * b


import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincareMath:
    """
    Fixed Poincaré Ball Math with boundary guards to prevent NaN.
    """

    def __init__(self, c=1.0, eps=1e-5):
        # Ensure c is a tensor for consistency, but don't re-wrap it later
        if not isinstance(c, torch.Tensor):
            self.c = torch.tensor(c)
        else:
            self.c = c
        self.eps = eps

    def _mobius_add_impl(self, x, y):
        # Internal implementation to avoid redundant re-calculations
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        return num / (denom.clamp(min=self.eps))

    def mobius_add(self, x, y):
        return self._mobius_add_impl(x, y)

    def dist(self, x, y):
        """Geodesic Distance with boundary clamping"""
        neg_x = -x
        diff = self._mobius_add_impl(neg_x, y)
        norm_diff = diff.norm(dim=-1, p=2)

        # Guard: Ensure argument to atanh is in [0, 1 - eps]
        # dist = 2/sqrt(c) * atanh(sqrt(c) * ||-x + y||)
        sqrt_c = torch.sqrt(self.c)
        # Clamp argument to avoid Inf/NaN at boundary
        arg = torch.clamp(sqrt_c * norm_diff, min=0.0, max=1.0 - self.eps)
        dist = 2.0 / sqrt_c * torch.atanh(arg)
        return dist

    def logmap0(self, y):
        """
        Logarithmic map at origin (Manifold -> Tangent)
        Fixed: Removed torch.tensor(self.c) wrapping
        Fixed: Added boundary check for y
        """
        y_norm = y.norm(dim=-1, keepdim=True)

        # Stability: Ensure y is strictly inside the ball
        # ||y|| < 1/sqrt(c)
        sqrt_c = torch.sqrt(self.c)
        max_norm = (1.0 - self.eps) / sqrt_c

        # If input is outside, project it back to boundary (crucial for training stability)
        cond = y_norm > max_norm
        if cond.any():
            # Renormalize invalid vectors to sit just inside the boundary
            y = torch.where(cond, y / y_norm * max_norm, y)
            y_norm = torch.where(cond, max_norm, y_norm)

        y_norm = y_norm.clamp(min=self.eps)

        # Calculation: scale = atanh(sqrt(c)*||y||) / (sqrt(c)*||y||)
        arg = torch.clamp(sqrt_c * y_norm, max=1.0 - self.eps)
        scale = torch.atanh(arg) / (sqrt_c * y_norm)
        return scale * y

    def expmap0(self, v):
        """
        Exponential map at origin (Tangent -> Manifold)
        Fixed: Removed torch.tensor(self.c) wrapping
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = torch.sqrt(self.c)

        # Calculation: scale = tanh(sqrt(c)*||v||) / (sqrt(c)*||v||)
        # tanh is safe for large inputs (saturates to 1), no clamping needed for arg
        scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return scale * v

    def einstein_midpoint(self, x, weights):
        """
        Weighted Einstein Midpoint Aggregation via Klein Model
        """
        # 1. Poincaré -> Klein
        # k = 2p / (1 + c||p||^2)
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        denom_p2k = 1.0 + self.c * x2
        x_klein = 2 * x / denom_p2k.clamp(min=self.eps)

        # 2. Gamma Factor (Lorentz factor)
        # gamma = 1 / sqrt(1 - c||k||^2)  <-- Klein metric gamma
        # OR derived from Poincare: gamma = (1 + c||p||^2) / (1 - c||p||^2)
        # Using Poincare derivation for stability:
        denom_gamma = (1.0 - self.c * x2).clamp(min=self.eps)
        gamma = (1.0 + self.c * x2) / denom_gamma

        # 3. Weighted Average in Klein
        # weights shape: (Nodes, Hyperedges) -> (N, E)
        # x_klein: (N, D) -> (N, 1, D)
        # gamma: (N, 1) -> (N, 1, 1)

        # Numerator: sum_i (w_i * gamma_i * k_i)
        w_gamma = weights.unsqueeze(-1) * gamma.unsqueeze(1)  # (N, E, 1)
        num = (w_gamma * x_klein.unsqueeze(1)).sum(dim=0)  # (E, D)

        # Denominator: sum_i (w_i * gamma_i)
        den = w_gamma.sum(dim=0).clamp(min=self.eps)  # (E, 1)

        midpoint_klein = num / den

        # 4. Klein -> Poincaré
        # p = k / (1 + sqrt(1 - c||k||^2))
        k2 = midpoint_klein.pow(2).sum(dim=-1, keepdim=True)

        # Stability: Ensure Klein vector is valid (c*k2 < 1)
        sqrt_c = torch.sqrt(self.c)
        max_k_norm = (1.0 - self.eps) / sqrt_c
        cond_k = k2 > (max_k_norm ** 2)
        if cond_k.any():
            ratio = max_k_norm / (k2.sqrt() + self.eps)
            midpoint_klein = torch.where(cond_k, midpoint_klein * ratio, midpoint_klein)
            k2 = torch.where(cond_k, max_k_norm ** 2, k2)

        sqrt_term = torch.sqrt((1.0 - self.c * k2).clamp(min=self.eps))
        midpoint_poincare = midpoint_klein / (1.0 + sqrt_term)

        return midpoint_poincare


class SaturationGate(nn.Module):
    """
    基于文稿 Eq. 12-14 的饱和度门控机制
    用于动态筛选高质量的 Hyperedge 连接
    """

    def __init__(self, saturation_rate=1.0):
        super().__init__()
        self.phi = saturation_rate

    def forward(self, attn_score):
        """
        attn_score: 未归一化的注意力分数 (N, E)
        """
        # 1. 计算每个节点的平均连接强度 (Adaptive Threshold)
        # Eq: mean(A_i) [cite: 222]
        mean_attn = attn_score.mean(dim=1, keepdim=True)

        # 2. ReLU 截断 (Hard Cut)
        # Eq: ReLU(A_ij - mean(A_i)) [cite: 223]
        diff = attn_score - mean_attn
        relu_out = F.relu(diff)

        # 3. Tanh 压缩与饱和控制
        # Eq: tanh(phi * ReLU(...)) [cite: 221]
        gate_val = torch.tanh(self.phi * relu_out)

        # 4. 最终概率
        # Eq: P_ij = Sigmoid(A_ij) * Gate_ij [cite: 220]
        prob = torch.sigmoid(attn_score) * gate_val

        return prob


class HConstructor(nn.Module):
    """
    符合 Dysformer Methodology 的动态双曲超图构造器
    """

    def __init__(self, num_edges, f_dim, c_init=1.0, learnable_c=True):
        super().__init__()
        self.num_edges = num_edges
        self.f_dim = f_dim

        # 曲率设置 [cite: 234-235]
        if learnable_c:
            self.c = nn.Parameter(torch.tensor([c_init]))
        else:
            self.register_buffer("c", torch.tensor([c_init]))

        self.math = PoincareMath(c=self.c)

        # 超边质心 (Learnable Hyperedge Centroids)
        # 初始化在切空间，forward时映射到流形 [cite: 206]
        self.edges_mu_tan = nn.Parameter(torch.randn(num_edges, f_dim) * 0.01)

        # 距离注意力的参数 [cite: 210-214]
        # score = -beta * dist - gamma
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # 饱和度门控
        self.saturation_gate = SaturationGate(saturation_rate=5.0)  # phi parameter

        # 变换矩阵 (Tangent space operations) [cite: 213]
        self.W_q = nn.Linear(f_dim, f_dim)
        self.W_k = nn.Linear(f_dim, f_dim)

    def forward(self, x, args=None):
        """
        x: Node features in Poincaré Ball (N, D)
        """
        device = x.device
        N, D = x.shape

        # 确保曲率为正 (Softplus) [cite: 235]
        curr_c = F.softplus(self.c)
        self.math.c = curr_c

        # 1. 映射超边质心到 Poincaré Ball
        # Exp_o(mu)
        edges = self.math.mobius_add(torch.zeros_like(self.edges_mu_tan),
                                     torch.tanh(torch.sqrt(curr_c) * self.edges_mu_tan / 2) / (torch.sqrt(curr_c)))

        # 2. 计算节点与超边质心的亲和度 (Geometric Affinity)
        # Step 1: Map to tangent space for linear transform [cite: 213]
        # x_tan = Log_o(x) approx x (if close to origin) or standard Logmap
        # 这里为了简便，假设 W_q, W_k 直接作用在切空间特征，如果输入 x 已经在流形上，
        # 严谨做法是: x_tan = logmap0(x), x_trans = W(x_tan), x_proj = expmap0(x_trans)
        # Dysformer 文中 [cite: 213] 提到 "operating in the tangent space... through exponential and logarithmic mappings"

        # 实现简化版 Distance Attention
        # Raw geometric affinity: s(i,j) = -d(x_i, c_j)

        # 计算距离矩阵 (N, E)
        # Expand dims for broadcasting: x(N, 1, D), edges(1, E, D)
        dist_mat = self.math.dist(x.unsqueeze(1).expand(-1, self.num_edges, -1),
                                  edges.unsqueeze(0).expand(N, -1, -1))

        # 3. 计算注意力分数
        # Eq: -beta * dist + gamma [cite: 212] (Note: source says -beta*d - gamma or similar linear transform)
        attn_score = -self.beta * dist_mat + self.gamma

        # 4. 饱和度门控获取关联矩阵 H
        # [cite: 219-224]
        H = self.saturation_gate(attn_score)  # (N, E)

        # 5. 节点 -> 超边聚合 (Einstein Midpoint)
        # [cite: 166-176]
        # 将节点特征聚合，更新超边特征
        # 注意：在 Dysformer 中，这一步产生的 edges_updated 是用于下一轮消息传递的特征
        edges_updated = self.math.einstein_midpoint(x, H)  # (E, D)

        # 如果需要保留原始 H (未经过门控的 raw score 用于可视化)
        H_raw = attn_score

        return edges_updated, H, H_raw


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. 基础几何工具 (Poincaré Ball Model) - 对应论文 Section 2.2
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. Fixed PoincareMath (With Numerical Stability Guards)
# =============================================================================
class PoincareMath:
    """
    Fixed Poincaré Ball Math with boundary guards to prevent NaN.
    """

    def __init__(self, c=1.0, eps=1e-5):
        # Fix: Do not re-wrap c if it is already a tensor to preserve gradients
        if not isinstance(c, torch.Tensor):
            self.c = torch.tensor(c)
        else:
            self.c = c
        self.eps = eps

    def _mobius_add_impl(self, x, y):
        # Internal implementation to avoid redundant re-calculations
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        # Fix: Clamp denominator to avoid division by zero
        return num / (denom.clamp(min=self.eps))

    def mobius_add(self, x, y):
        """Möbius Addition [cite: 106]"""
        return self._mobius_add_impl(x, y)

    def dist(self, x, y):
        """Geodesic Distance with boundary clamping [cite: 103]"""
        neg_x = -x
        diff = self._mobius_add_impl(neg_x, y)
        norm_diff = diff.norm(dim=-1, p=2)

        # Guard: Ensure argument to atanh is strictly < 1
        # dist = 2/sqrt(c) * atanh(sqrt(c) * ||-x + y||)
        sqrt_c = torch.sqrt(self.c)
        # Fix: Clamp argument to avoid Inf/NaN at boundary [cite: 101]
        arg = torch.clamp(sqrt_c * norm_diff, min=0.0, max=1.0 - self.eps)
        dist = 2.0 / sqrt_c * torch.atanh(arg)
        return dist

    def logmap0(self, y):
        """
        Logarithmic map at origin (Manifold -> Tangent) [cite: 112]
        Fixed: Removed torch.tensor(self.c) wrapping
        Fixed: Added boundary check/projection for y
        """
        y_norm = y.norm(dim=-1, keepdim=True)

        # Stability: Ensure y is strictly inside the ball ||y|| < 1/sqrt(c)
        sqrt_c = torch.sqrt(self.c)
        max_norm = (1.0 - self.eps) / sqrt_c

        # If input is outside due to drift, project it back to boundary
        cond = y_norm > max_norm
        if cond.any():
            # Renormalize invalid vectors to sit just inside the boundary
            y = torch.where(cond, y / y_norm * max_norm, y)
            y_norm = torch.where(cond, max_norm, y_norm)

        y_norm = y_norm.clamp(min=self.eps)

        # Calculation: scale = atanh(sqrt(c)*||y||) / (sqrt(c)*||y||)
        arg = torch.clamp(sqrt_c * y_norm, max=1.0 - self.eps)
        scale = torch.atanh(arg) / (sqrt_c * y_norm)
        return scale * y

    def expmap0(self, v):
        """
        Exponential map at origin (Tangent -> Manifold) [cite: 111]
        Fixed: Removed torch.tensor(self.c) wrapping
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = torch.sqrt(self.c)

        # Calculation: scale = tanh(sqrt(c)*||v||) / (sqrt(c)*||v||)
        # tanh is safe for large inputs (saturates to 1)
        scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return scale * v

    def einstein_midpoint(self, x, weights):
        """
        Weighted Einstein Midpoint Aggregation via Klein Model [cite: 166]
        """
        # 1. Poincaré -> Klein
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        denom_p2k = 1.0 + self.c * x2
        x_klein = 2 * x / denom_p2k.clamp(min=self.eps)

        # 2. Gamma Factor (Lorentz factor)
        denom_gamma = (1.0 - self.c * x2).clamp(min=self.eps)
        gamma = (1.0 + self.c * x2) / denom_gamma

        # 3. Weighted Average in Klein
        # weights shape: (Nodes, Hyperedges) -> (N, E)
        w_gamma = weights.unsqueeze(-1) * gamma.unsqueeze(1)  # (N, E, 1)
        num = (w_gamma * x_klein.unsqueeze(1)).sum(dim=0)  # (E, D)
        den = w_gamma.sum(dim=0).clamp(min=self.eps)  # (E, 1)

        midpoint_klein = num / den

        # 4. Klein -> Poincaré
        k2 = midpoint_klein.pow(2).sum(dim=-1, keepdim=True)

        # Stability: Ensure Klein vector is valid (c*k2 < 1)
        sqrt_c = torch.sqrt(self.c)
        max_k_norm = (1.0 - self.eps) / sqrt_c
        cond_k = k2 > (max_k_norm ** 2)
        if cond_k.any():
            ratio = max_k_norm / (k2.sqrt() + self.eps)
            midpoint_klein = torch.where(cond_k, midpoint_klein * ratio, midpoint_klein)
            k2 = torch.where(cond_k, max_k_norm ** 2, k2)

        sqrt_term = torch.sqrt((1.0 - self.c * k2).clamp(min=self.eps))
        midpoint_poincare = midpoint_klein / (1.0 + sqrt_term)

        return midpoint_poincare


# =============================================================================
# 2. Updated Dysformer Components (Using Robust PoincareMath)
# =============================================================================

class SaturationGate(nn.Module):
    """
    Saturation-based dynamic structure optimization mechanism.
    Filters redundant connections based on average affinity. [cite: 219-224]
    """

    def __init__(self, phi=1.0):
        super().__init__()
        self.phi = phi

    def forward(self, attn_score):
        # [cite: 222] Adaptive threshold
        mean_attn = attn_score.mean(dim=1, keepdim=True)
        # [cite: 223] Hard cut
        diff = attn_score - mean_attn
        relu_out = F.relu(diff)
        # [cite: 221] Saturation
        gate_val = torch.tanh(self.phi * relu_out)
        # [cite: 220] Gating
        prob = torch.sigmoid(attn_score) * gate_val
        return prob


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


class GeometricGatingUnit(nn.Module):
    """
    Geometric Gating Unit (GGU).
    Fuses spatial and spectral streams in tangent space. [cite: 250]
    """

    def __init__(self, dim):
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, dim)

    def forward(self, h_spatial_tan, h_spectral_tan):
        combined = torch.cat([h_spatial_tan, h_spectral_tan], dim=-1)
        z = torch.sigmoid(self.gate_linear(combined))  # [cite: 252]
        h_fused_tan = z * h_spatial_tan + (1 - z) * h_spectral_tan  # [cite: 257]
        return h_fused_tan


class DysformerLayer(nn.Module):
    """
    Single layer of Dysformer comprising Spatial, Spectral streams and GGU. [cite: 7]
    """

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


class Dysformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.in_channels
        self.hidden_dim = args.hidden_channels
        self.out_dim = args.out_channels
        self.num_layers = getattr(args, 'num_layers', 2)
        self.k_neighbors = getattr(args, 'num_edges', 10)

        # 先定义 input_map（或直接用你原来的 input_linear）
        self.input_map = nn.Linear(self.in_dim, self.hidden_dim)

        # 如果后面 forward 用的是 input_linear，就做个别名（任选其一）
        self.input_linear = self.input_map

        # 再定义 encoder（别在定义 input_map 前引用它）
        self.encoder = nn.ModuleList([self.input_map])

        # 下面保持不变
        self.curvatures = nn.Parameter(torch.ones(self.num_layers + 1))

        self.layers = nn.ModuleList([
            DysformerLayer(self.hidden_dim, self.hidden_dim, self.k_neighbors)
            for _ in range(self.num_layers)
        ])
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)


    def forward(self, dataset, epoch=0, perturb=None):
        x = dataset.graph['node_feat']

        if perturb is not None:
            # Implement perturbation logic if needed
            pass

        # Euclidean -> Hyperbolic [cite: 74]
        c_0 = F.softplus(self.curvatures[0])
        math0 = PoincareMath(c=c_0)

        x = self.input_linear(x)
        x = math0.expmap0(x)

        # Dysformer Layers
        for i, layer in enumerate(self.layers):
            c_in = F.softplus(self.curvatures[i])
            c_out = F.softplus(self.curvatures[i + 1])
            x = layer(x, c_in, c_out)

        # Classification in Tangent Space [cite: 246]
        c_final = F.softplus(self.curvatures[-1])
        math_final = PoincareMath(c=c_final)
        x_tan = math_final.logmap0(x)

        out = self.classifier(x_tan)
        return out










# ------------------------------
#  动态超图构造器（无显式正则版）
# ------------------------------
'''
class HConstructor(nn.Module):
    """
    动态超图构造器（增强版，无需修改损失函数）
    -----------------------------------------------------------
    • 曲率 k 为可学习 Parameter，正则化（若需要）由优化器 weight_decay 处理
    • 其余：动态扩/缩边、显存友好注意力、热核小波、多激活、邻接缓存
    """

    # ---------- 初始化 ----------
    def __init__(
        self,
        num_edges: int,
        f_dim: int,
        iters: int = 1,
        eps: float = 1e-8,
        hidden_dim: int = 256,
        *,
        learnable_k: bool = True,
        k_init: float = 1.0,
        act: str = "gelu",
        adj_cache: bool = True,
    ):
        super().__init__()
        self.num_edges = int(num_edges)
        self.f_dim     = int(f_dim)
        self.eps       = float(eps)
        self.iters     = int(iters)

        # ---------- 超边原型 ----------
        self.edges_mu        = nn.Parameter(torch.randn(max(1, self.num_edges), self.f_dim) * 0.02)
        self.edges_logsigma  = nn.Parameter(torch.zeros(max(1, self.num_edges), self.f_dim))

        self._need_expand = False
        self._need_shrink = False

        # ---------- 曲率 k ----------
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer("k", torch.tensor(float(k_init)), persistent=False)

        # ---------- 节点自注意力 ----------
        self.v_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.v_k = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.v_v = nn.Linear(self.f_dim, self.f_dim, bias=False)

        # ---------- V→E ----------
        self.ve_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.ve_k = nn.Linear(self.f_dim, self.f_dim, bias=False)

        # ---------- 超边自注意力 ----------
        self.e_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.e_k = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.e_v = nn.Linear(self.f_dim, self.f_dim, bias=False)

        # ---------- 归一化 & 融合 ----------
        self.norm_x = nn.LayerNorm(self.f_dim)
        self.norm_e = nn.LayerNorm(self.f_dim)

        # 激活选择
        act = act.lower()
        if act == "gelu":
            act_layer = nn.GELU()
        elif act == "swiglu":
            act_layer = SwiGLU(2 * self.f_dim, hidden_dim)
        else:
            act_layer = nn.ReLU(inplace=False)

        hid = max(self.f_dim, int(hidden_dim))
        self.edge_update = nn.Sequential(
            nn.Linear(2 * self.f_dim, hid),
            act_layer,
            nn.Linear(hid, self.f_dim),
        )
        self.node_proj  = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.nodes_fuse = nn.Linear(self.f_dim, self.f_dim, bias=False)

        # 多头参数
        self.h        = 4
        self.dropout  = nn.Dropout(p=0.0)

        # 邻接缓存
        self._cache_adj_enabled = bool(adj_cache)
        self._cached_H_pattern: torch.Tensor | None = None
        self._cached_Av: torch.Tensor | None = None
        self._cached_Ae: torch.Tensor | None = None

    # ---------- 工具函数 ----------
    @staticmethod
    def _get(args, name, default):
        return getattr(args, name, default)

    def mask_attn(self, attn: torch.Tensor, k: int, dim: int = -1):
        """top-k 稀疏化 + L1 归一化"""
        if k <= 0:
            return attn
        k = min(k, attn.size(dim))
        topk = torch.topk(attn, k, dim=dim)
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask = mask.scatter(dim, topk.indices, True)
        attn_pruned = attn.masked_fill(~mask, 0.0)
        denom = attn_pruned.sum(dim=dim, keepdim=True) + 1e-9
        return attn_pruned / denom

    # ---------- 动态增删边 ----------
    def _mark_expand(self): self._need_expand = True
    def _mark_shrink(self): self._need_shrink = True

    @torch.no_grad()
    def _expand_parameters(self):
        cur_n = self.edges_mu.size(0)
        if self.num_edges <= cur_n:
            self._need_expand = False
            return
        new_rows = self.num_edges - cur_n
        mu_pad = self.edges_mu[-1:].repeat(new_rows, 1).detach().clone()
        ls_pad = self.edges_logsigma[-1:].repeat(new_rows, 1).detach().clone()
        self.edges_mu       = nn.Parameter(torch.cat([self.edges_mu.detach(), mu_pad], dim=0))
        self.edges_logsigma = nn.Parameter(torch.cat([self.edges_logsigma.detach(), ls_pad], dim=0))
        self._need_expand = False

    @torch.no_grad()
    def _shrink_parameters(self):
        cur_n = self.edges_mu.size(0)
        if self.num_edges >= cur_n:
            self._need_shrink = False
            return
        self.edges_mu       = nn.Parameter(self.edges_mu[: self.num_edges].detach().clone())
        self.edges_logsigma = nn.Parameter(self.edges_logsigma[: self.num_edges].detach().clone())
        self._need_shrink = False

    def _adjust_edges(self, s_level: float, args):
        cur_epoch  = self._get(args, "epoch", 0)
        edge_warm  = self._get(args, "edge_warm", 0)
        if cur_epoch < edge_warm or not self._get(args, "use_dynamic", True):
            return
        up, low = self._get(args, "up_bound", 0.95), self._get(args, "low_bound", 0.60)
        min_e   = self._get(args, "min_num_edges", 4)
        if   s_level > float(up):
            self.num_edges += 1
            if self.num_edges > self.edges_mu.size(0):
                self._mark_expand()
        elif s_level < float(low):
            new_num = max(self.num_edges - 1, int(min_e))
            if new_num < self.num_edges:
                self.num_edges = new_num
                self._mark_shrink()

    # ---------- Lorentz 距离 ----------
    @staticmethod
    def euc2lorentz(x: torch.Tensor, k: float = 1.0):
        x2 = (x ** 2).sum(dim=-1, keepdim=True)
        t  = torch.sqrt(x2 + k)
        return torch.cat([t, x], dim=-1)

    @staticmethod
    def acosh_safe(x: torch.Tensor, eps: float = 1e-6):
        x = torch.clamp(x, min=1.0 + eps)
        return torch.log(x + torch.sqrt(x - 1.0) * torch.sqrt(x + 1.0))

    def hyperbolic_score(self, Q, K, k: float):
        Hh, Nq, Dh = Q.shape
        Nk = K.shape[1]
        Ql = self.euc2lorentz(Q.reshape(-1, Dh), k).reshape(Hh, Nq, Dh + 1)
        Kl = self.euc2lorentz(K.reshape(-1, Dh), k).reshape(Hh, Nk, Dh + 1)
        tQ, sQ = Ql[..., :1], Ql[..., 1:]
        tK, sK = Kl[..., :1], Kl[..., 1:]
        Lij = -(tQ @ tK.transpose(-1, -2)).squeeze(-3) + (sQ @ sK.transpose(-1, -2))
        cosh_d = -Lij / max(k, 1e-6)
        dij = self.acosh_safe(cosh_d)
        return -(dij ** 2)

    # ---------- 多头工具 ----------
    def _split_heads(self, x, h): return x.view(x.size(0), h, -1).transpose(0, 1).contiguous()
    def _merge_heads(self, x):    return x.transpose(0, 1).contiguous().view(x.size(1), -1)

    # ---------- 图归一化 / 小波 ----------
    @staticmethod
    def _norm_adj(A, eps=1e-9):
        deg = A.sum(dim=-1) + eps
        inv_sqrt = torch.pow(deg, -0.5)
        return (A * inv_sqrt.unsqueeze(-1)) * inv_sqrt.unsqueeze(-2)

    @staticmethod
    def _degree_norm(H, eps=1e-9):
        Dv = H.sum(dim=1, keepdim=True) + eps
        De = H.sum(dim=0, keepdim=True) + eps
        return (H / De) / torch.sqrt(Dv)

    def _heat_wavelet(self, A_norm, X, scales=(0.25, 0.5, 1.0), order: int = 2):
        I = torch.eye(A_norm.size(0), device=X.device, dtype=X.dtype)
        L = I - A_norm
        outs = []
        for s in scales:
            Y = X - s * (L @ X)
            if order >= 2: Y = Y + (s ** 2) / 2.0 * (L @ (L @ X))
            if order >= 3: Y = Y - (s ** 3) / 6.0 * (L @ (L @ (L @ X)))
            outs.append(Y)
        return torch.stack(outs, -2).sum(-2)

    # ---------- 分块注意力 ----------
    def _memory_efficient_attn_2d(
        self, Q, K, V, use_hyper=False, k_curv=1.0,
        q_chunk_size=256, k_chunk_size=256, work_dtype=torch.float32
    ):
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
            ke = min(ks + k_chunk_size, Nk)
            Kb, Vb = Kw[:, ks:ke], Vw[:, ks:ke]
            scores = (
                self.hyperbolic_score(Qw, Kb, k_curv) if use_hyper
                else torch.matmul(Qw, Kb.transpose(-1, -2)) / math.sqrt(Dh)
            )
            m_block = scores.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_block)
            exp_m_diff_prev = torch.exp(m - m_new)
            s, out = s * exp_m_diff_prev, out * exp_m_diff_prev
            exp_scores = torch.exp(scores - m_new)
            s += exp_scores.sum(dim=-1, keepdim=True)
            out += torch.matmul(exp_scores, Vb)
            m = m_new
            ks = ke
        return (out / (s + 1e-9)).to(Q.dtype)

    # =========================================================
    #  forward
    # =========================================================
    def forward(self, inputs: torch.Tensor, args):
        if self._need_expand:  self._expand_parameters()
        if self._need_shrink:  self._shrink_parameters()

        x = self.norm_x(inputs)
        N, D = x.shape
        heads = int(self._get(args, "attn_heads", 4))
        k_e   = int(self._get(args, "k_e", 16))
        use_h = bool(self._get(args, "use_hyper", True))
        k_curv = float(self.k)
        iters = int(self._get(args, "iters", self.iters))
        scales = self._get(args, "wavelet_scales", (0.25, 0.5, 1.0))
        order  = int(self._get(args, "wavelet_order", 2))

        assert D % heads == 0, "D % heads must == 0"
        E = self.num_edges
        sigma = self.edges_logsigma.exp()[:E]
        edges = self.edges_mu[:E] + sigma * torch.randn_like(sigma)

        H = H_raw = None
        for _ in range(iters):
            # (1) 节点自注意力
            Qv = self._split_heads(self.v_q(x), heads)
            Kv = self._split_heads(self.v_k(x), heads)
            Vv = self._split_heads(self.v_v(x), heads)
            x_sa = self.node_proj(self._merge_heads(
                self._memory_efficient_attn_2d(Qv, Kv, Vv, use_h, k_curv)
            )) + x

            # (2) V→E
            Q_ve = self._split_heads(self.ve_q(x_sa), heads)
            K_ve = self._split_heads(self.ve_k(self.norm_e(edges)), heads)
            score = (
                self.hyperbolic_score(Q_ve, K_ve, k_curv) if use_h
                else torch.matmul(Q_ve, K_ve.transpose(-1, -2)) / math.sqrt(Q_ve.size(-1))
            )
            H_raw = score.mean(0)
            H = self.mask_attn(F.softmax(H_raw, dim=-1), k_e)
            E_from_V = torch.matmul(H.t(), x_sa)

            # (2.5) 超边自注意力
            norm_edges = self.norm_e(edges)
            Qe = self._split_heads(self.e_q(norm_edges), heads)
            Ke = self._split_heads(self.e_k(norm_edges), heads)
            Ve = self._split_heads(self.e_v(norm_edges), heads)
            E_self = self._merge_heads(
                self._memory_efficient_attn_2d(Qe, Ke, Ve, use_h, k_curv)
            )
            edges = self.norm_e(self.edge_update(torch.cat([E_from_V, E_self], -1)) + edges)

            # (3) E→V
            Hn = self._degree_norm(H)
            X_from_E = torch.matmul(Hn, edges)

            # (4) 热核小波
            H_pattern = (H > 0)
            if (
                self._cache_adj_enabled
                and self._cached_H_pattern is not None
                and torch.equal(H_pattern, self._cached_H_pattern)
            ):
                Av, Ae = self._cached_Av, self._cached_Ae
            else:
                Av = self._norm_adj(H @ H.t())
                Ae = self._norm_adj(H.t() @ H)
                if self._cache_adj_enabled:
                    self._cached_H_pattern = H_pattern.detach().clone()
                    self._cached_Av, self._cached_Ae = Av.detach().clone(), Ae.detach().clone()

            x_sa_w  = self._heat_wavelet(Av, x_sa,     scales, order)
            x_e2v_w = self._heat_wavelet(Av, X_from_E, scales, order)
            e_w     = self._heat_wavelet(Ae, edges,    scales, order)
            x       = self.norm_x(self.nodes_fuse(x_sa_w + x_e2v_w + Hn @ e_w) + x)
            edges   = self.norm_e(edges + e_w)

            # (5) 动态饱和度
            with torch.no_grad():
                empty = (H.sum(0) == 0).sum()
                s_level = 1.0 - empty.float() / max(E, 1)
                self._adjust_edges(float(s_level), args)

            if self.num_edges > edges.size(0):  # 刚扩边，先零 padding
                pad_rows = self.num_edges - edges.size(0)
                pad = self.edges_mu[edges.size(0):edges.size(0)+pad_rows]
                edges = torch.cat([edges, pad.to(edges.dtype)], 0)

        return edges, H, H_raw


'''


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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# —— 若工程里已有该函数，可删掉这里的定义；保持一致的 Lorentz 嵌入 —— #
def euc2lorentz(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """
    欧氏坐标 → Lorentz 坐标（时间维 + 空间维）
    x: (..., D)
    return: (..., D+1)
    """
    t = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) + k)
    return torch.cat([t, x], dim=-1)
'''

class Dysformer(nn.Module):
    """
    超曲面 Transformer 主干（Lorentz 流形）
    支持：
      • Transformer 分支（TransConv）
      • 可选 GraphConv 分支
      • 可选动态超图分支 (HGNN_conv)
    全部在超曲面上解码。

    本版变更：
      1) 特征级融合：x_euc 与 x_dh 的融合权重改为可学习（w_dh ∈ [0,1]）
      2) logit 级融合：Transformer 与 Graph 的融合权重改为可学习（w_graph ∈ [0,1]）
    """

    # -------------------------------
    def __init__(self, args):
        super().__init__()
        self.args = args

        # --------- manifolds ---------
        self.manifold_in = TrainableLorentz(c_init=args.k_in,     c_max=args.c_max)
        #self.manifold_in = Lorentz(k=(10))
        self.manifold_hidden = TrainableLorentz(c_init=args.k_hidden, c_max=args.c_max)
        self.manifold_out = TrainableLorentz(c_init=args.k_out,   c_max=args.c_max)
        #self.manifold_out = Lorentz(k=(10))
        # --------- channels ----------
        self.in_channels     = args.in_channels      # Euclidean feat dim
        self.hidden_channels = args.hidden_channels  # Hyperbolic hidden dim
        self.out_channels    = args.out_channels     # 类别数

        # --------- 分支开关 ----------
        self.use_graph    = bool(getattr(args, "use_graph", False))
        self.use_dhyper   = bool(getattr(args, "use_dhyper", False))  # 动态超图

        # --------- 输入映射 ----------
        self.input_map = nn.Linear(self.in_channels, self.hidden_channels, bias=False)

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
        self.graph_conv = GraphConv(self.hidden_channels, self.hidden_channels, args=args) \
                          if self.use_graph else None

        # --------- 动态超图分支（可选） ----------
        # 这里沿用你项目中的 HGNN_conv；仅在 forward 中做可学习权重融合
        self.dhgnn_conv = DHGNN_conv(self.hidden_channels, self.hidden_channels, args.num_edges) \
                          if self.use_dhyper else None

        # --------- 超曲面解码器 ----------
        self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        self.decode_graph = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)

        # --------- 可学习融合权重 ----------
        # 初值读取自 args（若缺省则为 0.5），并映射到实数域参数 w*，前向用 sigmoid(w*) 得到 [0,1]
        dh_init = float(getattr(args, "dh_weight", 0.5))
        graph_init = float(getattr(args, "graph_weight", 0.5))
        # 为防 0/1 的 logit 溢出，做一次裁剪
        dh_init = min(max(dh_init, 1e-4), 1 - 1e-4)
        graph_init = min(max(graph_init, 1e-4), 1 - 1e-4)

        # 学习到的标量参数（全局的 batch 共享权重；若需按样本/时间步动态，可改成小 MLP 门控）
        self._w_dh_logit = nn.Parameter(torch.tensor(math.log(dh_init / (1.0 - dh_init)), dtype=torch.float32))
        self._w_graph_logit = nn.Parameter(torch.tensor(math.log(graph_init / (1.0 - graph_init)), dtype=torch.float32))

    # -------------------------------
    def forward(self, dataset, *, epoch: int = 0):
        """
        前向计算：
          - 只有在 use_graph=True 且 dataset.graph['edge_index'] 不为 None 时才使用图分支
          - 动态超图与原始特征先做“特征级可学习融合”
          - Transformer 与 Graph 的 logits 再做“logit 级可学习融合”
        """
        node_feat = dataset.graph['node_feat']  # (N, F)
        device = node_feat.device

        # --------- 安全获取 edge_index ----------
        if self.use_graph and 'edge_index' in dataset.graph and dataset.graph['edge_index'] is not None:
            edge_index = dataset.graph['edge_index'].to(device)
        else:
            edge_index = None

        # --------- 动态超图：特征级可学习融合 ----------
        x_euc = self.input_map(node_feat)  # (N, H)
        if self.use_dhyper and self.dhgnn_conv is not None:
            # 通过 HGNN 得到动态超图增强特征
            x_dh, _, _ = self.dhgnn_conv(x_euc, args=self.args)  # (N, H)

            # 可学习权重（标量）
            w_dh = torch.sigmoid(self._w_dh_logit)  # in [0,1]
            # 广播到特征维度（标量 * 张量），无需显式 expand
            x_euc = (1.0 - w_dh) * x_euc + w_dh * x_dh
        # 没开动态超图，则直接用 x_euc

        # --------- Transformer 分支 ----------
        x_tr = self.trans_conv(x_euc)  # (N, H+1) — 已在超曲面

        # --------- Graph 分支（可选，自动处理 2D/3D） ----------
        def _repeat_edge_index_over_time(edge_index, T: int, N: int, device, dtype=None):
            """
            将原始 edge_index (2, E) 复制到每个时间步，并在节点编号上加 t*N 的偏移，
            形成块对角大图；返回 (2, T*E)。
            """
            assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 必须是 (2, E)"
            row, col = edge_index[0].to(device), edge_index[1].to(device)
            if dtype is not None:
                row = row.to(dtype)
                col = col.to(dtype)
            offsets = (torch.arange(T, device=device, dtype=row.dtype).unsqueeze(1)) * N  # (T,1)
            row_t = row.unsqueeze(0) + offsets  # (T,E)
            col_t = col.unsqueeze(0) + offsets  # (T,E)
            return torch.stack([row_t.reshape(-1), col_t.reshape(-1)], dim=0)  # (2, T*E)

        def _sanitize_edge_index_for_size(edge_index, N_target: int):
            """
            过滤掉 >= N_target 的索引，保证所有条目落在 [0, N_target-1]。
            返回过滤后的 edge_index 以及保留比例（便于日志定位）。
            """
            row, col = edge_index[0], edge_index[1]
            mask = (row >= 0) & (row < N_target) & (col >= 0) & (col < N_target)
            if mask.all():
                return edge_index, 1.0
            ei_new = torch.stack([row[mask], col[mask]], dim=0)
            keep_ratio = mask.float().mean().item()
            return ei_new, keep_ratio

        if edge_index is not None and self.graph_conv is not None:
            ei0 = edge_index.to(x_euc.device)
            if ei0.dtype != torch.long:
                ei0 = ei0.long()

            if x_euc.dim() == 2:
                # (N,H)
                x_gc_in = x_euc
                N_feat = x_gc_in.size(0)

                ei_use, _ = _sanitize_edge_index_for_size(ei0, N_feat)
                x_gc_euc = self.graph_conv(x_gc_in, ei_use)  # (N,H)
                x_gc_hyp = euc2lorentz(x_gc_euc, k=self.manifold_out.k)

            elif x_euc.dim() == 3:
                # (T,N,H)
                T, N_step, Hdim = x_euc.shape
                x_gc_in = x_euc.reshape(T * N_step, Hdim)  # (TN, H)
                ei_time = _repeat_edge_index_over_time(ei0, T=T, N=N_step, device=x_euc.device, dtype=torch.long)

                N_feat = x_gc_in.size(0)
                ei_use, _ = _sanitize_edge_index_for_size(ei_time, N_feat)

                x_gc_flat = self.graph_conv(x_gc_in, ei_use)          # (TN,H)
                x_gc_hyp = euc2lorentz(x_gc_flat, k=self.manifold_out.k)  # (TN,H+1)
            else:
                raise ValueError(f"x_euc 期望 2D 或 3D，实际 {x_euc.shape}")
        else:
            x_gc_hyp = None

        # --------- 超曲面解码 + logit 级可学习融合 ----------
        tr_logits = self.decode_trans(x_tr)  # (N, C) 或 (TN, C)

        if x_gc_hyp is not None:
            gc_logits = self.decode_graph(x_gc_hyp)  # (N, C) 或 (TN, C)
            w_graph = torch.sigmoid(self._w_graph_logit)  # in [0,1]
            out = (1.0 - w_graph) * tr_logits + w_graph * gc_logits
        else:
            out = tr_logits

        if getattr(self.args, "return_emb", False):
            return out, x_tr  # logits, embedding（Transformer 分支的超曲面表征）
        else:
            return out

    # -------------------------------
    def get_attentions(self, x_euc):
        """返回 Transformer 中间层注意力 (L, N, N)"""
        return self.trans_conv.get_attentions(x_euc)

    # -------------------------------
    def reset_parameters(self):
        # 若需重置融合权重，可在此处设置为初始 logit
        with torch.no_grad():
            # 可根据 args 重置，否则保持当前已学到的值
            if hasattr(self.args, "dh_weight"):
                v = min(max(float(self.args.dh_weight), 1e-4), 1-1e-4)
                self._w_dh_logit.copy_(torch.tensor(math.log(v/(1-v))))
            if hasattr(self.args, "graph_weight"):
                v = min(max(float(self.args.graph_weight), 1e-4), 1-1e-4)
                self._w_graph_logit.copy_(torch.tensor(math.log(v/(1-v))))

    # -------------------------------
    def get_fusion_weights(self):
        """
        便于日志/可视化：返回当前两处融合权重（介于 0~1）
        {
          'w_dh': 动态超图特征占比,
          'w_graph': Graph 分支 logits 占比
        }
        """
        with torch.no_grad():
            return {
                "w_dh":    torch.sigmoid(self._w_dh_logit).item(),
                "w_graph": torch.sigmoid(self._w_graph_logit).item(),
            }


'''

'''

class Dysformer(nn.Module):
    """
    超曲面 Transformer 主干（Lorentz 流形）
    支持：
      • Transformer 分支（TransConv）
      • 可选 GraphConv 分支
      • 可选动态超图分支 (HGNN_conv)
    全部在超曲面上解码。

    本版变更：
      1) 特征级融合：x_euc 与 x_dh 的融合权重改为可学习（w_dh ∈ [0,1]）
      2) logit 级融合：Transformer 与 Graph 的融合权重改为可学习（w_graph ∈ [0,1]）
      3) 增加扰动实验接口（perturb），可对特征与图结构施加噪声 / 随机删边/加边
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # --------- manifolds ---------
        self.manifold_in = TrainableLorentz(c_init=args.k_in,     c_max=args.c_max)
        self.manifold_hidden = TrainableLorentz(c_init=args.k_hidden, c_max=args.c_max)
        self.manifold_out = TrainableLorentz(c_init=args.k_out,   c_max=args.c_max)

        # --------- channels ----------
        self.in_channels     = args.in_channels      # Euclidean feat dim
        self.hidden_channels = args.hidden_channels  # Hyperbolic hidden dim
        self.out_channels    = args.out_channels     # 类别数

        # --------- 分支开关 ----------
        self.use_graph    = bool(getattr(args, "use_graph", False))
        self.use_dhyper   = bool(getattr(args, "use_dhyper", False))  # 动态超图

        # --------- 输入映射 ----------
        self.input_map = nn.Linear(self.in_channels, self.hidden_channels, bias=False)

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
        self.graph_conv = GraphConv(self.hidden_channels, self.hidden_channels, args=args) \
                          if self.use_graph else None

        # --------- 动态超图分支（可选） ----------
        self.dhgnn_conv = DHGNN_conv(self.hidden_channels, self.hidden_channels, args.num_edges) \
                          if self.use_dhyper else None

        # --------- 超曲面解码器 ----------
        self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        self.decode_graph = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)

        # --------- 可学习融合权重 ----------
        dh_init = float(getattr(args, "dh_weight", 0.5))
        graph_init = float(getattr(args, "graph_weight", 0.5))
        dh_init = min(max(dh_init, 1e-4), 1 - 1e-4)
        graph_init = min(max(graph_init, 1e-4), 1 - 1e-4)

        self._w_dh_logit = nn.Parameter(
            torch.tensor(math.log(dh_init / (1.0 - dh_init)), dtype=torch.float32)
        )
        self._w_graph_logit = nn.Parameter(
            torch.tensor(math.log(graph_init / (1.0 - graph_init)), dtype=torch.float32)
        )

    # =========================================================
    #  扰动工具函数：特征扰动
    # =========================================================
    def _perturb_features(self, x, cfg=None):
        """
        对输入的欧氏特征 x 添加扰动。
        x:
            可以是形状 (N, F) 或 (T, N, F) 的张量。
        cfg:
            dict 或 None，支持的 key 包括：
              - "feat_noise_std": float，高斯噪声标准差
              - "feat_drop_rate": float，[0,1]，随机将特征置零
              - "shuffle_feat":   bool，是否在样本维度上打乱特征（目前只支持 2D x）

        返回:
            扰动后的 x，形状与输入一致。
        """
        if cfg is None:
            cfg = {"feat_noise_std": 0.0, "feat_drop_rate": 0.99, "shuffle_feat": False}

        x_pert = x.clone()

        feat_noise_std = float(cfg.get("feat_noise_std", 0.0))
        feat_drop_rate = float(cfg.get("feat_drop_rate", 0.99))
        shuffle_feat   = bool(cfg.get("shuffle_feat", False))

        # 1) 高斯噪声
        if feat_noise_std > 0.0:
            noise = torch.randn_like(x_pert) * feat_noise_std
            x_pert = x_pert + noise

        # 2) 随机置零（类似 feature dropout）
        if 0.0 < feat_drop_rate < 1.0:
            mask = torch.rand_like(x_pert) > feat_drop_rate
            x_pert = x_pert * mask

        # 3) 在样本维度上打乱特征（每一列单独洗牌）
        if shuffle_feat:
            # 目前只支持二维特征 (N, F) 的打乱
            if x_pert.dim() != 2:
                raise ValueError("shuffle_feat 目前只支持形状为 (N, F) 的二维特征张量")
            N, F = x_pert.shape
            base_idx = torch.arange(N, device=x_pert.device)
            for j in range(F):
                perm = base_idx[torch.randperm(N)]
                x_pert[:, j] = x_pert[perm, j]

        return x_pert

    # =========================================================
    #  扰动工具函数：图结构扰动
    # =========================================================
    def _perturb_edges(self, edge_index, num_nodes, cfg=None):
        """
        对图结构 edge_index 进行扰动。
        edge_index:
            形状 (2, E) 的 long tensor。
        num_nodes:
            图中节点数（注意：是“每个时间步”的节点数，如果后面要做时间展开）。
        cfg:
            dict 或 None，支持的 key 包括：
              - "edge_drop_rate": float，[0,1]，随机删边比例
              - "edge_add_rate":  float，[0,1]，相对于当前边数要加的随机边比例

        返回:
            扰动后的 edge_index，形状仍为 (2, E')。
        """
        if edge_index is None or cfg is None:
            return edge_index

        ei = edge_index.clone()
        device = ei.device

        edge_drop_rate = float(cfg.get("edge_drop_rate", 0.0))
        edge_add_rate  = float(cfg.get("edge_add_rate", 0.0))

        # 1) 随机删边
        if 0.0 < edge_drop_rate < 1.0 and ei.size(1) > 0:
            E = ei.size(1)
            keep_mask = torch.rand(E, device=device) > edge_drop_rate
            # 防止全删光：保证至少保留一条
            if keep_mask.sum() == 0:
                rand_idx = torch.randint(0, E, (1,), device=device)
                keep_mask[rand_idx] = True
            ei = ei[:, keep_mask]

        # 2) 随机加边（均匀从所有可能的点对中采样）
        if edge_add_rate > 0.0 and num_nodes > 1:
            E_now = ei.size(1)
            add_E = int(E_now * edge_add_rate)
            if add_E > 0:
                src = torch.randint(0, num_nodes, (add_E,), device=device)
                dst = torch.randint(0, num_nodes, (add_E,), device=device)
                add_edges = torch.stack([src, dst], dim=0)
                ei = torch.cat([ei, add_edges], dim=1)

        return ei

    # =========================================================
    #  修改后的 forward，加入扰动实验接口
    # =========================================================
    def forward(self, dataset, *, epoch: int = 0, perturb: dict = None):
        """
        前向计算：
          - dataset.graph['node_feat'] : 节点特征，形状 (N,F) 或 (T,N,F)
          - dataset.graph['edge_index']: 原始图结构 (2,E)，可选
          - epoch: 当前训练轮数（如需做随 epoch 变化的策略可使用）
          - perturb: dict，用于扰动实验（可为 None）
            示例：
              perturb = {
                  "feat_noise_std": 0.1,
                  "feat_drop_rate": 0.2,
                  "shuffle_feat": False,
                  "edge_drop_rate": 0.3,
                  "edge_add_rate": 0.1,
              }
        """
        node_feat = dataset.graph['node_feat']  # (N, F) 或 (T, N, F)
        device = node_feat.device

        # --------- 对输入特征做扰动（若有配置） ----------
        node_feat = self._perturb_features(node_feat, perturb)

        # --------- 安全获取 edge_index ----------
        if self.use_graph and 'edge_index' in dataset.graph and dataset.graph['edge_index'] is not None:
            edge_index = dataset.graph['edge_index'].to(device)
        else:
            edge_index = None

        # --------- 若使用图分支，对边做扰动 ----------
        if edge_index is not None and perturb is not None:
            # 这里的 num_nodes 指的是“基础图”的节点数：
            # - 若 node_feat 为 (N,F)，则 num_nodes = N
            # - 若 node_feat 为 (T,N,F)，则 num_nodes = N（每个时间步一个 N 节点的子图）
            if node_feat.dim() == 2:
                num_nodes_for_graph = node_feat.size(0)
            elif node_feat.dim() == 3:
                num_nodes_for_graph = node_feat.size(1)
            else:
                raise ValueError(f"node_feat 期望 2D 或 3D，实际为 {node_feat.shape}")
            edge_index = self._perturb_edges(edge_index, num_nodes_for_graph, perturb)

        # --------- 动态超图：特征级可学习融合 ----------
        x_euc = self.input_map(node_feat)  # (N, H) 或 (T,N,H)
        if self.use_dhyper and self.dhgnn_conv is not None:
            x_dh, _, _ = self.dhgnn_conv(x_euc, args=self.args)  # (N, H) 或 (T,N,H)
            w_dh = torch.sigmoid(self._w_dh_logit)  # 标量 ∈ [0,1]
            x_euc = (1.0 - w_dh) * x_euc + w_dh * x_dh

        # --------- Transformer 分支 ----------
        x_tr = self.trans_conv(x_euc)  # (N, H+1) 或 (T*N, H+1)，取决于你的 TransConv 实现

        # --------- Graph 分支（可选，自动处理 2D/3D） ----------
        def _repeat_edge_index_over_time(edge_index, T: int, N: int, device, dtype=None):
            """
            将原始 edge_index (2, E) 复制到每个时间步，并在节点编号上加 t*N 的偏移，
            形成块对角大图；返回 (2, T*E)。
            """
            assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index 必须是 (2, E)"
            row, col = edge_index[0].to(device), edge_index[1].to(device)
            if dtype is not None:
                row = row.to(dtype)
                col = col.to(dtype)
            offsets = (torch.arange(T, device=device, dtype=row.dtype).unsqueeze(1)) * N  # (T,1)
            row_t = row.unsqueeze(0) + offsets  # (T,E)
            col_t = col.unsqueeze(0) + offsets  # (T,E)
            return torch.stack([row_t.reshape(-1), col_t.reshape(-1)], dim=0)  # (2, T*E)

        def _sanitize_edge_index_for_size(edge_index, N_target: int):
            """
            过滤掉 >= N_target 的索引，保证所有条目落在 [0, N_target-1]。
            返回过滤后的 edge_index 以及保留比例（便于日志定位）。
            """
            row, col = edge_index[0], edge_index[1]
            mask = (row >= 0) & (row < N_target) & (col >= 0) & (col < N_target)
            if mask.all():
                return edge_index, 1.0
            ei_new = torch.stack([row[mask], col[mask]], dim=0)
            keep_ratio = mask.float().mean().item()
            return ei_new, keep_ratio

        if edge_index is not None and self.graph_conv is not None:
            ei0 = edge_index.to(x_euc.device)
            if ei0.dtype != torch.long:
                ei0 = ei0.long()

            if x_euc.dim() == 2:
                # (N,H)
                x_gc_in = x_euc
                N_feat = x_gc_in.size(0)

                ei_use, _ = _sanitize_edge_index_for_size(ei0, N_feat)
                x_gc_euc = self.graph_conv(x_gc_in, ei_use)  # (N,H)
                x_gc_hyp = euc2lorentz(x_gc_euc, k=self.manifold_out.k)

            elif x_euc.dim() == 3:
                # (T,N,H)
                T, N_step, Hdim = x_euc.shape
                x_gc_in = x_euc.reshape(T * N_step, Hdim)  # (T*N, H)
                ei_time = _repeat_edge_index_over_time(
                    ei0, T=T, N=N_step, device=x_euc.device, dtype=torch.long
                )

                N_feat = x_gc_in.size(0)
                ei_use, _ = _sanitize_edge_index_for_size(ei_time, N_feat)

                x_gc_flat = self.graph_conv(x_gc_in, ei_use)          # (T*N,H)
                x_gc_hyp = euc2lorentz(x_gc_flat, k=self.manifold_out.k)  # (T*N,H+1)
            else:
                raise ValueError(f"x_euc 期望 2D 或 3D，实际 {x_euc.shape}")
        else:
            x_gc_hyp = None

        # --------- 超曲面解码 + logit 级可学习融合 ----------
        tr_logits = self.decode_trans(x_tr)  # (N, C) 或 (T*N, C)

        if x_gc_hyp is not None:
            gc_logits = self.decode_graph(x_gc_hyp)  # (N, C) 或 (T*N, C)
            w_graph = torch.sigmoid(self._w_graph_logit)  # 标量 ∈ [0,1]
            out = (1.0 - w_graph) * tr_logits + w_graph * gc_logits
        else:
            out = tr_logits

        if getattr(self.args, "return_emb", False):
            return out, x_tr  # logits, embedding（Transformer 分支的超曲面表征）
        else:
            return out
'''