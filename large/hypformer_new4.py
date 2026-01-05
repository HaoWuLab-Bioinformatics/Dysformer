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
import networkit as nk
from typing import Optional, List
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
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torch
import torch.nn as nn


from torch_sparse import SparseTensor

import torch
import torch.nn.functional as F
import faiss
import numpy as np

import torch
import torch.nn.functional as F
import faiss
import numpy as np
import torch, math, torch.nn.functional as F

# -------- (1) top-k 稀疏化 + L1 归一化 --------
def _topk_sparse_softmax(t, k, dim=-1):
    if k <= 0 or k >= t.size(dim):          # 不裁剪
        return F.softmax(t, dim=dim)
    topk = torch.topk(t, k, dim=dim)
    mask = torch.zeros_like(t, dtype=torch.bool)
    mask.scatter_(dim, topk.indices, True)
    t = t.masked_fill(~mask, -1e30)         # -∞
    out = F.softmax(t, dim=dim)
    return out

# -------- (2) 对称归一化  D_v^{-½} H D_e^{-1} --------
def _norm_he(H, eps=1e-9):
    if H.is_sparse:
        Dv = torch.sparse.sum(H, dim=1).to_dense() + eps
        De = torch.sparse.sum(H, dim=0).to_dense() + eps
        r, c = H.indices()
        val = H.values() * (Dv[r].pow(-0.5) * De[c].pow(-1.0))
        return torch.sparse_coo_tensor(H.indices(), val, H.size()).coalesce()
    else:
        Dv = H.sum(1, keepdim=True) + eps
        De = H.sum(0, keepdim=True) + eps
        return (H / De) / Dv.sqrt()

# -------- (3) 热核小波  I − sL + s²/2 L² --------
def _heat_wavelet(A_norm, X, scales=(.25,.5,1.), order=2):
    Lx  = X - torch.sparse.mm(A_norm, X)              # L·X
    outs = []
    for s in scales:
        Y = X - s * Lx                                # I − sL
        if order >= 2:
            L2x = Lx - torch.sparse.mm(A_norm, Lx)    # L²·X
            Y = Y + (s**2)/2 * L2x
        outs.append(Y)
    return torch.stack(outs, -2).sum(-2)

class ScalableHConstructor:
    """基于 SCAN 的超边构造器（支持亿级节点）。

    Parameters
    ----------
    epsilon : float, optional
        结构相似度阈值 (0‒1)。越高意味着更严格的相似性要求。
    mu : int, optional
        核心节点最少相似邻居数 (μ)。
    star : bool, optional
        若为 ``True``，对每个簇以 *星型* 方式写回 ``edge_index``（root<->member），大幅节省内存；
        若为 ``False``，则完全连接簇内所有节点（|C|² 规模，慎用）。
    min_hyperedge_size : int, optional
        过滤掉规模小于该值的簇。
    device : torch.device, optional
        返回的 ``edge_index`` 所在设备。
    """

    def __init__(self,
                 epsilon: float = 0.7,
                 mu: int = 2,
                 star: bool = True,
                 min_hyperedge_size: int = 3,
                 device: Optional[torch.device] = None):
        self.epsilon = epsilon
        self.mu = mu
        self.star = star
        self.min_hyperedge_size = min_hyperedge_size
        self.device = device or torch.device("cpu")

        # 输出属性
        self.edge_index: Optional[torch.Tensor] = None  # 2 × E′
        self._cluster_labels: Optional[List[int]] = None  # len == num_nodes

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def construct_graph(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """根据原始图 ``edge_index``（无向）构造超边并返回新的 ``edge_index``。

        Parameters
        ----------
        edge_index : torch.Tensor
            形状 ``[2, E]``，每列 ``[u, v]`` 表示一条无向边。
        num_nodes : int
            图的节点数量 (|V|)。

        Returns
        -------
        torch.Tensor
            新的 ``edge_index``（形状 ``[2, E′]``），其中每对索引来自同一簇。
        """
        # 1. ---------- 构建 NetworKit 图 ----------
        G = nk.graph.Graph(num_nodes, weighted=False, directed=False)
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        for u, v in zip(src, dst):
            if u != v:
                G.addEdge(int(u), int(v))
        G.removeSelfLoops()

        # 2. ---------- 运行 SCAN ----------
        # NetworKit 默认多线程，可根据需要设置线程数：
        # nk.setNumberOfThreads(<n>)
        #scan_algo = nk.community.SCAN(G, self.epsilon, self.mu)
        if hasattr(nk.community, "SCAN"):  # ① 优先 SCAN++
            algo = nk.community.SCAN(G, self.epsilon, self.mu)
            out_type = "scan"
        elif hasattr(nk.community, "PSCAN"):  # ② 退而求其次
            algo = nk.community.PSCAN(G, self.epsilon, self.mu)
            out_type = "scan"
        elif hasattr(nk.community, "PLP"):  # ③ 最后兜底：并行标签传播
            algo = nk.community.PLP(G)
            out_type = "partition"
        else:
            raise RuntimeError("NetworKit 未编译 SCAN / PSCAN / PLP，无法构造超图！")
        # --------------------------------------------------------------------

        algo.run()

        # --- 提取标签向量 -----------------------------------------------------
        if out_type == "scan":  # SCAN / PSCAN
            labels = algo.getClusterLabels()  # list[int]
            noise_id = nk.community.SCAN.NOISE_ID  # -1
        else:  # PLP
            labels = algo.getPartition().getVector()  # list[int]
            noise_id = None  # 没有噪声概念
        # --------------------------------------------------------------------

        self._cluster_labels = labels

        # 3. ---------- 聚簇汇总 ----------
        clusters: dict[int, list[int]] = {}

        for nid, cid in enumerate(labels):
            # 只有当 noise_id 存在且当前节点是噪声时才跳过
            if noise_id is not None and cid == noise_id:
                continue
            clusters.setdefault(cid, []).append(nid)

        # 4. ---------- 转回 edge_index ----------
        rows: List[int] = []
        cols: List[int] = []
        for nodes in clusters.values():
            if len(nodes) < self.min_hyperedge_size:
                continue  # 过滤太小的簇
            if self.star:
                root = nodes[0]  # 选簇中第一个节点为根
                for n in nodes[1:]:
                    rows.extend([root, n])  # root ↔ n（双向）
                    cols.extend([n, root])
            else:
                # 完全连接（可能非常大，慎用！）
                for i, u in enumerate(nodes):
                    for v in nodes[i + 1:]:
                        rows.extend([u, v, v, u])
                        cols.extend([v, u, u, v])

        self.edge_index = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        return self.edge_index

    # ------------------------------------------------------------------
    @property
    def cluster_labels(self):
        """返回最近一次构造得到的聚类标签列表（长度 = 节点数）。"""
        return self._cluster_labels

    # ------------------------------------------------------------------
    def save_labels(self, path: str) -> None:
        """将聚类标签保存为文本，每行一个整数。"""
        if self._cluster_labels is None:
            raise RuntimeError("No clustering available, call construct_graph() first.")
        with open(path, "w", encoding="utf-8") as fp:
            for lb in self._cluster_labels:
                fp.write(f"{lb}\n")

def _apply_Av_to_X(H, X, eps=1e-9):
    # 实现  Av @ X  ，其中 Av = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
    Dv = torch.sparse.sum(H, dim=1).to_dense().clamp_min(eps)        # (N,)
    De = torch.sparse.sum(H, dim=0).to_dense().clamp_min(eps)        # (E,)

    X1 = X / Dv.sqrt().unsqueeze(1)                                  # Dv^{-1/2} X
    T  = torch.sparse.mm(H.transpose(0,1), X1) / De.unsqueeze(1)     # De^{-1} H^T (Dv^{-1/2} X)
    Y  = torch.sparse.mm(H, T) / Dv.sqrt().unsqueeze(1)              # Dv^{-1/2} [ H (...) ]
    return Y

def _heat_wavelet_without_explicit_Av(H, X, scales=(0.25,0.5,1.0), order=2):
    # L = I - Av
    AvX = _apply_Av_to_X(H, X)
    Lx  = X - AvX
    outs = []
    for s in scales:
        Y = X - s * Lx
        if order >= 2:
            L2x = Lx - _apply_Av_to_X(H, Lx)  # L^2 x = (I-Av) Lx
            Y = Y + (s**2)/2.0 * L2x
        outs.append(Y)
    return torch.stack(outs, -2).sum(-2)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
from manifolds.layer import HypLinear  # 需已在工程中可用


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
from manifolds.layer import HypLinear  # 你的工程里已有

class TransConvLayer(nn.Module):
    """
    内存友好的 TransConv 层：
    1) 节点自注意：采用线性注意力（不构造 N×N）。
    2) V→E：只在超图稀疏位置打分并做分组 softmax（不构造 N×E）。
    3) E 表征：轻量线性映射（替代 E×E 全量注意）。
    4) 热核小波：用 Av 的稀疏算子实现，不显式构造 Av 矩阵。
    说明：V→E 动态权重默认不参与反传（可用 args.v2e_grad=1 打开），
    这样能避免稀疏乘法对稀疏值求梯度导致的显存爆炸。
    """
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

        # ==== 节点自注意（超曲）Q/K/V ====
        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        self.Wv = nn.ModuleList() if use_weight else None
        for _ in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            if use_weight:
                self.Wv.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        # 线性注意力的缩放等
        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))

        # 将线性注意力输出（欧氏空间部分）再做一次线性映射
        self.v_map_mlp = nn.Linear(out_channels, out_channels, bias=True)

        # ==== V→E 稀疏注意所需的线性投影（欧氏）====
        self.ve_q = nn.Linear(out_channels, out_channels, bias=False)
        self.ve_k = nn.Linear(out_channels, out_channels, bias=False)

        # ==== E 表征轻量变换（替代 E×E 全量注意）====
        self.e_v = nn.Linear(out_channels, out_channels, bias=True)

        # dropout
        drop_p = getattr(args, "trans_dropout", 0.0) if args is not None else 0.0
        self.dropout = nn.Dropout(p=drop_p)

        # 其它参数
        self.power_k = getattr(args, "power_k", 2.0) if args is not None else 2.0
        self.trans_heads_concat = getattr(args, "trans_heads_concat", 0) if args is not None else 0
        self.v2e_grad = getattr(args, "v2e_grad", 0) if args is not None else 0  # 0=不反传, 1=允许反传
        if self.trans_heads_concat:
            self.final_linear = nn.Linear(out_channels * self.num_heads, out_channels, bias=True)

    # ---------- 内部工具 ----------
    @staticmethod
    def _fp(x, p=2.0):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / (norm_x_p + 1e-12)) * (x ** p)

    @staticmethod
    def _norm_he(H, eps=1e-9):
        # 返回 Dv^{-1/2} H De^{-1} 的稀疏张量（用于 E→V 时左乘）
        Dv = torch.sparse.sum(H, dim=1).to_dense().clamp_min(eps)  # (N,)
        De = torch.sparse.sum(H, dim=0).to_dense().clamp_min(eps)  # (E,)
        r, c = H.indices()
        val = H.values() * (Dv[r].pow(-0.5) * De[c].pow(-1.0))
        return torch.sparse_coo_tensor(H.indices(), val, H.size(), device=H.device).coalesce()

    @staticmethod
    def _apply_Av_to_X(H, X, eps=1e-9):
        # Av @ X，Av = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}，但不显式构造 Av
        Dv = torch.sparse.sum(H, dim=1).to_dense().clamp_min(eps)    # (N,)
        De = torch.sparse.sum(H, dim=0).to_dense().clamp_min(eps)    # (E,)
        X1 = X / Dv.sqrt().unsqueeze(1)                              # Dv^{-1/2} X
        T  = torch.sparse.mm(H.transpose(0, 1), X1) / De.unsqueeze(1)  # De^{-1} H^T (...)
        Y  = torch.sparse.mm(H, T) / Dv.sqrt().unsqueeze(1)          # Dv^{-1/2} H (...)
        return Y

    @classmethod
    def _heat_wavelet_without_explicit_Av(cls, H, X, scales=(0.25, 0.5, 1.0), order=2):
        # L = I - Av；使用泰勒展开  I − sL + s²/2 L²
        AvX = cls._apply_Av_to_X(H, X)
        Lx = X - AvX
        outs = []
        for s in scales:
            Y = X - s * Lx
            if order >= 2:
                L2x = Lx - cls._apply_Av_to_X(H, Lx)  # (I-Av)Lx
                Y = Y + (s * s) * 0.5 * L2x
            outs.append(Y)
        return torch.stack(outs, dim=-2).sum(dim=-2)

    # ---------- 线性注意力：在超曲输出上做特征映射（不构造 N×N） ----------
    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
        # 输入形状： [N, H, D+1] 的洛伦兹坐标（time|space）
        qs = hyp_qs[..., 1:]  # [N, H, D]
        ks = hyp_ks[..., 1:]  # [N, H, D]
        v  = hyp_vs[..., 1:]  # [N, H, D]

        phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_qs = self._fp(phi_qs, p=self.power_k)
        phi_ks = self._fp(phi_ks, p=self.power_k)

        # K^T V（按各头聚合），输出 [H, D, D]；不会产生 N×N
        kTv = torch.einsum('nhm,nhd->hmd', phi_ks, v)          # [H, D, D]
        numer = torch.einsum('nhm,hmd->nhd', phi_qs, kTv)      # [N, H, D]
        denom = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks)).unsqueeze(-1)  # [N,H,1]
        attn_out = numer / (denom + 1e-6)                      # [N, H, D]

        # 每头映射并聚合
        vss = self.v_map_mlp(v)                                # [N, H, D]
        attn_out = attn_out + vss
        if self.trans_heads_concat:
            attn_out = self.final_linear(attn_out.reshape(attn_out.size(0), -1))  # [N, H*D]->[N,D]
        else:
            attn_out = attn_out.mean(dim=1)  # [N, D]

        # 拼回洛伦兹坐标（time|space）
        time = ((attn_out ** 2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        hyp = torch.cat([time, attn_out], dim=-1)  # [N, D+1]
        if output_attn:
            # 不再显式返回 N×N 注意力权重
            return hyp, None
        return hyp

    # ---------- 主前向 ----------
    def forward(
        self,
        x,                          # [N, D+1]  (洛伦兹坐标)
        H_sparse=None,              # 稀疏 (N,E)，若为 None 不跑 V-E/E-V
        *,
        topk_self=0,                # 保留接口（未在此实现里用到）
        topk_v2e=0,                 # 保留接口（如需逐节点 top-k，可在此基础上扩展）
        wavelet_scales=(0.25, 0.5, 1.0),
        wavelet_order=2,
        output_attn: bool = False
    ):
        # ===== 1) 节点自注意（线性注意力，O(ND)）=====
        # 分头计算 Q/K/V（HypLinear 输出为洛伦兹向量）
        qs = torch.stack([self.Wq[i](x) for i in range(self.num_heads)], dim=1)  # [N,H,D+1]
        ks = torch.stack([self.Wk[i](x) for i in range(self.num_heads)], dim=1)  # [N,H,D+1]
        if self.use_weight:
            vs = torch.stack([self.Wv[i](x) for i in range(self.num_heads)], dim=1)  # [N,H,D+1]
        else:
            vs = qs

        # 线性注意力输出（仍为洛伦兹坐标）
        x_sa_hyp = self.linear_focus_attention(qs, ks, vs, output_attn=False)  # [N, D+1]
        # 用欧氏部分继续后续计算
        x_sa = self.manifold.logmap0(x_sa_hyp)[..., 1:]  # [N, D]
        x_sa = self.v_map_mlp(x_sa)                      # [N, D]

        # ===== 2) 无超边：直接返回（映回洛伦兹）=====
        if H_sparse is None:
            time = ((x_sa ** 2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
            return torch.cat([time, x_sa], dim=-1)

        # ===== 3) V→E 稀疏注意：只在 H 的非零处打分并做分组 softmax =====
        # 先把节点表示聚合到超边（度归一化平均）
        deg_e = torch.sparse.sum(H_sparse, dim=0).to_dense().clamp_min(1.0)           # [E]
        e_repr = torch.sparse.mm(H_sparse.transpose(0, 1), x_sa) / deg_e.unsqueeze(1) # [E, D]

        # 线性投影得到 Q/K，并按头切分
        Q_all = self.ve_q(x_sa)    # [N, D]
        K_all = self.ve_k(e_repr)  # [E, D]
        assert Q_all.size(-1) % self.num_heads == 0, "out_channels 必须能被 num_heads 整除"
        Dh = Q_all.size(-1) // self.num_heads

        def split_HD(t):  # [N_or_E, D] -> [H, N_or_E, Dh]
            return t.view(t.size(0), self.num_heads, Dh).transpose(0, 1).contiguous()

        QH = split_HD(Q_all)   # [H, N, Dh]
        KH = split_HD(K_all)   # [H, E, Dh]

        row_idx, col_idx = H_sparse.indices()  # [nnz], [nnz]
        weights_per_head = []
        for h in range(self.num_heads):
            qh = QH[h]  # [N, Dh]
            kh = KH[h]  # [E, Dh]

            # 只在稀疏位置计算打分 s_ij = <q[n], k[e]> / sqrt(Dh)
            s_vals = (qh[row_idx] * kh[col_idx]).sum(-1) / math.sqrt(Dh)  # [nnz]

            # 按节点做稳定 softmax: 对每个 n 的出边做 softmax
            s_max, _ = scatter_max(s_vals, row_idx)             # [N]
            s_exp = torch.exp(s_vals - s_max[row_idx])
            denom = scatter_add(s_exp, row_idx)                 # [N]
            w_vals = s_exp / (denom[row_idx] + 1e-9)            # [nnz]

            weights_per_head.append(w_vals)

        # 多头平均权重
        H_val = torch.stack(weights_per_head, dim=0).mean(0)  # [nnz]
        if not self.v2e_grad:
            # 默认：阻断 V→E 权重的梯度，避免反传到稀疏乘法导致显存爆炸
            H_val = H_val.detach()
        H_new = torch.sparse_coo_tensor(
            torch.stack([row_idx, col_idx], dim=0), H_val, H_sparse.size(), device=x.device
        ).coalesce()

        # ===== 4) 轻量 E 变换（不做 E×E 全注意力，避免 O(E^2)）=====
        e_sa = self.e_v(e_repr)          # [E, D]
        e_sa = F.relu(e_sa, inplace=True)
        e_sa = self.dropout(e_sa)

        # ===== 5) E→V 回流 + 热核小波（不显式构造 Av）=====
        H_norm = self._norm_he(H_new)                      # 稀疏、已归一化
        x_fromE = torch.sparse.mm(H_norm, e_sa)            # [N, D]
        x_wave  = self._heat_wavelet_without_explicit_Av(H_new, x_sa,
                                                         scales=wavelet_scales,
                                                         order=wavelet_order)  # [N, D]

        # ===== 6) 残差融合并映回洛伦兹 =====
        x_out_euc = x_sa + x_fromE + x_wave                # [N, D]
        if self.trans_heads_concat:
            # 此实现里节点自注意已做均值聚合；如需 concat，请确保维度吻合
            pass
        time = ((x_out_euc ** 2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        x_out = torch.cat([time, x_out_euc], dim=-1)       # [N, D+1]

        if output_attn:
            return x_out, None
        return x_out





class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, hidden_channels, num_layers=2,
                 num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True, args=None):
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

    def forward(self, x_input, *, H_sparse=None):
        layer_ = []

        # ——(1) Euclid → Hyper——
        x = self.fcs[0](x_input, x_manifold='euc')
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        # ——(2) 多层 TransConvLayer——
        for i, conv in enumerate(self.convs):
            # ❷ 这里把 H_sparse 传进去；TransConvLayer.forward 已经支持
            x = conv(x, H_sparse=H_sparse)        # <-- 关键改动
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)

        # ——(3) 输出层——
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
        return torch.stack(attentions, dim=0)  # [layer num, N, N]
class Dysformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.5, aggregate='add',args=None):
        super().__init__()
        self.manifold_in = Lorentz(k=float(args.k_in))
        # self.manifold_hidden = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.decoder_type = args.decoder_type
        self.hconstructor = ScalableHConstructor(          # 一次性构造
            epsilon=0.7, mu=2, star=True, min_hyperedge_size=3,
            device=torch.device(args.device)
        )
        self.manifold_out = Lorentz(k=float(args.k_out))
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.hyper_weight = graph_weight                   # 超图权重
        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act, args)
        self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn, gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)

        self.aggregate = aggregate
        self.use_edge_loss = False
        self.gnn_use_bn = gnn_use_bn

        if self.decoder_type == 'euc':
            self.decode_trans = nn.Linear(self.hidden_channels, self.out_channels)
            self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.decoder_type == 'hyp':
            self.decode_graph = HypLinear(self.manifold_out, self.hidden_channels, self.hidden_channels)
            self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        else:
            raise NotImplementedError
        self.hypergraph_conv = GraphConv(
            in_channels, hidden_channels,
            gnn_num_layers, gnn_dropout,
            gnn_use_bn, gnn_use_residual,
            gnn_use_weight, gnn_use_init, gnn_use_act
        )
        # ② 超图-专用解码器
        if self.decoder_type == 'euc':
            self.decode_hyper = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.decoder_type == 'hyp':
            self.decode_hyper = HypLinear(self.manifold_out, self.hidden_channels, self.hidden_channels)
        # ③ 权重
        self.hyper_weight = 0.2      # ← 外部可单独传，默认 0.2

    def forward(self, x, edge_index):
        # ----------------------------------------------------------
        # A. 把超边 edge_index → 稀疏 H  (只在第一次 forward 时创建)
        # ----------------------------------------------------------
        if self.hconstructor.edge_index is None:
            self.hconstructor.construct_graph(edge_index, x.size(0))
        hyper_edge_index = self.hconstructor.edge_index  # (2, E′)

        # ❶ 构成稀疏 H  (N × E′)
        row, col = hyper_edge_index
        val = torch.ones_like(row, dtype=torch.float32)
        H_sparse = torch.sparse_coo_tensor(
            torch.stack([row, col]), val,
            size=(x.size(0), int(col.max()) + 1), device=x.device
        ).coalesce()

        # ----------------------------------------------------------
        # ① Transformer 分支 —— 把 H_sparse 传进去
        # ----------------------------------------------------------
        x1 = self.trans_conv(x, H_sparse=H_sparse)  # <-- 关键改动

        # ----------------------------------------------------------
        # ② 普通图 GNN 分支（可选）
        # ----------------------------------------------------------
        x2 = self.graph_conv(x, edge_index) if self.use_graph else None

        # ----------------------------------------------------------
        # ③ 超图 GNN 分支（沿用原版 GraphConv；也可删除）
        # ----------------------------------------------------------
        x3 = self.hypergraph_conv(x, hyper_edge_index)

        # ----------------------------------------------------------
        # ④ 三路融合（完全不动）
        # ----------------------------------------------------------
        if self.decoder_type == 'euc':
            z1 = self.decode_trans(self.manifold_out.logmap0(x1)[..., 1:])
            z2 = self.decode_graph(x2) if x2 is not None else 0
            z3 = self.decode_hyper(x3)
            weight_g = self.graph_weight
            weight_h = self.hyper_weight
            x_out = (1 - weight_g - weight_h) * z1 + weight_g * z2 + weight_h * z3

        elif self.decoder_type == 'hyp':
            z_graph_hyp = self.decode_graph(x2, x_manifold='euc') if x2 is not None else None
            z_hyper_hyp = self.decode_hyper(x3, x_manifold='euc')
            parts = [(1 - self.graph_weight - self.hyper_weight) * x1]
            if z_graph_hyp is not None:
                parts.append(self.graph_weight * z_graph_hyp)
            parts.append(self.hyper_weight * z_hyper_hyp)
            z_stack = torch.stack(parts, dim=1)
            z_mid = self.manifold_out.mid_point(z_stack)
            x_out = self.decode_trans(z_mid)
        else:
            raise NotImplementedError

        return x_out

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def reset_parameters(self):
        # self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
        # self.fc.reset_parameters()