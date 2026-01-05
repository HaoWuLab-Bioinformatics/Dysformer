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


class HConstructor(nn.Module):
    """
    动态超图构造器（稳定版：不在当前计算图内改写 Parameter）
    -----------------------------------------------------------
    结构流程：节点 SA → V→E → E→V → 热核小波 → 融合
    特性：
      • 支持基于饱和度阈值的动态增删超边：
          - 增边动作仅标记 (_mark_expand)，真正扩张参数推迟到下一次 forward
          - 因此不会在反向过程中产生 in-place Parameter 改写
      • 所有算子避免原地写入，防止 autograd 版本冲突
      • 内置稳定的“分块流式 softmax”注意力实现：_memory_efficient_attn_2d
    """

    # ---------- 初始化 ----------
    def __init__(self, num_edges, f_dim, iters=1, eps=1e-8, hidden_dim=256):
        super().__init__()
        self.num_edges = int(num_edges)
        self.f_dim     = int(f_dim)
        self.eps       = eps
        self.iters     = int(iters)
        self.scale     = self.f_dim ** -0.5

        # 可学习的超边原型（均值 + log σ）
        self.edges_mu        = nn.Parameter(torch.randn(max(1, self.num_edges), self.f_dim) * 0.02)
        self.edges_logsigma  = nn.Parameter(torch.zeros(max(1, self.num_edges), self.f_dim))

        # 标记：下一次 forward 是否需要真正扩张 Parameter
        self._need_expand = False

        # 节点自注意力
        self.v_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.v_k = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.v_v = nn.Linear(self.f_dim, self.f_dim, bias=False)

        # V→E 交互
        self.ve_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.ve_k = nn.Linear(self.f_dim, self.f_dim, bias=False)

        # 超边自注意力
        self.e_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.e_k = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.e_v = nn.Linear(self.f_dim, self.f_dim, bias=False)

        # 归一化与融合
        self.norm_x = nn.LayerNorm(self.f_dim)
        self.norm_e = nn.LayerNorm(self.f_dim)

        hid = max(self.f_dim, int(hidden_dim))
        self.edge_update = nn.Sequential(
            nn.Linear(2 * self.f_dim, hid),
            nn.ReLU(inplace=False),   # 避免原地写
            nn.Linear(hid, self.f_dim),
        )
        self.node_proj  = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.nodes_fuse = nn.Linear(self.f_dim, self.f_dim, bias=False)

        # 默认多头
        self.h        = 4
        self.dropout  = nn.Dropout(p=0.0)
        self._last_H  = None   # 可用于调试缓存（未使用）

    # ---------- 工具 ----------
    @staticmethod
    def _get(args, name, default):
        return getattr(args, name, default)

    def mask_attn(self, attn: torch.Tensor, k: int, dim: int = -1):
        """在 dim 上保留每行 top-k，并重新 L1 归一化（不做原地改写）"""
        if k <= 0:
            return attn
        k = min(k, attn.size(dim))
        topk = torch.topk(attn, k, dim=dim)
        mask = torch.zeros_like(attn, dtype=torch.bool)
        # 对独立新建的 mask 写入不影响梯度版本
        mask = mask.scatter(dim, topk.indices, True)
        attn_pruned = attn.masked_fill(~mask, 0.0)
        denom = attn_pruned.sum(dim=dim, keepdim=True) + 1e-9
        return attn_pruned / denom

    # ---------- 动态边数调节 ----------
    def _mark_expand(self):
        """仅做标记；真正扩张推迟到下一次 forward"""
        self._need_expand = True

    @torch.no_grad()
    def _expand_parameters(self):
        """
        把 edges_mu / edges_logsigma 扩到 self.num_edges 行（no-grad 环境）
        使用重新赋值 nn.Parameter 的方式，避免 .data 带来的潜在版本问题
        注意：优化器的 param list 会在下一次创建优化器或手动更新时刷新
        """
        cur_n, feat = self.edges_mu.shape
        if self.num_edges <= cur_n:
            self._need_expand = False
            return

        new_rows = self.num_edges - cur_n
        mu_pad = self.edges_mu[-1:].repeat(new_rows, 1).detach().clone()
        ls_pad = self.edges_logsigma[-1:].repeat(new_rows, 1).detach().clone()

        new_mu = torch.cat([self.edges_mu.detach(),       mu_pad], dim=0)
        new_ls = torch.cat([self.edges_logsigma.detach(), ls_pad], dim=0)

        # 重新注册为 Parameter（非原地）
        self.edges_mu       = nn.Parameter(new_mu)
        self.edges_logsigma = nn.Parameter(new_ls)
        self._need_expand = False

    def _adjust_edges(self, s_level: float, args):
        """根据饱和度 s_level ∈ [0,1] 动态增删超边"""
        cur_epoch  = self._get(args, "epoch", 0)
        edge_warm  = self._get(args, "edge_warm", 0)
        if cur_epoch < edge_warm or not self._get(args, "use_dynamic", True):
            return
        up, low = self._get(args, "up_bound", 0.95), self._get(args, "low_bound", 0.60)
        min_e   = self._get(args, "min_num_edges", 4)

        if   s_level > float(up):  self.num_edges += 1
        elif s_level < float(low): self.num_edges = max(self.num_edges - 1, int(min_e))
        else:                      return

        # 只打标记，扩边延迟到下一次 forward
        if self.num_edges > self.edges_mu.size(0):
            self._mark_expand()

    # ---------- Lorentz 辅助 ----------
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
        """
        返回 -(d_Lorentz(Q,K)^2)，作为注意力 logits。
        Q: (Hh, Nq, Dh)  K: (Hh, Nk, Dh)
        """
        Hh, Nq, Dh = Q.shape
        Nk = K.shape[1]
        Ql = self.euc2lorentz(Q.reshape(-1, Dh), k).reshape(Hh, Nq, Dh + 1)
        Kl = self.euc2lorentz(K.reshape(-1, Dh), k).reshape(Hh, Nk, Dh + 1)
        tQ, sQ = Ql[..., :1], Ql[..., 1:]
        tK, sK = Kl[..., :1], Kl[..., 1:]
        m1 = -(tQ @ tK.transpose(-1, -2)).squeeze(-3)   # (Hh,Nq,Nk)
        m2 =  (sQ @ sK.transpose(-1, -2))               # (Hh,Nq,Nk)
        Lij = m1 + m2
        cosh_d = -Lij / max(k, 1e-6)
        dij = self.acosh_safe(cosh_d)
        return -(dij ** 2)

    # ---------- 多头工具 ----------
    def _split_heads(self, x, h):
        N, D = x.shape
        Dh = D // h
        return x.view(N, h, Dh).transpose(0, 1).contiguous()  # (Hh,N,Dh)

    def _merge_heads(self, x):
        Hh, N, Dh = x.shape
        return x.transpose(0, 1).contiguous().view(N, Hh * Dh)

    # ---------- 归一化 / 小波 ----------
    def _norm_adj(self, A, eps=1e-9):
        deg = A.sum(dim=-1) + eps
        inv_sqrt = torch.pow(deg, -0.5)
        A1 = A * inv_sqrt.unsqueeze(-1)
        A2 = A1 * inv_sqrt.unsqueeze(-2)
        return A2

    def _degree_norm(self, H, eps=1e-9):
        Dv = H.sum(dim=1, keepdim=True) + eps
        De = H.sum(dim=0, keepdim=True) + eps
        Hn = H / De
        return Hn / torch.sqrt(Dv)

    def _heat_wavelet(self, A_norm, X, scales=(0.25, 0.5, 1.0),
                      order: int = 2, aggregate: str = "sum"):
        N = A_norm.size(0)
        I = torch.eye(N, device=X.device, dtype=X.dtype)
        L = I - A_norm
        outs = []
        for s in scales:
            Y = X - s * (L @ X)
            if order >= 2:
                Y = Y + (s ** 2) / 2.0 * (L @ (L @ X))
            if order >= 3:
                Y = Y - (s ** 3) / 6.0 * (L @ (L @ (L @ X)))
            outs.append(Y)
        if aggregate == "concat":
            return torch.cat(outs, -1)
        return torch.stack(outs, -2).sum(-2)

    # ---------- 稳定/节省显存的二维注意力 ----------
    def _memory_efficient_attn_2d(
        self,
        Q: torch.Tensor,  # (Hh,Nq,Dh)
        K: torch.Tensor,  # (Hh,Nk,Dh)
        V: torch.Tensor,  # (Hh,Nk,Dh)
        use_hyper: bool = False,
        k_curv: float = 1.0,
        q_chunk_size: int = 256,
        k_chunk_size: int = 256,
        work_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        以“分块流式 softmax”方式计算注意力输出，避免一次性构建 (Nq×Nk) 的大矩阵。
        返回形状：(Hh,Nq,Dh)
        关键：全程避免原地写入（_ 结尾的 API），确保 autograd 版本不冲突。
        """
        Hh, Nq, Dh = Q.shape
        Nk = K.shape[1]
        device = Q.device

        # 为稳定性，低精度输入在内部转换为 float32 计算
        compute_dtype = torch.float32 if work_dtype in (torch.float16, torch.bfloat16) else work_dtype

        Qw = Q.to(compute_dtype)
        Kw = K.to(compute_dtype)
        Vw = V.to(compute_dtype)

        # 状态（流式 softmax 的“最大值-和”的更新）
        m = torch.full((Hh, Nq, 1), float("-inf"), dtype=compute_dtype, device=device)
        s = torch.zeros((Hh, Nq, 1), dtype=compute_dtype, device=device)
        out_accum = torch.zeros((Hh, Nq, Dh), dtype=compute_dtype, device=device)

        # 以 K/V 为外环分块（Nq 全量一次处理，显存更稳）
        ks = 0
        while ks < Nk:
            ke = min(ks + int(k_chunk_size), Nk)
            Kb = Kw[:, ks:ke, :]                           # (Hh, kb, Dh)
            Vb = Vw[:, ks:ke, :]                           # (Hh, kb, Dh)

            if use_hyper:
                scores = self.hyperbolic_score(Qw, Kb, float(k_curv))  # (Hh,Nq,kb)
            else:
                scores = torch.matmul(Qw, Kb.transpose(-1, -2)) * (1.0 / math.sqrt(Dh))

            # 当前块的行最大值
            m_block = scores.max(dim=-1, keepdim=True).values          # (Hh,Nq,1)
            m_new = torch.maximum(m, m_block)                           # (Hh,Nq,1)

            # 旧量缩放到新基准
            exp_m_diff_prev = torch.exp(m - m_new)                      # (Hh,Nq,1)
            s_scaled   = s * exp_m_diff_prev
            out_scaled = out_accum * exp_m_diff_prev

            # 新块贡献
            exp_scores = torch.exp(scores - m_new)                      # (Hh,Nq,kb)
            s_block = exp_scores.sum(dim=-1, keepdim=True)              # (Hh,Nq,1)
            out_block = torch.matmul(exp_scores, Vb)                    # (Hh,Nq,Dh)

            # 更新（非原地）
            s         = s_scaled   + s_block
            out_accum = out_scaled + out_block
            m         = m_new

            ks = ke

        # 归一化得到最终输出
        attn_out = out_accum / (s + 1e-9)                               # (Hh,Nq,Dh)
        return attn_out.to(Q.dtype)

    # =========================================================
    #  主 forward
    # =========================================================
    def forward(self, inputs: torch.Tensor, args):
        # 若上轮 forward 标记了扩边，这里先执行（no-grad 环境且不在当前计算图）
        if self._need_expand:
            self._expand_parameters()

        x = self.norm_x(inputs)                       # (N,D)
        N, D = x.shape
        device = x.device

        # ————— 读取 args —————
        iters  = int(self._get(args, "iters", self.iters))
        k_e    = int(self._get(args, "k_e", 16))
        heads  = int(self._get(args, "attn_heads", 4))
        use_h  = bool(self._get(args, "use_hyper", True))
        k_curv = float(self._get(args, "k_hidden", getattr(args, "k", 1.0)))
        scales = self._get(args, "wavelet_scales", (0.25, 0.5, 1.0))
        order  = int(self._get(args, "wavelet_order", 2))

        self.h = heads
        assert D % heads == 0, f"特征维 {D} 不能被多头数 {heads} 整除"

        # ————— 采样超边原型 —————
        E = int(self.num_edges)
        mu     = self.edges_mu[:E]
        sigma  = self.edges_logsigma.exp()[:E]
        edges  = mu + sigma * torch.randn_like(mu)

        H      = None
        H_raw  = None

        # ===== 迭代 iters 次 =====
        for _ in range(iters):
            # (1) 节点自注意力
            Qv = self._split_heads(self.v_q(x), heads)  # (Hh,N,Dh)
            Kv = self._split_heads(self.v_k(x), heads)
            Vv = self._split_heads(self.v_v(x), heads)

            Zv = self._memory_efficient_attn_2d(
                Q=Qv, K=Kv, V=Vv,
                use_hyper=use_h, k_curv=k_curv,
                q_chunk_size=256, k_chunk_size=256,
                work_dtype=torch.float32  # 使用稳定精度计算注意力
            )  # (Hh,N,Dh)
            x_sa = self.node_proj(self._merge_heads(Zv)) + x  # 残差：非原地

            # (2) V→E 构图
            Q_ve = self._split_heads(self.ve_q(x_sa), heads)         # (Hh,N,Dh)
            K_ve = self._split_heads(self.ve_k(self.norm_e(edges)), heads)  # (Hh,E,Dh)

            score_ve = (self.hyperbolic_score(Q_ve, K_ve, k_curv)
                        if use_h else
                        torch.matmul(Q_ve, K_ve.transpose(-1, -2)) / math.sqrt(Q_ve.size(-1)))
            H_raw = score_ve.mean(0)                    # (N,E)
            H_prob = F.softmax(H_raw, dim=-1)
            H      = self.mask_attn(H_prob, k=k_e, dim=-1)  # (N,E)

            # 聚合节点到超边
            E_from_V = torch.matmul(H.transpose(0, 1), x_sa)        # (E,D)

            # (2.5) 超边自注意力
            norm_edges = self.norm_e(edges)
            Qe = self._split_heads(self.e_q(norm_edges), heads)
            Ke = self._split_heads(self.e_k(norm_edges), heads)
            Ve = self._split_heads(self.e_v(norm_edges), heads)
            E_self = self._memory_efficient_attn_2d(
                Q=Qe, K=Ke, V=Ve,
                use_hyper=use_h, k_curv=k_curv,
                q_chunk_size=256, k_chunk_size=256,
                work_dtype=torch.float32
            )  # (Hh,E,Dh)
            E_self_m = self._merge_heads(E_self)  # (E,D)

            # 更新超边（非原地）
            edges = self.norm_e(self.edge_update(torch.cat([E_from_V, E_self_m], dim=-1)) + edges)

            # (3) E→V
            Hn = self._degree_norm(H)
            X_from_E = torch.matmul(Hn, edges)       # (N,D)

            # (4) 小波三路聚合
            Av = self._norm_adj(torch.matmul(H, H.transpose(0, 1)))
            Ae = self._norm_adj(torch.matmul(H.transpose(0, 1), H))

            x_sa_w  = self._heat_wavelet(Av, x_sa,     scales, order, "sum")
            x_e2v_w = self._heat_wavelet(Av, X_from_E, scales, order, "sum")
            e_w     = self._heat_wavelet(Ae, edges,    scales, order, "sum")
            X_from_eW = torch.matmul(Hn, e_w)

            x = self.norm_x(self.nodes_fuse(x_sa_w + x_e2v_w + X_from_eW) + x)
            edges = self.norm_e(edges + e_w)

            # (5) 动态饱和度调节（no-grad）
            with torch.no_grad():
                de = (H > 0).float().sum(0)                  # (E,)
                empty = (de == 0).sum()
                s_level = 1.0 - empty.float() / max(E, 1)
                self._adjust_edges(float(s_level), args)

            # 如增边标记已打，先在本轮运行时对 edges 张量零填充保持维度
            if self.num_edges > edges.size(0):
                pad_rows = self.num_edges - edges.size(0)
                if self.edges_mu.size(0) >= self.num_edges:
                    pad = self.edges_mu[edges.size(0): edges.size(0) + pad_rows]
                else:
                    pad = self.edges_mu[-1:].repeat(pad_rows, 1)
                edges = torch.cat([edges, pad.to(edges.dtype)], dim=0)

        return edges, H, H_raw





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
        # 这里沿用你项目中的 HGNN_conv；仅在 forward 中做可学习权重融合
        self.dhgnn_conv = HGNN_conv(self.hidden_channels, self.hidden_channels, args.num_edges) \
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
