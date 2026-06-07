
# -*- coding: utf-8 -*-
"""
Dysformer-Fusion-HDPC-Large
===========================

Drop-in replacement for the original fusion version.

Main changes:
1. Keep the Fusion training / forward logic:
   input_map -> dynamic hypergraph spatial-spectral blocks -> Lorentz TransConv
   -> optional GraphConv branch -> learnable logit fusion.
2. Replace H-KNN / dense N x N construction with scalable H-DPC
   (Hyperbolic Density Peak Clustering) incidence construction.
3. Avoid explicit dense N x N / E x E matrices in the dynamic hypergraph block:
   - linear focused node attention,
   - sparse incidence H,
   - sparse hypergraph heat-wavelet operator without materializing Av = H H^T.

The code assumes your project already provides:
    manifolds.hyp_layer.{TrainableLorentz, HypCLS, HypLinear, HypLayerNorm,
                         HypActivation, HypDropout}
    gnns.GraphConv

Typical usage:
    from Dysformer_fusion_hdpc_large import Dysformer
    model = Dysformer(args)
    logits = model(dataset, epoch=epoch)
"""

import math
import os
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Project compatibility imports
# ---------------------------------------------------------------------------
# Some Dysformer projects put Lorentz layers in `manifolds.hyp_layer`, while
# your large-scale project uses `manifolds.layer` + `manifolds.lorentz`.
# This block supports both layouts.  No change to parse.py / main.py is needed.
try:
    from manifolds.hyp_layer import (
        TrainableLorentz,
        HypCLS,
        HypLinear,
        HypLayerNorm,
        HypActivation,
        HypDropout,
    )
except Exception:
    from manifolds.layer import HypCLS, HypLinear, HypLayerNorm, HypActivation, HypDropout
    from manifolds.lorentz import Lorentz

    class TrainableLorentz(Lorentz):
        """
        Compatibility wrapper for projects that do not provide
        manifolds.hyp_layer.TrainableLorentz.

        It intentionally keeps the same constructor used by the Fusion code:
            TrainableLorentz(c_init=..., c_max=...)
        and maps c_init to Lorentz(k=...).
        """
        def __init__(self, c_init=1.0, c_max=None, k=None):
            kk = c_init if k is None else k
            super().__init__(k=float(kk))
            self.c_max = c_max

try:
    from gnns import GraphConv
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This file expects your project module `gnns.py` to export GraphConv."
    ) from exc


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _get(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _as_tuple(x: Any, default: Tuple[float, ...] = (0.25, 0.5, 1.0)) -> Tuple[float, ...]:
    if x is None:
        return default
    if isinstance(x, str):
        return tuple(float(v.strip()) for v in x.split(",") if v.strip())
    if isinstance(x, Iterable):
        return tuple(float(v) for v in x)
    return (float(x),)


def _safe_heads(dim: int, requested_heads: int) -> int:
    requested_heads = max(1, int(requested_heads))
    if dim % requested_heads == 0:
        return requested_heads
    for h in range(min(requested_heads, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


def _manifold_k(manifold: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = getattr(manifold, "k", None)
    if k is None:
        k = getattr(manifold, "c", 1.0)
    if isinstance(k, torch.Tensor):
        return k.to(device=device, dtype=dtype)
    return torch.tensor(float(k), device=device, dtype=dtype)


def euc2lorentz(x: torch.Tensor, k: Any = 1.0) -> torch.Tensor:
    if isinstance(k, torch.Tensor):
        k_t = k.to(device=x.device, dtype=x.dtype)
    else:
        k_t = torch.tensor(float(k), device=x.device, dtype=x.dtype)
    t = torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True) + k_t, min=1e-6))
    return torch.cat([t, x], dim=-1)


def _focus_power_feature(x: torch.Tensor, p: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    # x should be non-negative when this is called.
    xp = torch.pow(x, p)
    norm_x = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
    norm_xp = torch.norm(xp, p=2, dim=-1, keepdim=True).clamp_min(eps)
    return (norm_x / norm_xp) * xp


# ---------------------------------------------------------------------------
# Sparse hypergraph operators
# ---------------------------------------------------------------------------

def _coalesce(H: torch.Tensor) -> torch.Tensor:
    return H.coalesce() if H.is_sparse else H.to_sparse_coo().coalesce()


def _sparse_degrees(H: torch.Tensor, eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    H = _coalesce(H)
    rows, cols = H.indices()
    vals = H.values()
    Dv = torch.zeros(H.size(0), dtype=vals.dtype, device=vals.device)
    De = torch.zeros(H.size(1), dtype=vals.dtype, device=vals.device)
    if vals.numel() > 0:
        Dv.index_add_(0, rows, vals)
        De.index_add_(0, cols, vals)
    return Dv.clamp_min(eps), De.clamp_min(eps)


def _sparse_col_mean(H: torch.Tensor, X: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    H = _coalesce(H)
    _, De = _sparse_degrees(H, eps=eps)
    return torch.sparse.mm(H.transpose(0, 1).coalesce(), X) / De.unsqueeze(-1)


def _degree_norm_incidence(H: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Return sparse Dv^{-1/2} H De^{-1}."""
    H = _coalesce(H)
    rows, cols = H.indices()
    vals = H.values()
    Dv, De = _sparse_degrees(H, eps=eps)
    vals_norm = vals * Dv[rows].pow(-0.5) * De[cols].pow(-1.0)
    return torch.sparse_coo_tensor(
        H.indices(), vals_norm, H.size(), device=H.device, dtype=vals.dtype
    ).coalesce()


def _apply_hypergraph_av(H: torch.Tensor, X: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Apply Av @ X without materializing Av:
        Av = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
    H: sparse [N, E], X: dense [N, D]
    """
    H = _coalesce(H)
    Dv, De = _sparse_degrees(H, eps=eps)
    X1 = X / Dv.sqrt().unsqueeze(-1)
    T = torch.sparse.mm(H.transpose(0, 1).coalesce(), X1) / De.unsqueeze(-1)
    Y = torch.sparse.mm(H, T) / Dv.sqrt().unsqueeze(-1)
    return Y


import scipy.special as sp


def _heat_wavelet_sparse(
        H: torch.Tensor,
        X: torch.Tensor,
        scales: Tuple[float, ...] = (0.25, 0.5, 1.0),
        order: int = 3,
        eps: float = 1e-9,
) -> torch.Tensor:
    """
    Heat-kernel wavelet approximation using Chebyshev polynomials.
    Works entirely with sparse incidence matrix H.
    """
    if H.size(0) == 0 or H.size(1) == 0 or H._nnz() == 0:
        return X

    outs: List[torch.Tensor] = []

    for s in scales:
        s_val = float(s)

        # 1. Calculate Chebyshev coefficients (scalar math on CPU, safe for autograd)
        coeffs = []
        for k in range(order + 1):
            bessel_val = sp.iv(k, s_val)
            if k == 0:
                c_k = math.exp(-s_val) * bessel_val
            else:
                c_k = 2.0 * math.exp(-s_val) * ((-1) ** k) * bessel_val
            coeffs.append(c_k)

        # 2. Chebyshev polynomial recurrence
        T_0 = X
        Y = coeffs[0] * T_0

        if order >= 1:
            # T_1(L_tilde)X = -Av @ X
            T_1 = -_apply_hypergraph_av(H, X, eps=eps)
            Y = Y + coeffs[1] * T_1

            T_prev = T_1
            T_prev2 = T_0

            for k in range(2, order + 1):
                # T_k(L_tilde)X = -2 * Av @ T_{k-1} - T_{k-2}
                Av_T_prev = _apply_hypergraph_av(H, T_prev, eps=eps)
                T_curr = -2.0 * Av_T_prev - T_prev2
                Y = Y + coeffs[k] * T_curr

                # Update history
                T_prev2 = T_prev
                T_prev = T_curr

        outs.append(Y)

    return torch.stack(outs, dim=-2).sum(dim=-2)

# ---------------------------------------------------------------------------
# Poincare math and fusion units
# ---------------------------------------------------------------------------

class PoincareMath:
    """Numerically stable Poincare ball operations."""

    def __init__(self, c: Any = 1.0, eps: float = 1e-6):
        self.c = c if isinstance(c, torch.Tensor) else torch.tensor(float(c))
        self.eps = float(eps)

    def set_c(self, c: Any) -> None:
        self.c = c if isinstance(c, torch.Tensor) else torch.tensor(float(c))

    def _c_like(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.c, torch.Tensor):
            return self.c.to(device=x.device, dtype=x.dtype)
        return torch.tensor(float(self.c), device=x.device, dtype=x.dtype)

    def _sqrt_c(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self._c_like(x), min=self.eps))

    def proj(self, x: torch.Tensor, safe_margin: float = 1e-3) -> torch.Tensor:
        sqrt_c = self._sqrt_c(x)
        maxnorm = (1.0 - safe_margin) / sqrt_c
        norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True).clamp_min(self.eps))
        return torch.where(norm > maxnorm, x * (maxnorm / norm), x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self._c_like(x)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
        return self.proj(num / den.clamp_min(self.eps))

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        c = self._c_like(v)
        sqrt_c = torch.sqrt(torch.clamp(c, min=self.eps))
        v_norm = torch.sqrt((v * v).sum(dim=-1, keepdim=True).clamp_min(self.eps))
        out = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        return self.proj(out)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        y = self.proj(y)
        c = self._c_like(y)
        sqrt_c = torch.sqrt(torch.clamp(c, min=self.eps))
        y_norm = torch.sqrt((y * y).sum(dim=-1, keepdim=True).clamp_min(self.eps))
        arg = torch.clamp(sqrt_c * y_norm, min=0.0, max=1.0 - self.eps)
        return torch.atanh(arg) * y / (sqrt_c * y_norm)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = self.mobius_add(-x, y)
        c = self._c_like(diff)
        sqrt_c = torch.sqrt(torch.clamp(c, min=self.eps))
        diff_norm = torch.sqrt((diff * diff).sum(dim=-1).clamp_min(self.eps))
        arg = torch.clamp(sqrt_c * diff_norm, min=0.0, max=1.0 - self.eps)
        return 2.0 * torch.atanh(arg) / sqrt_c


class SaturationGate(nn.Module):
    """Saturation gate kept from the fusion code."""

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
        return torch.sigmoid(score) * gate


class GeometricGatingUnit(nn.Module):
    """Geometric gate for spatial / spectral fusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, h_spatial: torch.Tensor, h_spectral: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.gate(torch.cat([h_spatial, h_spectral], dim=-1)))
        return z * h_spatial + (1.0 - z) * h_spectral


class AcoshSafe(torch.autograd.Function):
    """Safe acosh with gradient clipping."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-6):
        ctx.eps = eps
        x_clamp = torch.clamp(x, min=1.0 + eps)
        ctx.save_for_backward(x_clamp)
        return torch.log(x_clamp + torch.sqrt(x_clamp - 1.0) * torch.sqrt(x_clamp + 1.0))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x_clamp,) = ctx.saved_tensors
        denom = torch.sqrt(torch.clamp(x_clamp * x_clamp - 1.0, min=ctx.eps))
        grad_x = torch.clamp(grad_output / denom, -100.0, 100.0)
        return grad_x, None


# ---------------------------------------------------------------------------
# Scalable H-DPC dynamic hypergraph constructor
# ---------------------------------------------------------------------------

class HConstructorHDPC(nn.Module):
    """
    Fusion-style dynamic hypergraph block with H-DPC construction.

    Unlike H-KNN, H-DPC first selects density-peak centers in hyperbolic
    feature space and then builds a sparse node-to-center incidence matrix H.
    The final hypergraph is not produced by connecting each node to its k
    nearest nodes.
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
        adj_cache: bool = False,
        sat_beta: float = 5.0,
        sat_tau_mode: str = "mean",
    ):
        super().__init__()
        self.num_edges = int(num_edges)
        self.f_dim = int(f_dim)
        self.iters = int(iters)
        self.eps = float(eps)

        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer("k", torch.tensor(float(k_init)), persistent=False)

        self.pmath = PoincareMath(c=1.0, eps=self.eps)

        # Fusion-style projections.
        self.v_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.v_k = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.v_v = nn.Linear(self.f_dim, self.f_dim, bias=False)

        self.ve_q = nn.Linear(self.f_dim, self.f_dim, bias=False)

        self.e_q = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.e_k = nn.Linear(self.f_dim, self.f_dim, bias=False)
        self.e_v = nn.Linear(self.f_dim, self.f_dim, bias=False)

        self.norm_x = nn.LayerNorm(self.f_dim)
        self.norm_e = nn.LayerNorm(self.f_dim)

        act_layer: nn.Module
        if str(act).lower() == "relu":
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

        # Kept from the original fusion constructor. For default binary DPC
        # assignment these parameters only affect optional multi-center weights.
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.sat_gate = SaturationGate(phi=sat_beta, tau_mode=sat_tau_mode)

        # Learnable edge seed is added to DPC cluster means. It preserves the
        # trainable hyperedge-prototype flavor of the fusion code without using
        # H-KNN.
        self.edge_seed = nn.Parameter(torch.randn(max(1, self.num_edges), self.f_dim) * 0.02)

        self._last_center_idx: Optional[torch.Tensor] = None
        self._cache_adj_enabled = bool(adj_cache)

        self.ggu = GeometricGatingUnit(self.f_dim)  # 原本的 Node GGU
        self.ggu_edge = GeometricGatingUnit(self.f_dim)  # 新增的 Hyperedge GGU
    def _positive_k(self) -> torch.Tensor:
        return F.softplus(self.k) + self.eps

    def _edge_seed_for(self, E: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if E <= self.edge_seed.size(0):
            return self.edge_seed[:E].to(device=device, dtype=dtype)
        rep = int(math.ceil(E / self.edge_seed.size(0)))
        return self.edge_seed.repeat(rep, 1)[:E].to(device=device, dtype=dtype)

    @staticmethod
    def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
        return x.view(x.size(0), heads, -1).transpose(0, 1).contiguous()

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1).contiguous().view(x.size(1), -1)

    @staticmethod
    def euc2lorentz(x: torch.Tensor, k: Any = 1.0) -> torch.Tensor:
        return euc2lorentz(x, k=k)

    def hyperbolic_score(self, Q: torch.Tensor, K: torch.Tensor, k: float) -> torch.Tensor:
        Hh, Nq, Dh = Q.shape
        Nk = K.shape[1]
        Ql = self.euc2lorentz(Q.reshape(-1, Dh), k).reshape(Hh, Nq, Dh + 1)
        Kl = self.euc2lorentz(K.reshape(-1, Dh), k).reshape(Hh, Nk, Dh + 1)

        tQ, sQ = Ql[..., :1], Ql[..., 1:]
        tK, sK = Kl[..., :1], Kl[..., 1:]
        lor = -torch.matmul(tQ, tK.transpose(-1, -2)) + torch.matmul(sQ, sK.transpose(-1, -2))
        cosh_d = torch.clamp(-lor / max(float(k), self.eps), min=1.0 + self.eps)
        d = AcoshSafe.apply(cosh_d, self.eps)
        return -(d ** 2)

    def _linear_focus_attention_2d(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        power_k: float = 2.0,
    ) -> torch.Tensor:
        """
        Linear focused attention for Euclidean tensors.
        Q/K/V: [heads, N, Dh]
        Return: [heads, N, Dh]
        """
        Qp = (F.relu(Q) + 1e-6)
        Kp = (F.relu(K) + 1e-6)
        Qp = _focus_power_feature(Qp, p=float(power_k))
        Kp = _focus_power_feature(Kp, p=float(power_k))

        kTv = torch.einsum("hnd,hnm->hdm", Kp, V)
        numerator = torch.einsum("hnd,hdm->hnm", Qp, kTv)
        k_sum = Kp.sum(dim=1)
        denominator = torch.einsum("hnd,hd->hn", Qp, k_sum).unsqueeze(-1)
        return numerator / (denominator + 1e-6)

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
        """
        Chunked softmax attention. Used only for hyperedge self-attention where
        E is small. Node self-attention uses linear focused attention.
        """
        Hh, Nq, Dh = Q.shape
        Nk = K.shape[1]
        device = Q.device
        compute_dtype = torch.float32 if work_dtype in (torch.float16, torch.bfloat16) else work_dtype
        Qw, Kw, Vw = Q.to(compute_dtype), K.to(compute_dtype), V.to(compute_dtype)

        m = torch.full((Hh, Nq, 1), -float("inf"), dtype=compute_dtype, device=device)
        s = torch.zeros((Hh, Nq, 1), dtype=compute_dtype, device=device)
        out = torch.zeros((Hh, Nq, Dh), dtype=compute_dtype, device=device)

        for ks in range(0, Nk, int(k_chunk_size)):
            ke = min(ks + int(k_chunk_size), Nk)
            Kb, Vb = Kw[:, ks:ke], Vw[:, ks:ke]
            scores = (
                self.hyperbolic_score(Qw, Kb, k_curv)
                if use_hyper
                else torch.matmul(Qw, Kb.transpose(-1, -2)) / math.sqrt(Dh)
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

        return (out / (s + 1e-9)).to(Q.dtype)

    def _distance(self, A: torch.Tensor, B: torch.Tensor, metric: str) -> torch.Tensor:
        metric = str(metric).lower()
        if metric == "cosine":
            A1 = F.normalize(A.float(), p=2, dim=-1)
            B1 = F.normalize(B.float(), p=2, dim=-1)
            return (1.0 - A1 @ B1.transpose(0, 1)).clamp_min(0.0)
        if metric == "hyperbolic":
            return self.pmath.dist(A.float().unsqueeze(1), B.float().unsqueeze(0))
        # Euclidean default.
        return torch.cdist(A.float(), B.float(), p=2)

    def _block_distance(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        metric: str,
        chunk_size: int,
    ) -> torch.Tensor:
        outs = []
        chunk_size = max(1, int(chunk_size))
        for st in range(0, A.size(0), chunk_size):
            outs.append(self._distance(A[st: st + chunk_size], B, metric=metric))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def _sample_indices(self, N: int, args: Any, device: torch.device) -> torch.Tensor:
        S = min(N, int(_get(args, "hdpc_sample_size", 4096)))
        if S >= N:
            return torch.arange(N, device=device, dtype=torch.long)

        mode = str(_get(args, "hdpc_sample_mode", "stride")).lower()
        if mode == "random":
            return torch.randperm(N, device=device)[:S].sort().values

        # Deterministic uniform sample; avoids epoch-to-epoch sampling noise.
        idx = torch.linspace(0, N - 1, steps=S, device=device)
        return idx.round().long().unique()

    @torch.no_grad()
    def _select_dpc_centers(self, dpc_x: torch.Tensor, args: Any) -> torch.Tensor:
        N = dpc_x.size(0)
        device = dpc_x.device
        metric = str(_get(args, "hdpc_metric", "hyperbolic")).lower()
        chunk = int(_get(args, "hdpc_dpc_chunk_size", 1024))

        sample_idx = self._sample_indices(N, args, device)
        sample_x = dpc_x[sample_idx]
        S = sample_x.size(0)
        E = max(1, min(int(self.num_edges), S))

        if S <= 1:
            return sample_idx[:1]

        dmat = self._block_distance(sample_x, sample_x, metric=metric, chunk_size=chunk)
        dmat.fill_diagonal_(0.0)

        valid = dmat[dmat > self.eps]
        if valid.numel() == 0:
            return sample_idx[:E]

        dc_mode = str(_get(args, "hdpc_dc_mode", "quantile")).lower()
        if dc_mode == "mean":
            dc = valid.mean()
        elif dc_mode == "median":
            dc = valid.median()
        else:
            q = float(_get(args, "hdpc_dc_quantile", 0.02))
            q = min(max(q, 0.001), 0.5)
            max_q = int(_get(args, "hdpc_quantile_max_elems", 2_000_000))
            if valid.numel() > max_q:
                # Deterministic sub-sample for quantile estimation.
                take = torch.linspace(0, valid.numel() - 1, steps=max_q, device=device).round().long()
                valid_q = valid[take]
            else:
                valid_q = valid
            dc = torch.quantile(valid_q, q)

        dc = dc.clamp_min(self.eps)
        rho = torch.exp(-((dmat / dc) ** 2)).sum(dim=1) - 1.0

        # DPC delta: distance to nearest point with higher density.
        higher = rho.unsqueeze(0) > rho.unsqueeze(1)  # row i, col j: rho[j] > rho[i]
        inf = torch.tensor(float("inf"), device=device, dtype=dmat.dtype)
        dist_to_higher = dmat.masked_fill(~higher, inf)
        delta, _ = dist_to_higher.min(dim=1)
        no_higher = torch.isinf(delta)
        if no_higher.any():
            delta = torch.where(no_higher, dmat.max(dim=1).values, delta)

        rho_n = (rho - rho.min()) / (rho.max() - rho.min() + self.eps)
        delta_n = (delta - delta.min()) / (delta.max() - delta.min() + self.eps)
        gamma = rho_n * delta_n

        if torch.all(gamma <= self.eps):
            gamma = rho_n + delta_n

        center_local = torch.topk(gamma, k=E, largest=True).indices
        return sample_idx[center_local]

    @torch.no_grad()
    def _build_hdpc_incidence(
        self,
        dpc_x: torch.Tensor,
        center_idx: torch.Tensor,
        args: Any,
        out_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build sparse H by assigning every node to DPC center(s).
        Default assign_topk = 1, i.e. hard DPC clustering.
        """
        N = dpc_x.size(0)
        E = int(center_idx.numel())
        device = dpc_x.device
        metric = str(_get(args, "hdpc_metric", "hyperbolic")).lower()
        assign_topk = max(1, int(_get(args, "hdpc_assign_topk", 1)))
        assign_topk = min(assign_topk, E)
        chunk = int(_get(args, "hdpc_assign_chunk_size", 32768))
        tau = float(_get(args, "hdpc_assign_tau", 1.0))

        centers = dpc_x[center_idx]
        rows_all: List[torch.Tensor] = []
        cols_all: List[torch.Tensor] = []
        vals_all: List[torch.Tensor] = []
        raw_all: List[torch.Tensor] = []

        for st in range(0, N, max(1, chunk)):
            ed = min(st + max(1, chunk), N)
            dist = self._distance(dpc_x[st:ed], centers, metric=metric)

            if assign_topk == 1:
                min_dist, cols = dist.min(dim=1)
                rows = torch.arange(st, ed, device=device, dtype=torch.long)
                vals = torch.ones_like(min_dist, dtype=out_dtype)
                raw = (-min_dist).to(out_dtype)
            else:
                min_dist, cols = torch.topk(dist, k=assign_topk, dim=1, largest=False)
                raw_score = -self.beta.detach().abs().float() * min_dist + self.gamma.detach().float()
                if bool(_get(args, "hdpc_use_saturation_gate", False)):
                    vals = self.sat_gate(raw_score).to(out_dtype)
                    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-9)
                else:
                    vals = F.softmax(raw_score / max(tau, self.eps), dim=1).to(out_dtype)
                raw = raw_score.to(out_dtype)
                rows = torch.arange(st, ed, device=device, dtype=torch.long).unsqueeze(1).expand(-1, assign_topk)

            rows_all.append(rows.reshape(-1))
            cols_all.append(cols.reshape(-1).long())
            vals_all.append(vals.reshape(-1))
            raw_all.append(raw.reshape(-1))

        rows_cat = torch.cat(rows_all, dim=0)
        cols_cat = torch.cat(cols_all, dim=0)
        vals_cat = torch.cat(vals_all, dim=0)
        raw_cat = torch.cat(raw_all, dim=0)

        indices = torch.stack([rows_cat, cols_cat], dim=0)
        H = torch.sparse_coo_tensor(indices, vals_cat, size=(N, E), device=device, dtype=out_dtype).coalesce()
        H_raw = torch.sparse_coo_tensor(indices, raw_cat, size=(N, E), device=device, dtype=out_dtype).coalesce()
        return H, H_raw

    def _adjust_edges(self, s_level: float, args: Any) -> None:
        cur_epoch = int(_get(args, "epoch", 0))
        edge_warm = int(_get(args, "edge_warm", 0))
        if cur_epoch < edge_warm or not bool(_get(args, "use_dynamic", True)):
            return

        up = float(_get(args, "up_bound", 0.95))
        low = float(_get(args, "low_bound", 0.60))
        min_e = int(_get(args, "min_num_edges", 4))
        max_e = int(_get(args, "max_num_edges", max(self.num_edges, int(_get(args, "num_edges", self.num_edges)))))

        if s_level > up and self.num_edges < max_e:
            self.num_edges += 1
        elif s_level < low:
            self.num_edges = max(self.num_edges - 1, min_e)

    def _prepare_dpc_space(self, x: torch.Tensor, k_curv: torch.Tensor, args: Any) -> torch.Tensor:
        metric = str(_get(args, "hdpc_metric", "hyperbolic")).lower()
        z = self.ve_q(x)
        if metric == "hyperbolic":
            self.pmath.set_c(k_curv.detach())
            return self.pmath.expmap0(z).detach()
        if metric == "cosine" or bool(_get(args, "hdpc_l2_normalize", True)):
            return F.normalize(z, p=2, dim=-1).detach()
        return z.detach()

    def forward(self, inputs: torch.Tensor, args: Any, return_node_feat: bool = False):
        original_shape = None
        if inputs.dim() == 3:
            original_shape = inputs.shape[:-1]
            x = inputs.reshape(-1, inputs.size(-1))
        elif inputs.dim() == 2:
            x = inputs
        else:
            raise ValueError(f"HConstructorHDPC expects 2D or 3D input, got shape {tuple(inputs.shape)}")

        x = self.norm_x(x)
        N, D = x.shape
        heads = _safe_heads(D, int(_get(args, "attn_heads", 4)))
        power_k = float(_get(args, "power_k", 2.0))
        use_h_edge = bool(_get(args, "use_hyper_edge_attn", False))
        scales = _as_tuple(_get(args, "wavelet_scales", (0.25, 0.5, 1.0)))
        order = int(_get(args, "wavelet_order", 2))
        iters = int(_get(args, "ss_iters", self.iters))

        k_curv = self._positive_k()
        k_val = float(k_curv.detach().item())
        self.pmath.set_c(k_curv.detach())

        H: Optional[torch.Tensor] = None
        H_raw: Optional[torch.Tensor] = None
        edges: Optional[torch.Tensor] = None
        x_out = x

        for _ in range(max(1, iters)):
            # 1) Large-scale node self-attention: linear focused attention.
            Qv = self._split_heads(self.v_q(x_out), heads)
            Kv = self._split_heads(self.v_k(x_out), heads)
            Vv = self._split_heads(self.v_v(x_out), heads)
            x_sa = self.node_proj(
                self._merge_heads(self._linear_focus_attention_2d(Qv, Kv, Vv, power_k=power_k))
            ) + x_out

            # 2) H-DPC construction in hyperbolic / normalized feature space.
            dpc_x = self._prepare_dpc_space(x_sa, k_curv=k_curv, args=args)
            center_idx = self._select_dpc_centers(dpc_x, args=args)
            self._last_center_idx = center_idx.detach()
            H, H_raw = self._build_hdpc_incidence(dpc_x, center_idx, args=args, out_dtype=x_sa.dtype)

            # 2.5) V -> E aggregation: DPC cluster means + learnable seed.
            edges_from_v = _sparse_col_mean(H, x_sa, eps=self.eps)
            E = edges_from_v.size(0)
            seed_scale = float(_get(args, "hdpc_seed_scale", 1.0))
            edges = self.norm_e(edges_from_v + seed_scale * self._edge_seed_for(E, x_sa.dtype, x_sa.device))

            # 3) Hyperedge self-attention. E is small; chunked attention is safe.
            e_heads = _safe_heads(D, heads)
            norm_edges = self.norm_e(edges)
            Qe = self._split_heads(self.e_q(norm_edges), e_heads)
            Ke = self._split_heads(self.e_k(norm_edges), e_heads)
            Ve = self._split_heads(self.e_v(norm_edges), e_heads)
            if bool(_get(args, "edge_linear_attention", True)):
                E_self = self._merge_heads(self._linear_focus_attention_2d(Qe, Ke, Ve, power_k=power_k))
            else:
                E_self = self._merge_heads(
                    self._memory_efficient_attn_2d(
                        Qe,
                        Ke,
                        Ve,
                        use_hyper=use_h_edge,
                        k_curv=k_val,
                        k_chunk_size=int(_get(args, "edge_attn_chunk_size", 256)),
                    )
                )
            edges = self.norm_e(self.edge_update(torch.cat([edges_from_v, E_self], dim=-1)) + edges)

            # 4) E -> V spatial stream.
            Hn = _degree_norm_incidence(H, eps=self.eps)
            x_spatial = torch.sparse.mm(Hn, edges)

            # 5) Sparse HGWT spectral streams, no dense Av/Ae.
            x_spectral = _heat_wavelet_sparse(H, x_sa, scales=scales, order=order, eps=self.eps)
            e_spectral = _heat_wavelet_sparse(H.transpose(0, 1).coalesce(), edges, scales=scales, order=order, eps=self.eps)

            # 6) Fusion.
            x_fused = self.ggu(x_spatial, x_spectral)
            x_out = self.norm_x(self.nodes_fuse(x_fused) + x_sa)
            #edges = self.norm_e(edges + e_spectral)
            # 将空间流 edges 与 频谱流 e_spectral 门控融合
            e_fused = self.ggu_edge(edges, e_spectral)
            # 加上残差并 LayerNorm
            edges = self.norm_e(e_fused + edges)
            # 7) Dynamic edge-number adjustment kept from Fusion.
            with torch.no_grad():
                _, De = _sparse_degrees(H, eps=self.eps)
                empty = (De <= self.eps * 10).sum().float()
                s_level = 1.0 - empty / max(float(De.numel()), 1.0)
                self._adjust_edges(float(s_level), args)

        assert edges is not None and H is not None and H_raw is not None

        if original_shape is not None:
            x_out_return = x_out.view(*original_shape, D)
        else:
            x_out_return = x_out

        if return_node_feat:
            return x_out_return, edges, H, H_raw
        return edges, H, H_raw


# ---------------------------------------------------------------------------
# Lorentz Transformer branch, scalable linear attention
# ---------------------------------------------------------------------------

class TransConvLayer(nn.Module):
    def __init__(
        self,
        manifold: Any,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        use_weight: bool = True,
        args: Any = None,
    ):
        super().__init__()
        self.manifold = manifold
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)
        self.use_weight = bool(use_weight)
        self.attention_type = str(_get(args, "attention_type", "linear_focused"))

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        self.Wv = nn.ModuleList() if use_weight else None

        for _ in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            if use_weight:
                self.Wv.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(self.out_channels)], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(self.out_channels, self.out_channels, bias=True)

        self.power_k = float(_get(args, "power_k", 2.0))
        self.trans_heads_concat = int(_get(args, "trans_heads_concat", 0))
        if self.trans_heads_concat:
            self.final_linear = nn.Linear(self.out_channels * self.num_heads, self.out_channels, bias=True)

    @staticmethod
    def fp(x: torch.Tensor, p: float = 2.0) -> torch.Tensor:
        return _focus_power_feature(x, p=p)

    def full_attention(self, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, output_attn: bool = False):
        # Warning: O(N^2). Keep only for small data or debugging.
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))
        att_weight = att_weight / self.scale + self.bias
        att_weight = nn.Softmax(dim=-1)(att_weight)
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)
        att_output = att_output.transpose(0, 1)
        att_output = self.manifold.mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        return att_output

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn: bool = False):
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        v = hyp_vs[..., 1:]

        phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        k_transpose_v = torch.einsum("nhm,nhd->hmd", phi_ks, v)
        numerator = torch.einsum("nhm,hmd->nhd", phi_qs, k_transpose_v)
        denominator = torch.einsum("nhd,hd->nh", phi_qs, torch.einsum("nhd->hd", phi_ks)).unsqueeze(-1)
        attn_output = numerator / (denominator + 1e-6)

        attn_output = attn_output + self.v_map_mlp(v)

        if self.trans_heads_concat:
            attn_output = self.final_linear(attn_output.reshape(attn_output.size(0), -1))
        else:
            attn_output = attn_output.mean(dim=1)

        k = _manifold_k(self.manifold, attn_output.device, attn_output.dtype)
        attn_output_time = torch.sqrt(torch.clamp((attn_output ** 2).sum(dim=-1, keepdim=True) + k, min=1e-6))
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

        if output_attn:
            return attn_output, None
        return attn_output

    def forward(
        self,
        query_input: torch.Tensor,
        source_input: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        output_attn: bool = False,
    ):
        q_list, k_list, v_list = [], [], []
        for h in range(self.num_heads):
            q_list.append(self.Wq[h](query_input))
            k_list.append(self.Wk[h](source_input))
            if self.use_weight:
                assert self.Wv is not None
                v_list.append(self.Wv[h](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)
        key = torch.stack(k_list, dim=1)
        value = torch.stack(v_list, dim=1)

        if self.attention_type == "full":
            attention_output, attn = self.full_attention(query, key, value, output_attn=True)
        else:
            attention_output, attn = self.linear_focus_attention(query, key, value, output_attn=True)

        if output_attn:
            return attention_output, attn
        return attention_output


class TransConv(nn.Module):
    def __init__(
        self,
        manifold_in: Any,
        manifold_hidden: Any,
        manifold_out: Any,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.5,
        use_bn: bool = True,
        use_residual: bool = True,
        use_weight: bool = True,
        use_act: bool = True,
        args: Any = None,
    ):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_bn = bool(use_bn)
        self.residual = bool(use_residual)
        self.use_act = bool(use_act)
        self.use_weight = bool(use_weight)

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = bool(_get(args, "add_positional_encoding", False))
        self.positional_encoding = HypLinear(
            self.manifold_in,
            self.in_channels,
            self.hidden_channels,
            self.manifold_hidden,
        )
        device = _get(args, "device", "cpu")
        self.epsilon = torch.tensor([1.0], device=device)

        for _ in range(self.num_layers):
            self.convs.append(
                TransConvLayer(
                    self.manifold_hidden,
                    self.hidden_channels,
                    self.hidden_channels,
                    num_heads=self.num_heads,
                    use_weight=self.use_weight,
                    args=args,
                )
            )
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels, self.hidden_channels, self.manifold_out))

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        original_shape = None
        if x_input.dim() == 3:
            original_shape = x_input.shape[:-1]
            x_input = x_input.reshape(-1, x_input.size(-1))

        layer_: List[torch.Tensor] = []

        x = self.fcs[0](x_input, x_manifold="euc")
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold="euc")
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            new_x = conv(x, x)
            if self.residual:
                new_x = self.manifold_hidden.mid_point(torch.stack((new_x, layer_[i]), dim=1))
            if self.use_bn:
                new_x = self.bns[i + 1](new_x)
            x = new_x
            layer_.append(x)

        x = self.fcs[-1](x)

        if original_shape is not None:
            x = x.view(*original_shape, x.size(-1))
        return x

    def get_attentions(self, x: torch.Tensor):
        original_shape = None
        if x.dim() == 3:
            original_shape = x.shape[:-1]
            x = x.reshape(-1, x.size(-1))

        layer_, attentions = [], []
        x = self.fcs[0](x, x_manifold="euc")
        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
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

        # Linear attention returns None attention to avoid N x N memory.
        if any(a is None for a in attentions):
            return None
        out = torch.stack(attentions, dim=0)
        if original_shape is not None:
            return out
        return out


# ---------------------------------------------------------------------------
# Dysformer: keep fusion training logic, replace constructor with H-DPC
# ---------------------------------------------------------------------------

def _build_graph_conv(in_dim: int, out_dim: int, args: Any) -> nn.Module:
    """Support both GraphConv signatures used in different Dysformer versions."""
    try:
        return GraphConv(in_dim, out_dim, args=args)
    except TypeError:
        return GraphConv(
            in_dim,
            out_dim,
            int(_get(args, "gnn_num_layers", 1)),
            float(_get(args, "gnn_dropout", 0.5)),
            bool(_get(args, "gnn_use_bn", True)),
            bool(_get(args, "gnn_use_residual", True)),
            bool(_get(args, "gnn_use_weight", True)),
            bool(_get(args, "gnn_use_init", False)),
            bool(_get(args, "gnn_use_act", True)),
        )


class Dysformer(nn.Module):
    """
    Fusion training logic:
      node_feat -> input_map/input_norm
      -> ss_blocks (Fusion spatial-spectral dynamic hypergraph, now H-DPC)
      -> TransConv -> HypCLS
      -> optional GraphConv branch -> learnable fusion of logits.

    This class is intended to replace the original fusion Dysformer class.
    """

    def __init__(self, *init_args: Any, **kwargs: Any):
        """
        Compatible constructors:

        1) Fusion/new interface:
            Dysformer(args)

        2) Original large-project interface used by parse.py:
            Dysformer(in_channels, hidden_channels, out_channels,
                      graph_weight=..., aggregate=..., args=args, ...)
        """
        super().__init__()

        if len(init_args) == 1 and "args" not in kwargs and not isinstance(init_args[0], (int, float)):
            args = init_args[0]
        else:
            args = kwargs.pop("args", None)
            if args is None:
                from types import SimpleNamespace
                args = SimpleNamespace()

            if len(init_args) >= 1:
                setattr(args, "in_channels", int(init_args[0]))
            if len(init_args) >= 2:
                setattr(args, "hidden_channels", int(init_args[1]))
            if len(init_args) >= 3:
                setattr(args, "out_channels", int(init_args[2]))

            # Preserve keyword settings passed by parse.py, for example
            # graph_weight, aggregate, use_graph, trans_num_layers, etc.
            for key, value in kwargs.items():
                if value is not None:
                    setattr(args, key, value)

        self.args = args

        c_max = float(_get(args, "c_max", 10.0))
        self.manifold_in = TrainableLorentz(c_init=float(_get(args, "k_in", 1.0)), c_max=c_max)
        self.manifold_hidden = TrainableLorentz(c_init=float(_get(args, "k_hidden", 1.0)), c_max=c_max)
        self.manifold_out = TrainableLorentz(c_init=float(_get(args, "k_out", 1.0)), c_max=c_max)

        self.in_channels = int(_get(args, "in_channels", _get(args, "feat_dim", 0)))
        self.hidden_channels = int(_get(args, "hidden_channels", _get(args, "dim", 256)))
        self.out_channels = int(_get(args, "out_channels", _get(args, "n_classes", 0)))

        if self.in_channels <= 0:
            raise ValueError("args.in_channels or args.feat_dim must be set.")
        if self.out_channels <= 0:
            raise ValueError("args.out_channels or args.n_classes must be set.")

        self.use_graph = bool(_get(args, "use_graph", False))
        self.use_dhyper = bool(_get(args, "use_dhyper", True))
        self.ss_layers = int(_get(args, "ss_layers", 1))

        self.input_map = nn.Linear(self.in_channels, self.hidden_channels, bias=False)
        self.input_norm = nn.LayerNorm(self.hidden_channels)

        self.ss_blocks = nn.ModuleList()
        if self.use_dhyper:
            for _ in range(self.ss_layers):
                self.ss_blocks.append(
                    HConstructorHDPC(
                        num_edges=int(_get(args, "num_edges", 16)),
                        f_dim=self.hidden_channels,
                        iters=int(_get(args, "ss_iters", 1)),
                        hidden_dim=max(
                            self.hidden_channels,
                            int(_get(args, "ss_hidden_dim", self.hidden_channels)),
                        ),
                        learnable_k=True,
                        k_init=float(_get(args, "k_hidden", 1.0)),
                        act=str(_get(args, "ss_act", "gelu")),
                        adj_cache=False,
                        sat_beta=float(_get(args, "sat_beta", 5.0)),
                        sat_tau_mode=str(_get(args, "sat_tau_mode", "mean")),
                    )
                )

        dh_init = float(_get(args, "dh_weight", 0.5))
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
            num_layers=int(_get(args, "trans_num_layers", _get(args, "num_layers", 1))),
            num_heads=int(_get(args, "trans_num_heads", _get(args, "num_heads", 1))),
            dropout=float(_get(args, "trans_dropout", _get(args, "dropout", 0.5))),
            use_bn=bool(_get(args, "trans_use_bn", True)),
            use_residual=bool(_get(args, "trans_use_residual", True)),
            use_weight=bool(_get(args, "trans_use_weight", True)),
            use_act=bool(_get(args, "trans_use_act", True)),
            args=args,
        )

        self.graph_conv = _build_graph_conv(self.hidden_channels, self.hidden_channels, args) if self.use_graph else None

        self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        self.decode_graph = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)

        graph_init = float(_get(args, "graph_weight", 0.5))
        graph_init = min(max(graph_init, 1e-4), 1.0 - 1e-4)
        self._w_graph_logit = nn.Parameter(
            torch.tensor(math.log(graph_init / (1.0 - graph_init)), dtype=torch.float32)
        )

        self.last_H_list: List[torch.Tensor] = []
        self.last_H_raw_list: List[torch.Tensor] = []

    @staticmethod
    def _repeat_edge_index_over_time(edge_index: torch.Tensor, T: int, N: int, device, dtype=torch.long):
        row = edge_index[0].to(device=device, dtype=dtype)
        col = edge_index[1].to(device=device, dtype=dtype)
        offsets = torch.arange(T, device=device, dtype=dtype).unsqueeze(1) * N
        row_t = row.unsqueeze(0) + offsets
        col_t = col.unsqueeze(0) + offsets
        return torch.stack([row_t.reshape(-1), col_t.reshape(-1)], dim=0)

    @staticmethod
    def _sanitize_edge_index_for_size(edge_index: torch.Tensor, N_target: int):
        row, col = edge_index[0], edge_index[1]
        mask = (row >= 0) & (row < N_target) & (col >= 0) & (col < N_target)
        if bool(mask.all()):
            return edge_index, 1.0
        return torch.stack([row[mask], col[mask]], dim=0), mask.float().mean().item()

    @staticmethod
    def _decode(decoder: nn.Module, x_hyp: torch.Tensor) -> torch.Tensor:
        if x_hyp.dim() == 3:
            shp = x_hyp.shape[:-1]
            y = decoder(x_hyp.reshape(-1, x_hyp.size(-1)))
            return y.view(*shp, y.size(-1))
        return decoder(x_hyp)

    def forward(self, dataset: Any, edge_index: Optional[torch.Tensor] = None, *, epoch: int = 0):
        """
        Compatible forwards:

        1) Fusion style:
            model(dataset, epoch=epoch), where dataset.graph contains node_feat / edge_index

        2) Original large-project style:
            model(x, edge_index)
        """
        if hasattr(dataset, "graph") and isinstance(dataset.graph, dict):
            node_feat = dataset.graph["node_feat"]
            if edge_index is None and "edge_index" in dataset.graph:
                edge_index = dataset.graph["edge_index"]
        else:
            node_feat = dataset

        device = node_feat.device
        self.args.epoch = epoch

        x_euc = self.input_norm(self.input_map(node_feat))
        H_list: List[torch.Tensor] = []
        H_raw_list: List[torch.Tensor] = []

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
        tr_logits = self._decode(self.decode_trans, x_tr)

        if edge_index is not None:
            edge_index = edge_index.to(device)
        elif self.use_graph and hasattr(dataset, "graph") and isinstance(dataset.graph, dict) and dataset.graph.get("edge_index", None) is not None:
            edge_index = dataset.graph["edge_index"].to(device)

        if edge_index is not None and self.graph_conv is not None:
            ei0 = edge_index.long()
            if x_euc.dim() == 2:
                ei_use, _ = self._sanitize_edge_index_for_size(ei0, x_euc.size(0))
                x_gc_euc = self.graph_conv(x_euc, ei_use)
                k_out = _manifold_k(self.manifold_out, x_gc_euc.device, x_gc_euc.dtype)
                x_gc_hyp = euc2lorentz(x_gc_euc, k=k_out)
            elif x_euc.dim() == 3:
                T, N_step, Hdim = x_euc.shape
                x_gc_in = x_euc.reshape(T * N_step, Hdim)
                ei_time = self._repeat_edge_index_over_time(ei0, T=T, N=N_step, device=x_euc.device)
                ei_use, _ = self._sanitize_edge_index_for_size(ei_time, x_gc_in.size(0))
                x_gc_flat = self.graph_conv(x_gc_in, ei_use)
                k_out = _manifold_k(self.manifold_out, x_gc_flat.device, x_gc_flat.dtype)
                x_gc_hyp = euc2lorentz(x_gc_flat, k=k_out).view(T, N_step, -1)
            else:
                raise ValueError(f"x_euc should be 2D or 3D, got {tuple(x_euc.shape)}")

            gc_logits = self._decode(self.decode_graph, x_gc_hyp)
            w_graph = torch.sigmoid(self._w_graph_logit)
            out = (1.0 - w_graph) * tr_logits + w_graph * gc_logits
        else:
            out = tr_logits

        if bool(_get(self.args, "return_emb", False)):
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
                "hdpc_num_edges": [block.num_edges for block in self.ss_blocks],
            }

    def reset_parameters(self):
        if hasattr(self.input_map, "reset_parameters"):
            self.input_map.reset_parameters()
        if self.graph_conv is not None and hasattr(self.graph_conv, "reset_parameters"):
            self.graph_conv.reset_parameters()
