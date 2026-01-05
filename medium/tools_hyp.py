# medium/tools_hyp.py
import math
import torch
import torch.nn.functional as F

def hyp_weighted_sum(manifold, X_hyp: torch.Tensor, W: torch.Tensor):
    X_tan = manifold.logmap0(X_hyp)
    if X_tan.dim() == 2 and W.dim() == 1:
        out_tan = torch.einsum('m,md->d', W, X_tan)
    elif X_tan.dim() == 3 and W.dim() == 2:
        out_tan = torch.einsum('bm,bmd->bd', W, X_tan)
    else:
        raise ValueError(f"Shape mismatch: X_tan={X_tan.shape}, W={W.shape}")
    out = manifold.expmap0(out_tan)
    return manifold.proj(out)

def merge_heads_hyp(manifold, X_hyp: torch.Tensor):
    X_tan = manifold.logmap0(X_hyp)
    out_tan = X_tan.mean(dim=0)
    out = manifold.expmap0(out_tan)
    return manifold.proj(out)

def memory_efficient_hyp_attn(
    manifold,
    Q_hyp: torch.Tensor,   # (Hh,Nq,Dm)
    K_hyp: torch.Tensor,   # (Hh,Nk,Dm)
    V_hyp: torch.Tensor,   # (Hh,Nk,Dm)
    q_chunk_size: int = 256,
    k_chunk_size: int = 256,
    work_dtype: torch.dtype = torch.float32,
):
    Hh, Nq, Dm = Q_hyp.shape
    Nk = K_hyp.shape[1]
    device = Q_hyp.device

    Q_t = manifold.logmap0(Q_hyp).to(work_dtype)
    K_t = manifold.logmap0(K_hyp).to(work_dtype)
    V_t = manifold.logmap0(V_hyp).to(work_dtype)

    m = torch.full((Hh, Nq, 1), float("-inf"), dtype=work_dtype, device=device)
    s = torch.zeros((Hh, Nq, 1), dtype=work_dtype, device=device)
    out_accum = torch.zeros((Hh, Nq, Dm), dtype=work_dtype, device=device)

    ks = 0
    scale = 1.0 / math.sqrt(Dm)
    while ks < Nk:
        ke = min(ks + int(k_chunk_size), Nk)
        Kb = K_t[:, ks:ke, :]
        Vb = V_t[:, ks:ke, :]
        scores = torch.matmul(Q_t, Kb.transpose(-1, -2)) * scale  # (Hh,Nq,kb)

        m_block = scores.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m, m_block)

        exp_m_diff_prev = torch.exp(m - m_new)
        s_scaled   = s * exp_m_diff_prev
        out_scaled = out_accum * exp_m_diff_prev

        exp_scores = torch.exp(scores - m_new)
        s_block    = exp_scores.sum(dim=-1, keepdim=True)
        out_block  = torch.matmul(exp_scores, Vb)

        s         = s_scaled   + s_block
        out_accum = out_scaled + out_block
        m         = m_new
        ks = ke

    out_tan = out_accum / (s + 1e-9)
    out_hyp = manifold.expmap0(out_tan)
    return manifold.proj(out_hyp)
