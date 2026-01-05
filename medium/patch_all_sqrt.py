# utils/safe_sqrt_jit.py
# -------------------------------------------------
# 用“定参”函数替换 torch.sqrt / Tensor.sqrt，JIT 兼容。
# -------------------------------------------------
import torch
EPS = 1e-12

# 先保存原始算子以备调试
torch._orig_sqrt = torch.sqrt

# JIT 可以编译的安全实现：只有一个位置参数，无 *args / **kwargs
def safe_sqrt(x):
    # clamp 在前，避免负数 / 0
    return torch._orig_sqrt(x.clamp_min(EPS))

# 全局替换
torch.sqrt = safe_sqrt
setattr(torch.Tensor, "sqrt", safe_sqrt)

print("[safe_sqrt_jit] torch.sqrt 已替换为 JIT-安全版")
