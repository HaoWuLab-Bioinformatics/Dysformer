import pdb

import torch.nn as nn
import torch.nn.functional
import torch.nn.init as init
from manifolds.lorentz import Lorentz
import math
from geoopt import ManifoldParameter
from geoopt.optim.rsgd import RiemannianSGD
from geoopt.optim.radam import RiemannianAdam
import torch.nn.functional as F


class HypLayerNorm(nn.Module):
    def __init__(self, manifold, in_features, manifold_out=None):
        super(HypLayerNorm, self).__init__()
        self.in_features = in_features
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.layer(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        # Adjust for a different manifold if specified
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypNormalization(nn.Module):
    def __init__(self, manifold, manifold_out=None):
        super(HypNormalization, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        # Adjust for a different manifold if specified
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypActivation(nn.Module):
    def __init__(self, manifold, activation, manifold_out=None):
        super(HypActivation, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.activation = activation

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.activation(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        # Adjust for a different manifold if specified
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypDropout(nn.Module):
    def __init__(self, manifold, dropout, manifold_out=None):
        super(HypDropout, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=False):
        if training:
            x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)

            # Adjust for a different manifold if specified
            if self.manifold_out is not None:
                x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypLinear(nn.Module):
    """
    Parameters:
        manifold (manifold): The manifold to use for the linear transformation.
        in_features (int): The size of each input sample.
        out_features (int): The size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        dropout (float, optional): The dropout probability. Default is 0.1.
    """

    def __init__(self, manifold, in_features, out_features, bias=True, manifold_out=None):
        super().__init__()
        self.in_features = in_features + 1  # + 1 for time dimension
        self.out_features = out_features
        self.bias = bias
        self.manifold = manifold
        self.manifold_out = manifold_out

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp'):
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        x_space = self.linear(x)

        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        # Adjust for a different manifold if specified
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypCLS(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        cls_emb = self.manifold.random_normal((self.out_channels, self.in_channels + 1), mean=0,
                                              std=1. / math.sqrt(self.in_channels + 1))
        self.cls = ManifoldParameter(cls_emb, self.manifold, requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def forward(self, x, x_manifold='hyp', return_type='neg_dist'):
        if x_manifold != 'hyp':
            x = self.manifold.expmap0(torch.cat([torch.zeros_like(x)[..., 0:1], x], dim=-1))  # project to Lorentz

        dist = -2 * self.manifold.k - 2 * self.cinner(x, self.cls) + self.bias
        dist = dist.clamp(min=0)

        if return_type == 'neg_dist':
            return - dist
        elif return_type == 'prob':
            return 10 / (1.0 + dist)
        elif return_type == 'neg_log_prob':
            return - 10 * torch.log(1.0 + dist)
        else:
            raise NotImplementedError


class Optimizer(object):
    def __init__(self, model, args):
        # Extract optimizer types and parameters from arguments
        euc_optimizer_type = getattr(args, 'euc_optimizer_type', args.optimizer_type)  # Euclidean optimizer type
        hyp_optimizer_type = getattr(args, 'hyp_optimizer_type', args.hyp_optimizer_type)  # Hyperbolic optimizer type
        euc_lr = getattr(args, 'euc_lr', args.lr)  # Euclidean learning rate
        hyp_lr = getattr(args, 'hyp_lr', args.hyp_lr)  # Hyperbolic learning rate
        euc_weight_decay = getattr(args, 'euc_weight_decay', args.weight_decay)  # Euclidean weight decay
        hyp_weight_decay = getattr(args, 'hyp_weight_decay', args.hyp_weight_decay)  # Hyperbolic weight decay

        # Separate parameters for Euclidean and Hyperbolic parts of the model
        euc_params = [p for n, p in model.named_parameters() if
                      p.requires_grad and not isinstance(p, ManifoldParameter)]  # Euclidean parameters
        hyp_params = [p for n, p in model.named_parameters() if
                      p.requires_grad and isinstance(p, ManifoldParameter)]  # Hyperbolic parameters

        # Print the number of Euclidean and Hyperbolic parameters
        # print(f">> Number of Euclidean parameters: {sum(p.numel() for p in euc_params)}")
        # print(f">> Number of Hyperbolic parameters: {sum(p.numel() for p in hyp_params)}")
        # Initialize Euclidean optimizer

        if euc_optimizer_type == 'adam':
            optimizer_euc = torch.optim.Adam(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)
        elif euc_optimizer_type == 'sgd':
            optimizer_euc = torch.optim.SGD(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)
        else:
            raise NotImplementedError("Unsupported Euclidean optimizer type")

        # Initialize Hyperbolic optimizer if there are Hyperbolic parameters
        if hyp_params:
            if hyp_optimizer_type == 'radam':
                optimizer_hyp = RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=50, weight_decay=hyp_weight_decay)
            elif hyp_optimizer_type == 'rsgd':
                optimizer_hyp = RiemannianSGD(hyp_params, lr=hyp_lr, stabilize=50, weight_decay=hyp_weight_decay)
            else:
                raise NotImplementedError("Unsupported Hyperbolic optimizer type")

            # Store both optimizers
            self.optimizer = [optimizer_euc, optimizer_hyp]
        else:
            # Store only Euclidean optimizer if there are no Hyperbolic parameters
            self.optimizer = [optimizer_euc]

    def step(self):
        # Perform optimization step for each optimizer
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self):
        # Reset gradients to zero for each optimizer
        for optimizer in self.optimizer:
            optimizer.zero_grad()


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.lorentz import Lorentz  # 继承原 Lorentz

__all__ = ["TrainableLorentz"]


class TrainableLorentz(Lorentz):
    r"""
    Lorentz 流形 (时间坐标为第 0 维) 的「**可学习曲率**」版本。
    - 内部用可训练参数 ``self._logc``            （ log(c) 形式，确保 c>0 ）
    - 每一步 **clamp** 在 ``(1e‑3, c_max)`` 内防止梯度爆炸/消失
    """

    def __init__(self, c_init: float = 1.0, *, c_max: float = 10.0):
        # 先给父类随便一个正数（占位），随后会被 property 覆盖

        super().__init__(k=float(c_init))
        self._logc = nn.Parameter(torch.tensor(math.log(c_init), dtype=torch.float32))
        self.c_max = float(c_max)

    # ---- 动态曲率 ----
    @property
    def k(self) -> torch.Tensor:
        """保持向后兼容：依旧叫 k（=c）"""
        # softplus 确保 >0，再 clamp 上限
        c_val = F.softplus(self._logc)
        return c_val.clamp(max=self.c_max)

    # 训练过程中想看数值，用 .item()
    def item(self) -> float:
        return float(self.k.detach())


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class TrainableLorentz(Lorentz):
    def __init__(self, c_init: float = 1.0, *, c_max: float = 10.0):
        super().__init__()
        self.c_max = float(c_max)

        # ====== 原有初始化逻辑 ======
        self.register_buffer("_c_in_fixed", torch.tensor(float(c_init)))
        self._logc_in = None

        self._logc_out = nn.Parameter(torch.zeros(()))   # log(1)=0
        self.register_buffer("_c_out_fixed", None)

        # ====== ★ 关键：初始化完强行设置 ★ ======
        #self.set_inputc(10.0, trainable=False)   # 固定常量
        self.set_inputc("auto")  # 固定常量
        #self.set_outputc("auto")                 # 可学习

        self.set_outputc(0.0, trainable=False)

    # ---------- 2. 曲率属性 ----------
    @property
    def inputc(self) -> torch.Tensor:
        if self._logc_in is not None:                      # 可学习
            return F.softplus(self._logc_in).clamp(1e-3, self.c_max)
        return self._c_in_fixed                           # 固定

    @property
    def outputc(self) -> torch.Tensor:
        if self._logc_out is not None:
            return F.softplus(self._logc_out).clamp(1e-3, self.c_max)
        return self._c_out_fixed

    # ---------- 3. 运行期手动修改 ----------
    @torch.no_grad()
    def set_inputc(self, value: float | str, *, trainable: bool = True):
        """
        手动设置输入曲率
        - value 为正数：固定常量；若 trainable=True 则转成可学习
        - value='auto'：改为可学习 (初值保持不变)
        """
        self._override_curvature(kind="in", value=value, trainable=trainable)

    @torch.no_grad()
    def set_outputc(self, value: float | str, *, trainable: bool = True):
        """
        同上，用于输出曲率
        """
        self._override_curvature(kind="out", value=value, trainable=trainable)

    # ---- 内部公共逻辑 ----
    def _override_curvature(self, *, kind: str, value, trainable: bool):
        log_name, fixed_name = f"_logc_{kind}", f"_c_{kind}_fixed"

        # 移除旧变量
        if getattr(self, log_name) is not None:
            self._parameters.pop(log_name)
        if getattr(self, fixed_name) is not None:
            self._buffers.pop(fixed_name)

        # 创建新变量
        if isinstance(value, str) and value.lower() == "auto":
            param = nn.Parameter(torch.zeros(()))          # log(1)=0
            setattr(self, log_name, param)
            self.register_buffer(fixed_name, None)
        elif isinstance(value, (int, float)) and value > 0:
            if trainable:
                param = nn.Parameter(torch.tensor(math.log(float(value))))
                setattr(self, log_name, param)
                self.register_buffer(fixed_name, None)
            else:
                self.register_buffer(fixed_name, torch.tensor(float(value)))
                setattr(self, log_name, None)
        else:
            warnings.warn(f"[TrainableLorentz] 非法曲率 {value}，忽略。")

    # ---------- 4. 前向映射 ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, spatial = x[..., -1:], x[..., :-1]
        scale = torch.sqrt(self.inputc / self.outputc)
        return torch.cat([spatial * scale, t], dim=-1)

    # ---------- 5. 调试辅助 ----------
    def curvature_values(self) -> tuple[float, float]:
        """返回当前 (inputc, outputc) 数值"""
        return float(self.inputc.detach()), float(self.outputc.detach())

'''