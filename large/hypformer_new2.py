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
class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        """
        Initializes a TransConvLayer instance.

        Args:
            manifold: The manifold to use for the layer.
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            num_heads: The number of attention heads.
            use_weight: Whether to use weights for the attention mechanism. Defaults to True.
            args: Additional arguments for the layer, including attention_type, power_k, and trans_heads_concat.

        Returns:
            None
        """
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
        # normalize input
        # qs = HypNormalization(self.manifold)(qs)
        # ks = HypNormalization(self.manifold)(ks)

        # negative squared distance (less than 0)
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))  # [H, N, N]
        att_weight = att_weight / self.scale + self.bias  # [H, N, N]

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
        phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [N, H, D]
        phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [N, H, D]

        phi_qs = self.fp(phi_qs, p=self.power_k)  # [N, H, D]
        phi_ks = self.fp(phi_ks, p=self.power_k)  # [N, H, D]

        # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)  # [H, D, D]

        # Step 2: Compute the kernel-transformed dot product of Q with the above result
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, D]

        # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))  # [N, H]
        denominator = denominator.unsqueeze(-1)  #

        # Step 4: Normalize the numerator with the denominator
        attn_output = numerator / (denominator + 1e-6)  # [N, H, D]

        # Map vs through v_map_mlp and ensure it is the correct shape
        vss = self.v_map_mlp(v)  # [N, H, D]
        attn_output = attn_output + vss  # preserve its rank, [N, H, D]

        if self.trans_heads_concat:
            attn_output = self.final_linear(attn_output.reshape(-1, self.num_heads * self.out_channels))
        else:
            attn_output = attn_output.mean(dim=1)

        attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

        if output_attn:
            # Calculate attention weights
            attention = torch.einsum('nhd,mhd->nmh', phi_qs, phi_ks)  # [N, M, H]
            attention = attention / (denominator.unsqueeze(1) + 1e-6)  # Normalize

            # Average attention across heads if needed
            attention = attention.mean(dim=-1)  # [N, M]
            return attn_output, attention
        else:
            return attn_output

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)  # [N, H, D]
        key = torch.stack(k_list, dim=1)  # [N, H, D]
        value = torch.stack(v_list, dim=1)  # [N, H, D]

        if output_attn:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(
                    query, key, value, output_attn)  # [N, H, D]
            elif self.attention_type == 'full':
                attention_output, attn = self.full_attention(
                    query, key, value, output_attn)
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(
                    query, key, value)  # [N, H, D]
            elif self.attention_type == 'full':
                attention_output = self.full_attention(
                    query, key, value)
            else:
                raise NotImplementedError

        final_output = attention_output
        # multi-head attention aggregation
        # final_output = self.manifold.mid_point(final_output)

        if output_attn:
            return final_output, attn
        else:
            return final_output


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

    def forward(self, x_input):
        layer_ = []

        # the original inputs are in Euclidean
        x = self.fcs[0](x_input, x_manifold='euc')
        # add positional embedding
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            # if self.use_act:
            #     x = self.activation(x)
            # # x = self.dropout(x, training=self.training)
            layer_.append(x)

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


# ==================== 主模型 HypFormer 替换为使用 FixedGraphConv ====================
class Dysformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.5, aggregate='add',args=None):
        """
        Initialize HypFormer model with hypergraph support.
        """
        super().__init__()
        # Initialize hyperbolic manifolds
        self.manifold_in = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.decoder_type = args.decoder_type

        self.manifold_out = Lorentz(k=float(args.k_out))
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        # Initialize hyperbolic Transformer convolution module (TransConv)
        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out,
                                    in_channels, hidden_channels,
                                    num_layers=trans_num_layers, num_heads=trans_num_heads,
                                    dropout=trans_dropout, use_bn=trans_use_bn,
                                    use_residual=trans_use_residual, use_weight=trans_use_weight,
                                    use_act=trans_use_act, args=args)
        # Decoder layers for classification (Euclidean or Hyperbolic)
        if self.decoder_type == 'euc':
            # Euclidean decoder: a linear layer after projecting hyperbolic features to Euclidean
            self.decode_trans = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.decoder_type == 'hyp':
            # Hyperbolic decoder: a hyperbolic linear layer to map to output dimension
            self.decode_trans = HypLinear(self.manifold_out, self.hidden_channels, self.out_channels)
        else:
            raise NotImplementedError(f"Unknown decoder_type: {self.decoder_type}")
        self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn,
                                    gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)

        # Initialize hypergraph constructor
        self.hconstructor = ScalableHConstructor(
            epsilon=0.7,  # 可调
            mu=2,
            star=True,
            min_hyperedge_size=3,
            device=torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
        )
        self.current_epoch = 0  # Track current training epoch for dynamic graph update

    def forward(self, x, edge_index,epoch):
        self.current_epoch = epoch

        """
        Forward pass of HypFormer with dynamic hypergraph integration.
        Every 20 epochs, the hypergraph structure is rebuilt using the previous epoch's node embeddings.
        Uses the hypergraph adjacency to perform graph convolution propagation.

        Args:
            x (Tensor): Input node features [N, in_channels] (Euclidean features).
        Returns:
            Tensor: Output predictions (logits or embeddings) for each node.
        """
        # If at the beginning (no graph yet), or every 20 epochs, update hypergraph structure
        if self.current_epoch == 0 or (self.current_epoch > 10 and self.current_epoch % 20 == 0):
            # Use the latest available node embeddings to construct hypergraph
            # For epoch 0 (initial), use raw input features; for subsequent updates, use cached embeddings from last epoch
            if self.current_epoch == 0:
                self.hconstructor.construct_graph(edge_index, x.size(0))  # 用原始 edge_index 初始化超图

            else:
                # Dynamic graph update using last epoch's embeddings
                # (last_node_embeds was cached at the end of previous forward pass)
                prev_embeds = self.hconstructor.last_node_embeds
                if prev_embeds is not None:
                    self.hconstructor.construct_graph(edge_index, x.size(0))
        # Perform hyperbolic Transformer propagation
        # (The TransConv module internally handles feature transformation and attention)
        x_hidden = self.trans_conv(x)  # hyperbolic hidden representations (Lorentz coordinates)
        # If a hypergraph (edge_index) is available, perform an explicit graph convolution propagation
        edge_index = self.hconstructor.edge_index
        if edge_index is not None:
            # Perform neighbor feature aggregation based on the hypergraph adjacency
            # Use hyperbolic operations: aggregate in Euclidean space then lift to hyperbolic
            x_space = x_hidden[..., 1:]  # spatial components of hyperbolic embeddings [N, hidden_channels]
            N = x_space.size(0)
            # Initialize neighbor sum tensor
            neighbor_sum = torch.zeros_like(x_space)
            # Sum neighbor features for each target node
            # edge_index[0] = target node indices, edge_index[1] = source neighbor indices
            neighbor_sum.index_add_(0, edge_index[0], x_space[edge_index[1]])
            # Compute degree (number of neighbors) for each node (count edges targeting each node)
            deg = torch.bincount(edge_index[0], minlength=N).unsqueeze(1).to(x_space.device)
            # Average the neighbor sums (avoid division by zero for isolated nodes)
            deg[deg == 0] = 1  # to prevent division by zero
            neighbor_avg = neighbor_sum / deg
            # Convert the Euclidean neighbor aggregation to hyperbolic (Lorentz) coordinates
            x_time = torch.sqrt((neighbor_avg ** 2).sum(dim=-1, keepdim=True) + self.manifold_out.k)
            x_neighbor = torch.cat([x_time, neighbor_avg], dim=-1)  # [N, hidden_channels+1]
            # Combine original hidden representations with neighbor-aggregated features in hyperbolic space
            x_hidden = self.manifold_out.mid_point(torch.stack((x_hidden, x_neighbor), dim=1))
        # Decode the hidden representation to output (classification logits or embeddings)
        if self.decoder_type == 'euc':
            # Project hyperbolic output to Euclidean space (log-map) and apply linear decoder
            x_out = self.decode_trans(self.manifold_out.logmap0(x_hidden)[..., 1:])
        elif self.decoder_type == 'hyp':
            # Apply hyperbolic linear decoder directly on hyperbolic features
            x_out = self.decode_trans(x_hidden)
        else:
            raise NotImplementedError
        # Cache the current hidden representations for next epoch's graph update
        self.hconstructor.last_node_embeds = x_hidden.detach()
        return x_out
    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def reset_parameters(self):
        # self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
        # self.fc.reset_parameters()