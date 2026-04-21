import torch
import torch.nn as nn

import config as cfg
from models.graph_layers import GraphAttentionLayer, GraphConvolution, MultiheadAttention


class MultiViewGNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_size: int,
        gat_hidden_size: int,
        num_heads: int = 1,
        dropout: float = 0.2,
        alpha_gat: float = 0.2,
        alpha_fusion: float = 0.8,
        beta_fusion: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_size = in_features
        self.out_size = out_size
        self.gat_hidden_size = gat_hidden_size
        self.alpha = alpha_fusion
        self.beta = beta_fusion

        self.GATLayer = GraphAttentionLayer(self.input_size, self.gat_hidden_size, dropout, alpha_gat)
        self.GCNLayer = GraphConvolution(self.input_size, self.gat_hidden_size)
        self.multihead_attn = MultiheadAttention(self.gat_hidden_size, self.gat_hidden_size, num_heads)
        self.linear_fusion = nn.Linear(self.gat_hidden_size, self.gat_hidden_size)

    def forward(self, all_adj, node_feat):
        # For new graphs we do not stack 3 copies; use full node count.
        node_num = node_feat.shape[0]

        hidden_GAT = []
        for adj in all_adj:
            h = self.GATLayer(node_feat, adj)
            hidden_GAT.append(h)

        hidden_self_att = []
        for h in hidden_GAT:
            h_in = h.unsqueeze(0)
            attn_output = self.multihead_attn(h_in)
            hidden_self_att.append(attn_output)

        hidden_fuse1 = []
        for h_att, h_base in zip(hidden_self_att, hidden_GAT):
            h = self.alpha * h_att + (1 - self.alpha) * h_base
            hidden_fuse1.append(h)

        view_hidden_final = []
        fusion_hidden = torch.zeros_like(hidden_fuse1[0], device=cfg.DEVICE)
        for h in hidden_fuse1:
            w = torch.sigmoid(self.linear_fusion(h))
            hid_now_view = w * h
            fusion_hidden += hid_now_view
            h_new = self.beta * hid_now_view + (1 - self.beta) * h
            view_hidden_final.append(h_new)

        # No slicing; retain all nodes
        return fusion_hidden.to(cfg.DEVICE), view_hidden_final


class MultiViewGNNBatch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_size: int,
        gat_hidden_size: int,
        num_heads: int = 1,
        dropout: float = 0.2,
        alpha_gat: float = 0.2,
        alpha_fusion: float = 0.8,
        beta_fusion: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_size = in_features
        self.out_size = out_size
        self.gat_hidden_size = gat_hidden_size
        self.alpha = alpha_fusion
        self.beta = beta_fusion

        self.multihead_attn = MultiheadAttention(self.gat_hidden_size, self.gat_hidden_size, num_heads)
        self.linear_fusion = nn.Linear(self.gat_hidden_size, self.gat_hidden_size)

    def forward(self, hidden_GAT: torch.Tensor):
        batchsz, views, seq_len, embsz = hidden_GAT.shape
        # For new graphs we do not stack 3 copies; use full node count.
        node_num = seq_len
        hidden = hidden_GAT.reshape((batchsz, views * seq_len, embsz))

        hidden_self_att = self.multihead_attn(hidden)
        hidden_fuse1 = self.alpha * hidden_self_att + (1 - self.alpha) * hidden
        hidden_fuse1 = hidden_fuse1.reshape(hidden_GAT.shape)
        hidden_fuse1 = hidden_fuse1.permute(1, 0, 2, 3)

        fusion_hidden = torch.zeros(hidden_fuse1[0].shape, device=cfg.DEVICE)
        view_hidden_final = []
        for h in hidden_fuse1:
            w = torch.sigmoid(self.linear_fusion(h))
            hid_now_view = w * h
            fusion_hidden += hid_now_view
            h_new = self.beta * hid_now_view + (1 - self.beta) * h
            view_hidden_final.append(h_new)

        # No slicing; retain all nodes
        return fusion_hidden.to(cfg.DEVICE), view_hidden_final
