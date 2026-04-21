import torch
import torch.nn as nn

import config as cfg


class AttentionLayer(nn.Module):
    def __init__(
        self,
        num_station: int,
        unlabel_size: int,
        label_size: int,
        hidden_size: int,
        p: float | None = None,
        eps: float | None = None,
        learnable_p: bool | None = None,
    ):
        super().__init__()
        self.num = num_station
        self.un_emb = unlabel_size
        self.emb = label_size
        self.hidden = hidden_size
        self.linear_1 = nn.Linear(self.un_emb + self.emb, self.hidden)
        self.linear_2 = nn.Linear(self.hidden, 1)
        if p is None:
            p = getattr(cfg, "IDW_P", 1.0)
        if eps is None:
            eps = getattr(cfg, "IDW_EPS", 1e-6)
        if learnable_p is None:
            learnable_p = getattr(cfg, "IDW_LEARNABLE_P", False)
        p = max(float(p), 1e-6)
        self.eps = float(eps)
        self.learnable_p = bool(learnable_p)
        if self.learnable_p:
            # Store log(p) to keep p positive via exp in forward.
            self.p_param = nn.Parameter(torch.tensor(float(p)).log())
            # forward:
            p = self.p_param.exp().clamp(0.1, 5.0)

        else:
            self.register_buffer("p_const", torch.tensor(p))

    def forward(self, unlabel_emb: torch.Tensor, label_emb: torch.Tensor, dis_lab: torch.Tensor):
        if isinstance(label_emb, (list, tuple)):
            label_emb = torch.stack(label_emb, dim=1)

        assert unlabel_emb.dim() == 2, f"unlabel_emb should be (B, H), got {unlabel_emb.shape}"
        assert label_emb.dim() == 3, f"label_emb should be (B, N, H), got {label_emb.shape}"
        assert dis_lab.dim() == 2, f"dis_lab should be (B, N), got {dis_lab.shape}"

        batch_size, num_station, _ = label_emb.shape
        assert (
            unlabel_emb.shape[0] == batch_size
        ), f"unlabel_emb batch mismatch: {unlabel_emb.shape[0]} vs {batch_size}"
        assert (
            dis_lab.shape[0] == batch_size and dis_lab.shape[1] == num_station
        ), f"dis_lab shape mismatch: {dis_lab.shape} vs (B={batch_size}, N={num_station})"

        unlabel_exp = unlabel_emb.unsqueeze(1).expand(-1, num_station, -1)
        pair_feat = torch.cat((label_emb, unlabel_exp), dim=-1)
        logits = self.linear_2(torch.relu(self.linear_1(pair_feat))).squeeze(-1)

        # IDW bias: w_i = 1 / (d_i + eps)^p, added as log-bias for stability.
        if self.learnable_p:
            p = self.p_param.exp()
        else:
            p = self.p_const
        dis = dis_lab.to(dtype=logits.dtype).clamp_min(self.eps)
        idw_log = -p * torch.log(dis)        # 等價但更穩、少一次 pow
        logits = logits + idw_log
        # optional:
        logits = logits.clamp(-50, 50)


        attention_score = torch.softmax(logits, dim=1)

        assert attention_score.shape == (batch_size, num_station), (
            f"attention_score should be (B, N), got {attention_score.shape}"
        )
        if torch.isfinite(attention_score).all():
            ones = torch.ones(batch_size, device=attention_score.device, dtype=attention_score.dtype)
            # assert torch.allclose(attention_score.sum(dim=1), ones, atol=1e-4, rtol=1e-4)
        return attention_score
