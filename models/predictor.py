from pathlib import Path
import time
from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.attention import AttentionLayer
from models.graph_layers import GraphAttentionLayer, GraphConvolution
from models.multiview_gnn import MultiViewGNNBatch
from utils.graph_loader import nodelist2indexlist, read_fusion_graph_cached


class GRUDecoder(nn.Module):
    def __init__(self, num_steps: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.num_steps = num_steps

    def forward(self, in_data: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        in_data = in_data.unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        result = []
        for _ in range(self.num_steps):
            output, hidden = self.lstm(in_data, hidden)
            output = self.out(output[-1])
            result.append(output)
            in_data = output.unsqueeze(0)
        result = torch.stack(result).squeeze(2).permute(1, 0)
        return result


class MultiViewPredictor(nn.Module):
    def __init__(
        self,
        num_station: int = cfg.K_NEIGHBOR,
        output_size: int = cfg.MODEL_OUTPUT_SIZE,
        add_labeled_embed: bool = cfg.ADD_LABELED_EMBED,
        alpha_multiview_fusion: float = cfg.ALPHA_MULTIVIEW_FUSION,
        graph_path: Path = cfg.GRAPH_PATH,
        view_num: int = cfg.VIEW_NUM,
        fuse_adj_method: str = cfg.FUSE_ADJ_METHOD,
    ) -> None:
        super().__init__()
        self.num_station = num_station
        self.prev_slot = cfg.HISTORICAL_T
        self.hidden_lstm = 32
        self.hidden_linear = 32
        self.hidden_gru = 32
        self.hidden_gnn = 32
        self.output_size = output_size
        self.add_labeled_embed = add_labeled_embed
        self.graph_path = Path(graph_path)
        self.view_num = view_num
        self.fuse_adj_method = fuse_adj_method

        self.unlabel_lstm_1 = nn.LSTM(len(cfg.MEO_COL), self.hidden_lstm)
        self.unlabel_linear_1 = nn.Linear(len(cfg.ST_COL_ALL), self.hidden_linear)
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear + self.hidden_lstm, self.hidden_linear * 2)
        self.label_lstm_1 = nn.LSTM(1 + len(cfg.MEO_COL), self.hidden_lstm)
        self.label_linear_1 = nn.Linear(len(cfg.ST_COL_ALL) + 1, self.hidden_linear)
        self.label_linear_2 = nn.Linear(self.hidden_linear + self.hidden_lstm + self.hidden_gnn, self.hidden_linear * 2)
        self.label_linear_3 = nn.Linear(self.hidden_gnn, self.hidden_linear * 2)

        self.stfgn = MultiViewGNNBatch(
            in_features=cfg.NODE_FEAT_DIM,
            out_size=self.hidden_gnn,
            gat_hidden_size=self.hidden_gnn,
            alpha_fusion=alpha_multiview_fusion,
        )
        self.GATLayer = GraphAttentionLayer(cfg.NODE_FEAT_DIM, self.hidden_gnn, 0.2, 0.2)
        self.GCNLayer = GraphConvolution(cfg.NODE_FEAT_DIM, self.hidden_gnn)

        self.idw_attention = AttentionLayer(num_station, self.hidden_linear * 2, self.hidden_linear * 2, 16)
        self.GRU = nn.GRUCell(self.hidden_gru + self.hidden_linear * 2 + self.hidden_linear * 2, self.hidden_gru, bias=True)
        self.GRU_DE = GRUDecoder(num_steps=output_size)
        self.liner_t = nn.Linear(self.hidden_gru, self.hidden_gru)
        self.output_fc = nn.Linear(self.hidden_gru, output_size)

    def forward(
        self,
        meo_unlabel: torch.Tensor,
        feature_unlabel: torch.Tensor,
        ovi_label: torch.Tensor,
        meo_label: torch.Tensor,
        feature_label: torch.Tensor,
        dis_label: torch.Tensor,
        h_t: torch.Tensor,
        timestamp: torch.Tensor,
        label_id_list: torch.Tensor,
        profile: bool = False,
    ) -> torch.Tensor:
        batch_size = meo_unlabel.shape[0]
        assert meo_unlabel.dim() == 3, f"meo_unlabel should be (B, T, F), got {meo_unlabel.shape}"
        assert feature_unlabel.dim() == 2, f"feature_unlabel should be (B, F), got {feature_unlabel.shape}"
        assert ovi_label.dim() == 3, f"ovi_label should be (B, N, T), got {ovi_label.shape}"
        assert meo_label.dim() == 4, f"meo_label should be (B, N, T, F), got {meo_label.shape}"
        assert feature_label.dim() == 3, f"feature_label should be (B, N, F), got {feature_label.shape}"
        assert dis_label.dim() == 2, f"dis_label should be (B, N), got {dis_label.shape}"
        assert timestamp.shape[0] == batch_size, f"timestamp batch mismatch: {timestamp.shape[0]} vs {batch_size}"
        assert label_id_list.shape[0] == batch_size, f"label_id_list batch mismatch: {label_id_list.shape[0]} vs {batch_size}"

        if profile:
            timers = {
                "graph_load": 0.0,
                "gat": 0.0,
                "lstm": 0.0,
                "attention": 0.0,
                "gru_loop": 0.0,
            }

            def _sync():
                if cfg.DEVICE.type == "cuda":
                    torch.cuda.synchronize(cfg.DEVICE)

            def _time_start():
                _sync()
                return time.perf_counter()

            def _time_end(start):
                _sync()
                return time.perf_counter() - start

        batch_graph_h = []
        gid_idx_list = []
        for i in range(len(timestamp)):
            t = int(timestamp[i].item())
            gids = label_id_list[i]
            if profile:
                t0 = _time_start()
            multi_view_adj, node_feat, node_id = read_fusion_graph_cached(
                t,
                path=self.graph_path,
                view_num=self.view_num,
                fuse_adj_method=self.fuse_adj_method,
                device=cfg.DEVICE,
            )
            if profile:
                timers["graph_load"] += _time_end(t0)
            gid_idx = nodelist2indexlist(gids.tolist(), node_id)
            gid_idx_list.append(gid_idx)

            if profile:
                t0 = _time_start()
            hidden_GAT = [self.GATLayer(node_feat, adj) for adj in multi_view_adj]
            if profile:
                timers["gat"] += _time_end(t0)
            batch_graph_h.append(torch.stack(hidden_GAT, dim=0))

        batch_graph_h = torch.stack(batch_graph_h, dim=0).float()

        if profile:
            t0 = _time_start()
        label_data_stfgn_batch, _ = self.stfgn(batch_graph_h)
        if profile:
            timers["gat"] += _time_end(t0)
        assert all(
            len(idx) == self.num_station for idx in gid_idx_list
        ), f"gid_idx_list should have {self.num_station} nodes per batch"
        gid_idx_tensor = torch.tensor(gid_idx_list, device=cfg.DEVICE, dtype=torch.long)
        assert gid_idx_tensor.dim() == 2, f"gid_idx_tensor should be (B, N), got {gid_idx_tensor.shape}"
        label_data_stfgn_batch = label_data_stfgn_batch.gather(
            dim=1,
            index=gid_idx_tensor.unsqueeze(2).expand(-1, -1, label_data_stfgn_batch.size(2)),
        )
        assert label_data_stfgn_batch.shape[:2] == (
            batch_size,
            self.num_station,
        ), f"label_data_stfgn_batch should be (B, N, H), got {label_data_stfgn_batch.shape}"

        if profile:
            t0 = _time_start()
        unlabel_time_data = meo_unlabel.permute(1, 0, 2)
        unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data)
        unlabel_time_data = unlabel_time_data.float()[-1]

        unlabel_fea_data = torch.relu(self.unlabel_linear_1(feature_unlabel))
        unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1)
        unlabel_data = torch.relu(self.unlabel_linear_2(unlabel_data))

        if self.add_labeled_embed:
            label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3)
            label_time_seq = label_time.permute(2, 0, 1, 3).reshape(
                label_time.size(2), batch_size * self.num_station, label_time.size(3)
            )
            label_time_out, _ = self.label_lstm_1(label_time_seq)
            label_time_data = label_time_out.float()[-1].reshape(batch_size, self.num_station, self.hidden_lstm)
            label_feature = torch.relu(self.label_linear_1(feature_label))
            label_data = torch.relu(
                self.label_linear_2(torch.cat([label_time_data, label_feature, label_data_stfgn_batch], dim=2))
            )
        else:
            label_data = torch.relu(self.label_linear_3(label_data_stfgn_batch))
        if profile:
            timers["lstm"] += _time_end(t0)

        assert unlabel_data.shape[0] == batch_size, f"unlabel_data batch mismatch: {unlabel_data.shape[0]} vs {batch_size}"
        assert label_data.shape[:2] == (
            batch_size,
            self.num_station,
        ), f"label_data should be (B, N, H), got {label_data.shape}"
        assert dis_label.shape[1] == self.num_station, f"dis_label N mismatch: {dis_label.shape[1]} vs {self.num_station}"

        if profile:
            t0 = _time_start()
        attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
        attention_out = torch.sum(attention_score.unsqueeze(2) * label_data, dim=1)
        sp_approximate = torch.relu(attention_out)
        if profile:
            timers["attention"] += _time_end(t0)

        if profile:
            t0 = _time_start()
        for _ in range(self.prev_slot):
            temp_approximate = F.relu(self.liner_t(h_t))
            X_feat = torch.cat([unlabel_data, temp_approximate, sp_approximate], dim=1)
            h_t = self.GRU(X_feat)
        if profile:
            timers["gru_loop"] += _time_end(t0)
            total = sum(timers.values())
            print(
                "[MultiViewPredictor] forward latency (s): "
                f"graph_load={timers['graph_load']:.6f}, "
                f"gat={timers['gat']:.6f}, "
                f"lstm={timers['lstm']:.6f}, "
                f"attention={timers['attention']:.6f}, "
                f"gru_loop={timers['gru_loop']:.6f}, "
                f"total={total:.6f}"
            )
        out = self.output_fc(h_t)
        return out
