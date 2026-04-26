from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

import config as cfg

_CACHE_PRINT_EVERY = 200
_CACHE_CALLS = 0


def _graph_cache_maxsize() -> int:
    return int(getattr(cfg, "GRAPH_CACHE_SIZE", 128))


def nodelist2indexlist(node_list: Sequence[int], node_id: np.ndarray) -> List[int]:
    return [int(np.where(node_id == node)[0][0]) for node in node_list]


def read_fusion_graph(
    t: int,
    path: Path = cfg.GRAPH_PATH,
    view_num: int = cfg.VIEW_NUM,
    fuse_adj_method: str = cfg.FUSE_ADJ_METHOD,
    device: torch.device = cfg.DEVICE,
):
    """Load multi-view adjacency and node features for timestamp t."""
    path = Path(path)
    sp_dist_adj = np.load(path / "adj_spatial_dist" / f"{t}.npy").astype(np.float32)
    sp_cluster_adj = np.load(path / "adj_spatial_cluster" / f"{t}.npy").astype(np.float32)
    tmep_adj = np.load(path / "adj_temporal" / f"{t}.npy").astype(np.float32)

    sp_dist_t = torch.from_numpy(sp_dist_adj).to(device)
    sp_cluster_t = torch.from_numpy(sp_cluster_adj).to(device)
    tmep_t = torch.from_numpy(tmep_adj).to(device)

    if view_num == 2:
        adj = [sp_dist_t, tmep_t]
    elif view_num == 3:
        adj = [sp_dist_t, sp_cluster_t, tmep_t]
    elif view_num == 4:
        if fuse_adj_method == "add":
            adj_fuse_t = sp_dist_t + sp_cluster_t + tmep_t
        elif fuse_adj_method == "cat":
            adj_fuse_np = np.load(path / "1view_4type" / f"{t}.npy").astype(np.float32)
            adj_fuse_t = torch.from_numpy(adj_fuse_np).to(device)
        else:
            raise ValueError(f"Unknown fuse_adj_method: {fuse_adj_method}")
        adj = [sp_dist_t, sp_cluster_t, tmep_t, adj_fuse_t]
    else:
        raise ValueError(f"Unsupported VIEW_NUM={view_num}")

    feat = np.load(path / "feat" / f"{t}.npy").astype(np.float32)
    feat_t = torch.from_numpy(feat).to(device)
    nid = np.load(path / "all_node_id.npy")
    return adj, feat_t, nid


@lru_cache(maxsize=_graph_cache_maxsize())
def _read_fusion_graph_cached_impl(
    t: int,
    path_str: str,
    view_num: int,
    fuse_adj_method: str,
    device_str: str,
):
    device = torch.device(device_str)
    path = Path(path_str)
    return read_fusion_graph(
        t,
        path=path,
        view_num=view_num,
        fuse_adj_method=fuse_adj_method,
        device=device,
    )


def read_fusion_graph_cached(
    t: int,
    path: Path = cfg.GRAPH_PATH,
    view_num: int = cfg.VIEW_NUM,
    fuse_adj_method: str = cfg.FUSE_ADJ_METHOD,
    device: torch.device = cfg.DEVICE,
    debug: bool = False,
):
    global _CACHE_CALLS
    path_str = str(Path(path).resolve())
    device_str = str(device)
    result = _read_fusion_graph_cached_impl(t, path_str, view_num, fuse_adj_method, device_str)
    if debug:
        _CACHE_CALLS += 1
        if _CACHE_CALLS % _CACHE_PRINT_EVERY == 0:
            info = _read_fusion_graph_cached_impl.cache_info()
            print(
                "[graph_loader] read_fusion_graph_cached cache_info: "
                f"hits={info.hits}, misses={info.misses}, maxsize={info.maxsize}, currsize={info.currsize}"
            )
    return result


def get_certain_node_batch(label_data_stfgn: torch.Tensor, gid_idx_list: Sequence[Sequence[int]]) -> torch.Tensor:
    embed = []
    for count in range(len(gid_idx_list)):
        gid_idx = gid_idx_list[count]
        tmp = [label_data_stfgn[count][idx].cpu().detach().numpy() for idx in gid_idx]
        embed.append(tmp)
    embed_np = np.array(embed, dtype=np.float32)
    return torch.from_numpy(embed_np).to(cfg.DEVICE)
