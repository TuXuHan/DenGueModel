#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchsummary import summary
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from fastdtw import fastdtw
import time
import matplotlib.pyplot as plt
import hdbscan
import math
import logging
import datetime
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# %%
import argparse
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--cv_k', type=int, default=0, help='k of cross validation')
args = parser.parse_args()
print('cv_k=',args.cv_k)
cv_k=args.cv_k
# cv_k=6
meo_col = ['測站氣壓(hPa)', '氣溫(℃)', '相對溼度(%)', '風速(m/s)', '降水量(mm)','測站最高氣壓(hPa)', 
                  '最高氣溫(℃)', '最大陣風(m/s)', '測站最低氣壓(hPa)','最低氣溫(℃)', '最小相對溼度(%)']
st_col = ['watersupply_hole','well', 'sewage_hole', 'underwater_con', 'pumping',
       'watersupply_others', 'watersupply_value', 'food_poi', 'rainwater_hole',
       'river', 'drainname', 'sewage_well', 'gaugingstation', 'underpass', 'watersupply_firehydrant']
st_col_all = st_col.copy()
dirs = ['B', 'T', 'L', 'R', 'RB', 'RT', 'LB', 'LT']
for grid_dir in dirs:
    col_name = pd.Series(st_col.copy()) + '_'+grid_dir
    col_name = col_name.tolist()
    st_col_all+=col_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('dataset_processed/label_id.txt','r') as f:
    lines = f.read().split('\n')[:-1]
labeled_id = [int(x) for x in lines]
# with open('dataset_processed/unlabel_train_id.txt','r') as f:
#     lines = f.read().split('\n')[:-1]
# train_id = [int(x) for x in lines]
# with open('dataset_processed/unlabel_valid_id.txt','r') as f:
#     lines = f.read().split('\n')[:-1]
# valid_id = [int(x) for x in lines]
# with open('dataset_processed/unlabel_test_id.txt','r') as f:
#     lines = f.read().split('\n')[:-1]
# test_id = [int(x) for x in lines]

split_id_list = []
for i in range(10):
    with open('dataset_processed/unlabeled_split_'+str(i)+'.txt','r') as f:
        lines = f.read().split('\n')[:-1]
    gid = [int(x) for x in lines]
    split_id_list.append(gid)

import numpy as np
k_val = (cv_k+8)%10
k_test = (cv_k+9)%10
if cv_k>=2:
    tmp = split_id_list[cv_k:]+split_id_list[:(cv_k-2)%10]
else:
    tmp = split_id_list[cv_k:cv_k+8]
print(len(tmp))
train_id = list(np.concatenate(tmp).flat)
valid_id = split_id_list[k_val]
test_id = split_id_list[k_test]

print('label',len(labeled_id))
print('train',len(train_id))
print('valid',len(valid_id))
print('test',len(test_id))



class station_data(Dataset) :
    def __init__(self, mode='train') : #pred_slot
#         self.target_unlabel = np.array(valid_index)[:,0].tolist()
#         self.target_label = np.array(valid_index)[:,1:].tolist()
        self.all_data = pd.read_csv('dataset_processed/all_processed_data_9box_nexty.csv')
        self.grid_df_neighbor = pd.read_csv('dataset_processed/grid_100neighbor_dist.csv')
        self.all_static_features = self.all_data[['id']+st_col_all].drop_duplicates().reset_index()
        
        self.label_id = labeled_id
        if mode=='train':
            self.unlabel_id = train_id
        elif mode=='test':
            self.unlabel_id = test_id
        else:
            self.unlabel_id = valid_id
            
        self.label_data = self.all_data[self.all_data['id'].isin(self.label_id)].reset_index()
        self.unlabel_data = self.all_data[self.all_data['id'].isin(self.unlabel_id)].reset_index()
        
        self.label_ovi_target = self.label_data.egg_num.values
        self.unlabel_ovi_target = self.unlabel_data.egg_num.values
        self.label_meo = self.label_data[['time','id']+meo_col]
        self.label_feature = self.all_static_features[self.all_static_features['id'].isin(self.label_id)].reset_index()
        self.unlabel_meo = self.unlabel_data[['time','id']+meo_col]
        self.unlabel_feature = self.all_static_features[self.all_static_features['id'].isin(self.unlabel_id)].reset_index()
        
        self.near_station = self.grid_df_neighbor[['grid_id']+nearest_col]
        self.near_station_dist = self.grid_df_neighbor[['grid_id']+near_dist_col]
        
        self.prev_slot = historical_T
        self.pred_slot = model_output_size #pred_slot
        
    def __getitem__(self, index) :
        unlabel_id = self.unlabel_data.id[index]
        label_id_list = self.near_station[self.near_station['grid_id']==unlabel_id][nearest_col].values[0]
        timestamp = self.unlabel_data.time[index]
        
        if timestamp>=(202206-self.pred_slot+1):
            # ovi_target = torch.from_numpy(np.zeros(self.pred_slot)).float().to(device)
            ovi_target = torch.zeros(self.pred_slot,device=device).float()
        else:
            ovi_target = torch.from_numpy(self.unlabel_ovi_target[index+1:index+self.pred_slot+1]).float().to(device)
            
        if timestamp<(201806+self.prev_slot):
            # meo_unlabel = torch.from_numpy(np.zeros((self.prev_slot,11))).float().to(device)
            meo_unlabel = torch.zeros( self.prev_slot,11 ,device=device).float()
        else:
            meo_unlabel = torch.from_numpy(self.unlabel_meo[index-self.prev_slot:index][meo_col].values).float().to(device)
        feature_unlabel = torch.from_numpy(self.unlabel_feature[self.unlabel_feature['id']==unlabel_id][st_col_all].values[0]).float().to(device)
    
        ovi_label, meo_label, feature_label = self.get_feats_label(label_id_list,timestamp)
        dis_label, inv_dis_label = self.get_dist(unlabel_id)
        feature_label_out = torch.cat((feature_label, dis_label.unsqueeze(1) ), 1)
        
        return ovi_target, meo_unlabel, feature_unlabel,    ovi_label,      meo_label,                             feature_label_out, inv_dis_label, torch.from_numpy(label_id_list).float().to(device), timestamp
     # shape: [pred_slot],[prev_slot,11],   [135], [k_neighbor,prev_slot], [k_neighbor,prev_slot,11], [k_neighbor,136], [k_neighbor], [k_neighbor]
     
    def get_feats_label(self, label_id_list, timestamp):
        meo_out = []
        ovi_out = []
        feat_out = []
        for gid in label_id_list:
            if timestamp < 201810:
                idx = self.label_meo[
                    (self.label_meo['time'] == timestamp) &
                    (self.label_meo['id'] == gid)
                ].index.values[0]

                tmp_len = self.prev_slot - (201810 - timestamp)

                # meo: (tmp_len, 11)，前面補 0，最後 reshape 回 (prev_slot, 11)
                tmp_meo = self.label_meo[idx - tmp_len:idx][meo_col].values      # (tmp_len, 11)
                pad_meo = np.zeros((self.prev_slot - tmp_len, tmp_meo.shape[1])) # (prev_slot - tmp_len, 11)
                meo = np.vstack([pad_meo, tmp_meo])                              # (prev_slot, 11)

                # ovi: (tmp_len,) 前面補 0 變 (prev_slot,)
                tmp_ovi = self.label_ovi_target[idx - tmp_len:idx]               # (tmp_len,)
                ovi = np.concatenate([np.zeros(self.prev_slot - tmp_len), tmp_ovi], axis=0)  # (prev_slot,)
            else:
                idx = self.label_meo[
                    (self.label_meo['time'] == timestamp) &
                    (self.label_meo['id'] == gid)
                ].index.values[0]

                meo = self.label_meo[idx - self.prev_slot:idx][meo_col].values    # (prev_slot, 11)
                ovi = self.label_ovi_target[idx - self.prev_slot:idx]            # (prev_slot,)

            feat = self.label_feature[self.label_feature['id'] == gid][st_col_all].values[0]

            meo_out.append(meo)    # 每個元素 shape: (prev_slot, 11)
            ovi_out.append(ovi)    # 每個元素 shape: (prev_slot,)
            feat_out.append(feat)  # 每個元素 shape: (len(st_col_all),)

        try:
            # ✅ 先轉成單一 numpy.ndarray，再轉 tensor（避免 Warning）
            ovi_np  = np.array(ovi_out, dtype=np.float32)          # (k_neighbor, prev_slot)
            meo_np  = np.array(meo_out, dtype=np.float32)          # (k_neighbor, prev_slot, 11)
            feat_np = np.array(feat_out, dtype=np.float32)         # (k_neighbor, len(st_col_all))

            final_ovi  = torch.from_numpy(ovi_np).to(device)       # (k_neighbor, prev_slot)
            final_meo  = torch.from_numpy(meo_np).to(device)       # (k_neighbor, prev_slot, 11)
            final_feat = torch.from_numpy(feat_np).to(device)      # (k_neighbor, len(st_col_all))
        except Exception as e:
            # 萬一上面哪裡 shape 出問題，保底用 0 tensor
            final_ovi  = torch.zeros(k_neighbor, self.prev_slot,       device=device).float()
            final_meo  = torch.zeros(k_neighbor, self.prev_slot, 11,   device=device).float()
            final_feat = torch.zeros(k_neighbor, len(st_col_all),      device=device).float()

        return final_ovi.float(), final_meo.float(), final_feat.float()

    
    def get_dist(self, unlabel_id):
        dist = self.near_station_dist[self.near_station_dist['grid_id']==unlabel_id][near_dist_col].values[0]
        inv_dist = 1./dist
        # return torch.from_numpy(np.array(dist)).float().to(device), torch.from_numpy(np.array(inv_dist)).float().to(device)
        return torch.tensor(dist,device=device).float(), torch.tensor(inv_dist,device=device).float()
        
    def __len__(self) :
        return len(self.unlabel_ovi_target)

# %%
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




# https://github.com/Diego999/pyGAT/blob/master/layers.py
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class Attention_layer(nn.Module) :
    def __init__(self, num_station, unlabel_size, label_size, hidden_size) :#10,64,64,16
        super(Attention_layer, self).__init__()
        self.num = num_station
        self.un_emb = unlabel_size
        self.emb = label_size
        self.hidden = hidden_size
        self.linear_1 = nn.Linear(self.un_emb+self.emb, self.hidden)
        self.linear_2 = nn.Linear(self.hidden, 1)
        
    def forward(self, unlabel_emb, label_emb, dis_lab) :
        # unlabel_emb torch.Size([128, 64])
        # label_emb torch.Size([128, 64]) *10
        # dis_lab torch.Size([128, 10])
        
#         label_1 = []
        label_2 = []
        for k in label_emb : 
#             label_1.append(nn.ReLU()(self.linear_1(torch.cat((k, unlabel_emb), 1))))
            tmp = nn.ReLU()(self.linear_1(torch.cat((k, unlabel_emb), 1))) # torch.Size([128, 64])+torch.Size([128, 64]) -> torch.Size([128, 128]) -> torch.Size([128, 16])
            label_2.append(self.linear_2(tmp)) # torch.Size([128, 16]) -> torch.Size([128, 1])
#         for k in label_1 : 
#             label_2.append(self.linear_2(k))
        attention_out_ori = torch.stack(label_2).squeeze().permute(1,0) #torch.Size([128, 1])->torch.Size([128, 10])
#         print('attention_out_ori',attention_out_ori.shape)#torch.Size([128, 10])
#         print('dis_lab',dis_lab.shape)#torch.Size([128, 10])
        attention_out = attention_out_ori * dis_lab #torch.Size([128, 10]) * torch.Size([128, 10]) -> torch.Size([128, 10])
#         print('attention_out',attention_out.shape)#torch.Size([128, 10])
        attention_score = nn.Softmax(dim=1)(attention_out) #torch.Size([128, 10])
        return attention_score #, nn.Softmax(dim=1)(attention_out_ori)

def nodelist2indexlist(node_list,node_id):
    index_list = [ np.where(node_id==node)[0][0] for node in node_list]
    return index_list
# nodelist2indexlist(node_list,node_id)


def read_fusion_graph(t, path='dataset_processed/graph_data/'):
    """
    讀取第 t 期的 multi-view adjacency 與節點特徵。
    這裡改成：
    1. np.load 後就轉成 float32
    2. 立刻轉成 torch.tensor(…, dtype=float32)
    3. 若要 fuse (add)，在 torch 這邊做加總，避免 NumPy 再 allocate 一個大陣列
    """

    # --- 1. 讀取並轉成 float32 numpy ---
    sp_dist_adj     = np.load(path + 'adj_spatial_dist/'   + str(t) + '.npy').astype(np.float32)
    sp_cluster_adj  = np.load(path + 'adj_spatial_cluster/'+ str(t) + '.npy').astype(np.float32)
    tmep_adj        = np.load(path + 'adj_temporal/'       + str(t) + '.npy').astype(np.float32)

    # --- 2. 轉成 torch.FloatTensor 放到對應 device ---
    sp_dist_t    = torch.from_numpy(sp_dist_adj).to(device)       # [N, N], float32
    sp_cluster_t = torch.from_numpy(sp_cluster_adj).to(device)
    tmep_t       = torch.from_numpy(tmep_adj).to(device)

    # --- 3. 根據 VIEW_NUM 組 adjacency list ---
    if VIEW_NUM == 2:
        adj = [sp_dist_t, tmep_t]

    elif VIEW_NUM == 3:
        adj = [sp_dist_t, sp_cluster_t, tmep_t]

    elif VIEW_NUM == 4:
        if fuse_adj_method == 'add':
            # ✅ 在 torch 裡 add，不在 NumPy 裡加，避免額外大 allocation
            adj_fuse_t = sp_dist_t + sp_cluster_t + tmep_t
        elif fuse_adj_method == 'cat':
            # 如果你有另外 precompute 的 fused adj
            adj_fuse_np = np.load(path + '1view_4type/' + str(t) + '.npy').astype(np.float32)
            adj_fuse_t  = torch.from_numpy(adj_fuse_np).to(device)
        else:
            raise ValueError(f"Unknown fuse_adj_method: {fuse_adj_method}")

        adj = [sp_dist_t, sp_cluster_t, tmep_t, adj_fuse_t]

    else:
        raise ValueError(f"Unsupported VIEW_NUM={VIEW_NUM}")

    # --- 4. node feature 一樣轉成 float32 torch tensor ---
    feat = np.load(path + 'feat/' + str(t) + '.npy').astype(np.float32)  # shape [N, F]
    feat_t = torch.from_numpy(feat).to(device)

    nid = np.load(path + 'all_node_id.npy')  # 這個只是 id，用 numpy 就好

    return adj, feat_t, nid





class MultiView_GNN(nn.Module):
    def __init__(self, in_features, out_size, gat_hidden_size, num_heads=1, dropout=0.2, alpha_gat=0.2, alpha_fusion=0.8, beta_fusion=0.5 ) :
        super(MultiView_GNN, self).__init__()
        self.input_size = in_features
        self.out_size = out_size
        self.gat_hidden_size = gat_hidden_size
        self.alpha = alpha_fusion
        self.beta = beta_fusion
        
        self.GATLayer = GraphAttentionLayer(self.input_size, self.gat_hidden_size, dropout, alpha_gat)
        self.GCNLayer = GraphConvolution(self.input_size, self.gat_hidden_size)
        self.multihead_attn = MultiheadAttention(self.gat_hidden_size,self.gat_hidden_size, num_heads)
#         self.multihead_attn = nn.MultiheadAttention(self.gat_hidden_size, num_heads)
        self.linear_fusion = nn.Linear(self.gat_hidden_size, self.gat_hidden_size)
        
    def forward(self, all_adj, node_feat):
#         print('enter MultiView_GNN()')
#         print('node_feat', node_feat.shape) #torch.Size([1608, 146])
        node_num = round(node_feat.shape[0]/3)
#         print('node num',node_num) #536
        
        hidden_GAT = []
        for adj in all_adj:
#             print('adj',adj.shape) #adj torch.Size([1608, 1608])
            h = self.GATLayer(node_feat, adj)
            # h = self.GCNLayer(node_feat, adj)
            hidden_GAT.append(h)
#         print('shape after GAT',h.shape) #torch.Size([1608, 32])
            
        hidden_self_att = []
        for h in hidden_GAT:
            h = h.unsqueeze(0)
#             h = h.transpose(1,0).unsqueeze(2) #batch_size, seq_length, _ = x.size()
            attn_output = self.multihead_attn(h) #h torch.Size([1608, 32])
#             attn_output = self.multihead_attn(h, h, h)
#             attn_output, attn_output_weights = multihead_attn(query, key, value)
            hidden_self_att.append(attn_output)
#         print('shape after self-att',attn_output.shape) #torch.Size([1, 1608, 32])
        
        hidden_fuse1 = []
        for i in range(len(hidden_self_att)):
            h = self.alpha*hidden_self_att[i]+(1-self.alpha)*hidden_GAT[i]
            hidden_fuse1.append(h)
#         print('shape after fusion', h.shape) #torch.Size([1, 1608, 32])
            
        view_hidden_final = []
        fusion_hidden = torch.zeros(h.shape,device=device)#.to(device=device)
#         print('fusion_hidden',fusion_hidden.shape) #torch.Size([1, 1608, 32])
        for h in hidden_fuse1:
            w = nn.Sigmoid()(self.linear_fusion(h))
            hid_now_view = w*h
            fusion_hidden += hid_now_view
            h_new = self.beta*hid_now_view + (1-self.beta)*h
            view_hidden_final.append(h_new)
#         print('fusion_hidden',fusion_hidden.shape) #torch.Size([1, 1608, 32])
        fusion_hidden = fusion_hidden[0,node_num*2:,:]#.squeeze()
#         print('shape final', h_new.shape) #torch.Size([1, 1608, 32])
#         print('output shape of MultiView_GNN', fusion_hidden.shape)# torch.Size([ 536, 32])
        
        return fusion_hidden.to(device), view_hidden_final


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
#         print('x.size()',x.size()) #torch.Size([32, 3, 1608, 32]) #torch.Size([1, 1608, 32])
#         seq_length , batch_size = x.size() #torch.Size([1608, 32])
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
#         print('qkv size',qkv.size()) #torch.Size([1, 1608, 96])

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim) #(32, 1608, 4, 12)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention




class MultiView_GNN_batch(nn.Module):
    def __init__(self, in_features, out_size, gat_hidden_size, num_heads=1, dropout=0.2, alpha_gat=0.2, alpha_fusion=0.8, beta_fusion=0.5 ) :
        super(MultiView_GNN_batch, self).__init__()
        self.input_size = in_features
        self.out_size = out_size
        self.gat_hidden_size = gat_hidden_size
        self.alpha = alpha_fusion
        self.beta = beta_fusion
        
#         self.GATLayer = GraphAttentionLayer(self.input_size, self.gat_hidden_size, dropout, alpha_gat)
        self.multihead_attn = MultiheadAttention(self.gat_hidden_size,self.gat_hidden_size, num_heads)
#         self.multihead_attn = nn.MultiheadAttention(self.gat_hidden_size, num_heads)
        self.linear_fusion = nn.Linear(self.gat_hidden_size, self.gat_hidden_size)
        
    def forward(self, hidden_GAT):

#         print('shape after GAT',hidden_GAT.shape) #torch.Size([32, 3, 1608, 32])
        batchsz, views, seq_len, embsz = hidden_GAT.shape
        node_num = round(seq_len/3)
        hidden = hidden_GAT.reshape((batchsz,views*seq_len, embsz)) #torch.Size([32, 4824, 32])

        hidden_self_att = self.multihead_attn(hidden)

        hidden_fuse1 = self.alpha*hidden_self_att+(1-self.alpha)*hidden
#         print('shape after fusion', hidden_fuse1.shape) #torch.Size([32, 4824, 32])
        hidden_fuse1 = hidden_fuse1.reshape(hidden_GAT.shape) #torch.Size([32, 3, 1608, 32])
        hidden_fuse1 = hidden_fuse1.permute(1,0,2,3) #torch.Size([3, 32, 1608, 32])
#         print('hidden_fuse1',hidden_fuse1.shape)
            
        view_hidden_final = []
        fusion_hidden = torch.zeros(hidden_fuse1[0].shape,device=device)#.to(device=device)
#         print('fusion_hidden',fusion_hidden.shape) #torch.Size([4824, 32])
        for h in hidden_fuse1:
            w = nn.Sigmoid()(self.linear_fusion(h))
            hid_now_view = w*h
            fusion_hidden += hid_now_view
#         print('fusion_hidden',fusion_hidden.shape) #torch.Size([32, 1608, 32])
        
        fusion_hidden = fusion_hidden[:,node_num*2:,:]#.squeeze()
#         print('shape final', h_new.shape) #torch.Size([1, 1608, 32])
#         print('output shape of MultiView_GNN', fusion_hidden.shape)# torch.Size([32, 536, 32])
        
        return fusion_hidden.to(device), view_hidden_final


def get_certain_node_batch(label_data_stfgn,gid_idx_list):
#     print('label_data_stfgn',label_data_stfgn.shape) #torch.Size([32, 536, 32])
    embed = []
    for count in range(len(gid_idx_list)):
        gid_idx = gid_idx_list[count]
#         print(gid_idx)
        tmp=[]
        for idx in gid_idx:
            tmp.append(label_data_stfgn[count][idx].cpu().detach().numpy())
        embed.append(tmp)
#     print('embed[0]',embed[0].shape,embed[0]) #32
    embed_np = np.array(embed, dtype=np.float32)  # (B, k_neighbor, hidden)
    embed_tensor = torch.from_numpy(embed_np).to(device)
#     print('embed_tensor',embed_tensor.shape) # torch.Size([32, 10, 32])
    return embed_tensor

# %%
# class GRU_EN(nn.Module) :
#     def __init__(self, input_size, hidden_dim, num_layers=1) :
#         super(GRU_EN, self).__init__()
#         self.input_size = input_size
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.lstm = nn.GRU(self.input_size, self.hidden_dim, num_layers = self.num_layers)
#         self.hidden = None
       
#     def init_hidden(self, batch_size):
#         return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
#                 torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())
    
#     def forward(self, in_data):
#         in_data = in_data.permute(1,0,2)
#         out, self.hidden = self.lstm(in_data, self.hidden)
#         return out, self.hidden

class GRU_DE(nn.Module) :
    def __init__(self, num_steps, hidden_dim=32, num_layers=1) :
        super(GRU_DE, self).__init__()
        self.lstm = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.num_steps = num_steps
    
    def forward(self, in_data, hidden):
        # print('in_data',in_data.shape)#(28,32)
        #batch_size, num_steps = outputs.shape
        #in_data = torch.tensor([[0.0]] * batch_size, dtype=torch.float).cuda()
        #in_data = in_data.unsqueeze(0) * batch_size
        in_data = in_data.unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        result = []
        for i in range(self.num_steps):
            output, hidden = self.lstm(in_data, hidden)
            output = self.out(output[-1])
            result.append(output)
            in_data = output.unsqueeze(0)
        result = torch.stack(result).squeeze(2).permute(1,0)
        return result
    
# class GRU_predictor(nn.Module) :
#     def __init__(self, input_size, hidden_dim, num_steps, num_layers=1) :
#         super(GRU_predictor, self).__init__()
#         self.input_size = input_size
#         self.hidden_dim = hidden_dim
#         self.num_steps = num_steps
#         self.num_layers = num_layers
#         self.linear = nn.Linear(self.hidden_dim, 1)
#         self.EN = GRU_EN(self.input_size, self.hidden_dim, self.num_layers)
#         self.DE = GRU_DE(self.num_steps, self.hidden_dim, self.num_layers)
        
#     def forward(self, in_data) :
#         in_data = in_data.unsqueeze(1).repeat(1,8,1)
#         self.EN.hidden = self.EN.init_hidden(in_data.shape[0])
#         init_out, hidden = self.EN(in_data)
#         init_out = nn.ReLU()(self.linear(init_out[-1].float()))
#         out = self.DE(init_out, in_data.shape[0], hidden)
#         #return torch.cat((init_out, out), 1) 
#         return out, hidden



class model(nn.Module) :
    def __init__(self, num_station=10, output_size=1) :
        super(model, self).__init__()
        self.num_station = num_station
        self.prev_slot = 4
        self.hidden_lstm = 32
        self.hidden_linear = 32
        self.hidden_gru = 32
        self.hidden_gnn = 32
        self.output_size = output_size
        
        self.unlabel_lstm_1 = nn.LSTM(11, self.hidden_lstm) #v
        self.unlabel_linear_1 = nn.Linear(135, self.hidden_linear)#v
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.label_lstm_1 = nn.LSTM(12, self.hidden_lstm)#v
        self.label_linear_1 = nn.Linear(136, self.hidden_linear)#v
        self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm+self.hidden_gnn, self.hidden_linear*2)
        self.label_linear_3 = nn.Linear(self.hidden_gnn, self.hidden_linear*2)
        
#         self.stfgn = MultiView_GNN(in_features=146, out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.stfgn = MultiView_GNN_batch(in_features=146, out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn, alpha_fusion=alpha_multiview_fusion)
        self.GATLayer = GraphAttentionLayer(146, self.hidden_gnn, 0.2, 0.2)
        self.GCNLayer = GraphConvolution(146, self.hidden_gnn)
        
        self.idw_attention = Attention_layer(num_station, self.hidden_linear*2, self.hidden_linear*2, 16)
        
        self.GRU = nn.GRUCell(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, bias=True) #nn.GRUCell(input_size, hidden_size)
        # self.GRU_2 = nn.GRUCell(self.hidden_gru , output_size, bias=True)
        self.GRU_DE = GRU_DE(num_steps=output_size)
        # self.GRU_EN_DE = GRU_predictor(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, output_size)
        self.liner_t = nn.Linear(self.hidden_gru, self.hidden_gru)
        self.output_fc = nn.Linear(self.hidden_gru ,output_size )
        
    def forward(self, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label, dis_label, h_t, timestamp, label_id_list) :#timestamp
        
#         batch_adj = []
#         batch_feat = []
        batch_graph_h = []
        gid_idx_list = []
        for i in range(len(timestamp)):
            t = timestamp[i].item()
#             print('timestamp',t)
            gids = label_id_list[i]
            multi_view_adj, node_feat, node_id = read_fusion_graph(t,graph_path)
            gid_idx = nodelist2indexlist(gids.tolist(),node_id)
            gid_idx_list.append(gid_idx)
            
            hidden_GAT = []
            for adj in multi_view_adj:
                h = self.GATLayer(node_feat, adj)
                # h = self.GCNLayer(node_feat.float(), adj.float())
                hidden_GAT.append(h.cpu().detach().numpy())
            batch_graph_h.append(hidden_GAT)
        batch_graph_h = np.array(batch_graph_h)                 # (B, V, N, D)
        batch_graph_h = torch.from_numpy(batch_graph_h).float().to(device)
#         print('batch_graph_h',batch_graph_h.shape)# torch.Size([32, 3, 1608, 32])
        
        label_data_stfgn_batch, _ = self.stfgn(batch_graph_h) 
        label_data_stfgn_batch = get_certain_node_batch(label_data_stfgn_batch,gid_idx_list) # torch.Size([32, 10, 32])
        label_data_stfgn_batch = label_data_stfgn_batch.permute(1,0,2) #[10,32,32]
#         print('label_data_stfgn_batch',label_data_stfgn_batch.shape) # torch.Size([32, 10, 32])

        for j in range(self.prev_slot):
            temp_approximate = F.relu(self.liner_t(h_t)) 
#             print('temp_approximate',temp_approximate.shape) #torch.Size([128, 32])

            unlabel_time_data = meo_unlabel.permute(1,0,2) # torch.Size([128, 4, 11]) ->  torch.Size([4, 128, 11])
            unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data) # torch.Size([4, 128, 11]) ->  torch.Size([4, 128, 32])
            unlabel_time_data = unlabel_time_data.float()[-1] # torch.Size([4, 128, 32]) ->  torch.Size([128, 32])
#             print('unlabel_time_data',unlabel_time_data.shape) #torch.Size([32, 32])
        
            unlabel_fea_data = nn.ReLU()(self.unlabel_linear_1(feature_unlabel)) # torch.Size([128, 135]) -> torch.Size([128, 32])
            unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1) # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64])
            unlabel_data = nn.ReLU()(self.unlabel_linear_2(unlabel_data)) # torch.Size([128, 64]) -> torch.Size([128, 64])
#             print('unlabel_data',unlabel_data.shape) #torch.Size([32, 64])

            if add_labeled_embed: 
                label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3) #torch.Size([128, 10, 4, 1]) + torch.Size([128, 10, 4, 11]) -> torch.Size([128, 10, 4, 12])
                label_time_data = []
                for i in range(self.num_station) :
                    lstm_tmp, _ = self.label_lstm_1(label_time[:,i,:,:].permute(1,0,2)) #torch.Size([128, 4, 12]) -> torch.Size([4, 128, 12]) -> torch.Size([4, 128, 32])
                    lstm_tmp = lstm_tmp.float()[-1] #torch.Size([4, 128, 32]) -> torch.Size([128, 32])
                    label_time_data.append(lstm_tmp)
    #             print('lstm_tmp',lstm_tmp.shape)
                label_feature = []
                for i in range(self.num_station) : #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])
                    label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
    #             print('label_feature[0]',label_feature[0].shape)
                label_data = []    
                for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                    label_data.append(nn.ReLU()(self.label_linear_2(torch.cat([label_time_data[i], label_feature[i],label_data_stfgn_batch[i]], 1))))
            
            else:
                label_data = []    
                for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                    label_data.append(nn.ReLU()(self.label_linear_3( label_data_stfgn_batch[i] )))
            
            # torch.Size([128, 64]) * self.num_station   
#             print('label_data',label_data.shape)

            attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            # attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            attention_out = []
            for n,i in enumerate(label_data) :
                attention_out.append(attention_score[:,n].unsqueeze(1)*i)
            attention_out = torch.sum(torch.stack(attention_out).permute(1,0,2), 1) #torch.Size([128, 64])
        
            sp_approximate = F.relu(attention_out) #torch.Size([128, 64])
#             print('sp_approximate',sp_approximate.shape) #torch.Size([128, 64])
            
            # torch.Size([128, 64]), torch.Size([128, 32]), torch.Size([128, 64])
            X_feat = torch.cat( [unlabel_data,temp_approximate,sp_approximate], dim=1 ) #torch.Size([128, 160])
#             print('X_feat',X_feat.shape) #torch.Size([128, 160])
            h_t = self.GRU(X_feat)
#             print('h_t',h_t.shape) #torch.Size([128, 32])
        out = self.output_fc(h_t)
        # out = self.GRU_DE(h_t,h_t)
#         print('out',out.shape) #torch.Size([128, 1])
        return out
        
        




class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
        
    def forward(self,yhat,y):
        return self.mae(yhat,y)

class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()
        
    def forward(self,yhat,y):
        return self.huber(yhat,y)



def testing(net, data_loader) :

    net.eval()
    loss_sum_rmse = 0.0
    loss_sum_mae = 0.0
    loss_sum_huber = 0.0
    
    with torch.no_grad() :
        for ovi_target, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label,label_id_list, timestamp in tqdm(data_loader):
            h_t = torch.zeros(ovi_target.shape[0],32, device=device)#.to(device=device)
            output = net(meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label, h_t, timestamp, label_id_list)
            # loss = loss_func_huber(output, ovi_target)
            rmse = loss_func_rmse(output, ovi_target)
            mae = loss_func_mae(output, ovi_target)
            loss = (rmse+mae)/2
            loss_sum_huber += loss#.item()
            loss_sum_rmse += rmse#.item()
            loss_sum_mae += mae#.item()
        avg_huber = loss_sum_huber/len(data_loader)
        avg_rmse = loss_sum_rmse/len(data_loader)
        avg_mae = loss_sum_mae/len(data_loader)
    return avg_huber, avg_rmse,avg_mae



if __name__ == '__main__':

    log_file_timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_filename = log_file_timestr+'.log' #datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
    logging.basicConfig(level=logging.INFO, filename='./log/'+log_filename, filemode='w',
	format='[%(asctime)s %(levelname)-8s] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)

    graph_path = 'dataset_processed/graph_data/v2/' #'dataset_processed/graph_data/'
    VIEW_NUM = 4
    fuse_adj_method='add'#'add'/'cat'
    k_neighbor = 10
    random_seed = 99
    batch_size=28
    shuffle=True
    historical_T = 4
    model_output_size=1
    lr=0.0005
    max_epoch = 20
    patient = 3
    load_weight = False #False/True
    load_weight_path = 'model/2022-11-22_16_42_05/tmp_save_model.pt'
    add_labeled_embed = False #False/True
    alpha_multiview_fusion = 0.3
    # cv_k=0

    # print("[Epoch %2d] Training RMSE : %.4f Training MAE : %.4f" %( epoch, avg_rmse, avg_mae))
    logging.info('Fusion+Multi-view')
    # logging.info('GRU Decoder')
    if add_labeled_embed:
        logging.info('with label embedding by mlp&lstm')
    else:
        logging.info('no label embedding by mlp&lstm')
    logging.info('hyper-parm')
    logging.info("VIEW_NUM = %3d" %( VIEW_NUM))
    if VIEW_NUM==4:
        logging.info("fuse_adj_method = "+fuse_adj_method)
    logging.info("k_neighbor = %3d" %( k_neighbor))
    logging.info("batch_size = %4d"%(batch_size))
    logging.info("historical_T = %2d"%(historical_T))
    logging.info("model_output_size = %2d"%(model_output_size))
    logging.info("learning_rate = %.5f"%(lr))
    logging.info("max_epoch = %3d"%(max_epoch))
    logging.info("patient = %2d"%(patient))
    logging.info("cv_k = %2d"%(cv_k))
    logging.info("alpha_multiview_fusion = %.2f"%(alpha_multiview_fusion))

    nearest_col = []
    near_dist_col = []
    for k in range(k_neighbor):
        nearest_col.append('nearest_'+str(k+1))
        near_dist_col.append('nearest_dist_'+str(k+1))

    train_dataset = station_data(mode='train')
    valid_dataset = station_data(mode='valid')
    test_dataset = station_data(mode='test')

    torch.manual_seed(random_seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False)#, pin_memory=True
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    net = model(num_station=k_neighbor, output_size=model_output_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)

    if not os.path.exists('./model/'+log_file_timestr+'/'):
        os.makedirs('./model/'+log_file_timestr+'/')
    if load_weight:
        # net.load_state_dict(torch.load('model/'+log_file_timestr+'/tmp_save_model.pt'))
        net.load_state_dict(torch.load(load_weight_path))

    loss_func_rmse = RMSELoss()
    loss_func_mae = MAELoss()
    loss_func_huber = HuberLoss()

    print("test performance before training...")
    time.sleep(1)
    test_huber_before, test_rmse_before, test_mae_before = testing(net, test_loader)
    print("Huber : %.4f RMSE : %.4f MAE : %.4f" %( test_huber_before, test_rmse_before, test_mae_before))#RMSE : 40.9928 MAE : 19.3708
    logging.info("Before training... Huber : %.4f  RMSE : %.4f  MAE : %.4f" %( test_huber_before, test_rmse_before, test_mae_before))
    torch.cuda.empty_cache()

    loss_history_tr_mae = []
    loss_history_val_mae = []
    loss_history_te_mae = []
    loss_history_tr_rmse = []
    loss_history_val_rmse = []
    loss_history_te_rmse = []
    loss_history_tr_huber = []
    loss_history_val_huber = []
    loss_history_te_huber = []
    
    patient_count=0
    best_epoch = -1
    best_valid_loss = 10000
    for epoch in range(max_epoch):
        try:
            # torch.cuda.empty_cache()
            # if epoch!=0:
            #     net.load_state_dict(torch.load('model/tmp_save_model.pt'))
            #     # torch.save(net.state_dict(), 'model/tmp_save_model.pt')
            net.train()
            running_loss_rmse = 0.0
            running_loss_mae = 0.0
            running_loss_huber = 0.0
            for ovi_target, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label,label_id_list, timestamp in tqdm(train_loader):
                optimizer.zero_grad()
                h_t = torch.zeros(ovi_target.shape[0],32,device=device)#.to(device=device)
                output = net(meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label, h_t, timestamp, label_id_list)
            #     print(output.shape)#torch.Size([128, 1])
                # loss = loss_func_huber(output, ovi_target)
                rmse = loss_func_rmse(output, ovi_target)
                mae = loss_func_mae(output, ovi_target)
                loss = (rmse+mae)/2
                loss.backward()
                optimizer.step()
                running_loss_huber += loss#.item()
                running_loss_rmse += rmse#.item()
                running_loss_mae += mae#.item()
            avg_huber = running_loss_huber/len(train_loader)
            avg_rmse = running_loss_rmse/len(train_loader)
            avg_mae = running_loss_mae/len(train_loader)
            torch.save(net.state_dict(), './model/'+log_file_timestr+'/tmp_save_model.pt')
            print("[Epoch %2d] Training Huber : %.4f Training RMSE : %.4f Training MAE : %.4f" %( epoch, avg_huber, avg_rmse, avg_mae))
            loss_history_tr_mae.append(avg_mae.clone().cpu().detach().numpy())
            loss_history_tr_rmse.append(avg_rmse.clone().cpu().detach().numpy())
            loss_history_tr_huber.append(avg_huber.clone().cpu().detach().numpy())
            logging.info("[Epoch %2d] Training Huber : %.4f Training RMSE : %.4f Training MAE : %.4f" %( epoch, avg_huber, avg_rmse, avg_mae))

            print("Start to validation...")
            time.sleep(1)
            valid_huber, valid_rmse, valid_mae = testing(net, valid_loader)
            print("[Epoch %2d] Validation Huber : %.4f Validation RMSE : %.4f Validation MAE : %.4f" %(epoch, valid_huber, valid_rmse, valid_mae))
            loss_history_val_mae.append(valid_mae.clone().cpu().detach().numpy())
            loss_history_val_rmse.append(valid_rmse.clone().cpu().detach().numpy())
            loss_history_val_huber.append(valid_huber.clone().cpu().detach().numpy())
            logging.info("[Epoch %2d] Validation Huber : %.4f Validation RMSE : %.4f Validation MAE : %.4f" %(epoch, valid_huber, valid_rmse, valid_mae))

            if valid_huber<best_valid_loss:
                best_valid_loss = valid_huber
                patient_count = 0
                torch.save(net.state_dict(), './model/'+log_file_timestr+'/best_wts_my_model.pt')
                print('Save model at epoch ',epoch)
                best_epoch = epoch
                logging.info('Save model at epoch '+str(epoch))
                
            else:
                patient_count+=1
                if patient_count == patient:
                    break
            time.sleep(1)

            print("Start to testing...")
            time.sleep(1)
            test_huber, test_rmse, test_mae = testing(net, test_loader)
            print("[Epoch %2d] Testing Huber : %.4f Testing RMSE : %.4f Testing MAE : %.4f" %(epoch, test_huber, test_rmse, test_mae))
            loss_history_te_mae.append(test_mae.clone().cpu().detach().numpy())
            loss_history_te_rmse.append(test_rmse.clone().cpu().detach().numpy())
            loss_history_te_huber.append(test_huber.clone().cpu().detach().numpy())
            logging.info("[Epoch %2d] Testing Huber : %.4f Testing RMSE : %.4f Testing MAE : %.4f" %(epoch, test_huber, test_rmse, test_mae))
            time.sleep(1)
            
        except Exception as e:
            torch.save(net.state_dict(), './model/'+log_file_timestr+'/tmp_save_model.pt')
            # logging.error("Catch an exception.", exc_info=True)
            # logging.exception('Catch an exception.')
            logging.exception('error in epoch'+str(epoch)) # print('error in epoch',epoch)
            # logging.error("error message:", exc_info=True) # print('error message:', str(e))
            break
    
    logging.info('best epoch: '+str(best_epoch))
    print('loss history: ')
    print('train RMSE', loss_history_tr_rmse)
    print('valid RMSE', loss_history_val_rmse)
    print('test RMSE', loss_history_te_rmse)
    print('train MAE', loss_history_tr_mae)
    print('valid MAE', loss_history_val_mae)
    print('test MAE', loss_history_te_mae)
    print('train Huber', loss_history_tr_huber)
    print('valid Huber', loss_history_val_huber)
    print('test Huber', loss_history_te_huber)

    print('best loss: ')
    print("[Epoch %2d] Training Huber : %.4f Training RMSE : %.4f Training MAE : %.4f" %( best_epoch , loss_history_tr_huber[best_epoch], loss_history_tr_rmse[best_epoch], loss_history_tr_mae[best_epoch]))
    print("[Epoch %2d] Validation Huber : %.4f Validation RMSE : %.4f Validation MAE : %.4f" %(best_epoch , loss_history_val_huber[best_epoch], loss_history_val_rmse[best_epoch], loss_history_val_mae[best_epoch]))
    print("[Epoch %2d] Testing Huber : %.4f Testing RMSE : %.4f Testing MAE : %.4f" %(best_epoch , loss_history_te_huber[best_epoch] , loss_history_te_rmse[best_epoch], loss_history_te_mae[best_epoch]))

    logging.info('best loss: ')
    logging.info("[Epoch %2d] Training Huber : %.4f Training RMSE : %.4f Training MAE : %.4f" %( best_epoch , loss_history_tr_huber[best_epoch], loss_history_tr_rmse[best_epoch], loss_history_tr_mae[best_epoch]))
    logging.info("[Epoch %2d] Validation Huber : %.4f Validation RMSE : %.4f Validation MAE : %.4f" %(best_epoch , loss_history_val_huber[best_epoch], loss_history_val_rmse[best_epoch], loss_history_val_mae[best_epoch]))
    logging.info("[Epoch %2d] Testing Huber : %.4f Testing RMSE : %.4f Testing MAE : %.4f" %(best_epoch , loss_history_te_huber[best_epoch] , loss_history_te_rmse[best_epoch], loss_history_te_mae[best_epoch]))

    if not os.path.exists('./fig/'+log_file_timestr+'/'):
        os.makedirs('./fig/'+log_file_timestr+'/')

    plt.plot(loss_history_tr_rmse,label='train RMSE')
    plt.plot(loss_history_val_rmse,label='valid RMSE')
    plt.legend(prop={'size': 10},loc='lower right')
    plt.savefig('./fig/'+log_file_timestr+'/rmse.png')
    plt.close()
    # plt.show()

    plt.plot(loss_history_tr_mae,label='train MAE')
    plt.plot(loss_history_val_mae,label='valid MAE')
    plt.legend(prop={'size': 10},loc='lower right')
    plt.savefig('./fig/'+log_file_timestr+'/mae.png')
    plt.close()
    # plt.show()

    plt.plot(loss_history_tr_huber,label='train Huber')
    plt.plot(loss_history_val_huber,label='valid Huber')
    plt.legend(prop={'size': 10},loc='lower right')
    plt.savefig('./fig/'+log_file_timestr+'/huber.png')
    plt.close()
    # plt.show()

    plt.plot(loss_history_tr_rmse,label='train RMSE')
    plt.plot(loss_history_val_rmse,label='valid RMSE')
    plt.plot(loss_history_tr_mae,label='train MAE')
    plt.plot(loss_history_val_mae,label='valid MAE')
    plt.plot(loss_history_tr_huber,label='train Huber')
    plt.plot(loss_history_val_huber,label='valid Huber')
    plt.legend(prop={'size': 10},loc='lower right')
    plt.savefig('./fig/'+log_file_timestr+'/rmse_mae_huber.png')
    plt.close()
    # plt.show()




    plt.plot(loss_history_tr_rmse,label='train RMSE')
    plt.plot(loss_history_val_rmse,label='valid RMSE')
    plt.plot(loss_history_te_rmse,label='test RMSE')
    plt.legend(prop={'size': 10},loc='lower right')
    plt.savefig('./fig/'+log_file_timestr+'/rmse_all.png')
    plt.close()
    # plt.show()

    plt.plot(loss_history_tr_mae,label='train MAE')
    plt.plot(loss_history_val_mae,label='valid MAE')
    plt.plot(loss_history_te_mae,label='test MAE')
    plt.legend(prop={'size': 10},loc='lower right')
    plt.savefig('./fig/'+log_file_timestr+'/mae_all.png')
    plt.close()
    # plt.show()

    plt.plot(loss_history_tr_huber,label='train Huber')
    plt.plot(loss_history_val_huber,label='valid Huber')
    plt.plot(loss_history_te_huber,label='test Huber')
    plt.legend(prop={'size': 10},loc='lower right')
    plt.savefig('./fig/'+log_file_timestr+'/huber_all.png')
    plt.close()
    # plt.show()

    plt.plot(loss_history_tr_rmse,label='train RMSE')
    plt.plot(loss_history_val_rmse,label='valid RMSE')
    plt.plot(loss_history_te_rmse,label='test RMSE')
    plt.plot(loss_history_tr_mae,label='train MAE')
    plt.plot(loss_history_val_mae,label='valid MAE')
    plt.plot(loss_history_te_mae,label='test MAE')
    plt.plot(loss_history_tr_huber,label='train Huber')
    plt.plot(loss_history_val_huber,label='valid Huber')
    plt.plot(loss_history_te_huber,label='test Huber')
    plt.legend(prop={'size': 10},loc='lower right')
    plt.savefig('./fig/'+log_file_timestr+'/rmse_mae_huber_all.png')
    plt.close()
    # plt.show()




    # net.load_state_dict(torch.load('./model/'+log_file_timestr+'/best_wts_my_model.pt'))
    # _, train_rmse, train_mae = testing(net, train_loader)
    # print('train')
    # print("MAE: %.2f" % train_mae)
    # print("RMSE: %.2f" % train_rmse)

    # _,valid_rmse, valid_mae = testing(net, valid_loader)
    # print('valid')
    # print("MAE: %.2f" % valid_mae)
    # print("RMSE: %.2f" % valid_rmse)

    # _,test_rmse, test_mae = testing(net, test_loader)
    # print('test')
    # print("MAE: %.2f" % test_mae)
    # print("RMSE: %.2f" % test_rmse)

    # logging.info('final evaluation ')
    # logging.info('training performance: MAE: %.2f, RMSE: %.2f'% (train_mae,train_rmse))
    # logging.info('validation performance: MAE: %.2f, RMSE: %.2f'% (valid_mae,valid_rmse))
    # logging.info('testing performance: MAE: %.2f, RMSE: %.2f'% (test_mae,test_rmse))


    # print("test performance before training...")
    # print("RMSE : %.4f MAE : %.4f" %( test_rmse_before, test_mae_before))

# %%
