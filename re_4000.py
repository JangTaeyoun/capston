# -*- coding: utf-8 -*-
import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# 0) 환경 설정 (재현성 & 성능)
# ---------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -----------------------------
# 1. 데이터 증강 (작은 노이즈)
# -----------------------------
def augment_features(features, noise_std=0.01):
    noise = torch.randn_like(features) * noise_std
    return features + noise


# ---------------------------------------------
# 2. source_type 기반 시계열 그래프 데이터셋
#    - 노드 선택은 past 기준
#    - 라벨은 future_df 조회, 없으면 0
#    - 엣지 생성: source_type 그룹 내 Top-K 페어링(폭증 방지)
# ---------------------------------------------
class SourceBasedSocialMediaGraphDataset(Dataset):
    def __init__(self, past_df, future_df, time_windows=10,
                 max_nodes=None, random_sample=True, pair_topk=5):
        self.past_df = past_df.copy()
        self.future_df = future_df.copy()
        self.time_windows = time_windows
        self.pair_topk = max(1, int(pair_topk))

        if 'source_type' not in self.past_df.columns:
            raise ValueError("source_type 컬럼이 없습니다. 데이터를 확인해주세요.")
        if 'video_id' not in self.past_df.columns:
            raise ValueError("video_id 컬럼이 없습니다. past_df를 확인해주세요.")

        # --- (A) 노드 선택은 past 기준 ---
        unique_videos = self.past_df['video_id'].unique().tolist()

        # 노드 제한
        if max_nodes and len(unique_videos) > max_nodes:
            if random_sample:
                selected_videos = np.random.choice(unique_videos, max_nodes, replace=False)
            else:
                top_videos = self.past_df.groupby('video_id', as_index=False)['views'].max() \
                                         .nlargest(max_nodes, 'views')['video_id'].tolist()
                selected_videos = top_videos
        else:
            selected_videos = unique_videos

        self.past_df = self.past_df[self.past_df['video_id'].isin(selected_videos)].reset_index(drop=True)
        # future_df는 라벨 계산에만 쓰므로 전체 유지(없으면 0 라벨로 처리)
        self.future_df = self.future_df.copy().reset_index(drop=True)

        self.video_to_node = {vid: idx for idx, vid in enumerate(sorted(set(self.past_df['video_id'])))}
        self.node_to_video = {idx: vid for vid, idx in self.video_to_node.items()}
        self.num_nodes = len(self.video_to_node)

        print(f"선택된 노드 수: {self.num_nodes}")
        print(f"랜덤 샘플링: {random_sample}")
        print("source_type 분포:")
        print(self.past_df['source_type'].value_counts().head(10))

        self.st_list = sorted(self.past_df['source_type'].unique().tolist())
        self.st_to_idx = {st: i for i, st in enumerate(self.st_list)}
        self.num_st = len(self.st_list)

        self.create_meaningful_temporal_graph()

    def _safe_ratio(self, a, b):
        a = float(a); b = float(b)
        return (a - b) / (max(a, b) + 1.0)

    def _label_from_future(self, dst_row):
        # dst_row: past의 단일 행
        dst_vid = dst_row['video_id']
        future_data = self.future_df[self.future_df['video_id'] == dst_vid]
        if len(future_data) == 0:
            return 0

        past_views = float(dst_row['views'])
        future_views = float(future_data.iloc[0]['views'])
        view_increase = future_views - past_views

        # 날짜 계산(가능한 경우 원본 날짜, 아니면 timestamp→date)
        if 'original_date' in dst_row and pd.notnull(dst_row['original_date']):
            try:
                first_date = pd.to_datetime(dst_row['original_date'])
            except Exception:
                first_date = pd.to_datetime('2025-05-10')
        else:
            if 'timestamp' in dst_row and pd.notnull(dst_row['timestamp']):
                try:
                    first_date = pd.to_datetime(dst_row['timestamp'])
                except Exception:
                    first_date = pd.to_datetime('2025-05-10')
            else:
                first_date = pd.to_datetime('2025-05-10')

        # (기존 정책 유지) 구간별 임계값
        d = first_date
        if pd.to_datetime('2025-05-10') <= d < pd.to_datetime('2025-05-18'):
            return 1 if view_increase >= 100000 else 0
        elif pd.to_datetime('2025-05-18') <= d < pd.to_datetime('2025-05-25'):
            return 1 if view_increase >= 80000 else 0
        elif pd.to_datetime('2025-05-25') <= d < pd.to_datetime('2025-06-01'):
            return 1 if view_increase >= 60000 else 0
        elif pd.to_datetime('2025-06-01') <= d < pd.to_datetime('2025-06-08'):
            return 1 if view_increase >= 50000 else 0
        elif pd.to_datetime('2025-06-08') <= d < pd.to_datetime('2025-06-15'):
            return 1 if view_increase >= 40000 else 0
        elif pd.to_datetime('2025-06-15') <= d < pd.to_datetime('2025-06-22'):
            return 1 if view_increase >= 30000 else 0
        elif pd.to_datetime('2025-06-22') <= d < pd.to_datetime('2025-06-30'):
            return 1 if view_increase >= 20000 else 0
        else:
            return 0

    def create_meaningful_temporal_graph(self):
        print("source_type 기반 의미 있는 시계열 그래프 데이터 생성 중...")
        timestamps = self.past_df['timestamp'].values if 'timestamp' in self.past_df.columns else np.arange(len(self.past_df))
        timestamps = np.nan_to_num(timestamps, nan=0.0)
        print(f"Timestamp 범위: {timestamps.min()} ~ {timestamps.max()}")
        time_bins = np.linspace(timestamps.min(), timestamps.max(), self.time_windows + 1)

        self.src_nodes, self.dst_nodes = [], []
        self.edge_features, self.timestamps, self.labels = [], [], []
        self.source_types = []

        # 창별 처리
        for i in range(self.time_windows):
            start_time, end_time = time_bins[i], time_bins[i + 1]
            mask = (timestamps >= start_time) & (timestamps < end_time)
            window_data = self.past_df[mask]
            if len(window_data) == 0:
                continue

            # source_type 그룹별 Top-K 페어링
            for source_type, group in window_data.groupby('source_type'):
                if len(group) < 2:
                    continue

                # 조회수 기준 정렬
                g_sorted = group.sort_values('views', ascending=False).reset_index(drop=True)
                # 상위 K와 하위 K 선택(겹치면 uniq)
                top_idx = list(range(min(self.pair_topk, len(g_sorted))))
                bot_idx = list(range(max(0, len(g_sorted) - self.pair_topk), len(g_sorted)))
                cand_idx = sorted(set(top_idx + bot_idx))
                if len(cand_idx) < 2:
                    continue

                vids = g_sorted.loc[cand_idx, 'video_id'].tolist()
                sub = g_sorted.loc[cand_idx].reset_index(drop=True)

                # 양방향 부분-전쌍(폭발 방지용 소수만)
                for j in range(len(sub)):
                    for k in range(len(sub)):
                        if j == k:
                            continue
                        src_vid = sub.iloc[j]['video_id']
                        dst_vid = sub.iloc[k]['video_id']
                        if src_vid not in self.video_to_node or dst_vid not in self.video_to_node:
                            continue

                        src_node = self.video_to_node[src_vid]
                        dst_node = self.video_to_node[dst_vid]

                        src_data = sub.iloc[j]
                        dst_data = sub.iloc[k]

                        views_diff    = self._safe_ratio(src_data['views'],    dst_data['views'])
                        likes_diff    = self._safe_ratio(src_data['likes'],    dst_data['likes'])
                        comments_diff = self._safe_ratio(src_data['comments'], dst_data['comments'])

                        t_norm = i / max(self.time_windows - 1, 1)

                        # ✅ 추가: 소스타입 one-hot
                        st_idx = self.st_to_idx[source_type]                 # 1-1에서 만든 매핑 사용
                        st_onehot = np.zeros(self.num_st, dtype=np.float32)
                        st_onehot[st_idx] = 1.0

                        # ✅ edge feature 확장: [diff 3] + [time 1] + [source_type one-hot]
                        edge_feat = [views_diff, likes_diff, comments_diff, t_norm] + st_onehot.tolist()

                        self.src_nodes.append(src_node)
                        self.dst_nodes.append(dst_node)
                        self.edge_features.append(edge_feat)
                        self.timestamps.append(float(i))
                        self.source_types.append(source_type)

                        viral_label = self._label_from_future(dst_data)
                        self.labels.append(viral_label)

        self.src_nodes = np.array(self.src_nodes, dtype=np.int64)
        self.dst_nodes = np.array(self.dst_nodes, dtype=np.int64)
        self.edge_features = np.array(self.edge_features, dtype=np.float32)
        self.timestamps = np.array(self.timestamps, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"생성된 그래프: {len(self.src_nodes)}개 엣지, {self.num_nodes}개 노드")
        print(f"바이럴 라벨 분포: {np.sum(self.labels)}/{len(self.labels)} ({np.mean(self.labels)*100:.1f}%)")
        source_type_counts = pd.Series(self.source_types).value_counts()
        print("\nsource_type별 엣지 분포 (상위 10개):")
        print(source_type_counts.head(10))
        if len(self.src_nodes) == 0:
            raise ValueError("엣지가 생성되지 않았습니다. 데이터를 확인해주세요.")

    def __len__(self):
        return len(self.src_nodes)

    def __getitem__(self, idx):
        return (
            self.src_nodes[idx],
            self.dst_nodes[idx],
            self.edge_features[idx],
            self.timestamps[idx],
            self.labels[idx]
        )


# ---------------------------------
# 2-1. 공출현 기반 희소 인접행렬 생성 (SCN용)
#      - Train 인덱스만 사용해 누수 차단
# ---------------------------------
def build_cooccurrence_adj_from_indices(dataset, indices, num_nodes, symmetric=True, add_self_loops=True):
    from collections import defaultdict
    counts = defaultdict(int)

    src_arr = dataset.src_nodes[indices]
    dst_arr = dataset.dst_nodes[indices]

    for s, d in zip(src_arr, dst_arr):
        s = int(s); d = int(d)
        counts[(s, d)] += 1
        if symmetric:
            counts[(d, s)] += 1

    if len(counts) == 0:
        # fall back to identity
        indices_t = torch.arange(num_nodes)
        i = torch.stack([indices_t, indices_t])
        v = torch.ones(num_nodes, dtype=torch.float32)
        return torch.sparse_coo_tensor(i, v, (num_nodes, num_nodes)).coalesce()

    rows, cols, vals = [], [], []
    for (r, c), v in counts.items():
        rows.append(r); cols.append(c); vals.append(float(v))

    i = torch.tensor([rows, cols], dtype=torch.long)
    v = torch.tensor(vals, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(i, v, (num_nodes, num_nodes)).coalesce()

    # Row-normalize
    row_sums = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1e-6)
    norm_v = adj.values() / row_sums[adj.indices()[0]]
    adj = torch.sparse_coo_tensor(adj.indices(), norm_v, (num_nodes, num_nodes)).coalesce()

    # Add self-loops and renormalize
    if add_self_loops:
        diag_idx = torch.arange(num_nodes, dtype=torch.long)
        diag_i = torch.stack([diag_idx, diag_idx])
        diag_v = torch.ones(num_nodes, dtype=torch.float32)
        eye = torch.sparse_coo_tensor(diag_i, diag_v, (num_nodes, num_nodes)).coalesce()

        adj = (adj + eye).coalesce()
        row_sums = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1e-6)
        v2 = adj.values() / row_sums[adj.indices()[0]]
        adj = torch.sparse_coo_tensor(adj.indices(), v2, (num_nodes, num_nodes)).coalesce()

    return adj


# -----------------------------
# 3. TGN Memory 모듈
# -----------------------------
class SourceBasedTGNMemory(nn.Module):
    def __init__(self, num_nodes, memory_dim, message_dim, edge_feat_dim=3, time_decay_alpha=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_nodes, memory_dim) * 0.1, requires_grad=False)
        self.register_buffer('last_update', torch.zeros(num_nodes))
        self.gru = nn.GRUCell(message_dim, memory_dim)
        self.message_encoder = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_feat_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        self.time_decay_alpha = time_decay_alpha

    def forward(self, src_nodes, dst_nodes, edge_feat, timestamps):
        device = self.memory.device
        src_nodes = src_nodes.long().to(device)
        dst_nodes = dst_nodes.long().to(device)
        edge_feat = edge_feat.to(device)
        timestamps = timestamps.to(device)

        batch_size = len(src_nodes)
        for i in range(batch_size):
            s = src_nodes[i].item()
            d = dst_nodes[i].item()
            t = timestamps[i].item()

            # 교체 후
            dt_s_t = torch.tensor(max(t - self.last_update[s].item(), 0.0),
                      device=self.memory.device, dtype=self.memory.dtype)
            dt_d_t = torch.tensor(max(t - self.last_update[d].item(), 0.0),
                      device=self.memory.device, dtype=self.memory.dtype)

            decay_s = torch.exp(-self.time_decay_alpha * dt_s_t)
            decay_d = torch.exp(-self.time_decay_alpha * dt_d_t)

            prev_src = (self.memory[s] * decay_s).unsqueeze(0)
            prev_dst = (self.memory[d] * decay_d).unsqueeze(0)


            msg_in = torch.cat([prev_src, prev_dst, edge_feat[i].unsqueeze(0)], dim=1)
            message = self.message_encoder(msg_in)

            upd_src = self.gru(message, prev_src)
            upd_dst = self.gru(message, prev_dst)

            # 교체 후
            with torch.no_grad():
                self.memory[s].copy_(upd_src.squeeze(0))
                self.memory[d].copy_(upd_dst.squeeze(0))
                 # last_update도 dtype/device 맞춰 안전하게 기록
                t_val = torch.as_tensor(t, device=self.last_update.device, dtype=self.last_update.dtype)
                self.last_update[s].copy_(t_val)
                self.last_update[d].copy_(t_val)


    def get_memory(self, node_indices):
        return self.memory[node_indices]

    def reset_memory(self):
        with torch.no_grad():
            self.memory.copy_(torch.randn_like(self.memory) * 0.1)
            self.last_update.zero_()



# --------------------------------
# 4. Graph Transformer Layer
# --------------------------------
class SourceBasedGraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads=4, ffn_dim=64, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


# -----------------------------
# 5. SCN Layer (희소/밀집 adj 지원)
# -----------------------------
class SourceBasedSCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        if hasattr(adj, "is_sparse") and adj.is_sparse:
            if adj.dtype != torch.float32:
                adj = adj.float()
            adj = adj.coalesce()
            orig_dtype = x.dtype
            if x.is_cuda:
                with torch.cuda.amp.autocast(False):
                    support = torch.sparse.mm(adj, x.float())
            else:
                support = torch.sparse.mm(adj, x.to(torch.float32))
            support = support.to(orig_dtype)
        else:
            if adj.shape[0] == adj.shape[1] and torch.allclose(
                adj, torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            ):
                support = x
            else:
                support = torch.mm(adj, x)

        out = self.linear(support)
        out = self.norm(out)
        return self.dropout(F.relu(out))


# ------------------------------------------------
# 6. 전체 모델
# ------------------------------------------------
class SourceBasedGT_TGN_SocialViral(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, memory_dim=24, message_dim=32, scn_dim=12, num_gt_layers=1,
                 update_memory_in_eval=True, edge_feat_dim=3, heads=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory = SourceBasedTGNMemory(num_nodes, memory_dim, message_dim, edge_feat_dim=edge_feat_dim, time_decay_alpha=0.1)
        self.node_emb = nn.Linear(node_feat_dim, memory_dim)
        self.mem_proj = nn.Linear(memory_dim * 2, memory_dim)

        self.gt_layers = nn.ModuleList([
            SourceBasedGraphTransformerLayer(memory_dim, heads=4, ffn_dim=64, dropout=0.3)
            for _ in range(num_gt_layers)
        ])
        self.scn = SourceBasedSCNLayer(memory_dim, scn_dim, dropout=0.3)

        self.edge_pred = nn.Sequential(
            nn.Linear(2 * scn_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

        self.update_memory_in_eval = update_memory_in_eval

        self.register_buffer("all_nodes_buf", torch.arange(num_nodes))

    def forward(self, src, dst, edge_feat, timestamps, node_features, adj):
        device = node_features.device
        batch_size = len(src)

        # === 예측용 스냅샷을 먼저 확보 ===
        node_embs = self.node_emb(node_features)                     # (N, M)
        all_nodes = self.all_nodes_buf.to(device)
        mem_snapshot = self.memory.get_memory(all_nodes).clone()     # 스냅샷 고정

        x0 = torch.cat([node_embs, mem_snapshot], dim=-1)            # (N, 2M)
        x0 = self.mem_proj(x0)                                       # (N, M)

        gt_input = x0.unsqueeze(0).expand(batch_size, -1, -1)        # (B, N, M)
        gt_out = gt_input
        for gt_layer in self.gt_layers:
            gt_out = gt_layer(gt_out)

        # 배치 차원 반복 (간단 명료 유지)
        scn_out = torch.stack([self.scn(gt_out[i], adj) for i in range(gt_out.size(0))], dim=0)  # (B, N, scn_dim)

        b_idx = torch.arange(src.size(0), device=device)
        src_emb = scn_out[b_idx, src]
        dst_emb = scn_out[b_idx, dst]
        logits = self.edge_pred(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)

        # === 예측을 만든 뒤에야 메모리를 업데이트 ===
        if self.training or self.update_memory_in_eval:
            self.memory(src, dst, edge_feat, timestamps)

        return logits


    def reset_memory(self):
        self.memory.reset_memory()


# ---------------------------------
# 7. 데이터 로딩 및 전처리
# ---------------------------------
def source_based_load_and_prepare_data(past_file, future_file):
    print("source_type 기반 데이터 로딩 중...")
    past_df = pd.read_csv(past_file, encoding='utf-8-sig')
    future_df = pd.read_csv(future_file, encoding='utf-8-sig')

    past_df.columns = [col.strip() for col in past_df.columns]
    future_df.columns = [col.strip() for col in future_df.columns]

    if 'source_type' not in past_df.columns:
        raise ValueError("past_df에 source_type 컬럼이 없습니다.")
    if 'video_id' not in past_df.columns:
        raise ValueError("past_df에 video_id 컬럼이 없습니다.")

    for col in ['views', 'likes', 'comments']:
        if col in past_df.columns:
            past_df[col] = pd.to_numeric(past_df[col], errors='coerce').fillna(0)
        if col in future_df.columns:
            future_df[col] = pd.to_numeric(future_df[col], errors='coerce').fillna(0)

    if 'timestamp' not in past_df.columns:
        past_df['timestamp'] = np.arange(len(past_df))
        past_df['original_date'] = pd.to_datetime('2025-05-10') + pd.to_timedelta(past_df['timestamp'], unit='D')
    else:
        past_df['original_date'] = pd.to_datetime(past_df['timestamp'], errors='coerce')
        past_df['timestamp'] = (past_df['original_date'] - past_df['original_date'].min()).dt.days
        if isinstance(past_df['timestamp'], pd.Series):
            past_df['timestamp'] = past_df['timestamp'].fillna(0)

    print(f"과거 데이터: {len(past_df)}개 영상")
    print(f"미래 데이터: {len(future_df)}개 영상")
    print(f"source_type 종류: {past_df['source_type'].nunique()}개")
    return past_df, future_df


# ------------------------------------------------------------
# 8. 시간순 학습 파이프라인 (누수 방지, BF16, NaN 가드)
# ------------------------------------------------------------
def source_based_train_gt_tgn_social(dataset, node_features, num_nodes,
                                     device='cpu', epochs=100, train_batch_size=32, val_batch_size=16, lr=0.001):
    print("Source-based GT-TGN+SCN 모델 학습 시작...")

    # 시간순 split (80% / 20%)
    order = np.argsort(dataset.timestamps)
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    train_indices = order[:train_size]
    val_indices   = order[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)

    pin = (device == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=val_batch_size,   shuffle=False, pin_memory=True, num_workers=2)

    # === (누수 차단) Train 엣지만으로 adj 생성 ===
    adj = build_cooccurrence_adj_from_indices(dataset, train_indices, num_nodes, symmetric=True, add_self_loops=True)

    # 🔹 추가: edge feature 차원 파악
    edge_feat_dim = dataset.edge_features.shape[1]  # 3 + 1(time) + num_st (또는 표준화 후 동일)

    # 모델
    model = SourceBasedGT_TGN_SocialViral(
        num_nodes=num_nodes,
        node_feat_dim=node_features.shape[1],
        memory_dim=24,
        message_dim=32,
        scn_dim=12,
        num_gt_layers=1,
        update_memory_in_eval=True,  # ✅ 검증/평가에서도 메모리 업데이트
        edge_feat_dim=edge_feat_dim,
        heads=2).to(device)

    # === (누수 차단) pos_weight도 Train 라벨로만 계산 ===
    train_labels_for_weight = dataset.labels[train_indices]
    num_pos = float(train_labels_for_weight.sum())
    num_neg = float(len(train_labels_for_weight) - num_pos)

    # ratio 계산 + 극단값 방지 클램프
    ratio = num_neg / max(num_pos, 1.0)   # 0으로 나누는 것 방지
    ratio = max(1.0, min(ratio, 20.0))
    pos_weight = torch.tensor([ratio], device=device)              # 상한값 20 (필요에 따라 10~50 조절 가능)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[info] pos_weight(ratio) = {ratio:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.7)

    best_val_auc = 0.0
    train_losses, val_metrics = [], []

    for epoch in range(epochs):
        # ---------------- Train ----------------
        model.train()
        model.reset_memory()
        total_loss = 0.0

        node_features_dev = node_features.to(device)
        adj_dev = adj.to(device)

        for src, dst, edge_feat, ts, labels in train_loader:
            src, dst = src.to(device), dst.to(device)
            edge_feat = torch.as_tensor(edge_feat, dtype=torch.float32, device=device)
            ts = torch.as_tensor(ts, dtype=torch.float32, device=device)
            labels = torch.as_tensor(labels, dtype=torch.float32, device=device)

            node_features_aug = augment_features(node_features_dev)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                pred = model(src, dst, edge_feat, ts, node_features_aug, adj_dev)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=50.0, neginf=-50.0)
                
                bce_loss = criterion(pred, labels)

                 # 2) 쌍대 순위 손실 (AUC 직접 밀어줌)
                pos_idx = (labels > 0.5).nonzero(as_tuple=False).squeeze(-1)
                neg_idx = (labels < 0.5).nonzero(as_tuple=False).squeeze(-1)
                pair_loss = torch.tensor(0.0, device=pred.device)
                if pos_idx.numel() > 0 and neg_idx.numel() > 0:
                    k = min(256, pos_idx.numel(), neg_idx.numel())  # 과도한 계산 방지용 샘플링
                    p = pos_idx[torch.randint(pos_idx.numel(), (k,), device=pred.device)]
                    n = neg_idx[torch.randint(neg_idx.numel(), (k,), device=pred.device)]
                    s_pos = pred[p]
                    s_neg = pred[n]
                    pair_loss = F.softplus(-(s_pos - s_neg)).mean()  # log(1+exp(-(pos-neg)))

                # 3) 최종 손실
                loss = bce_loss + 0.2 * pair_loss  # 람다 0.1~0.3에서 조정

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # ---------------- Validation ----------------
        model.eval()
        model.reset_memory()  # 초기화 후, eval에서도 forward가 메모리를 순차 업데이트함

        node_features_eval = node_features.to(device)
        adj_eval = adj.to(device)

        # 🔹 메모리 워밍업: Train 엣지를 한 번 흘려 메모리만 갱신 (예측 X)
        with torch.no_grad():
            for src_w, dst_w, edge_feat_w, ts_w, _ in train_loader:
                src_w, dst_w = src_w.to(device), dst_w.to(device)
                edge_feat_w = torch.as_tensor(edge_feat_w, dtype=torch.float32, device=device)
                ts_w = torch.as_tensor(ts_w, dtype=torch.float32, device=device)
                model.memory(src_w, dst_w, edge_feat_w, ts_w)

        val_preds, val_labels = [], []

        node_features_eval = node_features.to(device)
        adj_eval = adj.to(device)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            for src, dst, edge_feat, ts, labels in val_loader:
                src, dst = src.to(device), dst.to(device)
                edge_feat = torch.as_tensor(edge_feat, dtype=torch.float32, device=device)
                ts = torch.as_tensor(ts, dtype=torch.float32, device=device)
                labels = labels.to(device)

                pred = model(src, dst, edge_feat, ts, node_features_eval, adj_eval)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=50.0, neginf=-50.0)
                val_preds.append(torch.sigmoid(pred).float().cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_preds = np.concatenate(val_preds) if len(val_preds) > 0 else np.array([])
        val_labels = np.concatenate(val_labels) if len(val_labels) > 0 else np.array([])
        val_preds = np.nan_to_num(val_preds, nan=0.5, posinf=1.0, neginf=0.0)

        if val_labels.size > 0:
            val_preds_binary = (val_preds > 0.5).astype(int)
            print(f"[진단] Val label 분포: {np.bincount(val_labels.astype(int))}")
            print(f"[진단] Val 예측 분포: {np.bincount(val_preds_binary)}")
            print(f"[진단] Val 예측 확률 평균: {np.mean(val_preds):.4f}")

            try:
                val_auc = roc_auc_score(val_labels, val_preds) if np.unique(val_labels).size > 1 else 0.5
            except Exception:
                val_auc = 0.5
            val_accuracy = accuracy_score(val_labels, val_preds_binary)
            val_precision = precision_score(val_labels, val_preds_binary, zero_division=0)
            val_recall = recall_score(val_labels, val_preds_binary, zero_division=0)
            val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)

            train_losses.append(total_loss / max(len(train_loader), 1))
            val_metrics.append({'auc': val_auc, 'accuracy': val_accuracy, 'precision': val_precision,
                                'recall': val_recall, 'f1': val_f1})

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'best_source_gt_tgn_social_re_000.pth')

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | "
                  f"Val AUC: {val_auc:.4f} | Val Acc: {val_accuracy:.4f} | "
                  f"Val Prec: {val_precision:.4f} | Val Recall: {val_recall:.4f} | "
                  f"Val F1: {val_f1:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / max(len(train_loader), 1):.4f} | (검증 샘플 없음)")

        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f"최고 검증 AUC: {best_val_auc:.4f}")
    return model, train_losses, val_metrics, val_dataset, train_indices, val_indices, adj


# -------------------------------------------------
# 9. 평가 (val 전용, 메모리 고정 초기화 후 eval에서 업데이트)
# -------------------------------------------------
def source_based_evaluate_and_save_results(model, dataset_subset, node_features, adj, past_df, future_df,
                                           device='cpu', val_batch_size=16, out_csv='source_gt_tgn_social_viral_results(re_4000).csv',  warmup_indices=None):
    print("Source-based 예측 결과 평가 중 (Validation subset)...")
    model.eval()
    model.reset_memory()

    pin = (device == 'cuda')

    # 한 번만 디바이스로 올려두기
    node_features_dev = node_features.to(device)
    adj_dev = adj.to(device)

    # ✅ Train 엣지로만 워밍업 (검증 엣지 절대 X)
    if warmup_indices is not None and len(warmup_indices) > 0:
        warm_ds = Subset(dataset_subset.dataset, warmup_indices)
        warm_loader = DataLoader(warm_ds, batch_size=val_batch_size, shuffle=False,
                             pin_memory=pin, num_workers=0)
    with torch.no_grad():
        for src_w, dst_w, edge_feat_w, ts_w, _ in warm_loader:
            src_w, dst_w = src_w.to(device), dst_w.to(device)
            edge_feat_w = torch.as_tensor(edge_feat_w, dtype=torch.float32, device=device)
            ts_w = torch.as_tensor(ts_w, dtype=torch.float32, device=device)
            model.memory(src_w, dst_w, edge_feat_w, ts_w)  # 예측 X, 메모리만 업데이트

    loader = DataLoader(dataset_subset, batch_size=val_batch_size, shuffle=False, pin_memory=pin, num_workers=0)

    all_preds, all_labels, all_src, all_dst = [], [], [], []

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
        for src, dst, edge_feat, ts, labels in loader:
            src, dst = src.to(device), dst.to(device)
            edge_feat = torch.as_tensor(edge_feat, dtype=torch.float32, device=device)
            ts = torch.as_tensor(ts, dtype=torch.float32, device=device)

            pred = model(src, dst, edge_feat, ts, node_features_dev, adj_dev)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=50.0, neginf=-50.0)

            all_preds.append(torch.sigmoid(pred).float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_src.append(src.cpu().numpy())
            all_dst.append(dst.cpu().numpy())

    if device == 'cuda':
        torch.cuda.empty_cache()

    all_preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
    all_src = np.concatenate(all_src) if len(all_src) > 0 else np.array([])
    all_dst = np.concatenate(all_dst) if len(all_dst) > 0 else np.array([])
    all_preds = np.nan_to_num(all_preds, nan=0.5, posinf=1.0, neginf=0.0)

    if all_labels.size > 0:
        y_pred_bin = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, y_pred_bin)
        precision = precision_score(all_labels, y_pred_bin, zero_division=0)
        recall = recall_score(all_labels, y_pred_bin, zero_division=0)
        f1 = f1_score(all_labels, y_pred_bin, zero_division=0)
        auc = roc_auc_score(all_labels, all_preds) if np.unique(all_labels).size > 1 else None

        print("\n=== Source-based GT-TGN+SCN 모델 성능 (Validation) ===")
        print(f"정확도: {accuracy:.4f}")
        print(f"정밀도: {precision:.4f}")
        print(f"재현율: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("AUC: NA (single class)" if auc is None else f"AUC: {auc:.4f}")
        print("\n=== 상세 분류 보고서 ===")
        print(classification_report(all_labels, y_pred_bin))
    else:
        print("평가용 샘플이 없습니다.")

    # 결과 저장
    results_df = pd.DataFrame({
        'src_node_id': all_src.astype(int) if all_src.size > 0 else [],
        'dst_node_id': all_dst.astype(int) if all_dst.size > 0 else [],
        'src_video_id': [dataset_subset.dataset.node_to_video[int(s)] for s in all_src] if all_src.size > 0 else [],
        'dst_video_id': [dataset_subset.dataset.node_to_video[int(d)] for d in all_dst] if all_dst.size > 0 else [],
        'prediction_probability': all_preds,
        'actual_viral': all_labels,
        'predicted_viral': (all_preds > 0.5).astype(int)
    })

    if not results_df.empty:
        src_source_types, dst_source_types = [], []
        for src_vid, dst_vid in zip(results_df['src_video_id'], results_df['dst_video_id']):
            src_row = past_df[past_df['video_id'] == src_vid]
            dst_row = past_df[past_df['video_id'] == dst_vid]
            src_source_types.append(src_row['source_type'].iloc[0] if len(src_row) > 0 else 'Unknown')
            dst_source_types.append(dst_row['source_type'].iloc[0] if len(dst_row) > 0 else 'Unknown')

        results_df['src_source_type'] = src_source_types
        results_df['dst_source_type'] = dst_source_types
        results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"\n결과가 '{out_csv}'에 저장되었습니다.")

        # source_type별 성능 (AUC 단일클래스 가드)
        print("\n=== Source-type별 성능 분석 (Validation) ===")
        for st in results_df['src_source_type'].unique():
            if st == 'Unknown':
                continue
            mask = (results_df['src_source_type'] == st)
            if mask.sum() <= 10:
                continue
            subset = results_df[mask]
            acc = accuracy_score(subset['actual_viral'], subset['predicted_viral'])
            pos_rate = float(subset['actual_viral'].mean())
            if subset['actual_viral'].nunique() < 2:
                print(f"{st}: AUC=NA (single class), Acc={acc:.4f}, PosRate={pos_rate:.1%} (n={len(subset)})")
            else:
                subset_auc = roc_auc_score(subset['actual_viral'], subset['prediction_probability'])
                print(f"{st}: AUC={subset_auc:.4f}, Acc={acc:.4f} (n={len(subset)})")

    return results_df


# -----------------------------
# 10. main
# -----------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    try:
        # 1) 데이터 로딩
        past_df, future_df = source_based_load_and_prepare_data(
            'filter(5.10~6.30).csv',
            'filter(7.3).csv'
        )

        # 2) 노드 제한: 4000개 (past 기준), 시간창/학습 설정
        max_nodes = 4000
        time_windows = 5
        epochs = 100
        train_batch_size = 8
        val_batch_size = 4
        lr = 2.5e-4
        pair_topk = 40  # 그룹 내 상·하위 K만 페어링

        print(f"시간 윈도우: {time_windows} | 에폭: {epochs}")
        print(f"배치(Train/Val): {train_batch_size} / {val_batch_size} | lr: {lr}")

        # 3) 데이터셋 생성 (노드 선택은 past 기준, future는 라벨만)
        print("source_type 기반 시계열 그래프 데이터 생성 중...")
        dataset = SourceBasedSocialMediaGraphDataset(
            past_df=past_df,
            future_df=future_df,
            time_windows=time_windows,
            max_nodes=max_nodes,
            random_sample=True,
            pair_topk=pair_topk
        )

        # 4) 노드 특성 (views/likes/comments → 표준화 → 10차원 패딩)
        node_feature_cols = ['views', 'likes', 'comments']
        node_features = []
        for idx in range(dataset.num_nodes):
            vid = dataset.node_to_video[idx]
            row = past_df[past_df['video_id'] == vid]
            if len(row) > 0:
                vals = [row.iloc[0][c] if c in row.columns and pd.notnull(row.iloc[0][c]) else 0 for c in node_feature_cols]
                node_features.append(vals)
            else:
                node_features.append([0] * len(node_feature_cols))
        node_features = np.array(node_features, dtype=np.float32)
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        node_features = torch.tensor(node_features, dtype=torch.float32)
        if node_features.shape[1] < 10:
            pad = torch.zeros((node_features.shape[0], 10 - node_features.shape[1]), dtype=torch.float32)
            node_features = torch.cat([node_features, pad], dim=1)

        print(f"노드 특성 차원: {node_features.shape}")

        # 5) 학습 (시간순, 누수 방지)
        model, train_losses, val_metrics, val_dataset, train_indices, val_indices, adj = source_based_train_gt_tgn_social(
            dataset=dataset,
            node_features=node_features,
            num_nodes=dataset.num_nodes,
            device=device,
            epochs=epochs,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            lr=lr
        )

        # 6) 검증 세트 평가 및 저장 (adj는 train-only)
        results_df = source_based_evaluate_and_save_results(
            model=model,
            dataset_subset=val_dataset,
            node_features=node_features,
            adj=adj,
            past_df=past_df,
            future_df=future_df,
            device=device,
            val_batch_size=val_batch_size,
            out_csv='source_gt_tgn_social_viral_results(re_4000).csv',
            warmup_indices=train_indices  # ← 여기!
        )

        print("\n=== 완료 ===")
        print(f"검증 엣지 수: {len(results_df)}")
        if not results_df.empty:
            print(f"바이럴 라벨 비율: {results_df['actual_viral'].mean()*100:.1f}%")
            print(f"예측 바이럴 비율: {results_df['predicted_viral'].mean()*100:.1f}%")

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
