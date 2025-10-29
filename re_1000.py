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
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# ---------------------------------------------------------------------
# 전역 설정 (재현성 + matmul 성능)
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Ampere/Blackwell에서 matmul 최적화(선택)
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


# ---------------------------------------------------------------------
# 1) 간단한 데이터 증강: 노드 특성에 작은 가우시안 노이즈
# ---------------------------------------------------------------------
def augment_features(features: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    if noise_std <= 0:
        return features
    noise = torch.randn_like(features) * noise_std
    return features + noise


# ---------------------------------------------------------------------
# 2) source_type 기반 시계열 그래프 데이터셋
# ---------------------------------------------------------------------
class SourceBasedSocialMediaGraphDataset(Dataset):
    def __init__(
        self,
        past_df: pd.DataFrame,
        future_df: pd.DataFrame,
        time_windows: int = 10,
        max_nodes: int | None = None,
        random_sample: bool = False,
        top_by_views: bool = True,  # random_sample=False일 때 views 상위 우선
    ):
        self.past_df = past_df.copy()
        self.future_df = future_df.copy()
        self.time_windows = time_windows

        if 'source_type' not in self.past_df.columns:
            raise ValueError("source_type 컬럼이 없습니다. past_df를 확인하세요.")

        # 공통 video_id만 사용
        common_videos = list(set(self.past_df['video_id']) & set(self.future_df['video_id']))
        self.past_df = self.past_df[self.past_df['video_id'].isin(common_videos)].reset_index(drop=True)
        self.future_df = self.future_df[self.future_df['video_id'].isin(common_videos)].reset_index(drop=True)

        # max_nodes 적용
        selected_videos = common_videos
        if max_nodes and len(common_videos) > max_nodes:
            if random_sample:
                rng = np.random.default_rng(SEED)
                selected_videos = rng.choice(common_videos, max_nodes, replace=False).tolist()
            else:
                if top_by_views and 'views' in self.past_df.columns:
                    top = self.past_df[['video_id', 'views']].dropna()
                    top = top.sort_values('views', ascending=False).head(max_nodes)
                    selected_videos = top['video_id'].tolist()
                else:
                    selected_videos = common_videos[:max_nodes]

        # 필터 적용
        self.past_df = self.past_df[self.past_df['video_id'].isin(selected_videos)].reset_index(drop=True)
        self.future_df = self.future_df[self.future_df['video_id'].isin(selected_videos)].reset_index(drop=True)

        # 인덱스 매핑
        self.video_to_node = {vid: idx for idx, vid in enumerate(selected_videos)}
        self.node_to_video = {idx: vid for vid, idx in self.video_to_node.items()}
        self.num_nodes = len(selected_videos)

        print(f"선택된 노드 수: {self.num_nodes}")
        print(f"랜덤 샘플링: {random_sample}")
        print("source_type 분포(상위 10):")
        print(self.past_df['source_type'].value_counts().head(10))

        self._create_meaningful_temporal_graph()

    def _create_meaningful_temporal_graph(self):
        print("source_type 기반 의미 있는 시계열 그래프 데이터 생성 중...")

        # timestamp 준비
        if 'timestamp' in self.past_df.columns:
            timestamps = self.past_df['timestamp'].values
        else:
            timestamps = np.arange(len(self.past_df))
        timestamps = np.nan_to_num(timestamps, nan=0.0)

        print(f"Timestamp 범위: {np.min(timestamps)} ~ {np.max(timestamps)}")

        time_bins = np.linspace(np.min(timestamps), np.max(timestamps), self.time_windows + 1)

        src_nodes, dst_nodes = [], []
        edge_features, ts_list, labels = [], [], []
        edge_source_types = []

        # 윈도우별 + source_type 그룹 내부에서 완전쌍방향 엣지 생성
        for i in range(self.time_windows):
            start_time, end_time = time_bins[i], time_bins[i + 1]
            mask = (timestamps >= start_time) & (timestamps < end_time)
            window_data = self.past_df[mask]
            if len(window_data) == 0:
                continue

            for stype, group in window_data.groupby('source_type'):
                if len(group) < 2:
                    continue
                vids = group['video_id'].tolist()

                # 빠른 접근을 위해 인덱싱 캐시
                group_by_vid = {vid: group[group['video_id'] == vid].iloc[0] for vid in vids}

                for j, src_vid in enumerate(vids):
                    for k, dst_vid in enumerate(vids):
                        if j == k:
                            continue

                        s_node = self.video_to_node[src_vid]
                        d_node = self.video_to_node[dst_vid]
                        s_row = group_by_vid[src_vid]
                        d_row = group_by_vid[dst_vid]

                        # 엣지 특성(정규화된 차이값)
                        def safe_num(x):
                            try:
                                return float(x)
                            except Exception:
                                return 0.0

                        sv, sl, sc = safe_num(s_row.get('views', 0)), safe_num(s_row.get('likes', 0)), safe_num(s_row.get('comments', 0))
                        dv, dl, dc = safe_num(d_row.get('views', 0)), safe_num(d_row.get('likes', 0)), safe_num(d_row.get('comments', 0))

                        views_diff = (sv - dv) / (max(sv, dv) + 1.0)
                        likes_diff = (sl - dl) / (max(sl, dl) + 1.0)
                        comments_diff = (sc - dc) / (max(sc, dc) + 1.0)

                        src_nodes.append(s_node)
                        dst_nodes.append(d_node)
                        edge_features.append([views_diff, likes_diff, comments_diff])
                        ts_list.append(float(i))
                        edge_source_types.append(stype)

                        # 라벨 생성
                        future_row = self.future_df[self.future_df['video_id'] == dst_vid]
                        if len(future_row) > 0:
                            past_views = dv
                            future_views = safe_num(future_row.iloc[0].get('views', 0))
                            view_inc = future_views - past_views

                            # 날짜 처리
                            first_date = None
                            if 'original_date' in d_row and pd.notnull(d_row['original_date']):
                                try:
                                    first_date = pd.to_datetime(d_row['original_date'])
                                except Exception:
                                    first_date = None
                            if first_date is None:
                                try:
                                    first_date = pd.to_datetime(d_row.get('timestamp', None))
                                except Exception:
                                    first_date = None
                            if first_date is None:
                                first_date = pd.to_datetime('2025-05-10')

                            # 구간별 임계치
                            viral = 0
                            d = first_date
                            if pd.to_datetime('2025-05-10') <= d < pd.to_datetime('2025-05-18'):
                                viral = 1 if view_inc >= 100000 else 0
                            elif pd.to_datetime('2025-05-18') <= d < pd.to_datetime('2025-05-25'):
                                viral = 1 if view_inc >= 80000 else 0
                            elif pd.to_datetime('2025-05-25') <= d < pd.to_datetime('2025-06-01'):
                                viral = 1 if view_inc >= 60000 else 0
                            elif pd.to_datetime('2025-06-01') <= d < pd.to_datetime('2025-06-08'):
                                viral = 1 if view_inc >= 50000 else 0
                            elif pd.to_datetime('2025-06-08') <= d < pd.to_datetime('2025-06-15'):
                                viral = 1 if view_inc >= 40000 else 0
                            elif pd.to_datetime('2025-06-15') <= d < pd.to_datetime('2025-06-22'):
                                viral = 1 if view_inc >= 30000 else 0
                            elif pd.to_datetime('2025-06-22') <= d < pd.to_datetime('2025-06-30'):
                                viral = 1 if view_inc >= 20000 else 0
                        else:
                            viral = 0

                        labels.append(int(viral))

        self.src_nodes = np.asarray(src_nodes, dtype=np.int64)
        self.dst_nodes = np.asarray(dst_nodes, dtype=np.int64)
        self.edge_features = np.asarray(edge_features, dtype=np.float32)
        self.timestamps = np.asarray(ts_list, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.edge_source_types = edge_source_types

        print(f"생성된 그래프: {len(self.src_nodes)}개 엣지, {self.num_nodes}개 노드")
        pos = int(np.sum(self.labels))
        total = len(self.labels)
        rate = (pos / total * 100.0) if total > 0 else 0.0
        print(f"바이럴 라벨 분포: {pos}/{total} ({rate:.1f}%)")

        st_counts = pd.Series(self.edge_source_types).value_counts()
        print("\nsource_type별 엣지 분포 (상위 10개):")
        print(st_counts.head(10))

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
            self.labels[idx],
        )


# ---------------------------------------------------------------------
# 3) TGN Memory: register_buffer 사용 + eval 모드에서는 업데이트 금지
# ---------------------------------------------------------------------
class SourceBasedTGNMemory(nn.Module):
    def __init__(self, num_nodes, memory_dim, message_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_nodes, memory_dim) * 0.1, requires_grad=False)
        self.register_buffer('last_update', torch.zeros(num_nodes))
        self.gru = nn.GRUCell(message_dim, memory_dim)
        self.message_encoder = nn.Sequential(
            nn.Linear(memory_dim * 2 + 3, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

    def forward(self, src_nodes, dst_nodes, edge_feat, timestamps):
        # 검증/평가 단계에서는 메모리 업데이트 금지
        if not self.training:
            return

        batch_size = len(src_nodes)
        src_mem = self.memory[src_nodes]
        dst_mem = self.memory[dst_nodes]
        messages = self.message_encoder(torch.cat([src_mem, dst_mem, edge_feat], dim=1))

        for i in range(batch_size):
            s_idx = src_nodes[i]
            d_idx = dst_nodes[i]
            msg_i = messages[i].unsqueeze(0)

            prev_s = self.memory[s_idx].detach().unsqueeze(0)
            upd_s = self.gru(msg_i, prev_s).squeeze(0)
            self.memory.data[s_idx] = upd_s
            self.last_update[s_idx] = float(timestamps[i].item() if torch.is_tensor(timestamps[i]) else timestamps[i])

            prev_d = self.memory[d_idx].detach().unsqueeze(0)
            upd_d = self.gru(msg_i, prev_d).squeeze(0)
            self.memory.data[d_idx] = upd_d
            self.last_update[d_idx] = float(timestamps[i].item() if torch.is_tensor(timestamps[i]) else timestamps[i])

    def reset_memory(self):
        self.memory.data = torch.randn_like(self.memory.data) * 0.1
        self.last_update.zero_()


# ---------------------------------------------------------------------
# 4) Graph Transformer Layer
# ---------------------------------------------------------------------
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
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


# ---------------------------------------------------------------------
# 5) SCN Layer: 단위행렬이면 matmul 생략(assume_identity_adj 옵션)
# ---------------------------------------------------------------------
class SourceBasedSCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3, assume_identity_adj: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.assume_identity_adj = assume_identity_adj

    def forward(self, x, adj):
        # x: (N, D), adj: (N, N)
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        if adj.dim() != 2:
            adj = adj.view(adj.size(0), -1)

        if self.assume_identity_adj:
            support = x
        else:
            # 한번만 체크하고 싶다면 외부에서 flag를 넘겨주는게 베스트
            is_identity = (adj.shape[0] == adj.shape[1]) and torch.allclose(
                adj, torch.eye(adj.shape[0], device=adj.device)
            )
            support = x if is_identity else torch.mm(adj, x)

        out = self.linear(support)
        out = self.norm(out)
        return self.dropout(F.relu(out))


# ---------------------------------------------------------------------
# 6) Edge Predictor
# ---------------------------------------------------------------------
class SourceBasedEdgePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, src_emb, dst_emb):
        h = torch.cat([src_emb, dst_emb], dim=-1)
        return self.mlp(h).squeeze(-1)


# ---------------------------------------------------------------------
# 7) 전체 모델
# ---------------------------------------------------------------------
class SourceBasedGT_TGN_SocialViral(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_feat_dim,
        memory_dim=32,
        message_dim=32,
        scn_dim=16,
        num_gt_layers=1,
        assume_identity_adj: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory = SourceBasedTGNMemory(num_nodes, memory_dim, message_dim)
        self.node_emb = nn.Linear(node_feat_dim, memory_dim)
        self.gt_layers = nn.ModuleList([
            SourceBasedGraphTransformerLayer(memory_dim, heads=4, ffn_dim=64, dropout=0.3)
            for _ in range(num_gt_layers)
        ])
        self.scn = SourceBasedSCNLayer(memory_dim, scn_dim, dropout=0.3, assume_identity_adj=assume_identity_adj)
        self.edge_pred = SourceBasedEdgePredictor(scn_dim, hidden_dim=32)

    def forward(self, src, dst, edge_feat, timestamps, node_features, adj):
        batch_size = len(src)

        # 노드 임베딩
        node_embs = self.node_emb(node_features)  # (N, mem_dim)

        # 학습시에만 메모리 업데이트
        self.memory(src, dst, edge_feat, timestamps)

        # Graph Transformer: 각 배치 엣지에 대해 동일한 노드 임베딩을 복사
        gt_input = node_embs.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, D)
        gt_out = gt_input
        for gt_layer in self.gt_layers:
            gt_out = gt_layer(gt_out)

        # SCN: 배치별로 독립 처리
        scn_out = torch.stack([self.scn(gt_out[i], adj) for i in range(gt_out.size(0))], dim=0)  # (B, N, scn_dim)

        # 엣지별 임베딩 선택
        batch_indices = torch.arange(src.size(0), device=src.device)
        src_emb = scn_out[batch_indices, src]
        dst_emb = scn_out[batch_indices, dst]

        # 로짓 출력
        edge_logit = self.edge_pred(src_emb, dst_emb)
        return edge_logit

    def reset_memory(self):
        self.memory.reset_memory()


# ---------------------------------------------------------------------
# 8) 데이터 로딩/전처리
# ---------------------------------------------------------------------
def source_based_load_and_prepare_data(past_file: str, future_file: str):
    print("source_type 기반 데이터 로딩 중...")
    past_df = pd.read_csv(past_file, encoding='utf-8-sig')
    future_df = pd.read_csv(future_file, encoding='utf-8-sig')

    # 컬럼명 정리
    past_df.columns = [c.strip() for c in past_df.columns]
    future_df.columns = [c.strip() for c in future_df.columns]

    if 'source_type' not in past_df.columns:
        raise ValueError("past_df에 source_type 컬럼이 없습니다.")

    # 수치형 변환
    for c in ['views', 'likes', 'comments']:
        if c in past_df.columns:
            past_df[c] = pd.to_numeric(past_df[c], errors='coerce').fillna(0)
        if c in future_df.columns:
            future_df[c] = pd.to_numeric(future_df[c], errors='coerce').fillna(0)

    # timestamp 정규화
    if 'timestamp' not in past_df.columns:
        past_df['timestamp'] = np.arange(len(past_df))
    else:
        past_df['original_date'] = pd.to_datetime(past_df['timestamp'], errors='coerce')
        base = past_df['original_date'].min()
        past_df['timestamp'] = (past_df['original_date'] - base).dt.days
        if isinstance(past_df['timestamp'], pd.Series):
            past_df['timestamp'] = past_df['timestamp'].fillna(0)

    print(f"과거 데이터: {len(past_df)}개 영상")
    print(f"미래 데이터: {len(future_df)}개 영상")
    print(f"source_type 종류: {past_df['source_type'].nunique()}개")
    return past_df, future_df


def build_node_features_from_df(dataset: SourceBasedSocialMediaGraphDataset, past_df: pd.DataFrame) -> torch.Tensor:
    node_feature_cols = ['views', 'likes', 'comments']
    features = []
    for idx in range(dataset.num_nodes):
        vid = dataset.node_to_video[idx]
        row = past_df[past_df['video_id'] == vid]
        if len(row) > 0:
            vals = [float(row.iloc[0][c]) if (c in row.columns and pd.notnull(row.iloc[0][c])) else 0.0 for c in node_feature_cols]
        else:
            vals = [0.0, 0.0, 0.0]
        features.append(vals)
    features = np.asarray(features, dtype=np.float32)
    features = StandardScaler().fit_transform(features)
    x = torch.tensor(features, dtype=torch.float32)
    # 모델이 10차원 입력을 기대하면 padding
    if x.shape[1] < 10:
        pad = torch.zeros((x.shape[0], 10 - x.shape[1]), dtype=torch.float32)
        x = torch.cat([x, pad], dim=1)
    return x


# ---------------------------------------------------------------------
# 10) 학습 파이프라인
# ---------------------------------------------------------------------
def source_based_train_gt_tgn_social(
    dataset: SourceBasedSocialMediaGraphDataset,
    node_features: torch.Tensor,
    adj: torch.Tensor,
    num_nodes: int,
    device: str = 'cpu',
    epochs: int = 100,
    train_batch_size: int = 16,
    val_batch_size: int = 16,
    lr: float = 2e-3,
    assume_identity_adj: bool = True,
):
    print("Source-based GT-TGN+SCN 모델 학습 시작...")

    # 데이터 분할
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    pin = (device == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=val_batch_size,   shuffle=False, pin_memory=pin, num_workers=0)

    # 모델 초기화
    model = SourceBasedGT_TGN_SocialViral(
        num_nodes=num_nodes,
        node_feat_dim=node_features.shape[1],
        memory_dim=32,
        message_dim=32,
        scn_dim=16,
        num_gt_layers=1,
        assume_identity_adj=assume_identity_adj,
    ).to(device)

    # 손실/최적화
    pos = float(np.sum(dataset.labels))
    neg = float(len(dataset.labels) - pos)
    pos_weight_val = neg / max(pos, 1e-6)
    pos_weight = torch.tensor([pos_weight_val], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_val_auc = 0.0
    train_losses, val_metrics = [], []

    # 고정 텐서(디바이스에 미리 올려둠)
    node_features_dev = node_features.to(device, non_blocking=True)
    adj_dev = adj.to(device, non_blocking=True)

    for epoch in range(epochs):
        # -------------------- Train --------------------
        model.train()
        model.reset_memory()
        total_loss = 0.0

        for src, dst, edge_feat, ts, labels in train_loader:
            src = src.to(device, non_blocking=True)
            dst = dst.to(device, non_blocking=True)
            edge_feat = edge_feat.float().to(device, non_blocking=True)
            ts = ts.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            # 데이터 증강(노드 특성만)
            node_features_aug = augment_features(node_features_dev, noise_std=0.01)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                pred = model(src, dst, edge_feat, ts, node_features_aug, adj_dev)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=15.0, neginf=-15.0)
                loss = criterion(pred, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())

        # -------------------- Validation --------------------
        model.eval()  # 메모리 업데이트 안 함
        val_preds, val_labels = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            for src, dst, edge_feat, ts, labels in val_loader:
                src = src.to(device, non_blocking=True)
                dst = dst.to(device, non_blocking=True)
                edge_feat = edge_feat.float().to(device, non_blocking=True)
                ts = ts.float().to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                pred = model(src, dst, edge_feat, ts, node_features_dev, adj_dev)
                pred = torch.sigmoid(torch.nan_to_num(pred, nan=0.0))
                val_preds.append(pred.float().cpu().numpy())
                val_labels.append(labels.float().cpu().numpy())

        val_preds = np.concatenate(val_preds) if len(val_preds) > 0 else np.array([])
        val_labels = np.concatenate(val_labels) if len(val_labels) > 0 else np.array([])

        # 진단 로그
        if val_preds.size > 0:
            vb = (val_preds > 0.5).astype(int)
            print(f"[진단] Val label 분포: {np.bincount(val_labels.astype(int))}")
            print(f"[진단] Val 예측  분포: {np.bincount(vb)}")
            print(f"[진단] Val 예측 확률 평균: {np.mean(val_preds):.4f}")

            # 지표 계산 (AUC 예외 안전)
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except Exception:
                val_auc = 0.5  # 모든 값이 동일하거나 단일 클래스인 경우

            val_acc = accuracy_score(val_labels, vb)
            val_prec = precision_score(val_labels, vb, zero_division=0)
            val_rec = recall_score(val_labels, vb, zero_division=0)
            val_f1 = f1_score(val_labels, vb, zero_division=0)
        else:
            val_auc = 0.5
            val_acc = val_prec = val_rec = val_f1 = 0.0

        epoch_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(epoch_loss)
        val_metrics.append({'auc': val_auc, 'accuracy': val_acc, 'precision': val_prec, 'recall': val_rec, 'f1': val_f1})

        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_source_gt_tgn_social.pth')

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Prec: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")

        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f"최고 검증 AUC: {best_val_auc:.4f}")
    return model, train_losses, val_metrics


# ---------------------------------------------------------------------
# 11) 평가 및 저장
# ---------------------------------------------------------------------
def source_based_evaluate_and_save_results(
    model: SourceBasedGT_TGN_SocialViral,
    dataset: SourceBasedSocialMediaGraphDataset,
    node_features: torch.Tensor,
    adj: torch.Tensor,
    past_df: pd.DataFrame,
    future_df: pd.DataFrame,
    device: str = 'cpu',
    batch_size: int = 128,
    out_csv: str = 'source_gt_tgn_social_viral_results.csv',
):
    print("Source-based 예측 결과 평가 중...")
    model.eval()

    pin = (device == 'cuda')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=0)

    node_features_dev = node_features.to(device, non_blocking=True)
    adj_dev = adj.to(device, non_blocking=True)

    all_preds, all_labels, all_src, all_dst = [], [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == 'cuda')):
        for src, dst, edge_feat, ts, labels in loader:
            src = src.to(device, non_blocking=True)
            dst = dst.to(device, non_blocking=True)
            edge_feat = edge_feat.float().to(device, non_blocking=True)
            ts = ts.float().to(device, non_blocking=True)

            pred = model(src, dst, edge_feat, ts, node_features_dev, adj_dev)
            prob = torch.sigmoid(torch.nan_to_num(pred, nan=0.0)).float().cpu().numpy()

            all_preds.append(prob)
            all_labels.append(labels.numpy())
            all_src.append(src.cpu().numpy())
            all_dst.append(dst.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_src = np.concatenate(all_src)
    all_dst = np.concatenate(all_dst)

    # 성능
    vb = (all_preds > 0.5).astype(int)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.5
    acc = accuracy_score(all_labels, vb)
    prec = precision_score(all_labels, vb, zero_division=0)
    rec = recall_score(all_labels, vb, zero_division=0)
    f1 = f1_score(all_labels, vb, zero_division=0)

    print("\n=== Source-based GT-TGN+SCN 모델 성능 ===")
    print(f"정확도: {acc:.4f}")
    print(f"정밀도: {prec:.4f}")
    print(f"재현율: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n=== 상세 분류 보고서 ===")
    print(classification_report(all_labels, vb))

    # 저장용 DF
    results_df = pd.DataFrame({
        'src_video_id': [dataset.node_to_video[s] for s in all_src],
        'dst_video_id': [dataset.node_to_video[d] for d in all_dst],
        'prediction_probability': all_preds,
        'actual_viral': all_labels,
        'predicted_viral': vb,
    })

    # source_type 정보 부가
    src_types, dst_types = [], []
    pmap = past_df[['video_id', 'source_type']].drop_duplicates().set_index('video_id')['source_type'].to_dict()
    for svid, dvid in zip(results_df['src_video_id'], results_df['dst_video_id']):
        src_types.append(pmap.get(svid, 'Unknown'))
        dst_types.append(pmap.get(dvid, 'Unknown'))
    results_df['src_source_type'] = src_types
    results_df['dst_source_type'] = dst_types

    results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n결과가 '{out_csv}'에 저장되었습니다.")

    # source_type별 간단 성능
    print("\n=== Source-type별 성능 분석 (src 기준) ===")
    for st in results_df['src_source_type'].unique():
        if st == 'Unknown':
            continue
        mask = (results_df['src_source_type'] == st)
        if mask.sum() <= 10:
            continue
        sub = results_df[mask]
        try:
            sub_auc = roc_auc_score(sub['actual_viral'], sub['prediction_probability'])
        except Exception:
            sub_auc = 0.5
        sub_acc = accuracy_score(sub['actual_viral'], sub['predicted_viral'])
        print(f"{st}: AUC={sub_auc:.4f}, Acc={sub_acc:.4f} (n={len(sub)})")

    return results_df


# ---------------------------------------------------------------------
# 12) 메인
# ---------------------------------------------------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")

    try:
        # 1) 파일 경로
        past_csv = 'delete_node(5.10-6.30).csv'
        future_csv = 'delete_current(7.3).csv'

        # 2) 데이터 로딩
        past_df, future_df = source_based_load_and_prepare_data(past_csv, future_csv)

        # 3) 하이퍼파라미터
        max_nodes = 1000           # 1000 / 2000 / None(전체) 로 바꿔가며 실험 가능
        time_windows = 5
        epochs = 100
        train_batch = 16
        val_batch = 16
        lr = 2e-3
        assume_identity_adj = True  # adj = I 라면 True 유지 (SCN 빠른 경로)

        print(f"선택된 노드 수(목표): {max_nodes if max_nodes else '전체'}")
        print(f"시간 윈도우: {time_windows}")
        print(f"학습 에폭: {epochs} | 배치(학습/검증): {train_batch}/{val_batch} | 학습률: {lr}")

        # 4) 데이터셋 생성 (views 상위 우선 선택; 랜덤 샘플은 random_sample=True)
        dataset = SourceBasedSocialMediaGraphDataset(
            past_df=past_df,
            future_df=future_df,
            time_windows=time_windows,
            max_nodes=max_nodes,
            random_sample=False,   # 재현성 위해 False 권장
            top_by_views=True
        )

        # 5) 노드 특성/인접행렬
        node_features = build_node_features_from_df(dataset, past_df)
        adj = torch.eye(dataset.num_nodes).float()  # 단위행렬 (SCN에서 빠른 경로 사용)

        print(f"노드 특성 차원: {tuple(node_features.shape)}")
        print(f"인접 행렬 차원: {tuple(adj.shape)}")

        # 6) 학습
        model, train_losses, val_metrics = source_based_train_gt_tgn_social(
            dataset=dataset,
            node_features=node_features,
            adj=adj,
            num_nodes=dataset.num_nodes,
            device=device,
            epochs=epochs,
            train_batch_size=train_batch,
            val_batch_size=val_batch,
            lr=lr,
            assume_identity_adj=assume_identity_adj,
        )

        # 7) 평가/저장
        out_csv = f"source_gt_tgn_social_viral_results({dataset.num_nodes}).csv"
        results_df = source_based_evaluate_and_save_results(
            model=model,
            dataset=dataset,
            node_features=node_features,
            adj=adj,
            past_df=past_df,
            future_df=future_df,
            device=device,
            batch_size=128,
            out_csv=out_csv,
        )

        # 요약
        print("\n=== 최종 요약 ===")
        print(f"총 엣지 수: {len(results_df)}")
        print(f"바이럴 라벨 분포: {results_df['actual_viral'].sum()}/{len(results_df)} "
              f"({results_df['actual_viral'].mean()*100:.1f}%)")
        print(f"예측 바이럴 분포: {results_df['predicted_viral'].sum()}/{len(results_df)} "
              f"({results_df['predicted_viral'].mean()*100:.1f}%)")

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
