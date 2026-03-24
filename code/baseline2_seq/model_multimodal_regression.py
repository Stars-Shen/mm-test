from __future__ import annotations

"""
多模态回归模型（Baseline-2）
- 模态内 encoder（RR/ACT/SLEEP）
- late fusion（concat + MLP）
- 回归头（分数预测）

支持两种输入：
1) structured: [B, D, H, F]
2) event:      [B, D, T, F]
"""

from typing import Dict

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(1e-6)
    return num / den


def masked_attention_pooling(x: torch.Tensor, mask: torch.Tensor, scorer: nn.Module) -> torch.Tensor:
    """x: [N, L, E], mask: [N, L] -> [N, E]"""
    score = scorer(x).squeeze(-1)
    score = score.masked_fill(mask <= 0.5, -1e9)
    alpha = torch.softmax(score, dim=1)
    alpha = alpha * (mask > 0.5).to(alpha.dtype)
    alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pooled = (x * alpha.unsqueeze(-1)).sum(dim=1)
    return pooled


class StructuredDayEncoder(nn.Module):
    """[B,D,H,F] -> [B,D,E]，Linear + hour embedding + 1D CNN + masked attention pooling"""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        max_hour: int = 48,
        cnn_layers: int = 2,
        cnn_kernel_size: int = 3,
        use_hour_emb: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_hour_emb = use_hour_emb
        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.hour_emb = nn.Embedding(max_hour, embed_dim)
        self.pre_norm = nn.LayerNorm(embed_dim)

        cnn_layers = max(1, int(cnn_layers))
        cnn_kernel_size = max(1, int(cnn_kernel_size))
        padding = cnn_kernel_size // 2
        cnn_blocks = []
        for _ in range(cnn_layers):
            cnn_blocks.append(nn.Conv1d(embed_dim, embed_dim, kernel_size=cnn_kernel_size, padding=padding))
            cnn_blocks.append(nn.ReLU())
        self.cnn = nn.Sequential(*cnn_blocks)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor, hour_mask: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
        b, d, h, _ = x.shape
        n = b * d

        z = self.in_proj(x)  # [B,D,H,E]
        if self.use_hour_emb:
            hour_idx = torch.arange(h, device=x.device, dtype=torch.long).clamp_max(self.hour_emb.num_embeddings - 1)
            z = z + self.hour_emb(hour_idx).view(1, 1, h, self.embed_dim)
        z = self.pre_norm(z)

        z = z.reshape(n, h, self.embed_dim)
        hm = hour_mask.reshape(n, h)

        z = self.cnn(z.transpose(1, 2)).transpose(1, 2)  # [N,H,E]
        z = masked_attention_pooling(z, hm, self.scorer)  # [N,E]
        z = z.reshape(b, d, self.embed_dim)
        z = z * day_mask.unsqueeze(-1)
        return z


class EventDayEncoder(nn.Module):
    """[B,D,T,F] -> [B,D,E]，Linear + time feat proj + Transformer + masked attention pooling"""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        time_feat_dim: int = 2,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_time_feat: bool = True,
        max_tokens: int = 2048,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_time_feat = use_time_feat
        self.max_tokens = max(1, int(max_tokens))
        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.time_proj = nn.Linear(time_feat_dim, embed_dim)
        self.pre_norm = nn.LayerNorm(embed_dim)

        ff_mult = max(1, int(ff_mult))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor, step_mask: torch.Tensor, day_mask: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        b, d, t, _ = x.shape
        n = b * d

        if t == 0:   #对于batch = 1的情况下某个模态全为0，即T = 0 的情况
            return x.new_zeros((b, d, self.embed_dim))

        # RR token可能非常长（~6w），先做均匀下采样避免Transformer O(T^2)爆炸。
        if t > self.max_tokens:
            idx = torch.linspace(0, t - 1, steps=self.max_tokens, device=x.device).long()
            x = x[:, :, idx, :]
            step_mask = step_mask[:, :, idx]
            time_feat = time_feat[:, :, idx, :]
            t = self.max_tokens

        z = self.in_proj(x)
        if self.use_time_feat:
            z = z + self.time_proj(time_feat)
        z = self.pre_norm(z)

        z = z.reshape(n, t, self.embed_dim)
        sm = step_mask.reshape(n, t)

        # 避免某些行“全padding”时 Transformer 产生 NaN
        has_token = sm.sum(dim=1) > 0
        sm_for_encoder = sm.clone()
        if (~has_token).any() and t > 0:
            sm_for_encoder[~has_token, 0] = 1.0
            z = z.clone()
            z[~has_token, 0, :] = 0.0

        key_padding_mask = sm_for_encoder <= 0.5

        z = self.encoder(z, src_key_padding_mask=key_padding_mask)
        z = masked_attention_pooling(z, sm, self.scorer)
        z = z.reshape(b, d, self.embed_dim)
        z = z * day_mask.unsqueeze(-1)
        return z


class LateFusionRegressor(nn.Module):
    """[B,D,E]*3 -> [B,D]"""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, temporal_layers: int = 1):
        super().__init__()
        in_dim = embed_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.temporal_layers = max(0, int(temporal_layers))
        if self.temporal_layers > 0:
            self.day_gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=self.temporal_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.reg_head = nn.Linear(hidden_dim * 2, 1)
        else:
            self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, rr_e: torch.Tensor, act_e: torch.Tensor, sleep_e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rr_e, act_e, sleep_e], dim=-1)
        h = self.fusion(x)
        if self.temporal_layers > 0:
            h, _ = self.day_gru(h)
        y = self.reg_head(h).squeeze(-1)
        return y


class MultiModalStructuredRegressor(nn.Module):
    def __init__(
        self,
        rr_dim: int,
        act_dim: int,
        sleep_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        structured_max_hour: int = 48,
        structured_cnn_layers: int = 2,
        structured_cnn_kernel_size: int = 3,
        structured_use_hour_emb: bool = True,
        fusion_temporal_layers: int = 1,
    ):
        super().__init__()
        self.rr_encoder = StructuredDayEncoder(rr_dim, embed_dim, structured_max_hour, structured_cnn_layers, structured_cnn_kernel_size, structured_use_hour_emb)
        self.act_encoder = StructuredDayEncoder(act_dim, embed_dim, structured_max_hour, structured_cnn_layers, structured_cnn_kernel_size, structured_use_hour_emb)
        self.sleep_encoder = StructuredDayEncoder(sleep_dim, embed_dim, structured_max_hour, structured_cnn_layers, structured_cnn_kernel_size, structured_use_hour_emb)
        self.fusion_head = LateFusionRegressor(embed_dim, hidden_dim, temporal_layers=fusion_temporal_layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rr_hour_mask = batch.get('rr_hour_mask', batch['rr_mask'].unsqueeze(-1).expand(-1, -1, batch['rr'].shape[2]))
        act_hour_mask = batch.get('act_hour_mask', batch['act_mask'].unsqueeze(-1).expand(-1, -1, batch['act'].shape[2]))
        sleep_hour_mask = batch.get('sleep_hour_mask', batch['sleep_mask'].unsqueeze(-1).expand(-1, -1, batch['sleep'].shape[2]))

        rr_e = self.rr_encoder(batch['rr'], rr_hour_mask, batch['rr_mask'])
        act_e = self.act_encoder(batch['act'], act_hour_mask, batch['act_mask'])
        sleep_e = self.sleep_encoder(batch['sleep'], sleep_hour_mask, batch['sleep_mask'])
        pred = self.fusion_head(rr_e, act_e, sleep_e)
        return {'pred': pred, 'rr_emb': rr_e, 'act_emb': act_e, 'sleep_emb': sleep_e}


class MultiModalEventRegressor(nn.Module):
    def __init__(
        self,
        rr_dim: int,
        act_dim: int,
        sleep_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        event_time_feat_dim: int = 2,
        event_n_heads: int = 4,
        event_n_layers: int = 2,
        event_ff_mult: int = 4,
        event_dropout: float = 0.1,
        event_use_time_feat: bool = True,
        event_max_tokens: int = 2048,
        fusion_temporal_layers: int = 1,
    ):
        super().__init__()
        self.rr_encoder = EventDayEncoder(rr_dim, embed_dim, event_time_feat_dim, event_n_heads, event_n_layers, event_ff_mult, event_dropout, event_use_time_feat, max_tokens=event_max_tokens)
        self.act_encoder = EventDayEncoder(act_dim, embed_dim, event_time_feat_dim, event_n_heads, event_n_layers, event_ff_mult, event_dropout, event_use_time_feat, max_tokens=event_max_tokens)
        self.sleep_encoder = EventDayEncoder(sleep_dim, embed_dim, event_time_feat_dim, event_n_heads, event_n_layers, event_ff_mult, event_dropout, event_use_time_feat, max_tokens=event_max_tokens)
        self.fusion_head = LateFusionRegressor(embed_dim, hidden_dim, temporal_layers=fusion_temporal_layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rr_time_feat = batch.get('rr_time_feat', torch.zeros((*batch['rr'].shape[:3], 2), device=batch['rr'].device, dtype=batch['rr'].dtype))
        act_time_feat = batch.get('act_time_feat', torch.zeros((*batch['act'].shape[:3], 2), device=batch['act'].device, dtype=batch['act'].dtype))
        sleep_time_feat = batch.get('sleep_time_feat', torch.zeros((*batch['sleep'].shape[:3], 2), device=batch['sleep'].device, dtype=batch['sleep'].dtype))

        rr_e = self.rr_encoder(batch['rr'], batch['rr_step_mask'], batch['rr_day_mask'], rr_time_feat)
        act_e = self.act_encoder(batch['act'], batch['act_step_mask'], batch['act_day_mask'], act_time_feat)
        sleep_e = self.sleep_encoder(batch['sleep'], batch['sleep_step_mask'], batch['sleep_day_mask'], sleep_time_feat)
        pred = self.fusion_head(rr_e, act_e, sleep_e)
        return {'pred': pred, 'rr_emb': rr_e, 'act_emb': act_e, 'sleep_emb': sleep_e}
