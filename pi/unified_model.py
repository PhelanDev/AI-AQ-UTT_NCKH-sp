"""
HIERARCHICAL UNIFIED CENTRAL AI MODEL (SHARED BACKBONE + DUAL HEADS) - v2.3 FINAL
Mô hình AI Phân Tầng Thống Nhất kết hợp Tái Tạo Dị Thường & Dự Báo 24h
Phát triển cho Máy chủ Trung tâm Raspberry Pi 5 (Quad-core Cortex-A76)

Các nâng cấp đột phá:
1. Spatial Feature Dropout (p=0.15): Chống học vẹt cảm biến cục bộ.
2. Dual-Pooling Context: Kết hợp 60% bước cuối (nhạy tức thời) + 40% trung bình toàn chuỗi 24h (ổn định xu thế dài hạn).
3. BiGRU Cắt Gọt (hidden=32): Giảm 60% tham số (42.952 tham số), loại bỏ dung lượng học vẹt.
4. Linear Diurnal Residual + Unbounded Delta: Loại bỏ Dying ReLU, gradient mượt mà tối đa.
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pi.forecasting_engine import ChausalDilatedConvBlock, TemporalAttention, SeriesDecomp

class HierarchicalCentralAI(nn.Module):
    """
    Kiến trúc AI Phân Tầng Thống Nhất (Hierarchical Unified Central AI v3.1):
    - TẦNG 0 (Shared Backbone): Dilated TCN + Bi-GRU (32) + Temporal Attention + Dual-Pooling
    - TẦNG 1 (Anomaly Head): Tái tạo đặc trưng thời điểm hiện tại x_t (16 dims)
    - TẦNG 2 (Forecast Head): RevIN + DLinear Trend/Seasonal + Non-linear Residual Delta (24x2 dims)
    """

    def __init__(
        self,
        in_features: int = 16,
        tcn_channels: int = 32,
        gru_hidden: int = 32,
        out_steps: int = 24,
        out_targets: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_steps = out_steps
        self.out_targets = out_targets
        self.eps = 1e-5

        # === TẦNG 0: SHARED BACKBONE ===
        self.feature_drop = nn.Dropout1d(p=0.15)
        self.input_proj = nn.Linear(in_features, tcn_channels)
        self.tcn1 = ChausalDilatedConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=1)
        self.tcn2 = ChausalDilatedConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2)
        self.tcn3 = ChausalDilatedConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=4)

        self.bigru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(gru_hidden * 2)
        self.context_drop = nn.Dropout(p=0.25)

        # === TẦNG 1: ANOMALY RECONSTRUCTION HEAD ===
        self.recon_head = nn.Sequential(
            nn.Linear(gru_hidden * 2, 32),
            nn.SiLU(),
            nn.Dropout(p=0.20),
            nn.Linear(32, in_features),
        )

        # === TẦNG 2: 24H FORECASTING HEAD (RevIN + DLinear Trend-Seasonal + Multimodal Delta) ===
        self.decomp = SeriesDecomp(kernel_size=5)
        self.linear_trend = nn.Linear(out_steps, out_steps)
        self.linear_seasonal = nn.Linear(out_steps, out_steps)

        # Khởi tạo trọng số DLinear ưu tiên suy giảm hàm mũ (AR Exponential Decay)
        with torch.no_grad():
            self.linear_trend.weight.zero_()
            self.linear_trend.bias.zero_()
            self.linear_seasonal.weight.zero_()
            self.linear_seasonal.bias.zero_()
            for h in range(out_steps):
                decay = math.exp(-h / 12.0)
                self.linear_trend.weight[h, out_steps - 1] = decay
                self.linear_seasonal.weight[h, h % out_steps] += (1.0 - decay)

        self.forecast_head = nn.Sequential(
            nn.Linear(gru_hidden * 2, 48),
            nn.SiLU(),
            nn.Dropout(p=0.20),
            nn.Linear(48, out_steps * out_targets),
        )
        nn.init.zeros_(self.forecast_head[-1].bias)
        self.forecast_head[-1].weight.data.mul_(0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass đa nhiệm v3.1:
        Args:
            x: Tensor đầu vào shape (B, 24, 16) chuỗi thời gian 24h qua.
        Returns:
            forecast: Tensor dự báo tương lai shape (B, 24, 2) cho PM2.5 & AQI.
            recon: Tensor tái tạo thời điểm hiện tại shape (B, 16) để phát hiện dị thường.
        """
        b, t_in, f_dim = x.shape

        # 1. Spatial Feature Dropout (ngắt ngẫu nhiên kênh cảm biến khi train)
        x_trans = x.transpose(1, 2)
        x_drop = self.feature_drop(x_trans).transpose(1, 2)

        # 2. Shared Backbone
        h = self.input_proj(x_drop)
        h_conv = h.transpose(1, 2)
        h_tcn = self.tcn3(self.tcn2(self.tcn1(h_conv))).transpose(1, 2)

        gru_out, _ = self.bigru(h_tcn)
        att_out, _ = self.attention(gru_out)

        # Dual-Pooling: 0.6 bước cuối + 0.4 trung bình toàn chuỗi 24h
        context_z = self.context_drop(0.6 * att_out[:, -1, :] + 0.4 * torch.mean(att_out, dim=1))

        # 3. Nhánh Tái tạo Dị thường (Head 1)
        recon_current = self.recon_head(context_z)

        # 4. Nhánh Dự báo 24h Tương lai (Head 2 với RevIN + DLinear)
        idx_aqi = 15 if self.in_features >= 16 else 12
        x_targets = torch.stack([x[:, :, 0], x[:, :, idx_aqi]], dim=-1)  # (B, 24, 2)

        # Target-RevIN: Chuẩn hóa cửa sổ chống dịch chuyển phân phối theo mùa & vùng
        mu = x_targets.mean(dim=1, keepdim=True)  # (B, 1, 2)
        sigma = torch.sqrt(torch.var(x_targets, dim=1, keepdim=True, unbiased=False) + self.eps)  # (B, 1, 2)
        x_targets_norm = (x_targets - mu) / sigma

        # DLinear Trend & Seasonal Projection
        seasonal, trend = self.decomp(x_targets_norm)
        trend_pred = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)
        seasonal_pred = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        dlinear_base_norm = trend_pred + seasonal_pred  # (B, 24, 2)

        # Multimodal Non-linear Delta từ Backbone Context
        pred_delta_norm = self.forecast_head(context_z).view(b, self.out_steps, self.out_targets)

        # RevIN De-normalization & Physical Bounding
        forecast_norm = dlinear_base_norm + pred_delta_norm
        forecast_24h = forecast_norm * sigma + mu
        forecast_24h = F.relu(forecast_24h)

        return forecast_24h, recon_current
