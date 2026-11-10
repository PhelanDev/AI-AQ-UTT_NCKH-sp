"""
HIERARCHICAL UNIFIED CENTRAL AI MODEL (LITE GRU BACKBONE + DUAL HEADS)
Mô hình AI Phân Tầng Thống Nhất Tinh Giản cho Bản Sinh Viên AQ_Lite
Phát triển cho Máy chủ Trung tâm Raspberry Pi 5

Kiến trúc Chuẩn mực & Tinh gọn (Ponytail Style):
- Backbone: Vanilla GRU 1 tầng (32 hidden units) xử lý chuỗi 24 bước thời gian.
- Head 1 (Anomaly): Tái tạo đặc trưng thời điểm hiện tại (16 dims).
- Head 2 (Forecast): Dự báo chuỗi thời gian 24 giờ cho PM2.5 & AQI (24x2 dims).
- Tổng số tham số: ~6.500 params (giảm 85% so với bản 43k params), suy luận < 0.5 ms.
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalCentralAI(nn.Module):
    """
    Kiến trúc AI Phân Tầng Thống Nhất Bản Lite:
    - Shared Backbone: GRU 1 tầng (in_features -> 32)
    - Anomaly Reconstruction Head: Linear(32 -> in_features)
    - 24h Forecasting Head: Linear(32 -> 24 * 2)
    """

    def __init__(
        self,
        in_features: int = 16,
        tcn_channels: int = 32,  # Giữ lại tham số để tương thích ngược 100%
        gru_hidden: int = 32,
        out_steps: int = 24,
        out_targets: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.gru_hidden = gru_hidden
        self.out_steps = out_steps
        self.out_targets = out_targets

        # 1. Shared Backbone: Vanilla GRU 1 tầng
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # 2. Anomaly Reconstruction Head
        self.recon_head = nn.Sequential(
            nn.Linear(gru_hidden, in_features),
        )

        # 3. 24h Forecasting Head
        self.forecast_head = nn.Sequential(
            nn.Linear(gru_hidden, out_steps * out_targets),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass đa nhiệm:
        Args:
            x: Tensor đầu vào shape (B, 24, 16) chuỗi thời gian 24h qua.
        Returns:
            forecast: Tensor dự báo tương lai shape (B, 24, 2) cho PM2.5 & AQI.
            recon: Tensor tái tạo thời điểm hiện tại shape (B, 16) để phát hiện dị thường.
        """
        b, t_in, f_dim = x.shape

        # Đưa qua GRU Backbone
        gru_out, _ = self.gru(x)  # shape (B, 24, gru_hidden)
        last_context = gru_out[:, -1, :]  # shape (B, gru_hidden)

        # Nhánh 1: Tái tạo vector 16D hiện tại
        recon_current = self.recon_head(last_context)

        # Nhánh 2: Dự báo 24 giờ tương lai (PM2.5 và AQI)
        raw_forecast = self.forecast_head(last_context)
        forecast_24h = F.relu(raw_forecast.view(b, self.out_steps, self.out_targets))

        return forecast_24h, recon_current
