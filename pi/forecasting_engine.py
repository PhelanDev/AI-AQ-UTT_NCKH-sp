"""
TIME-SERIES FORECASTING NEURAL NETWORK ENGINE (DILATED TCN-BIGRU HYBRID ATTENTION)
Mạng Nơ-ron Chuỗi Thời gian Dự báo Đa bước PM2.5 & AQI (6h - 24h Tới)
Phát triển cho Máy chủ Trung tâm Raspberry Pi 5 (8GB RAM)
Thuộc Đề tài NCKH Quan trắc Chất lượng Không khí Đa điểm TinyML & Edge Server
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class ChausalDilatedConvBlock(nn.Module):
    """Khối Temporal Convolution (TCN) giãn cách thời gian kèm Residual Skip Connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        res = self.residual(x)
        out = self.conv1(x)

        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = out.transpose(1, 2)
        out = self.act1(self.norm1(out))
        out = out.transpose(1, 2)
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.act2(self.norm2(out))
        out = out.transpose(1, 2)
        return out + res

class TemporalAttention(nn.Module):
    """Cơ chế tự chú ý thời gian (Temporal Attention) trích xuất trọng số các thời điểm then chốt."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        q = self.query(h)
        k = self.key(h)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, h)
        return context, weights

class DilatedTCNBiGRUForecaster(nn.Module):
    """
    Kiến trúc Mạng nơ-ron Lai ghép: Dilated TCN + Bidirectional GRU + Attention.
    Dự báo đa bước thời gian (Multi-Horizon: t+1 đến t+24) cho PM2.5 và AQI.
    Tổng số tham số tối ưu (< 100,000 params), tính toán đa luồng mượt mà trên ARM Cortex-A76 Pi 5.
    """

    def __init__(
        self,
        in_features: int = 14,
        tcn_channels: int = 32,
        gru_hidden: int = 64,
        out_steps: int = 24,
        out_targets: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_steps = out_steps
        self.out_targets = out_targets

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

        self.decoder_dense = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, out_steps * out_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor đầu vào shape (B, T_in=24, in_features=14).
        Returns:
            Tensor dự báo shape (B, out_steps=24, out_targets=2) cho PM2.5 và VN-AQI.
        """
        b, t_in, f_dim = x.shape

        h = self.input_proj(x)

        h_conv = h.transpose(1, 2)
        h_tcn1 = self.tcn1(h_conv)
        h_tcn2 = self.tcn2(h_tcn1)
        h_tcn3 = self.tcn3(h_tcn2)
        h_seq = h_tcn3.transpose(1, 2)

        gru_out, _ = self.bigru(h_seq)

        att_out, _ = self.attention(gru_out)

        last_context = att_out[:, -1, :]

        pred_flat = self.decoder_dense(last_context)
        pred = pred_flat.view(b, self.out_steps, self.out_targets)

        return F.relu(pred)

class AsymmetricHuberLoss(nn.Module):
    """
    Hàm tổn thất Huber Bất đối xứng (Asymmetric Huber Loss) chuyên dụng cho chuỗi thời gian ô nhiễm.
    - Phạt nặng hơn khi mô hình dự báo THIẾU các đỉnh ô nhiễm nghiêm trọng (Under-prediction of PM2.5 spikes).
    - Phạt sai hướng biến thiên xu hướng (Trend penalty via Cosine Similarity).
    """

    def __init__(self, delta: float = 1.5, lambda_peak: float = 0.5, lambda_trend: float = 0.2):
        super().__init__()
        self.delta = delta
        self.lambda_peak = lambda_peak
        self.lambda_trend = lambda_trend

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_pred - y_true
        abs_error = torch.abs(error)

        is_small = abs_error <= self.delta
        huber = torch.where(
            is_small,
            0.5 * (abs_error ** 2),
            self.delta * (abs_error - 0.5 * self.delta)
        )

        peak_mask = (y_true > 50.0).float()
        under_pred = F.relu(y_true - y_pred)
        peak_penalty = self.lambda_peak * peak_mask * (under_pred ** 1.5)

        total_loss = torch.mean(huber + peak_penalty)

        if y_pred.shape[1] > 1 and self.lambda_trend > 0:
            diff_pred = y_pred[:, 1:, 0] - y_pred[:, :-1, 0]
            diff_true = y_true[:, 1:, 0] - y_true[:, :-1, 0]
            cos_sim = F.cosine_similarity(diff_pred, diff_true, dim=-1, eps=1e-6)
            trend_loss = self.lambda_trend * torch.mean(1.0 - cos_sim)
            total_loss = total_loss + trend_loss

        return total_loss

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính toán chỉ số kiểm định khoa học R2, RMSE, MAE."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-8 else 1.0
    r2 = max(-1.0, min(1.0, r2))

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    return {"r2": r2, "mae": mae, "rmse": rmse}

def train_central_forecaster(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "ai/models/central_forecaster.pt",
) -> Tuple[DilatedTCNBiGRUForecaster, Dict[str, float]]:
    """
    Huấn luyện mô hình Central Forecaster với dữ liệu chuỗi thời gian đa trạm.
    """
    device = torch.device("cpu")
    in_features = X_train.shape[-1]
    out_steps = Y_train.shape[1]
    out_targets = Y_train.shape[2]

    model = DilatedTCNBiGRUForecaster(
        in_features=in_features,
        tcn_channels=32,
        gru_hidden=64,
        out_steps=out_steps,
        out_targets=out_targets,
    ).to(device)

    criterion = AsymmetricHuberLoss(delta=1.5, lambda_peak=0.5, lambda_trend=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += float(loss.item())

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += float(loss.item())
                all_preds.append(out.cpu().numpy())
                all_trues.append(by.cpu().numpy())

        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)

        all_preds_np = np.concatenate(all_preds, axis=0)
        all_trues_np = np.concatenate(all_trues, axis=0)
        metrics = calculate_metrics(all_trues_np[:, :, 0], all_preds_np[:, :, 0])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return model, best_metrics
